"""TTS reward clients for ``--custom-rm-path slime_plugins.tts_reward.reward``.

``omni_reward_config`` accepts a mapping, inline JSON, or a JSON/YAML file::

    services:
      wer: {endpoint: http://asr:8000}
      sim: {endpoint: http://sim:8000}
    weights: {wer: 0.6, sim: 0.4}

Each enabled service supports ``timeout_s`` (120), ``max_retries`` (0 extra
attempts), ``max_inflight`` (4), ``audio_transport`` (path/upload), and
``max_audio_bytes`` (64 MiB). The rollout owns its audio artifacts; this hook
finishes all outstanding component readers before returning or raising.
Services remain external. Reported model metadata is recorded without identity
comparison. No model loader or deployment runtime is imported here.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import os
import re
import unicodedata
import uuid
import weakref
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import unquote, urlsplit

from slime.utils.http_utils import get, post

PROTOCOL = "voxrl.scoring.v1"
QUERY_VERSION = "moss_tts_v2_revision6_instruction_v1"
NORMALIZATION = "nfkc_casefold_no_punctuation_cjk_char_other_word_v1"
CHAR_LANGUAGES = {"zh", "zho", "chinese", "中文", "ja", "jpn", "japanese", "日语", "ko", "kor", "korean", "韩语"}
INSTRUCTION_FIELDS = (
    "Accent and dialect",
    "Age",
    "Appearance",
    "Applicable scenario",
    "Articulation clarity",
    "Attitude and stance",
    "Cognitive state",
    "Communicative act",
    "Emotion and psychological state",
    "Expression strategy",
    "Gender",
    "Listener context",
    "Occupation identity role",
    "Performance style",
    "Personality traits",
    "Phonation mode",
    "Physical scene",
    "Physical state",
    "Pitch",
    "Pitch range",
    "Prosodic variation",
    "Resonance placement",
    "Rhythm",
    "Roughness",
    "Speed",
    "Timbre brightness",
    "Timbre persona",
    "Timbre thickness",
    "Timbre warmth",
    "Vocal-fold tension",
)
_INSTRUCTION_KEYS = {key.casefold(): key for key in INSTRUCTION_FIELDS}


class ScoringError(RuntimeError):
    """A technical scoring failure; never converted into a numeric reward."""


def _finite(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ScoringError(f"{field} must be a finite number")
    return float(value)


def normalized_tokens(text: str, language: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith(("P", "S")))
    if language.lower().replace("_", "-").split("-")[0] in CHAR_LANGUAGES:
        return [ch for ch in text if not ch.isspace()]
    return text.split()


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, actual in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def instruction_queries(value, *, modality="speech") -> list[str]:
    """Match Prism's rendered global attributes and stable category order."""
    if not isinstance(value, Mapping):
        raise ValueError("JudgeCLAP requires a Prism global_instruction object")
    normalized = {}
    for key, attribute in value.items():
        if not isinstance(key, str):
            raise ValueError("Prism instruction keys must be strings")
        # Language conditions generation; it is not a JudgeCLAP attribute.
        if key.strip().casefold() == "language":
            continue
        category = _INSTRUCTION_KEYS.get(key.strip().casefold())
        if category is None or category in normalized:
            raise ValueError(f"unsupported or duplicate JudgeCLAP instruction field: {key!r}")
        if attribute is None:
            normalized[category] = None
            continue
        if not isinstance(attribute, str):
            raise ValueError(f"instruction.{category} must be a string or null")
        attribute = re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", attribute)).strip()
        pairs = {'"': '"', "'": "'", "“": "”", "‘": "’"}
        if len(attribute) >= 2 and pairs.get(attribute[0]) == attribute[-1]:
            opening, closing = attribute[0], attribute[-1]
            count = 2 if opening == closing else 1
            if attribute.count(opening) == count and attribute.count(closing) == count:
                attribute = attribute[1:-1].strip()
        if re.search(r"<\|[^<>]*\|>", attribute):
            raise ValueError(f"instruction.{category} contains a reserved special token")
        normalized[category] = attribute or None
    fields = sorted(normalized) if modality == "sing" else INSTRUCTION_FIELDS
    queries = [f"{key}: {normalized[key]}" for key in fields if normalized.get(key)]
    if not queries:
        raise ValueError("JudgeCLAP requires at least one rendered Prism global instruction attribute")
    return queries


def _reference(prompt, *, must_exist=True):
    references = prompt.get("references")
    if not isinstance(references, dict):
        raise ValueError("SIM references must map audio slots to URI strings")
    targets = prompt.get("reference_attributes")
    if not isinstance(targets, dict) or not isinstance(targets.get("timbre"), str):
        raise ValueError("SIM requires explicit reference_attributes.timbre")
    if targets["timbre"] == "none" or targets["timbre"] not in references:
        raise ValueError("SIM timbre reference must select an existing audio slot")
    uri = references[targets["timbre"]]
    if not isinstance(uri, str) or not uri.strip():
        raise ValueError("SIM timbre reference must be a nonempty URI string")
    parsed = urlsplit(uri)
    if parsed.scheme == "file" and parsed.netloc in {"", "localhost"}:
        path = Path(unquote(parsed.path))
    elif not parsed.scheme:
        path = Path(uri)
    else:
        raise ValueError("SIM reference URI must be localized before scoring")
    path = path.resolve(strict=must_exist)
    if must_exist and not path.is_file():
        raise ValueError("SIM reference audio must be a regular file")
    return targets["timbre"], path


class _Service:
    def __init__(self, options):
        if not isinstance(options, Mapping):
            raise ValueError("reward service configuration must be an object")
        unknown = set(options) - {
            "endpoint",
            "timeout_s",
            "max_retries",
            "max_inflight",
            "audio_transport",
            "max_audio_bytes",
            "auth_token_env",
        }
        if unknown:
            raise ValueError(f"unknown reward service options: {sorted(unknown)}")
        self.endpoint = options.get("endpoint")
        if not isinstance(self.endpoint, str):
            raise ValueError("reward service endpoint must be an HTTP(S) base URL")
        parsed = urlsplit(self.endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            raise ValueError("reward endpoint must be an HTTP(S) base URL without credentials, query, or fragment")
        auth_env = options.get("auth_token_env")
        self._request_auth = {}
        if auth_env is not None:
            if not isinstance(auth_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", auth_env):
                raise ValueError("auth_token_env must be an environment variable name")
            token = os.environ.get(auth_env)
            if not token or not token.strip():
                raise ValueError(f"Reward token environment {auth_env!r} is missing")
            self._request_auth = {"headers": {"Authorization": f"Bearer {token}"}}
        self.timeout = _finite(options.get("timeout_s", 120), "timeout_s")
        if self.timeout <= 0:
            raise ValueError("timeout_s must be positive")
        retries = options.get("max_retries", 0)
        inflight = options.get("max_inflight", 4)
        self.max_audio_bytes = options.get("max_audio_bytes", 64 * 1024 * 1024)
        for name, value, minimum in (
            ("max_retries", retries, 0),
            ("max_inflight", inflight, 1),
            ("max_audio_bytes", self.max_audio_bytes, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.attempts = retries + 1  # slime's max_retries counts attempts, not additional retries.
        self.transport = options.get("audio_transport", "path")
        if self.transport not in {"path", "upload"}:
            raise ValueError("audio_transport must be path or upload")
        self._info = None
        self._info_lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(inflight)

    def url(self, route):
        return self.endpoint.rstrip("/") + "/" + route

    async def info(self):
        async with self._info_lock:
            if self._info is None:
                body = await asyncio.wait_for(
                    get(self.url("info"), timeout=self.timeout, **self._request_auth), self.timeout
                )
                if not isinstance(body, dict) or body.get("protocol") != PROTOCOL:
                    raise ScoringError("incompatible scoring service protocol")
                identity = body.get("identity")
                if not isinstance(identity, dict):
                    raise ScoringError("scoring service identity must be an object")
                self._info = identity
            return self._info

    def audio(self, path):
        path = Path(path).resolve(strict=True)
        if not path.is_file():
            raise ValueError("audio must be a regular file")
        if self.transport == "path":
            return {"path": str(path)}
        if path.stat().st_size > self.max_audio_bytes:
            raise ValueError("audio exceeds configured upload byte limit")
        return {"data_b64": base64.b64encode(path.read_bytes()).decode("ascii"), "suffix": path.suffix.lower()}

    async def evaluate(self, route, request):
        async def run():
            async with self._slots:
                body = await post(
                    self.url(route), request, max_retries=self.attempts, timeout=self.timeout, **self._request_auth
                )
                if not isinstance(body, dict) or body.get("protocol") != PROTOCOL:
                    raise ScoringError("malformed scoring response protocol")
                if body.get("request_id") != request["request_id"]:
                    raise ScoringError("scoring response request_id mismatch")
                if not isinstance(body.get("result"), dict):
                    raise ScoringError("scoring response requires a result object")
                return body

        return await asyncio.wait_for(run(), self.timeout)


def _result(raw, transformed, body=None, **details):
    return {
        "raw": raw,
        "transformed": transformed,
        "status": "success",
        "identity": body.get("identity") if body else None,
        "details": details,
    }


def _vector(result, dimension):
    values = result.get("embedding")
    if (
        result.get("normalized") is not True
        or result.get("dimension") != dimension
        or not isinstance(values, list)
        or len(values) != dimension
    ):
        raise ScoringError("SIM embedding dimension/normalization contract differs")
    vector = tuple(_finite(value, "SIM embedding") for value in values)
    if abs(math.fsum(value * value for value in vector) - 1) > 1e-3:
        raise ScoringError("SIM embedding is not normalized")
    return vector


def _short_candidate(result):
    counts = [result.get(key) for key in ("num_samples", "min_samples", "sample_rate")]
    if (
        result.get("reason") != "too_short"
        or any(isinstance(v, bool) or not isinstance(v, int) for v in counts)
        or not 0 < counts[0] < counts[1]
        or counts[1:] != [1600, 16000]
        or "embedding" in result
    ):
        raise ScoringError("invalid short SIM embedding outcome")
    return {"reason": "too_short", "num_samples": counts[0], "min_samples": counts[1], "sample_rate": counts[2]}


class RewardScorer:
    """Score selected components while sharing frozen clients and reference caches.

    Component selection is independent of objective coefficients. Clients are
    initialized only when selected, before the first scoring await.
    """

    def __init__(self, services):
        if not isinstance(services, Mapping):
            raise ValueError("reward services must be a mapping")
        self._service_options = dict(services)
        self.services = {}
        self._reference_vectors = OrderedDict()

    async def wer(self, audio_path, language, expected, request_id):
        client = self.services["wer"]
        # Transcript targets and language hints never enter ASR decoding.
        body = await client.evaluate("score", {"request_id": request_id, "audio": client.audio(audio_path)})
        text = body["result"].get("text")
        if not isinstance(text, str):
            raise ScoringError("ASR result requires text")
        actual = normalized_tokens(text, language)
        errors = edit_distance(expected, actual)
        raw = errors / len(expected)
        return _result(
            raw,
            1 / (1 + raw),
            body,
            normalization=NORMALIZATION,
            language=language,
            asr_language_mode="auto",
            reward_transform="inverse_error",
            errors=errors,
            reference_tokens=len(expected),
            hypothesis_tokens=len(actual),
            transcript=text,
            request_id=request_id,
        )

    async def sim(self, audio_path, target, request_id):
        client = self.services["sim"]
        identity = await client.info()
        dimension = identity.get("embedding_dimension")
        if (
            identity.get("service_task") != "speaker_embedding"
            or isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 1
        ):
            raise ScoringError("SIM service requires a speaker_embedding identity with embedding_dimension")
        reference_id, reference = target
        before = reference.stat()

        def digest():
            with reference.open("rb") as stream:
                checksum = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(block)
                return checksum.hexdigest()

        checksum = await asyncio.to_thread(digest)
        key = checksum
        cached = self._reference_vectors.get(key)
        candidate_request = {"request_id": request_id + "-candidate", "audio": client.audio(audio_path)}
        requests = [client.evaluate("embed", candidate_request)]
        if cached is None:
            requests.append(
                client.evaluate("embed", {"request_id": request_id + "-reference", "audio": client.audio(reference)})
            )
        outcomes = await asyncio.gather(*requests, return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
        after = reference.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            self._reference_vectors.pop(key, None)
            raise ValueError("SIM reference audio changed while embedding")
        candidate = outcomes[0]
        if cached is None:
            target_result = outcomes[1]["result"]
            if target_result.get("status") == "invalid_audio":
                raise ValueError("SIM reference audio cannot produce a valid embedding")
            if target_result.get("status", "success") != "success":
                raise ScoringError("unknown SIM reference embedding outcome")
            cached = _vector(target_result, dimension)
        # Another concurrent sample can evict the entry while this sample is
        # waiting for its candidate request. Restore the validated value.
        self._reference_vectors[key] = cached
        self._reference_vectors.move_to_end(key)
        while len(self._reference_vectors) > 256:
            self._reference_vectors.popitem(last=False)
        result = candidate["result"]
        if result.get("status") == "invalid_audio":
            outcome = _result(None, 0, candidate, **_short_candidate(result))
            outcome["status"] = "invalid_audio"
            return outcome
        if result.get("status", "success") != "success":
            raise ScoringError("unknown SIM embedding outcome")
        vector = _vector(result, dimension)
        raw = max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(vector, cached, strict=True))))
        return _result(
            raw,
            max(0, raw),
            candidate,
            purpose="timbre",
            reference_id=reference_id,
            reference_sha256=checksum,
            embedding_dimension=dimension,
            request_id=request_id,
        )

    async def judge_clap(self, audio_path, queries, request_id):
        client = self.services["judge_clap"]
        identity = await client.info()
        categories = identity.get("supported_categories")
        if categories is not None:
            if not isinstance(categories, list) or any(not isinstance(item, str) for item in categories):
                raise ScoringError("invalid JudgeCLAP supported_categories")
            unsupported = {query.partition(":")[0] for query in queries} - set(categories)
            if unsupported:
                raise ValueError(f"JudgeCLAP does not support rendered categories: {sorted(unsupported)}")
        versions = identity.get("query_versions")
        if versions is not None and (not isinstance(versions, list) or QUERY_VERSION not in versions):
            raise ScoringError("JudgeCLAP service does not support the Prism v6 query contract")
        body = await client.evaluate(
            "score",
            {
                "request_id": request_id,
                "audio": client.audio(audio_path),
                "queries": queries,
                "query_version": QUERY_VERSION,
            },
        )
        scores = body["result"].get("scores")
        if not isinstance(scores, list) or len(scores) != len(queries):
            raise ScoringError("JudgeCLAP score count differs from query count")
        scores = [_finite(value, "JudgeCLAP score") for value in scores]
        if any(not 0 <= value <= 1 for value in scores):
            raise ScoringError("JudgeCLAP score outside [0,1]")
        mean = math.fsum(scores) / len(scores)
        return _result(
            mean,
            mean,
            body,
            query_version=QUERY_VERSION,
            queries=queries,
            attribute_scores=scores,
            attribute_count=len(scores),
            request_id=request_id,
        )

    async def score_components(self, sample, component_names):
        """Return complete component records without choosing an aggregate reward."""
        names = tuple(component_names)
        if not names or any(not isinstance(name, str) or name not in {"wer", "sim", "judge_clap"} for name in names):
            raise ValueError("select nonempty reward components from wer, sim, and judge_clap")
        if len(set(names)) != len(names):
            raise ValueError("selected reward components must be unique")
        missing = set(names) - self._service_options.keys()
        if missing:
            raise ValueError(f"missing reward services: {sorted(missing)}")
        for name in names:
            if name not in self.services:
                self.services[name] = _Service(self._service_options[name])
        metadata = sample.metadata
        if not isinstance(metadata, dict) or not isinstance(metadata.get("omni_prompt"), Mapping):
            raise ValueError("TTS rewards require sample.metadata.omni_prompt")
        metadata.pop("omni_reward", None)
        prompt = metadata["omni_prompt"]
        frames = metadata.get("omni_frames")
        if isinstance(frames, bool) or not isinstance(frames, int) or frames < 0:
            raise ValueError("TTS rewards require nonnegative integer metadata.omni_frames")
        inputs = {}
        if "wer" in names:
            target, language = metadata.get("transcript_target"), metadata.get("language")
            if not isinstance(target, str) or not isinstance(language, str) or not language.strip():
                raise ValueError("WER requires transcript_target and nonempty language")
            expected = normalized_tokens(target, language)
            if not expected:
                raise ValueError("WER transcript_target has no tokens after normalization")
            inputs["wer"] = (language, expected)
        if "sim" in names:
            inputs["sim"] = (_reference(prompt, must_exist=frames > 0),)
        if "judge_clap" in names:
            inputs["judge_clap"] = (
                instruction_queries(prompt.get("global_instruction"), modality=prompt.get("modality", "speech")),
            )
        if frames == 0:
            components = {
                name: {
                    "raw": None,
                    "transformed": 0.0,
                    "status": "invalid_audio",
                    "identity": None,
                    "details": {"reason": "zero_frame"},
                }
                for name in names
            }
        else:
            audio_path = metadata.get("omni_audio_path")
            if not isinstance(audio_path, (str, Path)) or not str(audio_path):
                raise ValueError("nonzero TTS frames require metadata.omni_audio_path")
            audio_path = Path(audio_path).resolve(strict=True)
            if not audio_path.is_file():
                raise ValueError("TTS candidate audio must be a regular file")
            request_id = uuid.uuid4().hex
            outcomes = await asyncio.gather(
                *(getattr(self, name)(audio_path, *inputs[name], request_id + "-" + name) for name in names),
                return_exceptions=True,
            )
            components = {
                name: outcome
                for name, outcome in zip(names, outcomes, strict=True)
                if not isinstance(outcome, BaseException)
            }
            failures = [
                (name, outcome)
                for name, outcome in zip(names, outcomes, strict=True)
                if isinstance(outcome, BaseException)
            ]
            if failures:
                metadata["omni_reward"] = {
                    "status": "error",
                    "components": components,
                    "errors": {name: type(error).__name__ for name, error in failures},
                }
                raise failures[0][1]
        return components


class _Rewards(RewardScorer):
    """Retain the original weighted-sum hook and configuration contract."""

    def __init__(self, config):
        if not isinstance(config, Mapping) or set(config) != {"services", "weights"}:
            raise ValueError("omni_reward_config requires services and weights mappings")
        weights, services = config["weights"], config["services"]
        if not isinstance(weights, Mapping) or not isinstance(services, Mapping):
            raise ValueError("reward services and weights must be mappings")
        if set(weights) - {"wer", "sim", "judge_clap"}:
            raise ValueError("supported reward components are wer, sim, and judge_clap")
        self.weights = {name: _finite(value, f"{name} weight") for name, value in weights.items()}
        if any(value < 0 for value in self.weights.values()) or not any(value > 0 for value in self.weights.values()):
            raise ValueError("reward weights must be nonnegative with at least one positive component")
        self.weights = {name: weight for name, weight in self.weights.items() if weight > 0}
        missing = set(self.weights) - set(services)
        if missing:
            raise ValueError(f"missing reward services: {sorted(missing)}")
        super().__init__({name: services[name] for name in self.weights})
        self.services = {name: _Service(services[name]) for name in self.weights}

    async def score(self, sample):
        components = await self.score_components(sample, self.weights)
        total = math.fsum(self.weights[name] * component["transformed"] for name, component in components.items())
        metadata = sample.metadata
        metadata["omni_reward"] = {
            "status": "success",
            "total": total,
            "weights": dict(self.weights),
            "components": components,
        }
        return total


def _config(value):
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, (str, Path)):
        raise ValueError("args.omni_reward_config must be a mapping, JSON string, or JSON/YAML path")
    if isinstance(value, str) and value.lstrip().startswith("{"):
        return json.loads(value)
    path = Path(value)
    content = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        return yaml.safe_load(content)
    return json.loads(content)


class RewardStateCache:
    """Own reward clients per event loop and configuration, outside serialized args."""

    def __init__(self, factory):
        self.factory = factory
        self._states = weakref.WeakKeyDictionary()

    def get(self, args):
        source = getattr(args, "omni_reward_config", None)
        if isinstance(source, Mapping):
            source = json.dumps(dict(source), sort_keys=True)
        elif isinstance(source, Path):
            source = str(source)
        if not isinstance(source, str):
            raise ValueError("args.omni_reward_config must be a mapping, JSON string, or JSON/YAML path")

        # Contended locks retain their loop even after all requests finish.
        # Also release clients from closed loops when used outside managed rollout.
        for owner in list(self._states):
            if owner.is_closed():
                del self._states[owner]
        states = self._states.setdefault(asyncio.get_running_loop(), {})
        if source not in states:
            states[source] = self.factory(_config(source))
        return states[source]

    def close(self, args):
        """Release this loop's clients after the owner has drained reward calls."""
        self._states.pop(asyncio.get_running_loop(), None)


_REWARD_STATES = RewardStateCache(_Rewards)


async def reward(args, sample, **kwargs):
    """Slime custom RM hook; return one scalar or ordered scalars for a batch."""
    pipeline = _REWARD_STATES.get(args)
    if isinstance(sample, list):
        outcomes = await asyncio.gather(*(pipeline.score(item) for item in sample), return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
        return outcomes
    return await pipeline.score(sample)


reward.close = _REWARD_STATES.close
