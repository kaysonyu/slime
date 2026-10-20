"""Adapt Prism samples to reusable audio reward component clients."""

import asyncio
import copy
import json
import re
import uuid
import weakref
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import unquote, urlsplit

from slime_plugins.audio_rewards import timing
from slime_plugins.audio_rewards.client import AudioRewardClient
from slime_plugins.audio_rewards.prosody import ProsodyScorer, normalize_language

QUERY_VERSION = "moss_tts_v2_revision6_instruction_v1"
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


def reference_audio(prompt, *, attribute="timbre", must_exist=True):
    label = "SIM" if attribute == "timbre" else attribute.upper()
    references = prompt.get("references")
    if not isinstance(references, dict):
        raise ValueError(f"{label} references must map audio slots to URI strings")
    targets = prompt.get("reference_attributes")
    if not isinstance(targets, dict) or not isinstance(targets.get(attribute), str):
        raise ValueError(f"{label} requires explicit reference_attributes.{attribute}")
    if targets[attribute] == "none" or targets[attribute] not in references:
        raise ValueError(f"{label} {attribute} reference must select an existing audio slot")
    uri = references[targets[attribute]]
    if not isinstance(uri, str) or not uri.strip():
        raise ValueError(f"{label} {attribute} reference must be a nonempty URI string")
    parsed = urlsplit(uri)
    if parsed.scheme == "file" and parsed.netloc in {"", "localhost"}:
        path = Path(unquote(parsed.path))
    elif not parsed.scheme:
        path = Path(uri)
    else:
        raise ValueError(f"{label} reference URI must be localized before scoring")
    path = path.resolve(strict=must_exist)
    if must_exist and not path.is_file():
        raise ValueError(f"{label} reference audio must be a regular file")
    return targets[attribute], path


def judge_queries(prompt):
    rendered = instruction_queries(prompt.get("global_instruction"), modality=prompt.get("modality", "speech"))
    caption_fields = {"Appearance", "Listener context", "Physical scene"}
    dropped = sorted({q.partition(":")[0] for q in rendered} & caption_fields)
    atomic = []
    for query in rendered:
        category, _, value = query.partition(":")
        if category not in caption_fields:
            atomic.extend(f"{category}: {part.strip()}" for part in value.split(" | ") if part.strip())
    if not atomic:
        raise ValueError("instruction has no attributes supported by this JudgeCLAP checkpoint")
    return atomic, dropped


def _accent_record(sim, reference_id):
    record = copy.deepcopy(sim)
    # This is a reused measurement, not an additional HTTP request/timing.
    record.pop("timing", None)
    record.setdefault("details", {}).update(purpose="accent", reference_id=reference_id, reused_from="sim")
    return record


class PrismRewardAdapter:
    """Validate Prism metadata and score requested components without aggregating."""

    def __init__(self, services, *, scorers=None, prosody=None, worker_index=0, worker_count=1):
        self.client = AudioRewardClient(
            services, scorers=scorers, worker_index=worker_index, worker_count=worker_count
        )

        self.prosody = ProsodyScorer(prosody)

    async def aclose(self):
        await asyncio.gather(self.client.aclose(), self.prosody.aclose())

    async def score_components(self, sample, component_names):
        """Return complete component records without choosing an aggregate reward."""
        names = tuple(component_names)
        if not names or any(
            not isinstance(name, str) or name not in {"wer", "sim", "accent", "prosody", "judge_clap"}
            for name in names
        ):
            raise ValueError("select nonempty reward components from wer, sim, accent, prosody, and judge_clap")
        if len(set(names)) != len(names):
            raise ValueError("selected reward components must be unique")
        requested = names
        if "accent" in names and "sim" not in names:
            names = (*names, "sim")
        for name in names:
            if name not in {"accent", "prosody"}:
                self.client.service(name)
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
            if (
                not isinstance(target, str)
                or not target.strip()
                or not isinstance(language, str)
                or not language.strip()
            ):
                raise ValueError("WER requires nonempty transcript_target and language")
            inputs["wer"] = (target, language)
        if "sim" in names:
            reference_id, reference = reference_audio(prompt, must_exist=frames > 0)
            inputs["sim"] = (reference,)
        if "accent" in names:
            attributes = prompt.get("reference_attributes", {})
            accent_id, _ = reference_audio(prompt, attribute="accent", must_exist=frames > 0)
            if accent_id != attributes.get("timbre"):
                raise ValueError("accent must share the timbre reference to reuse SIM")
        if "prosody" in names:
            prosody_id, reference = reference_audio(prompt, attribute="prosody", must_exist=frames > 0)
            inputs["prosody"] = (reference, normalize_language(metadata.get("language")))
        if "judge_clap" in names:
            queries, dropped = judge_queries(prompt)
            inputs["judge_clap"] = (queries,)
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
            components = {}

            async def score_one(name):
                with timing.capture() as profile:
                    profile["request_id"] = request_id + "-" + name
                    if name != "prosody":
                        service = self.client.service(name)
                        profile.update(
                            concurrency=service.total_concurrency,
                            worker_concurrency=service.concurrency,
                            worker_index=service.worker_index,
                            worker_count=service.worker_count,
                        )
                    try:
                        method = self.prosody.score if name == "prosody" else getattr(self.client, name)
                        kwargs = {"query_version": QUERY_VERSION} if name == "judge_clap" else {}
                        if name == "prosody":
                            kwargs["scorers"] = self.client.scorers
                        record = await method(audio_path, *inputs[name], request_id=profile["request_id"], **kwargs)
                        if name == "sim" and record["status"] == "success":
                            record["details"].update(purpose="timbre", reference_id=reference_id)
                        if name == "prosody":
                            record["details"].update(purpose="prosody", reference_id=prosody_id)
                        if name == "judge_clap":
                            record["details"]["dropped_categories"] = dropped
                        components[name] = record
                    except BaseException as error:
                        components[name] = {
                            "status": "cancelled" if isinstance(error, asyncio.CancelledError) else "error",
                            "error": type(error).__name__,
                        }
                        raise
                    finally:
                        components[name]["timing"] = profile
                return record

            scored_names = tuple(name for name in names if name != "accent")
            try:
                outcomes = await asyncio.gather(*(score_one(name) for name in scored_names), return_exceptions=True)
            except asyncio.CancelledError:
                metadata["omni_reward"] = {"status": "cancelled", "components": components}
                raise
            if "accent" in names:
                components["accent"] = _accent_record(components["sim"], accent_id)
            failures = [
                (name, outcome)
                for name, outcome in zip(scored_names, outcomes, strict=True)
                if isinstance(outcome, BaseException)
            ]
            if failures:
                metadata["omni_reward"] = {
                    "status": "error",
                    "components": components,
                    "errors": {name: type(error).__name__ for name, error in failures},
                }
                raise failures[0][1]
        if "accent" in names and frames == 0:
            components["accent"] = _accent_record(components["sim"], accent_id)
        return {name: components[name] for name in requested}


def load_reward_config(value):
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
        index = getattr(args, "rollout_worker_index", 0)
        count = getattr(args, "rollout_worker_count", 1)
        key = (source, index, count)
        if key not in states:
            states[key] = self.factory(load_reward_config(source), worker_index=index, worker_count=count)
        return states[key]

    async def close(self, args):
        """Release this loop's clients after the owner has drained reward calls."""
        states = self._states.pop(asyncio.get_running_loop(), {})
        await asyncio.gather(*(state.aclose() for state in states.values()))
