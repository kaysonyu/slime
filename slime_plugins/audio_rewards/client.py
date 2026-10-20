"""HTTP audio reward clients, independent of Prism samples and serving dependencies.

ASR uses native OpenAI transcriptions; SIM/JudgeCLAP use versioned embeddings.
The caller owns input files and chooses how component scores form an objective.
"""

import asyncio
import base64
import hashlib
import os
import re
from collections import OrderedDict
from collections.abc import Mapping
from functools import wraps
from pathlib import Path
from urllib.parse import urlsplit

from . import timing
from .cache import read_embeddings
from .concurrency import worker_budget
from .protocol import PROTOCOL, ScoringError, finite
from .scoring import ComponentScorers, component_result, normalized_vector
from .transport import RewardHTTPClient


async def _drain_on_cancel(awaitable):
    """Finish owned work before propagating cancellation, including repeated cancels."""
    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                pass
            except Exception:
                break
        # A late read failure must not replace cancellation or go unobserved.
        if not task.cancelled():
            task.exception()
        raise


def _owned_component(method):
    """Track the entire component, including local I/O outside service requests."""

    @wraps(method)
    async def run(self, *args, **kwargs):
        self._ensure_open()
        task = asyncio.create_task(method(self, *args, **kwargs))
        self._operations.add(task)
        try:
            return await asyncio.shield(task)
        finally:
            if not task.done():
                task.cancel()
            try:
                await _drain_on_cancel(asyncio.gather(task, return_exceptions=True))
            finally:
                self._operations.discard(task)

    return run


class ServiceClient:
    """One service's protocol, admission, timeout and owned HTTP lifetime."""

    def __init__(self, options, *, worker_index=0, worker_count=1):
        if not isinstance(options, Mapping):
            raise ValueError("reward service configuration must be an object")
        if "max_inflight" in options:
            raise ValueError("max_inflight was removed; set concurrency to the total budget for this training run")
        unknown = set(options) - {
            "endpoint",
            "timeout_s",
            "max_retries",
            "concurrency",
            "audio_transport",
            "auth_token_env",
            "api",
            "model",
            "embedding_cache",
            "max_completion_tokens",
        }
        if unknown:
            raise ValueError(f"unknown reward service options: {sorted(unknown)}")
        self.api = options.get("api")
        if self.api not in {"voxrl_embedding", "openai_audio"}:
            raise ValueError("reward api must explicitly select voxrl_embedding or openai_audio")
        self.model = options.get("model")
        self.max_completion_tokens = options.get("max_completion_tokens", 1024)
        if (
            isinstance(self.max_completion_tokens, bool)
            or not isinstance(self.max_completion_tokens, int)
            or self.max_completion_tokens <= 0
        ):
            raise ValueError("max_completion_tokens must be a positive integer")
        self.embedding_cache = options.get("embedding_cache")
        if self.embedding_cache is not None and (
            self.api != "voxrl_embedding" or not isinstance(self.embedding_cache, str)
        ):
            raise ValueError("embedding_cache requires a voxrl_embedding service and a file path")
        if self.api == "openai_audio" and (not isinstance(self.model, str) or not self.model.strip()):
            raise ValueError("openai_audio requires the served model name")
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
        self.timeout = finite(options.get("timeout_s", 120), "timeout_s")
        if self.timeout <= 0:
            raise ValueError("timeout_s must be positive")
        retries = options.get("max_retries", 0)
        self.total_concurrency = options.get("concurrency", 128 if self.api == "openai_audio" else 64)
        self.concurrency = worker_budget(self.total_concurrency, worker_index, worker_count)
        self.worker_index, self.worker_count = worker_index, worker_count
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError("max_retries must be an integer >= 0")
        self.attempts = retries + 1
        self.transport = options.get("audio_transport", "upload" if self.api == "openai_audio" else "path")
        if self.transport not in {"path", "upload"}:
            raise ValueError("audio_transport must be path or upload")
        if self.api == "openai_audio" and self.transport != "upload":
            raise ValueError("native OpenAI ASR requires audio_transport: upload")
        self._info = None
        self._info_lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(self.concurrency)
        self._requests = set()
        self._close_task = None
        self.http = RewardHTTPClient(self.concurrency)

    async def aclose(self):
        if self._close_task is None:

            async def close():
                pending = list(self._requests)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                await self.http.aclose()

            self._close_task = asyncio.create_task(close())
        await _drain_on_cancel(self._close_task)

    def _ensure_open(self):
        if self._close_task is not None:
            raise RuntimeError("reward service client is closed")

    async def _request(self, operation):
        self._ensure_open()
        task = asyncio.create_task(operation())
        self._requests.add(task)
        try:
            return await asyncio.wait_for(asyncio.shield(task), self.timeout)
        finally:
            # wait_for must not detach cleanup when its caller is cancelled again.
            if not task.done():
                task.cancel()
            try:
                await _drain_on_cancel(asyncio.gather(task, return_exceptions=True))
            finally:
                self._requests.discard(task)

    def url(self, route):
        return self.endpoint.rstrip("/") + "/" + route

    async def info(self):
        self._ensure_open()
        if self.api != "voxrl_embedding":
            raise ValueError("only embedding services expose /info")
        async with timing.acquire(self._info_lock, "identity_wait_s"):
            self._ensure_open()
            if self._info is None:
                with timing.phase("identity_http_s"):
                    body = await self._request(
                        lambda: self.http.get(self.url("info"), timeout=self.timeout, **self._request_auth)
                    )
                if not isinstance(body, dict) or body.get("protocol") != PROTOCOL:
                    raise ScoringError("incompatible embedding service protocol")
                dimension = body.get("dimension")
                if (
                    not isinstance(body.get("embedding_version"), str)
                    or not body["embedding_version"]
                    or isinstance(dimension, bool)
                    or not isinstance(dimension, int)
                    or dimension < 1
                    or not isinstance(body.get("normalized"), bool)
                    or body.get("dtype") != "float32"
                ):
                    raise ScoringError("invalid embedding service identity")
                self._info = body
            return self._info

    def audio(self, path):
        path = Path(path).resolve(strict=True)
        if not path.is_file():
            raise ValueError("audio must be a regular file")
        if self.transport == "path":
            return {"path": str(path)}
        return {"data_b64": base64.b64encode(path.read_bytes()).decode("ascii"), "suffix": path.suffix.lower()}

    async def transcribe(self, path, request_id):
        path = Path(path).resolve(strict=True)
        if not path.is_file():
            raise ValueError("ASR audio must be a regular file")

        async def run():
            async with timing.acquire(self._slots, "client_queue_s"):
                with timing.phase("prepare_s"):
                    content = await _drain_on_cancel(asyncio.to_thread(path.read_bytes))
                headers = {**self._request_auth.get("headers", {}), "X-Request-ID": request_id}
                with timing.phase("http_s"):
                    timing.count("http_requests")
                    body = await self.http.post_file(
                        self.url("v1/audio/transcriptions"),
                        fields={
                            "model": self.model,
                            "response_format": "json",
                            "temperature": "0",
                            "max_completion_tokens": str(self.max_completion_tokens),
                        },
                        filename=path.name,
                        content=content,
                        headers=headers,
                        timeout=self.timeout,
                        attempts=self.attempts,
                    )
                if not isinstance(body, dict) or not isinstance(body.get("text"), str):
                    raise ScoringError("OpenAI ASR response requires text")
                return {
                    "result": {"text": body["text"]},
                    "identity": {
                        "service_task": "asr",
                        "model_name": self.model,
                        "backend": "openai_audio",
                        "max_completion_tokens": self.max_completion_tokens,
                    },
                }

        return await self._request(run)

    async def embeddings(self, kind, inputs, request_id):
        identity = await self.info()

        async def run():
            async with timing.acquire(self._slots, "client_queue_s"):
                # Admission also bounds file reads and encoded upload buffers.
                prepared = inputs
                if kind == "audio":

                    def prepare():
                        return [
                            (
                                {**item, "audio": self.audio(item["audio"]["path"])}
                                if isinstance(item.get("audio"), dict) and set(item["audio"]) == {"path"}
                                else item
                            )
                            for item in inputs
                        ]

                    with timing.phase("prepare_s"):
                        prepared = await _drain_on_cancel(asyncio.to_thread(prepare))
                payload = {
                    "request_id": request_id,
                    "expected_embedding_version": identity["embedding_version"],
                    "inputs": prepared,
                }
                with timing.phase("http_s"):
                    timing.count("http_requests")
                    body = await self.http.post(
                        self.url(f"embed_{kind}"),
                        payload,
                        attempts=self.attempts,
                        timeout=self.timeout,
                        **self._request_auth,
                    )
            if not isinstance(body, dict) or body.get("protocol") != PROTOCOL or body.get("request_id") != request_id:
                raise ScoringError("invalid embedding response envelope")
            if any(body.get(k) != identity[k] for k in ["embedding_version", "dimension", "dtype", "normalized"]):
                raise ScoringError("embedding version or contract changed during scoring")
            results = body.get("results")
            if (
                not isinstance(results, list)
                or len(results) != len(inputs)
                or any(not isinstance(x, dict) for x in results)
                or [x.get("id") for x in results] != [x["id"] for x in inputs]
            ):
                raise ScoringError("embedding results differ from input ids/order")
            for result in results:
                if result.get("status") == "invalid_audio" and kind == "audio":
                    continue
                values = result.get("embedding")
                if (
                    result.get("status") != "success"
                    or not isinstance(values, list)
                    or len(values) != identity["dimension"]
                ):
                    raise ScoringError("invalid embedding result")
                for value in values:
                    finite(value, "embedding")
            return results

        return await self._request(run)

    async def embed_audio(self, path, request_id):
        identity = await self.info()
        result = (await self.embeddings("audio", [{"id": request_id, "audio": {"path": str(path)}}], request_id))[0]
        return {
            "identity": identity,
            "result": {**result, "dimension": identity["dimension"], "normalized": identity["normalized"]},
        }


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


class AudioRewardClient:
    """Orchestrate component inputs, shared vectors and configurable scoring.

    ServiceClient owns remote calls. ComponentScorers owns component math.
    The caller selects components and combines their records into a reward.
    """

    def __init__(self, services, *, scorers=None, worker_index=0, worker_count=1):
        if not isinstance(services, Mapping):
            raise ValueError("reward services must be a mapping")
        self._service_options = dict(services)
        self.scorers = ComponentScorers(scorers)
        self.worker_index, self.worker_count = worker_index, worker_count
        self.services = {}
        self._reference_vectors = OrderedDict()
        self._reference_digests = OrderedDict()
        self._reference_pending = {}
        self._text_vectors = OrderedDict()
        self._text_pending = {}
        self._operations = set()
        self._close_task = None

    async def aclose(self):
        if self._close_task is None:

            async def close():
                pending = self._operations | set(self._reference_pending.values()) | set(self._text_pending.values())
                for task in pending:
                    task.cancel()
                outcomes = await asyncio.gather(
                    asyncio.gather(*pending, return_exceptions=True),
                    *(service.aclose() for service in self.services.values()),
                    return_exceptions=True,
                )
                for outcome in outcomes:
                    if isinstance(outcome, BaseException):
                        raise outcome

            self._close_task = asyncio.create_task(close())
        await _drain_on_cancel(self._close_task)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    def _ensure_open(self):
        if self._close_task is not None:
            raise RuntimeError("audio reward client is closed")

    def _start_fill(self, pending, keys, operation):
        """Register shared work before yielding; the client owns its cancellation."""
        task = asyncio.create_task(operation)
        for key in keys:
            pending[key] = task

        def finished(done):
            for key in keys:
                pending.pop(key, None)
            # All waiters may have been cancelled before a shared fill fails.
            if not done.cancelled():
                done.exception()

        task.add_done_callback(finished)
        return task

    def service(self, name):
        self._ensure_open()
        if name not in {"wer", "sim", "judge_clap"}:
            raise ValueError(f"unknown audio reward component: {name}")
        if name not in self.services:
            if name not in self._service_options:
                raise ValueError(f"missing reward services: {name}")
            client = ServiceClient(
                self._service_options[name], worker_index=self.worker_index, worker_count=self.worker_count
            )
            expected = "openai_audio" if name == "wer" else "voxrl_embedding"
            if client.api != expected:
                raise ValueError(f"{name} requires api: {expected}")
            self.services[name] = client
        return self.services[name]

    @_owned_component
    async def wer(self, audio_path, target_text, language, request_id):
        client = self.service("wer")
        body = await client.transcribe(audio_path, request_id)
        result = await self.scorers.score(
            "wer",
            reference=target_text,
            hypothesis=body["result"]["text"],
            language=language,
            identity=body["identity"],
        )
        result["details"]["request_id"] = request_id
        return result

    @_owned_component
    async def sim(self, audio_path, reference_path, request_id):
        client = self.service("sim")
        identity = await client.info()
        dimension = identity.get("dimension")
        if (
            identity.get("service_task") != "speaker_embedding"
            or isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 1
        ):
            raise ScoringError("SIM service requires a speaker_embedding identity with dimension")
        reference = Path(reference_path).resolve(strict=True)
        before = reference.stat()

        def digest():
            with reference.open("rb") as stream:
                checksum = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(block)
                return checksum.hexdigest()

        stat_key = (str(reference), before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        checksum = self._reference_digests.get(stat_key)
        if checksum is None:
            with timing.phase("cache_lookup_s"):
                checksum = await _drain_on_cancel(asyncio.to_thread(digest))
            self._reference_digests[stat_key] = checksum
        self._reference_digests.move_to_end(stat_key)
        while len(self._reference_digests) > 4096:
            self._reference_digests.popitem(last=False)
        key = (checksum, identity.get("embedding_version"))
        cached = self._reference_vectors.get(key)
        if cached is None and client.embedding_cache:
            with timing.phase("cache_lookup_s"):
                found = await _drain_on_cancel(
                    asyncio.to_thread(read_embeddings, client.embedding_cache, identity, [checksum])
                )
            if checksum in found:
                cached = normalized_vector(
                    {"embedding": found[checksum], "dimension": dimension, "normalized": identity["normalized"]},
                    dimension,
                )
        requests = [client.embed_audio(audio_path, request_id + "-candidate")]
        timing.count("reference_cache_hit", int(cached is not None))
        timing.count("reference_cache_miss", int(cached is None))
        if cached is None:
            timing.count("reference_shared_fill", int(key in self._reference_pending))
            if key not in self._reference_pending:

                async def load_reference():
                    result = await client.embed_audio(reference, request_id + "-reference")
                    if result["result"].get("status", "success") == "success":
                        self._reference_vectors[key] = normalized_vector(result["result"], dimension)
                        while len(self._reference_vectors) > 4096:
                            self._reference_vectors.popitem(last=False)
                    return result

                self._start_fill(self._reference_pending, [key], load_reference())
            pending = self._reference_pending[key]

            async def wait_reference():
                with timing.phase("cache_wait_s"):
                    return await asyncio.shield(pending)

            requests.append(wait_reference())
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
            cached = normalized_vector(target_result, dimension)
        # Another concurrent sample can evict the entry while this sample is
        # waiting for its candidate request. Restore the validated value.
        self._reference_vectors[key] = cached
        self._reference_vectors.move_to_end(key)
        while len(self._reference_vectors) > 4096:
            self._reference_vectors.popitem(last=False)
        result = candidate["result"]
        if result.get("status") == "invalid_audio":
            outcome = component_result(None, 0, candidate, **_short_candidate(result))
            outcome["status"] = "invalid_audio"
            return outcome
        if result.get("status", "success") != "success":
            raise ScoringError("unknown SIM embedding outcome")
        vector = normalized_vector(result, dimension)
        record = await self.scorers.score("sim", candidate=vector, reference=cached, identity=candidate["identity"])
        record["details"].update(
            reference_sha256=checksum,
            embedding_dimension=dimension,
            request_id=request_id,
        )
        return record

    @_owned_component
    async def judge_clap(self, audio_path, queries, request_id, *, query_version):
        import numpy as np

        client = self.service("judge_clap")
        identity = await client.info()
        if identity.get("service_task") != "judge_clap" or identity.get("normalized") is not False:
            raise ScoringError("JudgeCLAP requires unnormalized audio/text embeddings")
        categories = identity.get("supported_categories")
        if not isinstance(categories, list) or any(not isinstance(x, str) for x in categories):
            raise ScoringError("invalid JudgeCLAP categories")
        versions = identity.get("query_versions")
        if not isinstance(versions, list) or query_version not in versions:
            raise ScoringError("incompatible JudgeCLAP query version")
        if not isinstance(queries, (list, tuple)) or not queries or any(not isinstance(q, str) for q in queries):
            raise ValueError("JudgeCLAP requires nonempty atomic text queries")
        for query in queries:
            category, separator, value = query.partition(":")
            if not separator or category not in categories or not value.strip():
                raise ValueError(f"JudgeCLAP does not support query: {query!r}")
        atomic = list(queries)
        version = identity["embedding_version"]

        def fp32_vector(values):
            with np.errstate(over="ignore"):
                vector = np.asarray(values, dtype=np.float32)
            if not np.isfinite(vector).all():
                raise ScoringError("embedding is not representable in FP32")
            return vector

        async def load_texts(chunk, start):
            vectors = {}
            if client.embedding_cache:
                with timing.phase("cache_lookup_s"):
                    found = await _drain_on_cancel(
                        asyncio.to_thread(read_embeddings, client.embedding_cache, identity, chunk)
                    )
                vectors.update((text, fp32_vector(vector)) for text, vector in found.items())
            missing = [text for text in chunk if text not in vectors]
            timing.count("text_cache_hit", len(vectors))
            timing.count("text_cache_miss", len(missing))
            if missing:
                results = await client.embeddings(
                    "text", [{"id": str(i), "text": t} for i, t in enumerate(missing)], f"{request_id}-text-{start}"
                )
                vectors.update(
                    (text, fp32_vector(result["embedding"])) for text, result in zip(missing, results, strict=True)
                )
            self._text_vectors.update(((version, text), vector) for text, vector in vectors.items())
            while len(self._text_vectors) > 8192:
                self._text_vectors.popitem(last=False)
            # Return values directly: another fill can evict them before a waiter resumes.
            return vectors

        async def texts():
            values, missing, pending = {}, [], set()
            # Cache selection and registration do not yield, so overlapping callers
            # share one fill (and its failure) without a lock across disk/network I/O.
            for text in dict.fromkeys(atomic):
                key = (version, text)
                if key in self._text_vectors:
                    values[text] = self._text_vectors[key]
                    self._text_vectors.move_to_end(key)
                elif key in self._text_pending:
                    pending.add(self._text_pending[key])
                    timing.count("text_shared_fill")
                    timing.count("text_cache_miss")
                else:
                    missing.append(text)
            timing.count("text_cache_hit", len(values))
            for start in range(0, len(missing), 64):
                chunk = missing[start : start + 64]
                pending.add(
                    self._start_fill(self._text_pending, [(version, text) for text in chunk], load_texts(chunk, start))
                )
            with timing.phase("cache_wait_s"):
                outcomes = await asyncio.gather(*(asyncio.shield(task) for task in pending), return_exceptions=True)
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
                values.update(outcome)
            return [values[text] for text in atomic]

        outcomes = await asyncio.gather(client.embed_audio(audio_path, request_id), texts(), return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
        body, text_vectors = outcomes
        if body["result"].get("status") != "success":
            raise ScoringError(f"JudgeCLAP audio cannot be encoded: {body['result'].get('reason')}")
        fp32_vector(body["result"]["embedding"])
        record = await self.scorers.score(
            "judge_clap",
            audio=body["result"]["embedding"],
            texts=text_vectors,
            queries=atomic,
            identity=body["identity"],
        )
        record["details"].update(
            query_version=query_version,
            queries=atomic,
            request_id=request_id,
            embedding_version=version,
        )
        return record
