"""Frozen teacher bindings and bounded batches of selected-action scores."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import os
import time
import weakref
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import httpx
import torch
import yaml

from slime.utils.http_utils import _wrap_ipv6, get
from slime_plugins.prism.http_client import _request
from slime_plugins.prism.protocol import LOGPROB_SEMANTICS, REPLAY_FIELDS, _config, _copy_wire_data, _integer, _tensor


@lru_cache(maxsize=32)
def _load_teacher_config(value):
    if value.lstrip().startswith("{"):
        config = json.loads(value)
    else:
        content = Path(value).read_text()
        config = json.loads(content) if Path(value).suffix.lower() == ".json" else yaml.safe_load(content)
    if (
        not isinstance(config, dict)
        or not isinstance(config.get("models", {}), dict)
        or not isinstance(config.get("by_domain", {}), dict)
    ):
        raise ValueError("Teacher configuration requires models and by_domain mappings")
    return config


@dataclass
class PendingScore:
    request: dict
    result: asyncio.Future


@dataclass(frozen=True)
class ScoreReceipt:
    value: dict
    batch_size: int
    request_seconds: float
    attempts: int


class TeacherClient:
    def __init__(self, args, configuration):
        self.args = copy.copy(args)
        self.args.omni_request_timeout_s = float(
            getattr(args, "omni_teacher_request_timeout_s", min(getattr(args, "omni_request_timeout_s", 600.0), 120.0))
        )
        self.args.omni_cancel_drain_timeout_s = min(getattr(args, "omni_cancel_drain_timeout_s", 30.0), 30.0)
        self.configuration = copy.deepcopy(_load_teacher_config(configuration))
        self.batch_size = int(getattr(args, "omni_teacher_batch_size", 8))
        self.wait_seconds = float(getattr(args, "omni_teacher_batch_wait_ms", 2.0)) / 1000
        self.max_inflight = int(getattr(args, "omni_teacher_max_inflight", 16))
        self.attempts = int(getattr(args, "omni_teacher_max_attempts", 2))
        if (
            not 1 <= self.batch_size <= 64
            or not 0 <= self.wait_seconds <= 1
            or self.max_inflight < 1
            or self.attempts < 1
        ):
            raise ValueError("Invalid frozen teacher batch, wait, concurrency, or retry limit")
        self.semaphore = asyncio.Semaphore(self.max_inflight)
        self.pending = {}
        self.timers = {}
        self.tasks = set()

    async def score(self, endpoint, binding, request):
        route = (endpoint, binding.get("auth_token_env"), binding["weight_sha256"])
        result = asyncio.get_running_loop().create_future()
        pending = self.pending.setdefault(route, [])
        pending.append(PendingScore(request, result))
        if len(pending) >= self.batch_size:
            self.dispatch(route)
        elif route not in self.timers:
            self.timers[route] = asyncio.get_running_loop().call_later(self.wait_seconds, self.dispatch, route)
        try:
            return await asyncio.shield(result)
        except asyncio.CancelledError:
            result.cancel()
            raise

    def dispatch(self, route):
        timer = self.timers.pop(route, None)
        if timer is not None:
            timer.cancel()
        pending = [item for item in self.pending.pop(route, []) if not item.result.done()]
        for begin in range(0, len(pending), self.batch_size):
            batch = pending[begin : begin + self.batch_size]
            task = asyncio.create_task(self.send(route, batch))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

    async def send(self, route, batch):
        endpoint, token_name, _ = route
        try:
            token = os.environ.get(token_name) if token_name else None
            if token_name and not token:
                raise ValueError(f"Teacher authentication environment {token_name!r} is missing")
            payload = {"samples": [item.request for item in batch]}
            async with self.semaphore:
                started = time.perf_counter()
                for attempt in range(self.attempts):
                    try:
                        response = await _request(
                            self.args,
                            endpoint + "/score_actions",
                            payload,
                            headers={"Authorization": f"Bearer {token}"} if token else None,
                        )
                        break
                    except (httpx.TransportError, httpx.HTTPStatusError, TimeoutError) as error:
                        retryable = not isinstance(error, httpx.HTTPStatusError) or error.response.status_code in {
                            429,
                            502,
                            503,
                            504,
                        }
                        if not retryable or attempt + 1 == self.attempts:
                            raise
                        await asyncio.sleep(min(0.1 * 2**attempt, 1.0))
            if not isinstance(response, dict) or type(response.get("version")) is not int or response["version"] != 1:
                raise ValueError("Prism teacher response version differs")
            results = response.get("results")
            if (
                not isinstance(results, list)
                or len(results) != len(batch)
                or any(not isinstance(row, dict) for row in results)
            ):
                raise ValueError("Prism teacher response result count differs")
            by_id = {row.get("sample_id"): row for row in results}
            if len(by_id) != len(batch) or set(by_id) != {item.request["sample_id"] for item in batch}:
                raise ValueError("Prism teacher response sample identities differ")
            for item in batch:
                if not item.result.done():
                    item.result.set_result(
                        ScoreReceipt(
                            by_id[item.request["sample_id"]], len(batch), time.perf_counter() - started, attempt + 1
                        )
                    )
        except BaseException as error:
            for item in batch:
                if not item.result.done():
                    if isinstance(error, asyncio.CancelledError):
                        item.result.cancel()
                    else:
                        item.result.set_exception(error)

    async def drain(self):
        for route in tuple(self.pending):
            self.dispatch(route)
        if self.tasks:
            await asyncio.gather(*tuple(self.tasks))

    async def close(self):
        for timer in self.timers.values():
            timer.cancel()
        self.timers.clear()
        for pending in self.pending.values():
            for item in pending:
                item.result.cancel()
        self.pending.clear()
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await asyncio.gather(*tuple(self.tasks), return_exceptions=True)


_CLIENTS = weakref.WeakKeyDictionary()


def teacher_client(args):
    source = args.omni_teacher_config
    source = json.dumps(source, sort_keys=True) if isinstance(source, dict) else source
    if not isinstance(source, str) or not source:
        raise ValueError("omni_teacher_config must be a JSON/YAML path or object")
    limits = tuple(
        getattr(args, name, default)
        for name, default in (
            ("omni_teacher_batch_size", 8),
            ("omni_teacher_batch_wait_ms", 2.0),
            ("omni_teacher_max_inflight", 16),
            ("omni_teacher_max_attempts", 2),
            ("omni_teacher_request_timeout_s", 120.0),
        )
    )
    key = (
        source,
        limits,
        getattr(args, "omni_model_name", "student"),
        repr(getattr(args, "sglang_model_routers", None)),
    )
    clients = _CLIENTS.setdefault(asyncio.get_running_loop(), {})
    if key not in clients:
        clients[key] = TeacherClient(args, source)
    return clients[key]


async def drain_teacher_clients(args):
    for client in _CLIENTS.get(asyncio.get_running_loop(), {}).values():
        await client.drain()


async def close_teacher_clients(args):
    for client in _CLIENTS.pop(asyncio.get_running_loop(), {}).values():
        await client.close()


def teacher_request(sample):
    trace = sample.metadata["omni_rollout"]
    sample_id = f"{sample.group_index}:{sample.index}:{trace['request_id']}"
    if sample.metadata.get("omni_replay_compact"):
        inputs = sample.multimodal_train_inputs
        rows = sample.metadata["omni_prompt_rows"]
        prompt = torch.cat((torch.tensor(sample.tokens[:rows])[:, None], inputs["audio_codes"][:rows]), dim=-1)
        successors = inputs["successor_audio_codes"][:rows].clone()
        edges = inputs["successor_audio_mask"][:rows].clone()
        successors[-1].zero_()
        edges[-1] = False
        replay = {
            "layout": "moss_prism_typed_v1",
            "prompt_rows": prompt.tolist(),
            "item_kind": inputs["item_kind"][:rows].tolist(),
            "audio_role": inputs["audio_role"][:rows].tolist(),
            "successor_audio_mask": edges.tolist(),
            "successor_audio_codes": successors.tolist(),
            "target_history_retention_mask": inputs["target_history_retention_mask"][:rows].tolist(),
        }
        codes = inputs["successor_audio_codes"][inputs["policy_code_mask"].any(-1)].tolist()
    else:
        replay = {name: trace["replay_inputs"][name] for name in REPLAY_FIELDS}
        codes = trace["action_streams"][0]["actions"]
    return _copy_wire_data(
        dict(
            version=2,
            sample_id=sample_id,
            **replay,
            codes=codes,
            finish_reason=trace["finish_reason"],
            temperature=1.0,
        )
    )


def teacher_request_hash(request):
    normalized = {key: value for key, value in request.items() if key != "sample_id"}
    normalized["temperature"] = float(normalized["temperature"])
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def attach_teacher_scores(sample, request, result, binding):
    """Align the teacher's selected-action scores with the student's trajectory."""
    identity = binding.get("model_identity")
    student = sample.metadata["omni_rollout"]["model_identity"]
    if not isinstance(identity, dict) or {k: v for k, v in identity.items() if k != "config_sha256"} != {
        k: v for k, v in student.items() if k != "config_sha256"
    }:
        raise ValueError("Prism teacher binding has incompatible replay identity")
    if identity.get("config_sha256") != student.get("config_sha256"):
        checkpoint = binding.get("checkpoint") or binding.get("model_path")
        if not checkpoint:
            raise ValueError("A different-config teacher requires local checkpoint assets for replay compatibility")
        teacher_config = _config(str(checkpoint))
        if (
            teacher_config.identity != identity
            or teacher_config.replay_geometry() != sample.metadata.get("omni_replay_geometry")
            or not sample.metadata.get("omni_tokenizer_sha256")
            or teacher_config.tokenizer_sha256 != sample.metadata["omni_tokenizer_sha256"]
        ):
            raise ValueError("Prism teacher checkpoint has incompatible geometry, special tokens or tokenizer")
    checksum = binding.get("weight_sha256")
    if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise ValueError("Prism teacher requires a pinned weight_sha256")
    if type(result.get("version")) is not int or type(result.get("temperature")) not in (int, float):
        raise ValueError("Prism teacher response version or temperature has invalid scalar types")
    expected = dict(
        version=2,
        sample_id=request["sample_id"],
        input_sha256=teacher_request_hash(request),
        temperature=1.0,
        finish_reason=request["finish_reason"],
        model_identity=identity,
        teacher_weight_sha256=checksum,
        logprob_semantics=LOGPROB_SEMANTICS,
        stop_semantics="pre_frame_threshold_v1",
    )
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("Prism teacher response identity, input hash, checksum or semantics differs")
    if not isinstance(result.get("weight_version"), str) or not result["weight_version"]:
        raise ValueError("Prism teacher weight version is missing")
    f, q = sample.response_length, student["n_vq"]
    stops = f + int(request["finish_reason"] == "stop")
    scores = _tensor(result.get("code_logprobs"), (f, q), torch.float32, "teacher code logprobs")
    selected = _tensor(result.get("selected_stop_logprobs"), (stops,), torch.float32, "teacher selected stop logprobs")
    if (scores > 1e-5).any() or (selected > 1e-5).any():
        raise ValueError("Teacher selected logprobs must be nonpositive")
    mm = sample.multimodal_train_inputs
    code_result = torch.zeros_like(mm["policy_rollout_code_logprobs"])
    stop_result = torch.zeros_like(mm["policy_rollout_stop_logprobs"])
    code_result[mm["policy_code_mask"].any(-1)] = scores
    stop_result[mm["policy_stop_mask"]] = selected
    mm["policy_teacher_code_logprobs"] = code_result
    mm["policy_teacher_stop_logprobs"] = stop_result
    sample.teacher_log_probs = scores.sum(-1).tolist()
    sample.metadata["omni_teacher_identity"] = copy.deepcopy(identity)
    sample.metadata["omni_teacher_weight_sha256"] = checksum
    sample.metadata["omni_teacher_weight_version"] = result["weight_version"]
    return sample


def _validate_frozen_deployment(args, deployment):
    """Reject trainable named models before discovering any teacher worker."""
    if not isinstance(deployment, str) or not deployment:
        raise ValueError("Prism teacher deployment must be a nonempty model name")
    if deployment == getattr(args, "omni_model_name", "student"):
        raise ValueError("A Prism teacher deployment cannot be the student model")
    configuration = getattr(args, "sglang_config", None)
    if configuration is None:
        # External/pre-resolved deployments may expose only their router map.
        # Pinned response identity and checksum remain mandatory below.
        return
    if isinstance(configuration, (str, Path)):
        from slime.backends.sglang_utils.sglang_config import SglangConfig

        configuration = SglangConfig.from_yaml(str(configuration))
    models = getattr(configuration, "models", None)
    if models is None:
        raise ValueError("Prism teacher requires a model-aware sglang_config")
    matches = [model for model in models if model.name == deployment]
    if len(matches) != 1:
        raise ValueError(f"Prism teacher deployment {deployment!r} must name one configured model")
    model = matches[0]
    if model.update_weights is None:
        model.resolve(args)
    if model.update_weights is not False:
        raise ValueError(f"Prism teacher deployment {deployment!r} must have update_weights=false")


async def _teacher_endpoint(args, sample, binding):
    """Discover a named teacher's live worker; the router has no score API."""
    selected = [name for name in ("endpoint", "deployment") if binding.get(name) is not None]
    if len(selected) != 1:
        raise ValueError("Prism teacher binding must specify exactly one of endpoint or deployment")
    if selected[0] == "endpoint":
        endpoint = binding["endpoint"]
        if not isinstance(endpoint, str) or not endpoint.startswith(("http://", "https://")):
            raise ValueError("Prism teacher endpoint must be HTTP(S)")
        return endpoint.rstrip("/")
    deployment = binding["deployment"]
    _validate_frozen_deployment(args, deployment)
    routers = getattr(args, "sglang_model_routers", None) or {}
    if deployment not in routers:
        raise ValueError(f"Prism teacher deployment {deployment!r} has no registered router")
    address = routers[deployment]
    if not isinstance(address, (tuple, list)) or len(address) != 2:
        raise ValueError("Named Omni model router must provide its (host, port)")
    host, port = address
    port = _integer(port, "teacher router port", minimum=1)
    timeout = float(getattr(args, "rollout_health_check_timeout", 10.0))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Teacher health timeout must be finite and positive")
    router = f"http://{_wrap_ipv6(str(host))}:{port}"
    body = await asyncio.wait_for(get(router + "/workers", timeout=timeout), timeout=timeout)
    if not isinstance(body, dict) or not isinstance(body.get("workers"), list):
        raise ValueError("Omni teacher router /workers returned an invalid worker list")
    workers = sorted(
        {
            worker["url"].rstrip("/")
            for worker in body["workers"]
            if isinstance(worker, dict)
            and worker.get("health_state") == "healthy"
            and worker.get("routable") is True
            and worker.get("disabled") is False
            and worker.get("model") in (None, deployment)
            and isinstance(worker.get("url"), str)
            and worker["url"].startswith(("http://", "https://"))
        }
    )
    if not workers:
        raise ValueError(f"Omni teacher deployment {deployment!r} has no healthy routable workers")
    digest = hashlib.sha256(json.dumps([sample.group_index, sample.index]).encode()).digest()
    start = int.from_bytes(digest[:8], "little") % len(workers)
    for offset in range(len(workers)):
        worker = workers[(start + offset) % len(workers)]
        try:
            health = await asyncio.wait_for(get(worker + "/health", timeout=timeout), timeout=timeout)
        except Exception:
            continue
        if isinstance(health, dict) and health.get("running") is True:
            return worker
    raise ValueError(f"Omni teacher deployment {deployment!r} has no reachable running workers")


async def score_teacher(args, sample, *, evaluation=False, **kwargs):
    """Per-sample hook: route an explicit domain to one frozen Prism teacher."""
    if evaluation:
        score_during_eval = getattr(args, "omni_eval_teacher_scoring", True)
        if type(score_during_eval) is not bool:
            raise ValueError("omni_eval_teacher_scoring must be Boolean")
        if not score_during_eval:
            return sample
    if not getattr(args, "omni_teacher_config", None):
        return sample
    client = teacher_client(args)
    config = client.configuration
    models, routes = config.get("models", {}), config.get("by_domain", {})
    domain = sample.metadata.get("domain")
    key = config.get("default_teacher") if domain is None else routes.get(domain)
    if key not in models:
        raise ValueError(f"Unmapped Prism teacher domain {domain!r}")
    binding = models[key]
    if not isinstance(binding, dict):
        raise ValueError("Prism teacher binding must be an object")
    sample.metadata["omni_teacher_key"] = key
    sample.metadata["omni_teacher_status"] = "pending"
    try:
        endpoint = await _teacher_endpoint(args, sample, binding)
        sample.metadata["omni_teacher_endpoint"] = endpoint
        request = teacher_request(sample)
        receipt = await client.score(endpoint, binding, request)
        attach_teacher_scores(sample, request, receipt.value, binding)
        sample.metadata.update(
            omni_teacher_batch_size=receipt.batch_size,
            omni_teacher_request_seconds=receipt.request_seconds,
            omni_teacher_attempts=receipt.attempts,
        )
    except BaseException as error:
        sample.metadata["omni_teacher_status"] = "cancelled" if isinstance(error, asyncio.CancelledError) else "error"
        sample.metadata["omni_teacher_error"] = type(error).__name__
        raise
    sample.metadata["omni_teacher_status"] = "success"
    if "deployment" in binding:
        sample.metadata["omni_teacher_deployment"] = binding["deployment"]
    return sample


score_teacher.flush = drain_teacher_clients
score_teacher.close = close_teacher_clients
