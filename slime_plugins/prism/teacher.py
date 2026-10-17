"""Frozen teacher bindings and bounded batches of selected-action scores."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from slime.rollout.teacher_client import BatchedTeacherClient, TeacherClientOptions
from slime.utils.http_utils import _wrap_ipv6, get
from slime_plugins.prism.http_client import _request
from slime_plugins.prism.protocol import LOGPROB_SEMANTICS, _copy_wire_data, _integer, _tensor


class TeacherClient(BatchedTeacherClient):
    def __init__(self, args, configuration):
        super().__init__(TeacherClientOptions.from_config(configuration))
        self.args = copy.copy(args)
        self.args.omni_request_timeout_s = float(self.options.request_timeout_s)
        self.args.omni_cancel_drain_timeout_s = min(getattr(args, "omni_cancel_drain_timeout_s", 30.0), 30.0)

    async def send_batch(self, route, requests):
        endpoint, token_name = route
        token = os.environ.get(token_name) if token_name else None
        if token_name and not token:
            raise ValueError(f"Teacher authentication environment {token_name!r} is missing")
        response = await _request(
            self.args,
            endpoint + "/score_actions",
            {"samples": requests},
            headers={"Authorization": f"Bearer {token}"} if token else None,
        )
        if not isinstance(response, dict) or type(response.get("version")) is not int or response["version"] != 1:
            raise ValueError("Prism teacher response version differs")
        results = response.get("results")
        if (
            not isinstance(results, list)
            or len(results) != len(requests)
            or any(not isinstance(row, dict) for row in results)
        ):
            raise ValueError("Prism teacher response result count differs")
        by_id = {row.get("sample_id"): row for row in results}
        if len(by_id) != len(requests) or set(by_id) != {request["sample_id"] for request in requests}:
            raise ValueError("Prism teacher response sample identities differ")
        return [by_id[request["sample_id"]] for request in requests]


def teacher_request(sample):
    trace = sample.metadata["omni_rollout"]
    sample_id = f"{sample.group_index}:{sample.index}:{trace['request_id']}"
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
    stream = sample.policy_actions["rvq"]
    valid = stream["valid_mask"]
    if (valid & stream["actions"].ne(inputs["successor_audio_codes"])).any():
        raise ValueError("RVQ actions disagree with successor conditioning")
    codes = stream["actions"][valid.any(-1)].tolist()
    if len(codes) != sample.response_length:
        raise ValueError("Prism valid actions disagree with the generated frame count")
    stops = sample.policy_actions["stop"]
    stop_values = stops["actions"][stops["valid_mask"]].tolist()
    expected = [0] * len(codes) + ([1] if trace["finish_reason"] == "stop" else [])
    if stop_values != expected:
        raise ValueError("Prism stop actions disagree with emitted frames and finish reason")
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


def attach_teacher_scores(sample, request, result):
    """Align the teacher's selected-action scores with the student's trajectory."""
    if type(result.get("version")) is not int or type(result.get("temperature")) not in (int, float):
        raise ValueError("Prism teacher response version or temperature has invalid scalar types")
    expected = dict(
        version=2,
        sample_id=request["sample_id"],
        input_sha256=teacher_request_hash(request),
        temperature=1.0,
        finish_reason=request["finish_reason"],
        logprob_semantics=LOGPROB_SEMANTICS,
        stop_semantics="pre_frame_threshold_v1",
    )
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("Prism teacher response request or scoring semantics differs")
    if not isinstance(result.get("weight_version"), str) or not result["weight_version"]:
        raise ValueError("Prism teacher weight version is missing")
    f, q = sample.response_length, sample.policy_actions["rvq"]["actions"].shape[-1]
    stops = f + int(request["finish_reason"] == "stop")
    scores = _tensor(result.get("code_logprobs"), (f, q), torch.float32, "teacher code logprobs")
    selected = _tensor(result.get("selected_stop_logprobs"), (stops,), torch.float32, "teacher selected stop logprobs")
    if (scores > 1e-5).any() or (selected > 1e-5).any():
        raise ValueError("Teacher selected logprobs must be nonpositive")
    streams = sample.policy_actions
    code_result = torch.zeros_like(streams["rvq"]["rollout_log_probs"])
    stop_result = torch.zeros_like(streams["stop"]["rollout_log_probs"])
    code_result[streams["rvq"]["valid_mask"].any(-1)] = scores
    stop_result[streams["stop"]["valid_mask"]] = selected
    streams["rvq"]["teacher_log_probs"] = code_result
    streams["stop"]["teacher_log_probs"] = stop_result
    sample.metadata["omni_teacher_identity"] = copy.deepcopy(result.get("model_identity"))
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


async def score_teacher(args, sample, *, binding, client, evaluation=False):
    """Score the frozen Prism teacher selected by the public OPD stage."""
    sample.metadata["omni_teacher_status"] = "pending"
    try:
        endpoint = await _teacher_endpoint(args, sample, binding)
        sample.metadata["omni_teacher_endpoint"] = endpoint
        request = teacher_request(sample)
        receipt = await client.score((endpoint, binding.get("auth_token_env")), request)
        attach_teacher_scores(sample, request, receipt.value)
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


score_teacher.create_client = TeacherClient
