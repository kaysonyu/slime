"""Score, freeze, and report row-aligned policy action streams."""

import math
from argparse import Namespace
from urllib.parse import quote

import torch

from slime.utils.policy import action_targets


def validate_action_policy_args(args: Namespace) -> None:
    """Check the supported execution modes for explicitly supplied action streams."""
    requirements = {
        "calculate_per_token_loss": False,
        "normalize_advantages": False,
        "use_critic": False,
        "use_opsm": False,
        "use_tis": False,
        "get_mismatch_metrics": False,
        "use_score_centering": False,
        "kl_coef": 0,
        "custom_pg_loss_reducer_function_path": None,
        "custom_advantage_function_path": None,
        "compute_advantages_and_returns": True,
        "opd_full_vocab": False,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "sequence_parallel": False,
    }
    for key, expected in requirements.items():
        value = getattr(args, key, expected)
        if value != expected:
            raise ValueError(f"Megatron action-stream training requires {key}={expected!r}; got {value!r}")
    if not (
        getattr(args, "advantage_estimator", None) == "grpo"
        and getattr(args, "loss_type", None) == "policy_loss"
        and getattr(args, "pg_loss_type", None) in (None, "ppo")
    ):
        raise ValueError("Action-stream scoring currently supports native GRPO/OPD with PPO clipping")
    weights = getattr(args, "policy_loss_weights", None) or {}
    if not isinstance(weights, dict):
        raise ValueError("policy_loss_weights must map stream names to nonnegative weights")
    for name, value in weights.items():
        if not isinstance(name, str) or not name or "/" in name or not math.isfinite(value) or value < 0:
            raise ValueError("Policy stream loss weights must be finite and nonnegative")


def policy_forward_kwargs(args, batch, model):
    """Pass replay targets only to models that explicitly accept this input contract."""
    while hasattr(model, "module"):
        model = model.module
    if not getattr(model, "accepts_policy_actions", False):
        return {}
    if not batch.get("policy_actions"):
        raise ValueError("A model with accepts_policy_actions=True requires action streams")
    return {
        "policy_actions": action_targets(batch["policy_actions"]),
        "policy_return_entropy": args.use_rollout_entropy or args.entropy_coef != 0,
    }


def get_policy_scores(args: Namespace, batch: dict, output: torch.Tensor) -> dict[str, list[torch.Tensor]]:
    """Extract row-aligned selected scores with the configured model callback."""
    from slime.utils.misc import load_function

    validate_action_policy_args(args)
    streams = batch.get("policy_actions")
    if not streams:
        raise ValueError("Policy scoring requires action streams")
    path = getattr(args, "custom_megatron_log_prob_function_path", None)
    if not path:
        raise ValueError("Explicit action streams require --custom-megatron-log-prob-function-path")
    extractor = load_function(path)
    scores = extractor(args, batch, output)
    if not isinstance(scores, dict):
        raise TypeError("Policy score extractors must return a dictionary of per-sample tensor lists")
    for name in streams:
        values = scores.get(f"policy/{name}/log_probs")
        if not isinstance(values, list) or len(values) != len(batch["total_lengths"]):
            raise ValueError(f"Policy scores for {name} must contain one tensor per sample")
    return {key: value for key, value in scores.items() if key.startswith("policy/")}


def grouped_opd_metrics(args, rollout_data):
    if not args.use_opd:
        return {}
    inputs = rollout_data.get("policy_actions")
    if not inputs:
        return {}
    source = "rollout" if args.use_rollout_logprobs else "old"
    domains = rollout_data.get("policy_domains", [None] * len(inputs))
    teachers = rollout_data.get("policy_teachers", [None] * len(inputs))
    masks = rollout_data.get("policy_sample_masks", [True] * len(inputs))
    totals, counts = {}, {}
    for streams, domain, teacher, enabled in zip(inputs, domains, teachers, masks, strict=True):
        for name, stream in streams.items():
            mask = stream["valid_mask"] & stream["loss_mask"]
            old = stream[f"{source}_log_probs"][mask].detach()
            if not old.numel():
                continue
            active = torch.as_tensor(enabled, device=old.device, dtype=old.dtype).reshape(())
            value = (old - stream["teacher_log_probs"][mask].detach()).mean() * active
            for field, label in (("domain", domain), ("teacher", teacher)):
                if label is not None:
                    key = f"{field}/{quote(str(label), safe='')}/{name}/opd_logratio_mean"
                    totals[key] = totals.get(key, 0) + value
                    counts[key] = counts.get(key, 0) + active
    if not totals:
        return {}
    keys = sorted(totals)
    values = torch.stack([torch.stack((totals[key], counts[key])) for key in keys]).cpu().tolist()
    return {key: (total, count) for key, (total, count) in zip(keys, values, strict=True) if count > 0}


def store_old_action_scores(rollout_data, scores, prefix="old"):
    inputs = rollout_data.get("policy_actions")
    for i, streams in enumerate(inputs):
        for name, stream in streams.items():
            stream[f"{prefix}_log_probs"] = scores[f"policy/{name}/log_probs"][i].detach().clone()
            if f"policy/{name}/entropy" in scores:
                stream[f"{prefix}_entropy"] = scores[f"policy/{name}/entropy"][i].detach().clone()


def policy_rollout_metrics(rollout_data):
    """Summarize actual action tensors; all means carry explicit sample counts."""
    trajectories = rollout_data["policy_actions"]
    enabled_samples = rollout_data.get("policy_sample_masks", [True] * len(trajectories))
    totals, counts = {}, {}
    for streams, enabled in zip(trajectories, enabled_samples, strict=True):
        for name, stream in streams.items():
            selected = stream["valid_mask"] & stream["loss_mask"]
            active = torch.as_tensor(enabled, device=selected.device, dtype=torch.float32).reshape(())
            for label, mask in (("valid_actions", stream["valid_mask"]), ("train_actions", selected)):
                key = f"policy/{name}/{label}"
                totals[key] = totals.get(key, 0) + mask.sum() * active
                counts[key] = counts.get(key, 0) + active
            for field in (
                "old_log_probs",
                "rollout_log_probs",
                "teacher_log_probs",
                "ref_log_probs",
                "old_entropy",
                "ref_entropy",
                "teacher_entropy",
                "advantages",
                "opd_reverse_kl",
            ):
                if field not in stream:
                    continue
                values = stream[field][selected].detach()
                if not values.numel():
                    continue
                key = f"policy/{name}/{field}"
                totals[key] = totals.get(key, 0) + values.mean() * active
                counts[key] = counts.get(key, 0) + active
    keys = sorted(totals)
    if not keys:
        return {}
    values = torch.stack([torch.stack((totals[key], counts[key])) for key in keys]).cpu().tolist()
    return {key: (total, count) for key, (total, count) in zip(keys, values, strict=True) if count > 0}
