"""Row-aligned policy actions, independent of the model's conditioning inputs."""

from typing import TypedDict

import torch


class ActionStream(TypedDict, total=False):
    """One action family; required trajectory fields are checked at the data boundary."""

    actions: torch.Tensor
    valid_mask: torch.Tensor
    loss_mask: torch.Tensor
    rollout_log_probs: torch.Tensor
    teacher_log_probs: torch.Tensor
    old_log_probs: torch.Tensor
    train_log_probs: torch.Tensor
    old_entropy: torch.Tensor
    ref_entropy: torch.Tensor
    teacher_entropy: torch.Tensor
    ref_log_probs: torch.Tensor
    advantages: torch.Tensor
    opd_reverse_kl: torch.Tensor


PolicyActions = dict[str, ActionStream]


def action_targets(streams: PolicyActions) -> PolicyActions:
    """Expose only replay targets to the model, keeping frozen scores in training data."""
    return {name: {field: stream[field] for field in ("actions", "valid_mask")} for name, stream in streams.items()}


def validate_policy_actions(streams: PolicyActions, rows: int) -> None:
    if not streams:
        raise ValueError("A policy trajectory must declare at least one action stream")
    for name, stream in streams.items():
        if not isinstance(name, str) or not name or "/" in name:
            raise ValueError("Policy stream names must be nonempty path components")
        required = {"actions", "valid_mask", "loss_mask"}
        if missing := required - stream.keys():
            raise ValueError(f"Policy stream {name} is missing {sorted(missing)}")
        valid = stream["valid_mask"]
        if not torch.is_tensor(valid) or valid.dtype != torch.bool or valid.ndim < 1 or valid.shape[0] != rows:
            raise ValueError(f"Policy stream {name} requires a boolean valid_mask aligned to {rows} sequence rows")
        for field, value in stream.items():
            if not torch.is_tensor(value) or value.shape != valid.shape:
                raise ValueError(f"Policy stream {name}/{field} must have shape {tuple(valid.shape)}")
            expected = (
                torch.long if field == "actions" else torch.bool if field in {"valid_mask", "loss_mask"} else None
            )
            if expected is not None and value.dtype != expected:
                raise ValueError(f"Policy stream {name}/{field} must have dtype {expected}")
            if expected is None and (not value.dtype.is_floating_point or not torch.isfinite(value[valid]).all()):
                raise ValueError(f"Policy stream {name}/{field} must contain finite floating-point action scores")
        if (stream["loss_mask"] & ~valid).any():
            raise ValueError(f"Policy stream {name} loss_mask selects an invalid action")
