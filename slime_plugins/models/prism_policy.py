# Copyright (c) 2025, OpenMOSS. All rights reserved.
"""Prism policy contracts."""

from __future__ import annotations

import torch
from torch import Tensor

from slime_plugins.models.prism_topology import PrismAudioRole, PrismItemKind


def selected_logprob_and_entropy(logits: Tensor, targets: Tensor, *, return_entropy=False):
    """FP32 temperature-one categorical scores, including the binary stop."""
    logprobs = logits.float().log_softmax(-1)
    selected = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    entropy = -(logprobs.exp() * logprobs).sum(-1) if return_entropy else None
    return selected, entropy


def _carrier_tensor(value, name, shape, dtype, device, *, pad_value=0):
    if torch.is_tensor(value) and value.ndim == len(shape) - 1:
        value = value.unsqueeze(0)
    if (
        torch.is_tensor(value)
        and value.ndim == len(shape)
        and value.shape[0] == 1
        and tuple(value.shape[2:]) == shape[2:]
        and value.shape[1] < shape[1]
    ):
        padding = torch.full(
            (1, shape[1] - value.shape[1], *shape[2:]), pad_value, dtype=value.dtype, device=value.device
        )
        value = torch.cat((value, padding), dim=1)
    if not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype != dtype or value.device != device:
        raise ValueError(f"{name} must be {dtype} {shape} on {device}")
    return value


def prepare_policy_inputs(
    input_ids: Tensor,
    *,
    audio_codes,
    item_kind,
    audio_role,
    successor_audio_codes,
    successor_audio_mask,
    target_history_retention_mask,
    policy_actions,
    n_vq,
    speech_vocab_size,
    text_vocab_size,
    text_pad_idx,
    audio_start_token_id=151652,
    cu_seqlens=None,
) -> dict[str, Tensor]:
    """Validate a chronological carrier without imposing SFT terminal labels.

    Reference and prompt-prefix successor edges are real conditioning even
    when their action masks are zero. A zero-frame sampled stop may live on
    the TEXT/NONE audio_start row; a length cap has no invented stop action.
    """
    if (
        not torch.is_tensor(input_ids)
        or input_ids.dtype != torch.int64
        or input_ids.ndim != 2
        or input_ids.shape[0] != 1
    ):
        raise ValueError("Prism input_ids must be int64 [1,T] text/control IDs")
    shape = tuple(input_ids.shape)
    rvq_shape = (*shape, n_vq)
    device = input_ids.device
    audio = _carrier_tensor(audio_codes, "audio_codes", rvq_shape, torch.int64, device)
    kinds = _carrier_tensor(item_kind, "item_kind", shape, torch.int64, device)
    roles = _carrier_tensor(audio_role, "audio_role", shape, torch.int64, device)
    codes = _carrier_tensor(successor_audio_codes, "successor_audio_codes", rvq_shape, torch.int64, device)
    edges = _carrier_tensor(successor_audio_mask, "successor_audio_mask", shape, torch.bool, device)
    actions = _carrier_tensor(policy_actions["rvq"]["actions"], "rvq actions", rvq_shape, torch.int64, device)
    code_mask = _carrier_tensor(policy_actions["rvq"]["valid_mask"], "rvq mask", rvq_shape, torch.bool, device)
    stop_mask = _carrier_tensor(policy_actions["stop"]["valid_mask"], "stop mask", shape, torch.bool, device)
    stops = _carrier_tensor(policy_actions["stop"]["actions"], "stop actions", shape, torch.int64, device)
    retention = (
        torch.ones(rvq_shape, dtype=torch.bool, device=device)
        if target_history_retention_mask is None
        else _carrier_tensor(
            target_history_retention_mask,
            "target_history_retention_mask",
            rvq_shape,
            torch.bool,
            device,
            pad_value=True,
        )
    )
    text = kinds.eq(PrismItemKind.TEXT)
    target = roles.eq(PrismAudioRole.TARGET)
    frame_actions = code_mask.any(-1)
    checks = (
        (code_mask & actions.ne(codes), "RVQ actions disagree with successor conditioning"),
        ((actions < 0) | (actions >= speech_vocab_size), "RVQ action outside vocabulary"),
        ((kinds < 0) | (kinds > 1), "invalid item_kind"),
        ((roles < 0) | (roles > 2) | text.ne(roles.eq(PrismAudioRole.NONE)), "invalid item/audio role"),
        ((input_ids < 0) | (input_ids >= text_vocab_size), "text ID outside vocabulary"),
        (~text & input_ids.ne(text_pad_idx), "audio rows require the text-pad filler"),
        (text.unsqueeze(-1) & audio.ne(0), "text rows require zero audio fillers"),
        (
            (audio < 0) | (audio >= speech_vocab_size) | (codes < 0) | (codes >= speech_vocab_size),
            "audio code outside vocabulary",
        ),
        (~edges.unsqueeze(-1) & codes.ne(0), "inactive successor payload must be zero"),
        (~target.unsqueeze(-1) & ~retention, "text/reference input must retain all channels"),
        (code_mask.ne(code_mask[..., :1]), "the policy must sample complete RVQ frames"),
        (
            frame_actions & (~edges | ~stop_mask | stops.ne(0)),
            "frame actions require a real edge and sampled continue",
        ),
        (stop_mask & stops.eq(0) & ~frame_actions, "a sampled continue must produce a complete frame"),
        (stop_mask & stops.eq(1) & edges, "a sampled stop cannot condition on an emitted successor"),
        ((stops < 0) | (stops > 1), "stop actions must be binary"),
        (
            stop_mask & ~(target | (text & input_ids.eq(audio_start_token_id))),
            "stop actions require target history or audio_start",
        ),
    )
    for invalid, message in checks:
        if bool(invalid.any()):
            raise ValueError(message)
    next_kind = torch.cat((kinds[:, 1:], torch.zeros_like(kinds[:, :1])), dim=1)
    next_role = torch.cat((roles[:, 1:], torch.zeros_like(roles[:, :1])), dim=1)
    next_codes = torch.cat((audio[:, 1:], torch.zeros_like(audio[:, :1])), dim=1)
    if bool((edges & (next_kind.ne(PrismItemKind.AUDIO) | codes.ne(next_codes).any(-1))).any()):
        raise ValueError("successor conditioning must contain the actual next audio frame")
    if bool((edges & ((~text & roles.ne(next_role)) | (text & input_ids.ne(audio_start_token_id)))).any()):
        raise ValueError("successor audio must remain within its reference/target span")
    if bool((frame_actions & next_role.ne(PrismAudioRole.TARGET)).any()):
        raise ValueError("policy frame actions cannot supervise a reference successor")
    if cu_seqlens is not None:
        boundaries = cu_seqlens.detach().cpu().tolist()
        if (
            len(boundaries) < 2
            or boundaries[0] != 0
            or boundaries[-1] != shape[1]
            or any(a >= b for a, b in zip(boundaries, boundaries[1:], strict=False))
        ):
            raise ValueError("packed boundaries must partition all chronological rows")
        ends = torch.tensor([position - 1 for position in boundaries[1:]], device=device)
        if bool(edges[0, ends].any()):
            raise ValueError("successor conditioning cannot cross a packed sequence boundary")
    return {
        "tokens": torch.cat((input_ids.unsqueeze(-1), audio), dim=-1),
        "item_kind": kinds,
        "audio_role": roles,
        "retention": retention,
        "successor_codes": codes,
        "successor_mask": edges,
        "code_mask": code_mask,
        "code_actions": actions,
        "stop_mask": stop_mask,
        "stop_actions": stops,
    }


def score_policy(args, batch, output: Tensor) -> dict[str, list[Tensor]]:
    """Slice packed scores without shifting rows or dropping stop-only actions.

    The row-aligned action arrays retain zero-frame terminal actions.
    Slime's forward-only callback owns its ``(empty_tensor, result)`` wrapper.
    """
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError("Prism policy output must have shape [1,T,C]")
    code_mask = batch["policy_actions"]["rvq"]["valid_mask"]
    if code_mask.ndim == 3:
        code_mask = code_mask.squeeze(0)
    n_vq = code_mask.shape[-1]
    width = n_vq + 1
    with_entropy = getattr(args, "use_rollout_entropy", False) or getattr(args, "entropy_coef", 0.0) > 0
    if output.shape[-1] not in (width, 2 * width):
        raise ValueError("Prism policy output has an unexpected action dimension")
    if with_entropy and output.shape[-1] != 2 * width:
        raise ValueError("Prism entropy was requested but the model did not return entropy columns")
    results = {"policy/rvq/log_probs": [], "policy/stop/log_probs": []}
    if output.shape[-1] == 2 * width:
        results.update({"policy/rvq/entropy": [], "policy/stop/entropy": []})
    offset = 0
    for length in batch["total_lengths"]:
        length = int(length)
        end = offset + length
        codes = output[0, offset:end, :n_vq]
        stops = output[0, offset:end, n_vq]
        results["policy/rvq/log_probs"].append(codes)
        results["policy/stop/log_probs"].append(stops)
        if output.shape[-1] == 2 * width:
            code_entropy = output[0, offset:end, width : width + n_vq]
            stop_entropy = output[0, offset:end, width + n_vq]
            results["policy/rvq/entropy"].append(code_entropy)
            results["policy/stop/entropy"].append(stop_entropy)
        offset = end
    if offset > output.shape[1] or offset != code_mask.shape[0]:
        raise ValueError("packed policy masks and total_lengths disagree")
    return results
