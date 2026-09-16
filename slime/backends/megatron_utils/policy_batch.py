"""Structured policy batching and CP ownership, independent of model family.

The source position of a prediction owns its complete action row. Input rows
and target rows are separate index spaces, including at CP boundaries.
"""

from dataclasses import dataclass, fields, replace

import torch


@dataclass
class PolicyBatch:
    input_rows: torch.Tensor
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    prediction_positions: torch.Tensor
    targets: torch.Tensor
    action_mask: torch.Tensor
    old_logprobs: torch.Tensor
    advantages: torch.Tensor
    temperatures: torch.Tensor
    sample_indices: torch.Tensor
    sample_denominators: torch.Tensor
    group_denominators: torch.Tensor
    action_row_indices: torch.Tensor
    global_row_indices: torch.Tensor
    prediction_counts: torch.Tensor
    teacher_logprobs: torch.Tensor | None = None
    group_names: tuple[str, ...] = ()

    def to(self, device):
        return replace(
            self,
            **{
                f.name: getattr(self, f.name).to(device, non_blocking=True)
                for f in fields(self)
                if torch.is_tensor(getattr(self, f.name))
            },
        )


def cp_row_indices(cu_seqlens, cp_size, cp_rank):
    """Megatron/TE's per-sequence zigzag layout, including physical padding."""
    if cp_size < 1 or not 0 <= cp_rank < cp_size:
        raise ValueError("Invalid CP rank/size")
    spans = []
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
        start, end = int(start), int(end)
        if cp_size == 1:
            spans.append(torch.arange(start, end))
            continue
        if (end - start) % (2 * cp_size):
            raise ValueError("Packed sequence lengths must be divisible by 2*CP")
        chunk = (end - start) // (2 * cp_size)
        for part in (cp_rank, 2 * cp_size - cp_rank - 1):
            spans.append(torch.arange(start + part * chunk, start + (part + 1) * chunk))
    return torch.cat(spans).long()


def shard_policy_batch(batch: PolicyBatch, cp_size=1, cp_rank=0):
    rows = cp_row_indices(batch.cu_seqlens, cp_size, cp_rank).to(batch.input_rows.device)
    lookup = torch.full((len(batch.input_rows),), -1, device=rows.device, dtype=torch.long)
    lookup[rows] = torch.arange(len(rows), device=rows.device)
    local_prediction_positions = lookup[batch.prediction_positions]
    selected = (local_prediction_positions >= 0).nonzero().flatten()
    action_fields = (
        "targets",
        "action_mask",
        "old_logprobs",
        "advantages",
        "temperatures",
        "sample_indices",
        "sample_denominators",
        "group_denominators",
        "action_row_indices",
        "teacher_logprobs",
    )
    return replace(
        batch,
        input_rows=batch.input_rows[rows],
        position_ids=batch.position_ids[rows],
        global_row_indices=batch.global_row_indices[rows],
        prediction_positions=local_prediction_positions[selected],
        **{key: getattr(batch, key)[selected] for key in action_fields if getattr(batch, key) is not None},
    )


def collate_policy_batch(
    samples, tensors, pad_row, temperatures, *, cp_size=1, cp_rank=0, pad_multiple=1, group_names=None
):
    """Pack model-provided rows and predictions, then apply the common CP route.

    ``tensors`` contains (input_rows, prediction_positions, targets, mask, old)
    for each sample. Channels remain intact; only the sequence axis is sharded.
    """
    import math

    if not samples or len(samples) != len(tensors) or len(samples) != len(temperatures):
        raise ValueError("A microbatch requires aligned non-empty sample/tensor lists")
    alignment = math.lcm(pad_multiple, 2 * cp_size if cp_size > 1 else 1)
    all_rows, all_positions, all_predictions, all_targets, all_masks, all_old = [], [], [], [], [], []
    all_advantages, all_temperatures, all_sample_indices, all_denominators, all_teachers = [], [], [], [], []
    all_group_denominators = []
    offsets = [0]
    has_teacher = [sample.teacher_scores is not None for sample in samples]
    if any(has_teacher) and not all(has_teacher):
        raise ValueError("Teacher scores must cover every sample in an MOPD microbatch")
    for index, (sample, pieces, temperature) in enumerate(zip(samples, tensors, temperatures, strict=True)):
        rows, positions, targets, mask, old = pieces
        if not (targets.shape == mask.shape == old.shape) or len(positions) != len(targets):
            raise ValueError("Actions, scores, masks and prediction positions are misaligned")
        if positions.numel() and (positions.min() < 0 or positions.max() >= len(rows)):
            raise ValueError("Prediction source is outside its sample")
        padding = (-len(rows)) % alignment
        all_predictions.append(positions + offsets[-1])
        all_rows.append(torch.cat((rows, pad_row.expand(padding, *rows.shape[1:]))))
        all_positions.append(torch.arange(len(rows) + padding))
        offsets.append(offsets[-1] + len(rows) + padding)
        all_targets.append(targets)
        all_masks.append(mask)
        all_old.append(old)
        all_advantages.append(torch.full_like(old, float(sample.advantage or 0)))
        all_temperatures.append(torch.as_tensor(temperature).float().expand_as(old))
        all_sample_indices.append(torch.full((len(targets),), index, dtype=torch.long))
        all_denominators.append(torch.full((len(targets),), int(mask.sum()), dtype=torch.float32))
        all_group_denominators.append(mask.sum(0).float().expand_as(old))
        if all(has_teacher):
            if sample.teacher_scores.shape != targets.shape:
                raise ValueError("Teacher scores do not match the exact student actions")
            all_teachers.append(sample.teacher_scores)
    targets = torch.cat(all_targets)
    group_names = tuple(group_names or (f"head_{index}" for index in range(targets.shape[1])))
    if len(group_names) != targets.shape[1] or len(set(group_names)) != len(group_names):
        raise ValueError("Action group names must uniquely describe every head")
    batch = PolicyBatch(
        input_rows=torch.cat(all_rows),
        position_ids=torch.cat(all_positions),
        cu_seqlens=torch.tensor(offsets, dtype=torch.int32),
        prediction_positions=torch.cat(all_predictions),
        targets=targets,
        action_mask=torch.cat(all_masks),
        old_logprobs=torch.cat(all_old),
        advantages=torch.cat(all_advantages),
        temperatures=torch.cat(all_temperatures),
        sample_indices=torch.cat(all_sample_indices),
        sample_denominators=torch.cat(all_denominators),
        group_denominators=torch.cat(all_group_denominators),
        action_row_indices=torch.arange(len(targets)),
        global_row_indices=torch.arange(offsets[-1]),
        prediction_counts=torch.tensor([len(pieces[1]) for pieces in tensors]),
        teacher_logprobs=torch.cat(all_teachers) if all_teachers else None,
        group_names=group_names,
    )
    return shard_policy_batch(batch, cp_size, cp_rank)


def sum_sample_means(values, batch: PolicyBatch):
    """CP-local contributions with complete-sample denominators."""
    values = torch.where(batch.action_mask, values, 0)
    return (values.sum(-1) / batch.sample_denominators.clamp_min(1)).sum()
