"""Action objectives and their Megatron gradient-scaling boundary."""

import torch
import torch.distributed as dist
from megatron.core import mpu


def collect_policy_scores(output, *, batch, **kwargs):
    """Reassemble selected action scores for diagnostics, never full-vocabulary logits."""
    full = output.new_zeros((int(batch.prediction_counts.sum()), output.shape[1]))
    full[batch.action_row_indices] = output
    if mpu.get_context_parallel_world_size() > 1:
        dist.all_reduce(full, group=mpu.get_context_parallel_group())
    return output.new_zeros(()), {
        "log_probs": [value.detach().cpu() for value in full.split(batch.prediction_counts.tolist())]
    }


def structured_policy_loss(args, batch, num_microbatches, step_global_batch_size, current):
    from slime.utils.ppo_utils import action_policy_terms, mopd_action_terms

    from .policy_batch import sum_sample_means

    mask = batch.action_mask
    if args.objective == "grpo":
        terms, ratio, clipped = action_policy_terms(
            current,
            batch.old_logprobs,
            batch.advantages,
            mask,
            eps_clip=args.eps_clip,
            eps_clip_high=args.eps_clip_high,
        )
    else:
        if batch.teacher_logprobs is None:
            raise ValueError("MOPD requires a teacher score for every action")
        terms, distillation_advantage = mopd_action_terms(
            current,
            batch.old_logprobs,
            batch.teacher_logprobs,
            mask,
            advantage_clip=args.mopd_advantage_clip,
        )
        ratio = (torch.where(mask, current - batch.old_logprobs, 0)).exp()
        clipped = torch.zeros_like(ratio)
    loss = sum_sample_means(terms, batch)
    # Preserve the global attention backward graph on ranks without trainable actions.
    loss = loss + current.sum() * 0
    metric_values = {
        "loss": loss.detach(),
        "pg_loss": loss.detach(),
        "ratio_mean": sum_sample_means(ratio.detach(), batch),
        "clip_fraction": sum_sample_means(clipped.detach(), batch),
        "behavior_logprob_abs_diff": sum_sample_means((current.detach() - batch.old_logprobs).abs(), batch),
    }
    if args.objective == "mopd":
        metric_values["mopd/advantage"] = sum_sample_means(distillation_advantage, batch)
        metric_values["mopd/advantage_abs"] = sum_sample_means(distillation_advantage.abs(), batch)
        metric_values["mopd/teacher_logprob"] = sum_sample_means(batch.teacher_logprobs, batch)
    nll = (torch.where(mask, -current.detach(), 0) / batch.group_denominators.clamp_min(1)).sum(0)
    for index, (group, value) in enumerate(zip(batch.group_names, nll, strict=True)):
        metric_values[f"model/{group}/nll"] = value
        metric_values[f"model/{group}/actions_per_sample"] = mask[:, index].sum().to(current.dtype)
    # Megatron's three-value loss contract divides by num_microbatches; DDP
    # averages over DP*CP. These factors yield the mean over complete samples.
    scaled = (
        loss * num_microbatches * mpu.get_data_parallel_world_size(with_context_parallel=True) / step_global_batch_size
    )
    return (
        scaled,
        current.new_tensor(1, dtype=torch.int64),
        {
            "keys": list(metric_values),
            "values": torch.stack([current.new_zeros(()), *metric_values.values()]),
        },
    )
