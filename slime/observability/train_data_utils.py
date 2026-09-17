"""Save already reassembled action-score diagnostics from the training actor."""

from pathlib import Path
import torch
from megatron.core import mpu


def save_debug_train_data(args, *, rollout_id, rollout_data):
    if args.save_debug_train_data is None:
        return
    if mpu.get_tensor_model_parallel_rank() != 0 or mpu.get_context_parallel_rank() != 0:
        return
    rank = mpu.get_data_parallel_rank(with_context_parallel=False)
    path = Path(args.save_debug_train_data.format(rollout_id=rollout_id, rank=rank))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"format_version": 1, "model_family": args.model_family, **rollout_data},
        path.with_name(f"dp{rank}-{path.name}"),
    )
