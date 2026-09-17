"""Microbatch iteration and common CP batching for native speech samples."""

from collections.abc import Sequence
from importlib import import_module

import torch
from megatron.core import mpu
from megatron.training.global_vars import get_args

from slime.utils import accelerator
from slime.utils.types import RolloutBatch
from .policy_batch import PolicyBatch, collate_policy_batch


def get_batch(data_iterator, keys=None, pad_multiplier=128, allgather_cp=False) -> PolicyBatch:
    args = get_args()
    if allgather_cp:
        raise ValueError("Speech batches currently use the standard Megatron zigzag CP layout")
    samples = data_iterator.get_next(["samples"])["samples"]
    adapter = import_module(f"slime_plugins.models.{args.model_family}.data")
    tensors, temperatures, pad_row, group_names = adapter.batch_inputs(samples, args.policy_config)
    return collate_policy_batch(
        samples,
        tensors,
        pad_row,
        temperatures,
        cp_size=mpu.get_context_parallel_world_size(),
        cp_rank=mpu.get_context_parallel_rank(),
        pad_multiple=pad_multiplier,
        group_names=group_names,
    ).to(accelerator.current_device())


class DataIterator:
    """Iterator over a rollout dict following an explicit micro-batch index schedule."""

    def __init__(
        self,
        rollout_data: RolloutBatch,
        micro_batch_indices: list[list[int]],
    ) -> None:
        """Initialize an iterator over ``rollout_data``.

        Args:
            rollout_data: Dict of per-sample fields for this DP rank.
            micro_batch_indices: List of mbs, each mbs being the local sample indices to select.
        """
        self.rollout_data = rollout_data
        self.micro_batch_indices = micro_batch_indices
        self.offset = 0

    def get_next(self, keys: Sequence[str]) -> dict[str, list[object] | None]:
        """Return the next micro-batch for the requested keys.

        Returns a dict mapping each key to a list subset (or None if absent).
        """
        batch = {}
        indices = self.micro_batch_indices[self.offset]
        for key in keys:
            vals = self.rollout_data.get(key, None)
            if vals is None:
                batch[key] = None
            else:
                batch[key] = [vals[i] for i in indices]
        self.offset += 1
        return batch

    def reset(self) -> "DataIterator":
        """Reset internal offset to the start and return self."""
        self.offset = 0
        return self


def get_data_iterator(rollout_data: RolloutBatch) -> list[DataIterator]:
    """Build one ``DataIterator`` per VPP stage from the pre-computed schedule in ``rollout_data``."""
    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
    micro_batch_indices = rollout_data["micro_batch_indices"]
    return [DataIterator(rollout_data, micro_batch_indices) for _ in range(vpp_size)]


def tensors_to_cpu(tensor_list):
    """Move a list of GPU tensors to CPU for Ray object store transfer.

    Args:
        tensor_list: List of GPU tensors, or None.

    Returns:
        List of CPU tensors (detached), or None if input is None.
    """
    if tensor_list is None:
        return None
    return [t.detach().cpu() for t in tensor_list]


def tensors_to_gpu(tensor_list, device=None):
    """Move a list of CPU tensors back to GPU.

    Args:
        tensor_list: List of CPU tensors, or None.
        device: Target CUDA device. If None, uses current device.

    Returns:
        List of GPU tensors, or None if input is None.
    """
    if tensor_list is None:
        return None
    if device is None:
        device = accelerator.current_device()
    return [t.to(device=device, dtype=torch.float32) for t in tensor_list]
