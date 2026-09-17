from argparse import Namespace
from collections.abc import Iterator, Sequence

import torch
import torch.distributed as dist
from megatron.core import mpu


def all_gather_param(name: str, param: torch.nn.Parameter) -> torch.Tensor:
    """
    All-gather TP-sharded param to full tensor. expert_bias→param,
    non-TP/duplicated/TP-size-1→param.data.
    Uses expert-TP for ".experts.", else regular-TP. linear_fc1 rechunked (GLU), linear_fc2 dim fix.
    """
    if "expert_bias" in name:
        return param

    assert hasattr(param, "tensor_model_parallel"), f"{name} does not have tensor_model_parallel attribute"
    if not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
        return param.data

    if ".experts." in name:
        tp_size = mpu.get_expert_tensor_parallel_world_size()
    else:
        tp_size = mpu.get_tensor_model_parallel_world_size()

    if tp_size == 1:
        return param.data

    if ".experts." in name:
        tp_group = mpu.get_expert_tensor_parallel_group()
    else:
        tp_group = mpu.get_tensor_model_parallel_group()

    param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
    dist.all_gather(param_partitions, param.data, group=tp_group)
    partition_dim = param.partition_dim
    assert param.partition_stride == 1 or (
        param.partition_stride == 2 and "linear_fc1" in name
    ), "partition_stride != 1 is not supported"
    # TODO: here we did an extra copy during concat, maybe merge this with convert_to_hf is better?
    # TODO: check only GLU is used.
    if "linear_fc1.weight" in name or "linear_fc1.bias" in name:
        param_partitions = [p.chunk(2, dim=0) for p in param_partitions]
        param_partitions = [p[0] for p in param_partitions] + [p[1] for p in param_partitions]
    # this is bug in megatron's grouped moe.
    if "linear_fc2.weight" in name:
        if partition_dim == 0:
            partition_dim = 1
    param = torch.cat(param_partitions, dim=partition_dim)
    return param


def named_params_and_buffers(args: Namespace, model: Sequence[torch.nn.Module]) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield policy parameters; supported speech backbones have PP=EP=1."""
    for module in model:
        yield from module.named_parameters()
