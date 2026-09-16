from __future__ import annotations

import socket
import time
from argparse import Namespace
from collections.abc import Iterator, Sequence
from importlib import import_module

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle
from tqdm import tqdm

from slime.utils import accelerator
from slime.utils.distributed_utils import get_gloo_group, init_process_group
from slime.utils.http_utils import _wrap_ipv6

from .common import all_gather_param, named_params_and_buffers


class UpdateWeightFromDistributed:
    """Gather native parameters, then publish complete HF weights to paused Omni stages."""

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.adapter = import_module(f"slime_plugins.models.{args.model_family}.weights")
        self.expected_names = self.adapter.expected_hf_names(args.hf_checkpoint)
        self.weight_version = 0
        self._model_update_groups = None

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
    ) -> None:
        """
        Create "slime-pp_{pp_rank}" if PP source (DP=TP=0). Lock prevents concurrent transfers.
        """
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock
        self._engine_gpu_counts = engine_gpu_counts

        # For TP:
        #   1. AllGather parameters to rank 0
        #   2. Broadcast parameters from rank 0 to all sglang engines
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        if self._is_pp_src_rank:
            self._group_name = f"slime-{self.args.weight_sync_session}-pp_{pp_rank}"

        if self._is_pp_src_rank:
            if self._model_update_groups is not None:
                disconnect_rollout_engines_from_distributed(
                    self._group_name, self._model_update_groups, self.rollout_engines
                )
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args,
                self._group_name,
                rollout_engines,
                engine_gpu_counts=engine_gpu_counts,
            )

    def disconnect_rollout_engines(self) -> None:
        if not getattr(self, "_is_pp_src_rank", False) or self._model_update_groups is None:
            return
        disconnect_rollout_engines_from_distributed(self._group_name, self._model_update_groups, self.rollout_engines)
        self._model_update_groups = None

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        Pause → transfer all tensors → resume. Omni invalidates its stage cache on refit.
        """
        self.weight_version += 1
        self._sent_names = set()

        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_pp_src_rank else None
        self._send_weights(pbar)
        if self._is_pp_src_rank and self._sent_names != self.expected_names:
            raise ValueError(
                f"Incomplete Omni weight publication: missing={sorted(self.expected_names - self._sent_names)}, "
                f"unexpected={sorted(self._sent_names - self.expected_names)}"
            )
        if pbar is not None:
            pbar.close()
        if dist.get_rank() == 0:
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _send_weights(self, pbar):
        for chunk in self._iter_chunks():
            names = [name for name, _ in chunk]
            if len(names) != len(set(names)) or self._sent_names.intersection(names):
                raise ValueError("Duplicate tensors in Omni weight publication")
            self._sent_names.update(names)
            self._update_bucket_weights_from_distributed(chunk, pbar=pbar)
        dist.barrier(group=get_gloo_group())

    def _iter_chunks(self) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """
        Yield broadcast-sized HF chunks of non-expert params: TP all-gather +
        HF convert per param, then bucket up to ``--update-weight-buffer-size``.
        Empty on non-PP-src ranks (they still join all_gather_param).
        """
        buffer_size = 0
        buffer: list[tuple[str, torch.Tensor]] = []
        for name, param in named_params_and_buffers(self.args, self.model):
            param = all_gather_param(name, param)
            if not self._is_pp_src_rank:
                continue
            hf_chunk = self.adapter.export_parameter(self.args, name, param)
            chunk_bytes = sum(t.numel() * t.element_size() for _, t in hf_chunk)
            if buffer and buffer_size + chunk_bytes > self.args.update_weight_buffer_size:
                yield buffer
                buffer = []
                buffer_size = 0
            buffer.extend(hf_chunk)
            buffer_size += chunk_bytes
        if buffer:
            yield buffer

    def _update_bucket_weights_from_distributed(
        self,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        pbar: tqdm | None = None,
        load_format: str | None = None,
    ) -> None:
        """
        Lock → transfer → clear → unlock → pbar++. Lock prevents communication deadlock.
        """
        # Lock the rollout engines to prevent communication deadlock.
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)

        try:
            refs = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.rollout_engines,
                converted_named_tensors,
                load_format=load_format,
            )
            ray.get(refs)
            converted_named_tensors.clear()
        finally:
            ray.get(self.rollout_engine_lock.release.remote())
        pbar.update(1)


def connect_rollout_engines_from_distributed(
    args: Namespace,
    group_name: str,
    rollout_engines: Sequence[ActorHandle],
    engine_gpu_counts: Sequence[int] | None = None,
) -> dist.ProcessGroup:
    """
    Create a device process group: training rank 0 + all engine GPUs. Blocks until joined.

    ``engine_gpu_counts`` gives the number of GPUs per engine.  When engines
    have heterogeneous TP sizes (e.g. prefill TP=2, decode TP=4), each engine
    occupies a different number of ranks in the process group.
    """
    if engine_gpu_counts is None:
        engine_gpu_counts = [args.rollout_num_gpus_per_engine] * len(rollout_engines)

    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    world_size = sum(engine_gpu_counts) + 1  # +1 for training rank 0

    # Compute cumulative rank offsets: engine i starts at cumulative[i] + 1.
    cumulative = [0]
    for c in engine_gpu_counts:
        cumulative.append(cumulative[-1] + c)

    backend = accelerator.weight_update_backend()
    refs = [
        engine.init_weights_update_group.remote(
            master_address=master_address,
            master_port=master_port,
            rank_offset=cumulative[i] + 1,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )
        for i, engine in enumerate(rollout_engines)
    ]
    model_update_groups = init_process_group(
        backend=backend,
        init_method=f"tcp://{_wrap_ipv6(master_address)}:{master_port}",
        world_size=world_size,
        rank=0,
        group_name=group_name,
    )
    ray.get(refs)
    return model_update_groups


def disconnect_rollout_engines_from_distributed(group_name, model_update_groups, rollout_engines):
    """
    Destroy the weight-update process group on training and engines.
    """
    refs = [engine.destroy_weights_update_group.remote(group_name) for engine in rollout_engines]
    dist.destroy_process_group(model_update_groups)
    ray.get(refs)


def update_weights_from_distributed(
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    load_format: str | None = None,
) -> list[ObjectRef]:
    """
    Send metadata through Ray and tensors through the configured transport.
    """
    refs = [
        engine.update_weights_from_distributed.remote(
            names=[name for name, _ in converted_named_tensors],
            dtypes=[param.dtype for _, param in converted_named_tensors],
            shapes=[param.shape for _, param in converted_named_tensors],
            group_name=group_name,
            weight_version=str(weight_version),
            load_format=load_format,
        )
        for engine in rollout_engines
    ]
    handles = []
    for _, param in converted_named_tensors:
        handles.append(dist.broadcast(param.data, 0, group=group, async_op=True))
    for handle in handles:
        handle.wait()

    return refs
