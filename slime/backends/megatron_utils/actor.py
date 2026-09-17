"""Megatron actor lifecycle for native speech trajectories."""

import logging

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu

from slime.observability.logging_utils import init_tracking
from slime.observability.profile_utils import TrainProfiler
from slime.observability.timer import Timer, inverse_timer, timer, with_defer
from slime.ray.train_actor import TrainRayActor
from slime.utils import accelerator
from slime.utils.memory_utils import clear_memory

from .data import get_data_iterator
from .initialize import init, is_megatron_main_rank
from .model import forward_only, initialize_model_and_optimizer, save, train
from .update_weight import create_weight_updater

logger = logging.getLogger(__name__)


class MegatronTrainRayActor(TrainRayActor):
    @with_defer(lambda: Timer().start("train_wait"))
    def init(self, args, role="actor", with_ref=False, with_opd_teacher=False):
        if role != "actor" or with_ref or with_opd_teacher:
            raise ValueError("TTS uses one trainable policy and external frozen teachers")
        if args.debug_rollout_only:
            self.args = args
            return 0
        super().init(args, role)
        init(args)
        if is_megatron_main_rank():
            init_tracking(args, primary=False, role=role)
        self.prof = TrainProfiler(args)
        self.model, self.optimizer, self.opt_param_scheduler, iteration = initialize_model_and_optimizer(args)
        self.train_parallel_config = {
            "dp_size": mpu.get_data_parallel_world_size(with_context_parallel=False),
            "cp_size": mpu.get_context_parallel_world_size(),
            "vpp_size": 1,
            "microbatch_group_size_per_vp_stage": 1,
        }
        start = 0 if args.initialization_checkpoint or args.finetune else iteration + 1
        if args.start_rollout_id is None:
            args.start_rollout_id = start
        self.weight_updater = create_weight_updater(
            args,
            self.model,
        )
        self.weight_updater.weight_version = start
        self.rollout_engines = None
        self.prof.on_init_end()
        clear_memory()
        return args.start_rollout_id

    def compute_log_prob(self, data_iterator, num_microbatches):
        return forward_only(self.args, self.model, data_iterator, num_microbatches)

    def train(self, rollout_id, rollout_data_ref, external_data=None):
        if self.args.debug_rollout_only:
            return
        selected = rollout_data_ref[mpu.get_data_parallel_rank(with_context_parallel=False)]
        data = ray.get(selected.inner)
        Timer().seq_lens = [sample.trajectory.sequence_length for sample in data["samples"]]
        iterator = get_data_iterator(data)
        counts, batch_sizes = data["num_microbatches"], data["global_batch_sizes"]
        if rollout_id == self.args.start_rollout_id or self.args.save_debug_train_data:
            scores = self.compute_log_prob(iterator, counts)["log_probs"]
            differences = []
            for sample, current in zip(data["samples"], scores, strict=True):
                _, _, _, mask, old = sample.trajectory.training_tensors(self.args.policy_config)
                differences.append((current - old)[mask].abs())
            values = torch.cat(differences)
            maximum = values.max().to(accelerator.current_device())
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            totals = torch.tensor([float(values.sum()), values.numel()], device=maximum.device, dtype=torch.float64)
            dist.all_reduce(totals)
            mean = float(totals[0] / totals[1].clamp_min(1))
            logger.info("Selected-action train/rollout absolute difference: max=%.6f mean=%.6f", maximum, mean)
            if float(maximum) > self.args.logprob_parity_tolerance or mean > self.args.logprob_parity_mean_tolerance:
                raise ValueError(f"Training/rollout score mismatch: max={float(maximum)}, mean={mean}")
            data["log_probs"] = scores
        with inverse_timer("train_wait"), timer("train"):
            train(rollout_id, self.model, self.optimizer, self.opt_param_scheduler, iterator, counts, batch_sizes)
        if self.args.save_debug_train_data:
            from slime.observability.train_data_utils import save_debug_train_data

            save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=data)
        self.prof.step(rollout_id)

    def save_model(self, rollout_id, force_sync=False):
        if self.args.debug_rollout_only:
            return
        save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)
        if force_sync and self.args.async_save:
            from megatron.training.async_utils import maybe_finalize_async_save

            maybe_finalize_async_save(blocking=True)

    def update_weights(self):
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return
        engines, lock, _, gpu_counts, *_ = ray.get(self.rollout_manager.get_updatable_engines_and_lock.remote())
        if self.rollout_engines is None:
            self.weight_updater.connect_rollout_engines(engines, lock, engine_gpu_counts=gpu_counts)
            self.rollout_engines = engines
        self.weight_updater.update_weights()
        if dist.get_rank() == 0:
            version = str(self.weight_updater.weight_version)
            infos = ray.get([engine.get_model_info.remote() for engine in engines])
            if any(str(info.get("weight_version")) != version for info in infos):
                raise RuntimeError("Omni replicas did not acknowledge the complete weight version")
            ray.get(self.rollout_manager.set_weight_version.remote(version))
        dist.barrier()

    def dispose(self):
        if self.args.debug_rollout_only:
            return
        if self.rollout_engines is not None:
            self.weight_updater.disconnect_rollout_engines()
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
