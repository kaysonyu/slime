"""Rollout lifecycle and DP planning for native Omni trajectories."""

import itertools
import math
import time

import ray
import torch

from slime.backends.sglang_omni_utils.client import OmniClient
from slime.observability import logging_utils
from slime.observability.rollout_data_utils import load_debug_rollout_data, save_debug_rollout_data
from slime.observability.rollout_metrics import log_tts_rollout_data
from slime.rollout.base_types import call_rollout_fn
from slime.rollout.data_source import RolloutDataSource
from slime.rollout.evaluation import evaluation_source
from slime.utils.dp_schedule import build_dp_schedule
from slime.utils.misc import Box, load_function

from .utils import Lock


@ray.remote
class RolloutManager:
    def __init__(self, args):
        logging_utils.configure_logger()
        self.args = args
        self.data_source = RolloutDataSource(args)
        self.generate_rollout = load_function(args.rollout_function_path)
        self.eval_generate_rollout = load_function(args.eval_function_path)
        self.args.teacher_versions = {}
        self.weight_version = None
        self.rollout_engine_lock = Lock.options(num_cpus=0.1).remote()
        control_actor = ray.remote(num_cpus=0.1)(OmniClient)
        self.rollout_engines = [
            control_actor.remote(url, args.omni_stage, args.omni_timeout) for url in args.omni_endpoints
        ]
        self.engine_gpu_counts = []
        if self.rollout_engines:
            infos = ray.get([engine.get_model_info.remote() for engine in self.rollout_engines])
            from importlib import import_module

            adapter = import_module(f"slime_plugins.models.{args.model_family}.data")
            for info in infos:
                if info.get("supports_weight_update") is not True or 2 not in info.get("rollout_schema_versions", []):
                    raise ValueError(
                        "Student Omni stages must support complete weight updates and exact trajectory schema v2"
                    )
                adapter.validate_model_identity(args.policy_config, info.get("model_identity", {}))
            self.engine_gpu_counts = [int(info.get("tp_size", info.get("tensor_parallel_size", 1))) for info in infos]
        self.train_parallel_config = None
        logging_utils.init_tracking(args, primary=False)

    def get_num_rollout_per_epoch(self):
        return max(1, math.ceil(len(self.data_source) / self.args.rollout_batch_size))

    def set_train_parallel_config(self, config):
        self.train_parallel_config = config

    def get_updatable_engines_and_lock(self):
        return self.rollout_engines, self.rollout_engine_lock, 0, self.engine_gpu_counts, [], []

    def set_weight_version(self, version):
        self.weight_version = str(version)
        self.args.expected_weight_version = self.weight_version

    def generate(self, rollout_id):
        started = time.monotonic()
        if self.args.load_debug_rollout_data:
            samples = load_debug_rollout_data(self.args.load_debug_rollout_data, rollout_id=rollout_id)
            metrics = {}
        else:
            output = call_rollout_fn(self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False)
            samples = list(itertools.chain.from_iterable(output.samples))
            metrics = output.metrics or {}
        if not samples or any(
            sample.trajectory is None
            or sample.remove_sample
            or sample.status not in (sample.Status.COMPLETED, sample.Status.TRUNCATED)
            for sample in samples
        ):
            raise ValueError("A TTS rollout must return non-empty native trajectories")
        if self.weight_version is not None and not self.args.load_debug_rollout_data:
            if any(sample.trajectory.weight_version != self.weight_version for sample in samples):
                raise ValueError("Generated trajectory does not match the published student version")
        save_debug_rollout_data(self.args.save_debug_rollout_data, samples, rollout_id=rollout_id, evaluation=False)
        log_tts_rollout_data(rollout_id, self.args, samples, metrics, time.monotonic() - started)
        if self.args.debug_rollout_only:
            return None
        groups = {}
        for sample in samples:
            if sample.group_index is None or sample.index is None:
                raise ValueError("Every rollout sample requires explicit sample and group identities")
            groups.setdefault(sample.group_index, []).append(sample)
        for group in groups.values():
            if self.args.objective == "grpo":
                if len(group) != self.args.n_samples_per_prompt:
                    raise ValueError("GRPO normalization requires the complete prompt group")
                rewards = torch.tensor([s.get_reward_value(self.args) for s in group], dtype=torch.float32)
                if not torch.isfinite(rewards).all():
                    raise ValueError("Non-finite rewards cannot enter training")
                advantages = rewards - rewards.mean()
                if self.args.grpo_std_normalization:
                    advantages = advantages / (rewards.std(unbiased=False) + 1e-6)
                for sample, advantage in zip(group, advantages.tolist(), strict=True):
                    sample.advantage = advantage
            else:
                for sample in group:
                    if sample.teacher_scores is None:
                        raise ValueError("MOPD sample is missing teacher action scores")
                    sample.advantage = 0.0
        return self._split_train_data_by_dp(samples)

    def _split_train_data_by_dp(self, samples):
        config = self.train_parallel_config
        if config is None:
            raise RuntimeError("Trainer must publish its parallel layout before rollout")
        cp_size = config["cp_size"]
        alignment = math.lcm(self.args.data_pad_size_multiplier, 2 * cp_size if cp_size > 1 else 1)
        lengths = [math.ceil(s.trajectory.sequence_length / alignment) * alignment for s in samples]
        rollout_ids = [s.index for s in samples]
        partitions, microbatches, counts, batch_sizes = build_dp_schedule(
            self.args,
            config,
            lengths,
            global_batch_size=self.args.global_batch_size,
            rollout_indices=rollout_ids,
        )
        if sum(len(p) for p in partitions) != len(samples):
            raise ValueError("TTS batch size must divide the rollout sample count; refusing to discard samples")
        return [
            Box(
                ray.put(
                    dict(
                        samples=[samples[i] for i in partition],
                        partition=partition,
                        num_microbatches=counts,
                        global_batch_sizes=batch_sizes,
                        micro_batch_indices=microbatches[rank],
                    )
                )
            )
            for rank, partition in enumerate(partitions)
        ]

    def eval(self, rollout_id):
        if self.args.debug_train_only:
            return
        for dataset in self.args.eval_datasets:
            eval_args, source = evaluation_source(self.args, dataset)
            started = time.monotonic()
            output = call_rollout_fn(self.eval_generate_rollout, eval_args, rollout_id, source, evaluation=True)
            for name, values in output.data.items():
                metrics = {
                    f"eval/{name}/{key.removeprefix('rollout/')}": value
                    for key, value in (output.metrics or {}).items()
                }
                log_tts_rollout_data(
                    rollout_id,
                    eval_args,
                    values["samples"],
                    metrics,
                    time.monotonic() - started,
                    prefix=f"eval/{name}",
                )

    def save(self, rollout_id):
        self.data_source.metadata["teacher_versions"] = self.args.teacher_versions
        self.data_source.save(rollout_id)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)
        self.args.teacher_versions = self.data_source.metadata.get("teacher_versions", {})

    def dispose(self):
        if self.rollout_engines:
            ray.get([engine.close.remote() for engine in self.rollout_engines])
            for engine in self.rollout_engines:
                ray.kill(engine)
        logging_utils.finish_tracking(self.args)
