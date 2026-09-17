"""TTS conditioning records and resumable prompt-group sampling."""

import copy
import logging
import os
import random
import tempfile
from pathlib import Path

import torch

from slime.rollout.tts_data import load_speech_manifest
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class TTSPromptDataset:
    """Raw conditioning records; model/Omni owns prompt rendering and codec work."""

    def __init__(self, path, seed, **options):
        self.seed = seed
        self.original, self.identity, self.source_sha256, self.has_references = load_speech_manifest(path, **options)
        self.samples = list(self.original)

    def shuffle(self, epoch):
        self.samples = list(self.original)
        random.Random(self.seed + epoch).shuffle(self.samples)

    def __len__(self):
        return len(self.samples)


class RolloutDataSource:
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        self.metadata = {}

        domains = None
        if getattr(args, "objective", "grpo") == "mopd":
            domains = {entry.partition("=")[0] for entry in args.mopd_teachers}
        self.dataset = (
            TTSPromptDataset(
                args.prompt_data,
                args.rollout_seed,
                language=getattr(args, "wer_language", "en"),
                reward_config=getattr(args, "reward_configuration", None) if domains is None else None,
                hf_checkpoint=getattr(args, "hf_checkpoint", None),
                metadata_overrides=getattr(args, "metadata_overrides", None),
                teacher_domains=domains,
            )
            if args.prompt_data
            else None
        )
        if self.dataset is not None and args.rollout_shuffle:
            self.dataset.shuffle(0)

    def get_samples(self, num_samples):
        if self.dataset is not None:
            prompt_samples = []
            while len(prompt_samples) < num_samples:
                remaining = num_samples - len(prompt_samples)
                count = min(remaining, len(self.dataset) - self.sample_offset)
                prompt_samples.extend(self.dataset.samples[self.sample_offset : self.sample_offset + count])
                self.sample_offset += count
                if self.sample_offset == len(self.dataset):
                    self.epoch_id += 1
                    self.sample_offset = 0
                    if self.args.rollout_shuffle:
                        self.dataset.shuffle(self.epoch_id)
        else:
            prompt_samples = [Sample() for _ in range(num_samples)]

        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return

        state_dict = {
            "state_version": 2,
            "dataset_sha256": self.dataset.identity if self.dataset is not None else None,
            "sampling_seed": self.args.rollout_seed,
            "sampling_shuffle": self.args.rollout_shuffle,
            "n_samples_per_prompt": self.args.n_samples_per_prompt,
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
        }
        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".sampler-", suffix=".tmp")
        try:
            with os.fdopen(descriptor, "wb") as stream:
                torch.save(state_dict, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset:
            return

        if self.args.load is None:
            return

        directory = Path(self.args.load)
        if directory.name.startswith("iter_"):
            directory = directory.parent
        path = directory / f"rollout/global_dataset_state_dict_{rollout_id}.pt"
        if not os.path.exists(path):
            if rollout_id is not None and rollout_id >= 0:
                raise FileNotFoundError(f"Native resume requires the matching rollout sampler state: {path}")
            logger.info(f"Checkpoint {path} does not exist.")
            return

        logger.info(f"load metadata from {path}")
        state_dict = torch.load(path, weights_only=True)
        expected = self.dataset.identity if self.dataset is not None else None
        if "state_version" not in state_dict:
            if self.dataset is not None and self.dataset.has_references:
                raise ValueError("Legacy sampler checkpoint cannot verify reference audio identity")
            expected = self.dataset.source_sha256 if self.dataset is not None else None
            logger.warning(
                "Restoring a legacy sampler: only JSONL/seed identity is available; WER now uses the migrated definition"
            )
        elif (
            state_dict["state_version"] != 2
            or state_dict.get("n_samples_per_prompt") != self.args.n_samples_per_prompt
        ):
            raise ValueError("Sampler checkpoint version or prompt fanout changed")
        if state_dict.get("dataset_sha256") != expected or state_dict.get("sampling_seed") != self.args.rollout_seed:
            raise ValueError("Resume requires the same prompt dataset and sampling seed")
        if state_dict.get("sampling_shuffle", False) != self.args.rollout_shuffle:
            raise ValueError("Resume requires the same prompt shuffle setting")
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})
        if any(
            type(value) is not int or value < 0
            for value in (self.sample_offset, self.epoch_id, self.sample_group_index, self.sample_index)
        ):
            raise ValueError("Invalid sampler checkpoint counters")
        if self.dataset is not None and self.sample_offset >= len(self.dataset):
            raise ValueError("Sampler offset is outside the dataset")

        if self.args.rollout_global_dataset and self.args.rollout_shuffle and self.dataset is not None:
            self.dataset.shuffle(self.epoch_id)

    def __len__(self) -> int:
        if self.dataset is None:
            return 0
        return len(self.dataset)
