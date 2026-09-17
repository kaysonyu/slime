"""Duration bucket policy shared by eager and accelerator backends."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .assets import TARGET_SAMPLE_RATE


@dataclass(frozen=True, slots=True, order=True)
class BucketSpec:
    seconds: int
    max_batch: int = 16
    sample_rate: int = TARGET_SAMPLE_RATE

    @property
    def num_samples(self) -> int:
        return self.seconds * self.sample_rate


DEFAULT_BUCKETS = tuple(BucketSpec(seconds) for seconds in (4, 8, 12, 16, 20, 24, 30))


class BucketPolicy:
    """Choose fixed duration tensors to limit padding in dynamic batches."""

    def __init__(self, buckets: Sequence[BucketSpec] = DEFAULT_BUCKETS) -> None:
        if not buckets:
            raise ValueError("at least one bucket is required")
        ordered = tuple(sorted(buckets))
        if len({bucket.seconds for bucket in ordered}) != len(ordered):
            raise ValueError("bucket durations must be unique")
        self.buckets = ordered

    @property
    def version(self) -> str:
        return "b" + "-".join(str(bucket.seconds) for bucket in self.buckets)

    def select(self, num_samples: int) -> BucketSpec:
        if num_samples <= 0:
            raise ValueError("audio length must be positive")
        for bucket in self.buckets:
            if num_samples <= bucket.num_samples:
                return bucket
        raise ValueError(f"audio has {num_samples} samples, maximum is {self.buckets[-1].num_samples}")

    def group_indices(self, lengths: Sequence[int]) -> dict[BucketSpec, list[int]]:
        result: dict[BucketSpec, list[int]] = {}
        for index, length in enumerate(lengths):
            result.setdefault(self.select(int(length)), []).append(index)
        return result

    @staticmethod
    def pad(
        waveforms: Sequence[torch.Tensor],
        bucket: BucketSpec,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad waveforms to one bucket and retain their true lengths."""
        if not waveforms:
            raise ValueError("cannot pad an empty batch")
        if len(waveforms) > bucket.max_batch:
            raise ValueError(f"bucket {bucket.seconds}s supports at most {bucket.max_batch} items")
        batch = torch.zeros(len(waveforms), bucket.num_samples, dtype=torch.float32)
        lengths = torch.empty(len(waveforms), dtype=torch.long)
        for index, waveform in enumerate(waveforms):
            if waveform.ndim != 1:
                raise ValueError(f"waveform must be 1-D, got {tuple(waveform.shape)}")
            size = int(waveform.numel())
            if size > bucket.num_samples:
                raise ValueError(f"waveform does not fit {bucket.seconds}s bucket")
            batch[index, :size] = waveform.detach().cpu().float()
            lengths[index] = size
        return batch, lengths
