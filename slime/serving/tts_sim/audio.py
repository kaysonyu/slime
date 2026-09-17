"""Allowlisted audio decoding and bounded 16-kHz preprocessing."""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torchaudio.transforms import Resample

from .assets import MAX_AUDIO_SAMPLES, MAX_AUDIO_SECONDS, TARGET_SAMPLE_RATE

PREPROCESS_VERSION = "first-channel-resample16k-front30s-v1"
SUPPORTED_SUFFIXES = frozenset({".wav", ".flac"})
MIN_AUDIO_SAMPLES = 1_600


class AudioInputError(ValueError):
    code = "invalid_audio"


class PathAccessError(AudioInputError):
    code = "path_access_denied"


class AudioDecodingError(AudioInputError):
    code = "audio_decode_failed"


class AudioTooShortError(AudioInputError):
    code = "audio_too_short"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


@dataclass(frozen=True, slots=True)
class FileSignature:
    real_path: str
    size: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    source: str
    original_sample_rate: int
    original_channels: int
    original_frames: int
    effective_samples: int
    truncated: bool

    @property
    def original_duration_seconds(self) -> float:
        return self.original_frames / self.original_sample_rate

    @property
    def effective_duration_seconds(self) -> float:
        return self.effective_samples / TARGET_SAMPLE_RATE


@dataclass(frozen=True, slots=True)
class LoadedAudio:
    waveform: torch.Tensor
    metadata: AudioMetadata
    signature: FileSignature

    @property
    def num_samples(self) -> int:
        return int(self.waveform.numel())


class AudioLoader:
    """Resolve, bound, decode, and resample allowlisted audio paths."""

    def __init__(
        self,
        allowed_roots: Iterable[Path],
        min_samples: int = MIN_AUDIO_SAMPLES,
    ) -> None:
        self.allowed_roots = tuple(Path(root).resolve() for root in allowed_roots)
        if not self.allowed_roots:
            raise ValueError("AudioLoader requires explicit allowed roots")
        self.min_samples = int(min_samples)
        self._resamplers: dict[int, Resample] = {}
        self._resampler_lock = threading.Lock()

    def resolve_path(self, value: str | os.PathLike[str]) -> Path:
        path = Path(value)
        if not path.is_absolute():
            raise PathAccessError(f"audio path must be absolute: {path}")
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"audio file does not exist: {path}") from exc
        if not resolved.is_file():
            raise PathAccessError(f"audio path is not a regular file: {resolved}")
        if self.allowed_roots and not any(_is_relative_to(resolved, root) for root in self.allowed_roots):
            roots = ", ".join(str(root) for root in self.allowed_roots)
            raise PathAccessError(f"resolved audio path {resolved} is outside configured allowlist roots: {roots}")
        if resolved.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise AudioDecodingError(f"unsupported audio extension {resolved.suffix!r}; expected WAV or FLAC")
        return resolved

    @staticmethod
    def file_signature(path: Path) -> FileSignature:
        stat = path.stat()
        return FileSignature(str(path), int(stat.st_size), int(stat.st_mtime_ns))

    def _resampler(self, source_rate: int) -> Resample:
        with self._resampler_lock:
            resampler = self._resamplers.get(source_rate)
            if resampler is None:
                resampler = Resample(orig_freq=source_rate, new_freq=TARGET_SAMPLE_RATE)
                self._resamplers[source_rate] = resampler
            return resampler

    @staticmethod
    def _prefix_frames(source_rate: int, total_frames: int) -> int:
        # The guard gives the resampling filter right-hand context so the first
        # 30 seconds match full-file-resample-then-slice at the crop boundary.
        guard = max(64, math.ceil(source_rate / TARGET_SAMPLE_RATE) * 64)
        wanted = math.ceil(MAX_AUDIO_SECONDS * source_rate) + guard
        return min(total_frames, wanted)

    def _decode_stream(
        self,
        stream: str,
        *,
        source_label: str,
        signature: FileSignature,
    ) -> LoadedAudio:
        """Decode only the bounded prefix and return a finite 16-kHz waveform."""
        try:
            with sf.SoundFile(stream) as handle:
                sample_rate = int(handle.samplerate)
                channels = int(handle.channels)
                total_frames = int(handle.frames)
                if sample_rate <= 0 or channels <= 0 or total_frames <= 0:
                    raise AudioDecodingError("audio metadata is empty or invalid")
                frames = self._prefix_frames(sample_rate, total_frames)
                decoded = handle.read(frames, dtype="float32", always_2d=True)
        except AudioInputError:
            raise
        except Exception as exc:
            raise AudioDecodingError(f"failed to decode {source_label}") from exc

        waveform_np = np.ascontiguousarray(decoded[:, 0], dtype=np.float32)
        if waveform_np.size == 0:
            raise AudioDecodingError(f"decoded empty audio from {source_label}")
        if not np.isfinite(waveform_np).all():
            raise AudioDecodingError(f"audio contains NaN or Inf: {source_label}")

        waveform = torch.from_numpy(waveform_np)
        if sample_rate != TARGET_SAMPLE_RATE:
            waveform = self._resampler(sample_rate)(waveform.unsqueeze(0)).squeeze(0)
        waveform = waveform[:MAX_AUDIO_SAMPLES].to(dtype=torch.float32).contiguous()
        if waveform.numel() < self.min_samples:
            raise AudioTooShortError(f"audio has {waveform.numel()} effective samples; minimum is {self.min_samples}")
        if not bool(torch.isfinite(waveform).all().item()):
            raise AudioDecodingError(f"resampled audio contains NaN or Inf: {source_label}")

        metadata = AudioMetadata(
            source=source_label,
            original_sample_rate=sample_rate,
            original_channels=channels,
            original_frames=total_frames,
            effective_samples=int(waveform.numel()),
            truncated=(total_frames / sample_rate) > MAX_AUDIO_SECONDS,
        )
        return LoadedAudio(waveform=waveform, metadata=metadata, signature=signature)

    def load_path(self, value: str | os.PathLike[str]) -> LoadedAudio:
        path = self.resolve_path(value)
        signature = self.file_signature(path)
        return self._decode_stream(
            str(path),
            source_label=str(path),
            signature=signature,
        )
