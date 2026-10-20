"""Audio transport and decoding, shared by the two frozen embedding services."""

import base64
import binascii
import io
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


class InvalidAudio(ValueError):
    def __init__(self, reason, **details):
        self.reason, self.details = reason, details
        super().__init__(reason)


@lru_cache(maxsize=16)
def _resampler(source_rate):
    import torchaudio

    return torchaudio.transforms.Resample(source_rate, SAMPLE_RATE)


def decode_audio(audio, max_seconds=120):
    """Decode full audio to mono 16 kHz FP32. No model-specific padding here."""
    if not isinstance(audio, dict) or set(audio) - {"path", "data_b64", "suffix"}:
        raise ValueError("audio must contain path or data_b64 (and optional suffix)")
    path, data = audio.get("path"), audio.get("data_b64")
    if (path is not None) == (data is not None):
        raise ValueError("audio requires exactly one of path and data_b64")
    if path is not None:
        if not isinstance(path, str) or not Path(path).is_file():
            raise ValueError("audio.path must name a readable regular file")
        source = path
    else:
        if not isinstance(data, str):
            raise ValueError("audio.data_b64 must be a string")
        try:
            raw = base64.b64decode(data, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("invalid audio.data_b64") from exc
        source = io.BytesIO(raw)
    try:
        with sf.SoundFile(source) as stream:
            if stream.frames / stream.samplerate > max_seconds:
                raise InvalidAudio("too_long", max_seconds=max_seconds)
            rate = stream.samplerate
            waveform = stream.read(dtype="float32", always_2d=True).mean(axis=1)
    except InvalidAudio:
        raise
    except (RuntimeError, ValueError) as exc:
        raise InvalidAudio("decode_error") from exc
    if not len(waveform):
        raise InvalidAudio("empty_audio", num_samples=0, sample_rate=SAMPLE_RATE)
    if not np.isfinite(waveform).all():
        raise InvalidAudio("nonfinite_audio")
    if rate != SAMPLE_RATE:
        import torch

        waveform = _resampler(rate)(torch.from_numpy(waveform)).numpy()
    return np.ascontiguousarray(waveform, dtype=np.float32)
