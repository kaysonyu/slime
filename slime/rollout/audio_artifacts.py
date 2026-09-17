"""Validate complete PCM WAVs and atomically publish model-independent artifacts."""

import base64
import binascii
import hashlib
import io
import os
import tempfile
import wave
from pathlib import Path

from slime.rollout.failures import RecoverableRolloutError


def persist_audio(audio, *, root, namespace, rollout_id, sample_id, attempt=0):
    if not isinstance(audio, dict) or not isinstance(audio.get("data"), str):
        raise ValueError("Omni audio requires a Base64 data string")
    try:
        data = base64.b64decode(audio["data"], validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Omni audio is malformed Base64") from error
    try:
        with wave.open(io.BytesIO(data), "rb") as stream:
            channels, width, rate, frames = (
                stream.getnchannels(),
                stream.getsampwidth(),
                stream.getframerate(),
                stream.getnframes(),
            )
            if stream.getcomptype() != "NONE" or channels < 1 or width not in (1, 2, 3, 4) or rate < 1 or frames < 1:
                raise RecoverableRolloutError("artifact.invalid_wav", "Expected nonempty PCM WAV")
            declared = audio.get("sample_rate")
            if declared is not None and (type(declared) is not int or declared != rate):
                raise RecoverableRolloutError("artifact.invalid_wav", "WAV sample rate differs from response metadata")
            frame_bytes = channels * width
            remaining = frames
            while remaining:
                block = stream.readframes(min(remaining, 65536))
                if not block or len(block) % frame_bytes:
                    raise RecoverableRolloutError("artifact.invalid_wav", "Truncated WAV frames")
                remaining -= len(block) // frame_bytes
    except (EOFError, wave.Error) as error:
        raise RecoverableRolloutError("artifact.invalid_wav", "Unreadable PCM WAV") from error
    if type(rollout_id) is not int or rollout_id < 0 or type(attempt) is not int or attempt < 0:
        raise ValueError("Audio rollout id and attempt must be non-negative integers")
    scope = hashlib.sha256(namespace.encode()).hexdigest()[:16]
    key = hashlib.sha256(f"{namespace}:{rollout_id}:{sample_id}:{attempt}".encode()).hexdigest()
    directory = Path(root) / scope / f"rollout_{rollout_id:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{key}.wav"
    fd, temporary = tempfile.mkstemp(prefix=".audio-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return str(destination), dict(
        audio_sha256=hashlib.sha256(data).hexdigest(),
        sample_rate=rate,
        audio_channels=channels,
        audio_frames=frames,
        artifact_namespace=namespace,
    )
