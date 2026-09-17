"""Validated WAV payload helpers."""

from __future__ import annotations

import base64
from pathlib import Path


def wav_data_url(path: str | Path) -> str:
    audio_path = Path(path)
    try:
        size = audio_path.stat().st_size
    except OSError as error:
        raise ValueError("WAV input is not a readable file.") from error
    if size <= 0:
        raise ValueError("WAV payload is empty.")
    try:
        raw = audio_path.read_bytes()
    except OSError as error:
        raise ValueError("WAV input could not be read.") from error
    if not raw:
        raise ValueError("WAV payload is empty.")
    return "data:audio/wav;base64," + base64.b64encode(raw).decode("ascii")


def wav_file_uri(path: str | Path) -> str:
    audio_path = Path(path)
    try:
        size = audio_path.stat().st_size
        resolved_path = audio_path.resolve(strict=True)
    except OSError as error:
        raise ValueError("WAV input is not a readable file.") from error
    if size <= 0:
        raise ValueError("WAV payload is empty.")
    return resolved_path.as_uri()
