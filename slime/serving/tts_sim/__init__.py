"""Torch WavLM/ECAPA speaker-similarity service and FastAPI surface."""

from .assets import (
    CANONICAL_CHECKPOINT,
    CANONICAL_CHECKPOINT_SHA256,
    EMBEDDING_DIMENSION,
    MODEL_NAME,
)
from .model import WavLMLargeECAPATDNN, load_wavlm_large_ecapa_tdnn

__all__ = [
    "CANONICAL_CHECKPOINT",
    "CANONICAL_CHECKPOINT_SHA256",
    "EMBEDDING_DIMENSION",
    "MODEL_NAME",
    "WavLMLargeECAPATDNN",
    "load_wavlm_large_ecapa_tdnn",
]
