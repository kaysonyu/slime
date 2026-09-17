"""Stable model asset identities and verification helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

MODEL_NAME = "wavlm_large_ecapa_tdnn"
EMBEDDING_DIMENSION = 256
TARGET_SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 30
MAX_AUDIO_SAMPLES = TARGET_SAMPLE_RATE * MAX_AUDIO_SECONDS

CANONICAL_CHECKPOINT = Path(
    "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/wavlm-ecapa-sim-seedtts/wavlm_large_finetune.pth"
)
CANONICAL_CHECKPOINT_SHA256 = "51f07e3b94d9e0262a6a675ef5a087be3dd09e8c62e9d886827f44f82fe7f94b"
SEED_TTS_EVAL_ROOT = Path(
    "/inspire/ssd/project/embodied-multimodality/public/kyu/src/emb/seed-tts-eval"
)
SEED_TTS_EVAL_COMMIT = "752f4297f090c46bb1a55a1f7439e5944ddefe8d"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA256 without loading a checkpoint into RAM."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_canonical_checkpoint(path: Path = CANONICAL_CHECKPOINT) -> str:
    """Fail closed when a different checkpoint is supplied under the known path."""

    if not path.is_file():
        raise FileNotFoundError(f"canonical checkpoint is missing: {path}")
    actual = sha256_file(path)
    if actual != CANONICAL_CHECKPOINT_SHA256:
        raise ValueError(
            f"checkpoint SHA256 mismatch for {path}: "
            f"expected {CANONICAL_CHECKPOINT_SHA256}, got {actual}"
        )
    return actual
