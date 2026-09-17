"""Embedding backend protocol, strict Torch model loader, and device pool."""

from __future__ import annotations

import concurrent.futures
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TypedDict, runtime_checkable

import torch

from .assets import (
    CANONICAL_CHECKPOINT,
    CANONICAL_CHECKPOINT_SHA256,
    EMBEDDING_DIMENSION,
    verify_canonical_checkpoint,
)
from .model import load_wavlm_large_ecapa_tdnn


class BackendDetails(TypedDict):
    type: str
    precision: str
    tf32_matmul: bool
    compile: bool


@runtime_checkable
class EmbeddingBackend(Protocol):
    name: str
    fingerprint: str
    checkpoint_sha256: str
    allow_tf32: bool
    precision: str
    details: BackendDetails

    @property
    def is_ready(self) -> bool: ...

    def embed(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor: ...

    def close(self) -> None: ...


class PyTorchEmbeddingBackend:
    """Strict WavLM/ECAPA inference backend for one CPU or CUDA device."""

    name = "pytorch_fp32"

    def __init__(
        self,
        checkpoint: Path = CANONICAL_CHECKPOINT,
        device: str = "cuda:0",
        verify_hash: bool = True,
        allow_tf32: bool = True,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("SIM supports only CPU or CUDA Torch devices")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"configured SIM device {self.device} requires CUDA, but CUDA is unavailable"
            )

        self.checkpoint_sha256 = (
            verify_canonical_checkpoint(self.checkpoint)
            if verify_hash
            else CANONICAL_CHECKPOINT_SHA256
        )
        self.allow_tf32 = bool(allow_tf32 and self.device.type == "cuda")
        self.precision = "fp32_tf32_matmul" if self.allow_tf32 else "fp32"
        self.details = {
            "type": "pytorch",
            "precision": self.precision,
            "tf32_matmul": self.allow_tf32,
            "compile": False,
        }
        self.fingerprint = f"{self.checkpoint_sha256}:{self.name}:tf32={int(self.allow_tf32)}"
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.allow_tf32

        self.model = (
            load_wavlm_large_ecapa_tdnn(checkpoint_path=str(self.checkpoint), map_location="cpu")
            .eval()
            .to(self.device)
        )
        self._inference_lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return True

    def embed(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim != 2 or lengths.ndim != 1:
            raise ValueError("backend expects rank-2 waveforms and rank-1 lengths")
        if waveforms.shape[0] != lengths.shape[0] or waveforms.shape[0] == 0:
            raise ValueError("backend received an empty or mismatched batch")

        device_waveforms = waveforms.to(self.device, dtype=torch.float32, non_blocking=True)
        device_lengths = lengths.to(self.device, dtype=torch.long, non_blocking=True)
        with self._inference_lock, torch.inference_mode():
            output = self.model(device_waveforms, device_lengths)
        output = output.detach().cpu().float()
        expected_shape = (int(waveforms.shape[0]), EMBEDDING_DIMENSION)
        if output.shape != expected_shape:
            raise RuntimeError(f"backend returned invalid embedding shape {tuple(output.shape)}")
        if not bool(torch.isfinite(output).all().item()):
            raise RuntimeError("backend returned NaN or Inf")
        return output

    def close(self) -> None:
        return None


class BackendPool:
    """Shard a batch over same-fingerprint backends and restore input order."""

    def __init__(self, backends: Sequence[EmbeddingBackend]) -> None:
        if not backends:
            raise ValueError("backend pool requires at least one backend")
        fingerprints = {backend.fingerprint for backend in backends}
        if len(fingerprints) != 1:
            raise ValueError("all pooled backends must use the same model fingerprint")

        self.backends = tuple(backends)
        first = self.backends[0]
        self.name = "pool[" + ",".join(backend.name for backend in self.backends) + "]"
        self.fingerprint = first.fingerprint
        self.checkpoint_sha256 = first.checkpoint_sha256
        self.allow_tf32 = first.allow_tf32
        self.precision = first.precision
        self.details: BackendDetails = {
            "type": first.details["type"],
            "precision": first.details["precision"],
            "tf32_matmul": first.details["tf32_matmul"],
            "compile": first.details["compile"],
        }
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.backends),
            thread_name_prefix="embedding-device",
        )
        self._inference_lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return all(backend.is_ready for backend in self.backends)

    @property
    def parallelism(self) -> int:
        return len(self.backends)

    def embed(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim != 2 or lengths.ndim != 1:
            raise ValueError("pool expects rank-2 waveforms and rank-1 lengths")
        if waveforms.shape[0] != lengths.shape[0] or waveforms.shape[0] == 0:
            raise ValueError("pool received an empty or mismatched batch")
        if len(self.backends) == 1:
            return self.backends[0].embed(waveforms, lengths)

        batch_size = int(waveforms.shape[0])
        shard_count = min(len(self.backends), batch_size)
        minimum_shard, larger_shards = divmod(batch_size, shard_count)
        with self._inference_lock:
            futures: list[concurrent.futures.Future[torch.Tensor]] = []
            start = 0
            for index, backend in enumerate(self.backends[:shard_count]):
                shard_size = minimum_shard + int(index < larger_shards)
                stop = start + shard_size
                futures.append(
                    self._executor.submit(
                        backend.embed,
                        waveforms[start:stop],
                        lengths[start:stop],
                    )
                )
                start = stop
            outputs = [future.result() for future in futures]
        return torch.cat(outputs, dim=0)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
        for backend in self.backends:
            backend.close()
