import struct
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient

from slime.serving.tts_sim import api
from slime.serving.tts_sim.api import create_app
from slime.serving.tts_sim.audio import PREPROCESS_VERSION, AudioLoader
from slime.serving.tts_sim.backend import BackendPool
from slime.serving.tts_sim.buckets import BucketPolicy
from slime.serving.tts_sim.cache import ReferenceEmbeddingCache
from slime.serving.tts_sim.config import Settings
from slime.serving.tts_sim.service import EmbeddingService

NUM_GPUS = 0


class FakeBackend:
    name = "test"
    fingerprint = "test-v1"
    checkpoint_sha256 = "test-checkpoint"
    is_ready = True
    allow_tf32 = False
    precision = "fp32"
    details = {
        "type": "test",
        "precision": "fp32",
        "tf32_matmul": False,
        "compile": False,
    }

    def embed(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        rows: list[torch.Tensor] = []
        for waveform, length in zip(waveforms, lengths, strict=True):
            valid = waveform[: int(length)]
            seed = torch.tensor([valid.mean(), valid.std(unbiased=False), float(length) / 480_000])
            rows.append(F.pad(seed, (0, 253)))
        return torch.stack(rows)

    def close(self) -> None:
        return None


class PoolBackend:
    name = "test"
    fingerprint = "pool-v1"
    checkpoint_sha256 = "test-checkpoint"
    is_ready = True
    allow_tf32 = False
    precision = "fp32"
    details = {
        "type": "test",
        "precision": "fp32",
        "tf32_matmul": False,
        "compile": False,
    }

    def __init__(self, device: str, barrier: threading.Barrier | None = None) -> None:
        self.device = device
        self.barrier = barrier
        self.batch_sizes: list[int] = []
        self.closed = False

    def embed(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        assert waveforms.shape[0] == lengths.shape[0]
        self.batch_sizes.append(int(waveforms.shape[0]))
        if self.barrier is not None:
            self.barrier.wait()
        return waveforms[:, :1].repeat(1, 256)

    def close(self) -> None:
        self.closed = True


def _write_wave(path: Path, frequency: float) -> None:
    positions = np.arange(3_200, dtype=np.float32) / 16_000
    sf.write(
        path,
        0.2 * np.sin(2 * np.pi * frequency * positions),
        16_000,
        subtype="FLOAT",
    )


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        checkpoint=tmp_path / "unused.pth",
        devices=("cpu",),
        allowed_roots=(tmp_path,),
        allow_tf32=False,
        per_device_batch=16,
        dynamic_delay_ms=1,
        max_queue_items=128,
        max_items_per_request=16,
        reference_cache_items=16,
        audio_workers=2,
        max_inflight_requests=8,
        api_key=None,
        host="127.0.0.1",
        port=8000,
        log_level="warning",
        access_log=False,
        source_tree="test-tree",
    )


def test_audio_loader_uses_left_channel_for_stereo_inputs(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    left_channel = np.linspace(-0.25, 0.25, 3_200, dtype=np.float32)
    right_channel = np.full(3_200, 0.75, dtype=np.float32)
    sf.write(
        path,
        np.column_stack((left_channel, right_channel)),
        16_000,
        subtype="FLOAT",
    )

    loaded = AudioLoader([tmp_path]).load_path(path)

    assert PREPROCESS_VERSION == "first-channel-resample16k-front30s-v1"
    assert loaded.metadata.original_channels == 2
    torch.testing.assert_close(loaded.waveform, torch.from_numpy(left_channel))
    assert not torch.allclose(loaded.waveform, torch.from_numpy(right_channel))


def test_audio_loader_does_not_limit_source_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "large.wav"
    channels = 2
    sample_width = 2
    sample_rate = 16_000
    block_align = channels * sample_width
    frames = (64 * 1024 * 1024) // block_align + 1
    data_size = frames * block_align
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        sample_rate * block_align,
        block_align,
        sample_width * 8,
        b"data",
        data_size,
    )
    with path.open("wb") as stream:
        stream.write(header)
        stream.seek(44 + data_size - 1)
        stream.write(b"\0")

    loaded = AudioLoader([tmp_path]).load_path(path)

    assert path.stat().st_size > 64 * 1024 * 1024
    assert loaded.metadata.truncated
    assert loaded.num_samples == 30 * sample_rate


def test_vendored_sim_service_exposes_only_sanitized_similarity_api(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.wav"
    candidate = tmp_path / "candidate.wav"
    missing = tmp_path / "private-missing.wav"
    _write_wave(reference, 220)
    _write_wave(candidate, 230)
    service = EmbeddingService(
        AudioLoader([tmp_path]),
        FakeBackend(),
        BucketPolicy(),
        ReferenceEmbeddingCache(16),
    )
    app = create_app(service=service, settings=_settings(tmp_path))
    payload = {
        "items": [
            {
                "id": "pair",
                "reference_path": str(reference),
                "candidate_path": str(candidate),
            }
        ]
    }

    with TestClient(app) as client:
        first = client.post("/v1/similarities", json=payload).json()["results"][0]
        second = client.post("/v1/similarities", json=payload).json()["results"][0]
        payload["items"][0]["candidate_path"] = str(missing)
        failed = client.post("/v1/similarities", json=payload).json()["results"][0]
        removed_route_status = client.post("/v1/embeddings", json={}).status_code
        model = client.get("/v1/model").json()

    assert first["error"] is None
    assert not first["reference_cache_hit"]
    assert second["reference_cache_hit"]
    assert failed["error"]["code"] == "file_not_found"
    assert str(missing) not in failed["error"]["message"]
    assert removed_route_status == 404
    assert model["backend"] == "pytorch"
    assert model["source_tree"] == "test-tree"
    assert model["devices"] == ["cpu"]
    assert model["parallelism"] == 1
    assert model["per_device_batch"] == 16
    assert model["global_batch"] == 16
    assert "device" not in model
    assert "max_batch" not in model
    assert "allowed_roots" not in model


def test_vendored_sim_service_does_not_limit_http_request_bytes(tmp_path: Path) -> None:
    service = EmbeddingService(
        AudioLoader([tmp_path]),
        FakeBackend(),
        BucketPolicy(),
        ReferenceEmbeddingCache(16),
    )
    app = create_app(service=service, settings=_settings(tmp_path))

    with TestClient(app) as client:
        response = client.request(
            "GET",
            "/health/live",
            content=b"x" * (4 * 1024 * 1024 + 1),
        )

    assert response.status_code == 200


@pytest.mark.parametrize(
    "name", ["WAVLM_SIM_BACKEND", "WAVLM_SIM_MAX_REQUEST_BYTES"]
)
def test_vendored_sim_service_rejects_removed_configuration(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.setenv(name, "1")

    with pytest.raises(ValueError, match="removed SIM configuration"):
        Settings.from_env()


def test_settings_expand_all_cuda_and_scale_global_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setenv("WAVLM_SIM_DEVICES", "all_cuda")
    monkeypatch.setenv("WAVLM_SIM_EXPECTED_CUDA_DEVICES", "4")
    monkeypatch.setenv("WAVLM_SIM_PER_DEVICE_BATCH", "16")
    monkeypatch.setenv("WAVLM_SIM_ALLOWED_ROOTS", str(tmp_path))

    settings = Settings.from_env()

    assert settings.devices == ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    assert settings.per_device_batch == 16
    assert settings.global_batch == 64


def test_settings_reject_visible_cuda_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setenv("WAVLM_SIM_DEVICES", "all_cuda")
    monkeypatch.setenv("WAVLM_SIM_EXPECTED_CUDA_DEVICES", "8")

    with pytest.raises(ValueError, match="expected 8 visible CUDA devices, found 4"):
        Settings.from_env()


def test_settings_rejects_incomplete_visible_cuda_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setenv("WAVLM_SIM_DEVICES", "cuda:0,cuda:2")
    monkeypatch.setenv("WAVLM_SIM_EXPECTED_CUDA_DEVICES", "2")

    with pytest.raises(ValueError, match="must select every visible CUDA device"):
        Settings.from_env()


def test_service_requires_explicit_project_audio_roots(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("WAVLM_SIM_ALLOWED_ROOTS", raising=False)
    monkeypatch.delenv("WAVLM_SIM_EXPECTED_CUDA_DEVICES", raising=False)
    monkeypatch.setenv("WAVLM_SIM_DEVICES", "cpu")
    with pytest.raises(ValueError, match="WAVLM_SIM_ALLOWED_ROOTS"):
        Settings.from_env()
    with pytest.raises(ValueError, match="explicit allowed roots"):
        AudioLoader([])


def test_backend_pool_shards_concurrently_and_preserves_order() -> None:
    barrier = threading.Barrier(3, timeout=2)
    backends = [PoolBackend(f"cuda:{index}", barrier) for index in range(3)]
    pool = BackendPool(backends)
    waveforms = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    lengths = torch.full((8,), 2, dtype=torch.long)

    output = pool.embed(waveforms, lengths)
    pool.close()

    assert [backend.batch_sizes for backend in backends] == [[3], [3], [2]]
    assert output[:, 0].tolist() == waveforms[:, 0].tolist()
    assert all(backend.closed for backend in backends)


def test_build_service_creates_one_model_per_device_and_one_shared_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings = Settings(
        checkpoint=settings.checkpoint,
        devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
        allowed_roots=settings.allowed_roots,
        allow_tf32=settings.allow_tf32,
        per_device_batch=settings.per_device_batch,
        dynamic_delay_ms=settings.dynamic_delay_ms,
        max_queue_items=settings.max_queue_items,
        max_items_per_request=settings.max_items_per_request,
        reference_cache_items=settings.reference_cache_items,
        audio_workers=settings.audio_workers,
        max_inflight_requests=settings.max_inflight_requests,
        api_key=settings.api_key,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        access_log=settings.access_log,
        source_tree=settings.source_tree,
    )
    constructor_calls: list[tuple[str, bool]] = []

    def fake_backend(*, device: str, verify_hash: bool, **_kwargs: Path | bool) -> PoolBackend:
        constructor_calls.append((device, verify_hash))
        return PoolBackend(device)

    monkeypatch.setattr(api, "PyTorchEmbeddingBackend", fake_backend)

    service = api.build_service(settings)
    try:
        assert constructor_calls == [
            ("cuda:0", True),
            ("cuda:1", False),
            ("cuda:2", False),
            ("cuda:3", False),
        ]
        assert service.backend.max_batch == 64
        assert service.reference_cache.max_items == settings.reference_cache_items
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
