"""Environment-backed serving settings and resource limits."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from .assets import CANONICAL_CHECKPOINT

MAX_REQUEST_ITEMS = 16

_REMOVED_ENVIRONMENT_VARIABLES = (
    "WAVLM_SIM_BACKEND",
    "WAVLM_SIM_DEVICE",
    "WAVLM_SIM_MAX_BATCH",
    "WAVLM_SIM_MAX_REQUEST_BYTES",
    "WAVLM_SIM_PYTORCH_COMPILE_BATCH",
    "WAVLM_SIM_PYTORCH_COMPILE_PREWARM",
    "WAVLM_SIM_TRT_ENGINE_DIR",
    "WAVLM_SIM_TRT_REQUIRE_PRODUCTION_GATE",
)


def _positive_int(name: str, default: int, *, maximum: int | None = None) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _optional_positive_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _removed_environment_variables() -> tuple[str, ...]:
    return tuple(name for name in _REMOVED_ENVIRONMENT_VARIABLES if name in os.environ)


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable serving limits parsed from ``WAVLM_SIM_*`` environment vars."""

    checkpoint: Path
    devices: tuple[str, ...]
    allowed_roots: tuple[Path, ...]
    allow_tf32: bool
    per_device_batch: int
    dynamic_delay_ms: float
    max_queue_items: int
    max_items_per_request: int
    reference_cache_items: int
    audio_workers: int
    max_inflight_requests: int
    api_key: str | None
    host: str
    port: int
    log_level: str
    access_log: bool
    source_tree: str

    @property
    def global_batch(self) -> int:
        return self.per_device_batch * len(self.devices)

    def __post_init__(self) -> None:
        if not self.devices or any(not device.strip() for device in self.devices):
            raise ValueError("devices must contain at least one non-empty device")
        if len(set(self.devices)) != len(self.devices):
            raise ValueError("devices must not contain duplicates")
        if not self.allowed_roots:
            raise ValueError("at least one allowed audio root is required")
        if not 1 <= self.per_device_batch <= MAX_REQUEST_ITEMS:
            raise ValueError(f"per_device_batch must be between 1 and {MAX_REQUEST_ITEMS}")
        if self.dynamic_delay_ms < 0:
            raise ValueError("dynamic_delay_ms cannot be negative")
        positive_values = {
            "max_queue_items": self.max_queue_items,
            "max_items_per_request": self.max_items_per_request,
            "reference_cache_items": self.reference_cache_items,
            "audio_workers": self.audio_workers,
            "max_inflight_requests": self.max_inflight_requests,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_items_per_request > MAX_REQUEST_ITEMS:
            raise ValueError(f"max_items_per_request must be at most {MAX_REQUEST_ITEMS}")
        if not 1 <= self.port <= 65_535:
            raise ValueError("service port is invalid")
        if not self.source_tree.strip():
            raise ValueError("source_tree cannot be empty")

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings and reject removed or internally inconsistent options."""
        removed = _removed_environment_variables()
        if removed:
            names = ", ".join(removed)
            raise ValueError(f"removed SIM configuration is set: {names}")

        raw_devices = os.getenv("WAVLM_SIM_DEVICES")
        if raw_devices is None:
            devices = ("cuda:0",) if torch.cuda.is_available() else ("cpu",)
        elif raw_devices.strip().lower() == "all_cuda":
            cuda_count = torch.cuda.device_count()
            if cuda_count <= 0:
                raise ValueError("WAVLM_SIM_DEVICES=all_cuda but CUDA is unavailable")
            devices = tuple(f"cuda:{index}" for index in range(cuda_count))
        else:
            devices = tuple(
                device.strip() for device in raw_devices.split(",") if device.strip()
            )

        expected_cuda_devices = _optional_positive_int(
            "WAVLM_SIM_EXPECTED_CUDA_DEVICES"
        )
        if expected_cuda_devices is not None:
            visible_cuda_devices = torch.cuda.device_count()
            if visible_cuda_devices != expected_cuda_devices:
                raise ValueError(
                    f"expected {expected_cuda_devices} visible CUDA devices, "
                    f"found {visible_cuda_devices}"
                )
            expected_devices = tuple(
                f"cuda:{index}" for index in range(expected_cuda_devices)
            )
            if devices != expected_devices:
                raise ValueError(
                    "WAVLM_SIM_DEVICES must select every visible CUDA device when "
                    "WAVLM_SIM_EXPECTED_CUDA_DEVICES is set"
                )

        raw_roots = os.getenv("WAVLM_SIM_ALLOWED_ROOTS")
        if raw_roots is None:
            raise ValueError("WAVLM_SIM_ALLOWED_ROOTS must explicitly name the shared audio roots")
        else:
            allowed_roots = tuple(
                Path(part).resolve() for part in raw_roots.split(os.pathsep) if part
            )
            if not allowed_roots:
                raise ValueError("WAVLM_SIM_ALLOWED_ROOTS cannot be empty")

        try:
            delay_ms = float(os.getenv("WAVLM_SIM_DYNAMIC_DELAY_MS", "5"))
        except ValueError as exc:
            raise ValueError("WAVLM_SIM_DYNAMIC_DELAY_MS must be a number") from exc
        if delay_ms < 0:
            raise ValueError("WAVLM_SIM_DYNAMIC_DELAY_MS cannot be negative")

        try:
            port = int(os.getenv("PORT", os.getenv("WAVLM_SIM_PORT", "8000")))
        except ValueError as exc:
            raise ValueError("service port must be an integer") from exc

        allow_tf32_default = any(device.startswith("cuda:") for device in devices)
        return cls(
            checkpoint=Path(os.getenv("WAVLM_SIM_CHECKPOINT", str(CANONICAL_CHECKPOINT))),
            devices=devices,
            allowed_roots=tuple(allowed_roots),
            allow_tf32=_boolean("WAVLM_SIM_ALLOW_TF32", allow_tf32_default),
            per_device_batch=_positive_int(
                "WAVLM_SIM_PER_DEVICE_BATCH",
                MAX_REQUEST_ITEMS,
                maximum=MAX_REQUEST_ITEMS,
            ),
            dynamic_delay_ms=delay_ms,
            max_queue_items=_positive_int("WAVLM_SIM_MAX_QUEUE_ITEMS", 8_192),
            max_items_per_request=_positive_int(
                "WAVLM_SIM_MAX_ITEMS_PER_REQUEST",
                MAX_REQUEST_ITEMS,
                maximum=MAX_REQUEST_ITEMS,
            ),
            reference_cache_items=_positive_int("WAVLM_SIM_REFERENCE_CACHE_ITEMS", 100_000),
            audio_workers=_positive_int("WAVLM_SIM_AUDIO_WORKERS", 16),
            max_inflight_requests=_positive_int("WAVLM_SIM_MAX_INFLIGHT_REQUESTS", 4_096),
            api_key=os.getenv("WAVLM_SIM_API_KEY") or None,
            host=os.getenv("WAVLM_SIM_HOST", "0.0.0.0"),
            port=port,
            log_level=os.getenv("WAVLM_SIM_LOG_LEVEL", "info"),
            access_log=_boolean("WAVLM_SIM_ACCESS_LOG", False),
            source_tree=os.getenv("WAVLM_SIM_SOURCE_TREE", "unknown"),
        )
