#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import tomllib
from endpoint import remove_endpoint, save_endpoint

ROOT = Path(__file__).resolve().parent
SLIME_ROOT = ROOT.parents[2]
KINDS = ("asr", "sim", "judge")
ALL_ACTIONS = ("plan", "deploy", "status", "wait", "endpoint", "stop", "delete")
# Bulk actions must not mutate platform objects. Endpoint refresh is allowed
# because each successful lookup only updates locked local state.
ALL_SAFE_ACTIONS = ("plan", "status", "endpoint")


# Typed configuration and lifecycle values.
class ConfigError(ValueError):
    pass


class CommandError(RuntimeError):
    pass


@dataclass(frozen=True)
class Quota:
    gpus: int
    cpus: int
    memory_gib: int


@dataclass(frozen=True)
class Platform:
    kind: str
    name: str
    workspace: str
    project: str
    group: str
    quota: Quota
    image: str
    replicas: int
    shm_gib: int
    priority: int
    custom_domain: str | None
    registry_model: str
    registry_version: str


@dataclass(frozen=True)
class Vllm:
    platform: Platform
    bin: str
    model_path: Path
    audio_root: Path
    served_name: str
    memory_fraction: float
    max_model_len: int


@dataclass(frozen=True)
class Sim:
    platform: Platform
    checkpoint: Path
    reference_roots: tuple[Path, ...]
    audio_root: Path
    per_device_batch: int
    delay_ms: float
    max_items: int
    audio_workers: int
    max_queue_items: int
    max_inflight: int
    cache_items: int
    allow_tf32: bool


Service: TypeAlias = Vllm | Sim


@dataclass(frozen=True)
class Config:
    asr: Vllm
    sim: Sim
    judge: Vllm

    def get(self, kind: str) -> Service:
        if kind == "asr":
            return self.asr
        if kind == "sim":
            return self.sim
        if kind == "judge":
            return self.judge
        raise ConfigError(f"unknown service kind: {kind}")


@dataclass(frozen=True)
class Settings:
    state_dir: Path
    bundle: Path | None
    inspire_cwd: Path
    attempts: int
    retry_delay_s: int
    poll_s: int
    wait_timeout_s: int
    stop_timeout_s: int
    delete_timeout_s: int

    @classmethod
    def from_env(cls) -> Settings:
        bundle_value = os.environ.get(
            "SERVING_ENDPOINT_BUNDLE",
            str(ROOT.parent / "endpoints.env"),
        )
        bundle = Path(bundle_value) if bundle_value else None
        state_dir = Path(os.environ.get("SERVING_STATE_DIR", str(ROOT / ".state")))
        inspire_cwd = Path(os.environ.get("SERVING_INSPIRE_CWD", "/tmp"))
        for label, path in (("SERVING_STATE_DIR", state_dir), ("SERVING_INSPIRE_CWD", inspire_cwd)):
            if not path.is_absolute():
                raise ConfigError(f"{label} must be an absolute path")
        if not inspire_cwd.is_dir():
            raise ConfigError("SERVING_INSPIRE_CWD must be an existing directory")
        if bundle is not None:
            if not bundle.is_absolute():
                raise ConfigError("SERVING_ENDPOINT_BUNDLE must be an absolute path")
            if not bundle.parent.is_dir():
                raise ConfigError("SERVING_ENDPOINT_BUNDLE parent directory must exist")
        return cls(
            state_dir=state_dir,
            bundle=bundle,
            inspire_cwd=inspire_cwd,
            attempts=_env_int("SERVING_READONLY_ATTEMPTS", 3, minimum=1),
            retry_delay_s=_env_int("SERVING_RETRY_DELAY_SECONDS", 2, minimum=0),
            poll_s=_env_int("SERVING_POLL_SECONDS", 10, minimum=1),
            wait_timeout_s=_env_int("SERVING_WAIT_TIMEOUT_SECONDS", 3600, minimum=1),
            stop_timeout_s=_env_int("SERVING_STOP_TIMEOUT_SECONDS", 600, minimum=1),
            delete_timeout_s=_env_int("SERVING_DELETE_TIMEOUT_SECONDS", 600, minimum=1),
        )


@dataclass(frozen=True)
class State:
    name: str
    status: str


@dataclass(frozen=True)
class Spec:
    command: str
    description: str


@dataclass(frozen=True)
class EndpointMeta:
    key: str
    api_path: str


ENDPOINTS = {
    "asr": EndpointMeta("TTS_ASR_URL", "/v1/chat/completions"),
    "sim": EndpointMeta("TTS_SIM_URL", "/v1/similarities"),
    "judge": EndpointMeta("TTS_JUDGE_URL", "/v1/chat/completions"),
}


# TOML and operational environment parsing.
def _env_int(name: str, default: int, *, minimum: int) -> int:
    value = os.environ.get(name, str(default))
    if not value.isdigit() or int(value) < minimum:
        raise ConfigError(f"{name} must be an integer greater than or equal to {minimum}")
    return int(value)


def _table(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a TOML table")
    return value


def _keys(table: Mapping[str, object], expected: set[str], label: str) -> None:
    missing = expected - set(table)
    unknown = set(table) - expected
    if missing:
        raise ConfigError(f"{label} is missing: {', '.join(sorted(missing))}")
    if unknown:
        raise ConfigError(f"{label} has unknown keys: {', '.join(sorted(unknown))}")


def _text(table: Mapping[str, object], key: str, label: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label}.{key} must be a non-empty string")
    return value.strip()


def _optional_text(table: Mapping[str, object], key: str, label: str) -> str | None:
    value = table.get(key)
    if not isinstance(value, str):
        raise ConfigError(f"{label}.{key} must be a string")
    return value.strip() or None


def _int(table: Mapping[str, object], key: str, label: str, *, minimum: int = 1) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{label}.{key} must be an integer greater than or equal to {minimum}")
    return value


def _float(table: Mapping[str, object], key: str, label: str, *, minimum: float = 0) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or value < minimum:
        raise ConfigError(f"{label}.{key} must be a number greater than or equal to {minimum}")
    return float(value)


def _bool(table: Mapping[str, object], key: str, label: str) -> bool:
    value = table.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"{label}.{key} must be a boolean")
    return value


def _path(table: Mapping[str, object], key: str, label: str) -> Path:
    return Path(_text(table, key, label))


PLATFORM_KEYS = {
    "service_name",
    "workspace",
    "project",
    "group",
    "quota",
    "image",
    "replicas",
    "shm_gib",
    "priority",
    "custom_domain",
    "registry_model",
    "registry_version",
}
VLLM_KEYS = PLATFORM_KEYS | {
    "vllm_bin",
    "model_path",
    "generated_audio_root",
    "served_model_name",
    "gpu_memory_utilization",
    "max_model_len",
}
SIM_KEYS = PLATFORM_KEYS | {
    "checkpoint",
    "reference_audio_roots",
    "generated_audio_root",
    "per_device_batch",
    "dynamic_delay_ms",
    "max_items_per_request",
    "audio_workers",
    "max_queue_items",
    "max_inflight_requests",
    "reference_cache_items",
    "allow_tf32",
}


def _platform(table: Mapping[str, object], kind: str) -> Platform:
    label = kind
    quota_table = _table(table.get("quota"), f"{label}.quota")
    _keys(quota_table, {"gpus", "cpus", "memory_gib"}, f"{label}.quota")
    quota = Quota(
        gpus=_int(quota_table, "gpus", f"{label}.quota"),
        cpus=_int(quota_table, "cpus", f"{label}.quota"),
        memory_gib=_int(quota_table, "memory_gib", f"{label}.quota"),
    )
    if quota.gpus not in {1, 4, 8}:
        raise ConfigError(f"{label}.quota.gpus must be one of 1, 4, or 8")
    replicas = _int(table, "replicas", label)
    shm_gib = _int(table, "shm_gib", label)
    priority = _int(table, "priority", label)
    if shm_gib > quota.memory_gib:
        raise ConfigError(f"{label}.shm_gib cannot exceed quota.memory_gib")
    if priority > 10:
        raise ConfigError(f"{label}.priority must be in 1..10")
    name = _text(table, "service_name", label)
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", name) or len(name) > 63:
        raise ConfigError(f"{label}.service_name must be a lowercase DNS label of at most 63 characters")
    custom_domain = _optional_text(table, "custom_domain", label)
    if custom_domain is not None and (
        not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", custom_domain) or len(custom_domain) > 63
    ):
        raise ConfigError(f"{label}.custom_domain must be a lowercase DNS label of at most 63 characters")
    return Platform(
        kind=kind,
        name=name,
        workspace=_text(table, "workspace", label),
        project=_text(table, "project", label),
        group=_text(table, "group", label),
        quota=quota,
        image=_text(table, "image", label),
        replicas=replicas,
        shm_gib=shm_gib,
        priority=priority,
        custom_domain=custom_domain,
        registry_model=_text(table, "registry_model", label),
        registry_version=_text(table, "registry_version", label),
    )


def _vllm(table: Mapping[str, object], kind: str) -> Vllm:
    _keys(table, VLLM_KEYS, kind)
    memory_fraction = _float(table, "gpu_memory_utilization", kind)
    if not 0 < memory_fraction < 1:
        raise ConfigError(f"{kind}.gpu_memory_utilization must be between 0 and 1")
    return Vllm(
        platform=_platform(table, kind),
        bin=_text(table, "vllm_bin", kind),
        model_path=_path(table, "model_path", kind),
        audio_root=_path(table, "generated_audio_root", kind),
        served_name=_text(table, "served_model_name", kind),
        memory_fraction=memory_fraction,
        max_model_len=_int(table, "max_model_len", kind),
    )


def _sim(table: Mapping[str, object]) -> Sim:
    label = "sim"
    _keys(table, SIM_KEYS, label)
    roots_value = table.get("reference_audio_roots")
    if not isinstance(roots_value, list) or not roots_value:
        raise ConfigError("sim.reference_audio_roots must be a non-empty array")
    roots: list[Path] = []
    for index, value in enumerate(roots_value):
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"sim.reference_audio_roots[{index}] must be a non-empty string")
        roots.append(Path(value.strip()))
    per_device_batch = _int(table, "per_device_batch", label)
    max_items = _int(table, "max_items_per_request", label)
    if per_device_batch > 16:
        raise ConfigError("sim.per_device_batch must be at most 16")
    if max_items > 16:
        raise ConfigError("sim.max_items_per_request must be at most 16")
    return Sim(
        platform=_platform(table, label),
        checkpoint=_path(table, "checkpoint", label),
        reference_roots=tuple(roots),
        audio_root=_path(table, "generated_audio_root", label),
        per_device_batch=per_device_batch,
        delay_ms=_float(table, "dynamic_delay_ms", label),
        max_items=max_items,
        audio_workers=_int(table, "audio_workers", label),
        max_queue_items=_int(table, "max_queue_items", label),
        max_inflight=_int(table, "max_inflight_requests", label),
        cache_items=_int(table, "reference_cache_items", label),
        allow_tf32=_bool(table, "allow_tf32", label),
    )


def load_config(path: Path) -> Config:
    if not path.is_file():
        raise ConfigError(f"config does not exist: {path}")
    try:
        with path.open("rb") as stream:
            raw: object = tomllib.load(stream)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"invalid TOML in {path}: {error}") from error
    root = _table(raw, "config")
    _keys(root, set(KINDS), "config")
    return Config(
        asr=_vllm(_table(root.get("asr"), "asr"), "asr"),
        sim=_sim(_table(root.get("sim"), "sim")),
        judge=_vllm(_table(root.get("judge"), "judge"), "judge"),
    )


# Process adapter for the installed Inspire CLI.
class Runner:
    def __init__(self, settings: Settings) -> None:
        inspire = shutil.which("inspire")
        if inspire is None:
            raise ConfigError("inspire CLI is not available")
        self.inspire = Path(inspire)
        self.settings = settings

    def read(self, command: Sequence[str], *, label: str = "inspire") -> str:
        # Only idempotent reads and platform dry-runs use this retry path.
        delay_s = self.settings.retry_delay_s
        for attempt in range(1, self.settings.attempts + 1):
            result = subprocess.run(
                command,
                cwd=self.settings.inspire_cwd,
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout
            if attempt == self.settings.attempts:
                for line in result.stderr.splitlines():
                    print(f"{label}: {line}", file=sys.stderr)
                raise CommandError(f"{label} command failed with exit code {result.returncode}")
            print(
                f"warning: {label} read failed; retrying attempt {attempt + 1}/{self.settings.attempts}",
                file=sys.stderr,
            )
            time.sleep(delay_s)
            delay_s *= 2
        raise AssertionError("retry loop did not return")

    def read_inspire(self, *args: str) -> str:
        return self.read((str(self.inspire), *args))

    def write(self, *args: str) -> None:
        # Create, stop, and delete are intentionally submitted once.
        result = subprocess.run(
            (str(self.inspire), *args),
            cwd=self.settings.inspire_cwd,
            check=False,
        )
        if result.returncode != 0:
            raise CommandError(f"inspire command failed with exit code {result.returncode}")

    def endpoint_url(self, service: Service) -> str:
        python = self.inspire.resolve().parent / "python"
        if not python.is_file() or not os.access(python, os.X_OK):
            raise ConfigError("cannot find the Inspire CLI Python runtime")
        platform = service.platform
        meta = ENDPOINTS[platform.kind]
        return self.read(
            (
                str(python),
                str(ROOT / "endpoint.py"),
                "--workspace",
                platform.workspace,
                "--name",
                platform.name,
                "--api-path",
                meta.api_path,
            ),
            label="endpoint",
        ).strip()


# Public Inspire JSON contains service state but intentionally omits URLs.
def _obj(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CommandError(f"{label} must be a JSON object")
    return value


def _state(payload: str, wanted_name: str) -> State:
    try:
        parsed: object = json.loads(payload)
    except json.JSONDecodeError as error:
        raise CommandError("Inspire returned invalid JSON") from error
    root = _obj(parsed, "response")
    if root.get("success") is not True:
        raise CommandError("Inspire response did not report success")
    data = _obj(root.get("data"), "response.data")
    items = data.get("items")
    if not isinstance(items, list):
        raise CommandError("response.data.items must be a list")
    matches: list[State] = []
    for index, value in enumerate(items):
        item = _obj(value, f"response.data.items[{index}]")
        name = item.get("name")
        status = item.get("status")
        if not isinstance(name, str) or not name.strip():
            raise CommandError(f"response.data.items[{index}].name must be a non-empty string")
        if not isinstance(status, str) or not status.strip():
            raise CommandError(f"response.data.items[{index}].status must be a non-empty string")
        if name.strip() == wanted_name:
            matches.append(State(name.strip(), status.strip()))
    if len(matches) > 1:
        raise CommandError(f"Inspire returned duplicate service name {wanted_name!r}")
    return matches[0] if matches else State(wanted_name, "ABSENT")


# Service-specific command construction and asset validation.
def _shared_dir(path: Path, label: str) -> None:
    if not path.is_absolute() or not str(path).startswith("/inspire/"):
        raise ConfigError(f"{label} must be an absolute shared /inspire path")
    if not path.is_dir():
        raise ConfigError(f"{label} is not a directory")


def _placeholder(value: str, label: str) -> None:
    if "REPLACE_ME" in value or "replace-me" in value:
        raise ConfigError(f"{label} still contains a REPLACE_ME placeholder")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repo), *args),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise CommandError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def _vllm_spec(service: Vllm) -> Spec:
    platform = service.platform
    _placeholder(platform.name, f"{platform.kind}.service_name")
    _placeholder(str(service.audio_root), f"{platform.kind}.generated_audio_root")
    if not (service.model_path / "config.json").is_file():
        raise ConfigError(f"{platform.kind}.model_path is missing config.json")
    if not (service.model_path / "model.safetensors.index.json").is_file():
        raise ConfigError(f"{platform.kind}.model_path is missing its weight index")
    _shared_dir(service.audio_root, f"{platform.kind}.generated_audio_root")
    command = shlex.join(
        (
            "exec",
            service.bin,
            "serve",
            str(service.model_path),
            "--served-model-name",
            service.served_name,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--dtype",
            "bfloat16",
            "--tensor-parallel-size",
            "1",
            "--data-parallel-size",
            str(platform.quota.gpus),
            "--gpu-memory-utilization",
            str(service.memory_fraction),
            "--max-model-len",
            str(service.max_model_len),
            "--allowed-local-media-path",
            str(service.audio_root),
        )
    )
    label = "ASR" if platform.kind == "asr" else "Judge"
    return Spec(command, f"MOSS-TTS {label} direct vLLM TP1 DP{platform.quota.gpus}")


def _sim_spec(service: Sim, *, clean: bool) -> Spec:
    platform = service.platform
    _placeholder(platform.name, "sim.service_name")
    _placeholder(str(service.audio_root), "sim.generated_audio_root")
    for root in service.reference_roots:
        _placeholder(str(root), "sim.reference_audio_roots")
    if shutil.which("git") is None:
        raise ConfigError("git is not available")
    source = SLIME_ROOT / "slime/serving/tts_sim/__main__.py"
    launcher = ROOT / "runtime/launch_sim.sh"
    if not source.is_file():
        raise ConfigError("Slime is missing its vendored SIM service entrypoint")
    if not service.checkpoint.is_file():
        raise ConfigError("sim.checkpoint is not a file")
    for index, root in enumerate(service.reference_roots):
        _shared_dir(root, f"sim.reference_audio_roots[{index}]")
    _shared_dir(service.audio_root, "sim.generated_audio_root")
    head = _git(SLIME_ROOT, "rev-parse", "HEAD")
    # Validate actual shared source bytes, so development trees do not require a commit.
    fingerprint = SLIME_ROOT / "tools/tts_source_fingerprint.py"
    tree = subprocess.check_output([sys.executable, str(fingerprint), str(source.parent)], text=True).strip()
    launcher_blob = _git(SLIME_ROOT, "hash-object", "--", str(launcher))
    allowed_roots = ":".join(str(root) for root in (*service.reference_roots, service.audio_root))
    env = {
        "EXPECTED_VISIBLE_GPUS": str(platform.quota.gpus),
        "SLIME_REPO": str(SLIME_ROOT),
        "SLIME_DEPLOY_SIM_TREE": tree,
        "SLIME_DEPLOY_LAUNCHER_BLOB": launcher_blob,
        "WAVLM_SIM_CHECKPOINT": str(service.checkpoint),
        "WAVLM_SIM_ALLOWED_ROOTS": allowed_roots,
        "WAVLM_SIM_PER_DEVICE_BATCH": str(service.per_device_batch),
        "WAVLM_SIM_DYNAMIC_DELAY_MS": str(service.delay_ms),
        "WAVLM_SIM_MAX_ITEMS_PER_REQUEST": str(service.max_items),
        "WAVLM_SIM_AUDIO_WORKERS": str(service.audio_workers),
        "WAVLM_SIM_MAX_QUEUE_ITEMS": str(service.max_queue_items),
        "WAVLM_SIM_MAX_INFLIGHT_REQUESTS": str(service.max_inflight),
        "WAVLM_SIM_REFERENCE_CACHE_ITEMS": str(service.cache_items),
        "WAVLM_SIM_ALLOW_TF32": "1" if service.allow_tf32 else "0",
        "SERVICE_PORT": "8000",
    }
    command = shlex.join(("exec", "env", *(f"{key}={value}" for key, value in env.items()), "bash", str(launcher)))
    description = (
        f"MOSS-TTS Torch SIM 1 process, {platform.quota.gpus} model replicas, "
        f"shared cache; slime={head}; sim-tree={tree}"
    )
    return Spec(command, description)


def _create_args(service: Service, *, clean: bool) -> list[str]:
    platform = service.platform
    spec = _vllm_spec(service) if isinstance(service, Vllm) else _sim_spec(service, clean=clean)
    quota = platform.quota
    args = [
        "serving",
        "create",
        "--name",
        platform.name,
        "--model",
        platform.registry_model,
        "--model-version",
        platform.registry_version,
        "--workspace",
        platform.workspace,
        "--project",
        platform.project,
        "--group",
        platform.group,
        "--quota",
        f"{quota.gpus},{quota.cpus},{quota.memory_gib}",
        "--image",
        platform.image,
        "--command",
        spec.command,
        "--port",
        "8000",
        "--replicas",
        str(platform.replicas),
        "--nodes-per-replica",
        "1",
        "--shm-size",
        str(platform.shm_gib),
        "--priority",
        str(platform.priority),
        "--description",
        spec.description,
    ]
    if platform.custom_domain is not None:
        args.extend(("--custom-domain", platform.custom_domain))
    return args


# Serving lifecycle behind the ACTION + TARGET interface.
class Manager:
    def __init__(self, config: Config, settings: Settings) -> None:
        self.config = config
        self.settings = settings
        self.runner = Runner(settings)

    def state(self, service: Service) -> State:
        platform = service.platform
        payload = self.runner.read_inspire(
            "--json",
            "serving",
            "list",
            "--workspace",
            platform.workspace,
            "--project",
            platform.project,
            "--keyword",
            platform.name,
            "--limit",
            "200",
        )
        return _state(payload, platform.name)

    def wait(self, service: Service, wanted: str, timeout_s: int) -> None:
        started = time.monotonic()
        while True:
            state = self.state(service)
            if state.status == wanted:
                return
            if wanted == "RUNNING" and state.status in {"FAILED", "STOPPED"}:
                raise CommandError(f"{state.name} entered terminal state {state.status} while waiting for RUNNING")
            if time.monotonic() - started >= timeout_s:
                raise CommandError(f"timed out waiting for {state.name} to reach {wanted}")
            time.sleep(self.settings.poll_s)

    def write_endpoint(self, service: Service) -> None:
        platform = service.platform
        meta = ENDPOINTS[platform.kind]
        state_file = self.settings.state_dir / platform.kind / f"{platform.name}.env"
        url = self.runner.endpoint_url(service)
        save_endpoint(meta.key, url, state_file, self.settings.bundle)
        if self.settings.bundle is not None:
            print(f"endpoint bundle: {self.settings.bundle}", file=sys.stderr)
        print(f"endpoint file: {state_file}", file=sys.stderr)

    def remove_endpoint(self, service: Service) -> None:
        platform = service.platform
        meta = ENDPOINTS[platform.kind]
        state_file = self.settings.state_dir / platform.kind / f"{platform.name}.env"
        result = remove_endpoint(meta.key, state_file, self.settings.bundle)
        if result.warning is not None:
            print(f"warning: {result.warning}", file=sys.stderr)

    def plan(self, service: Service) -> None:
        output = self.runner.read_inspire(*_create_args(service, clean=False), "--dry-run")
        print(output, end="")

    def deploy(self, service: Service) -> None:
        args = _create_args(service, clean=True)
        output = self.runner.read_inspire(*args, "--dry-run")
        print(output, end="")
        state = self.state(service)
        if state.status != "ABSENT":
            raise CommandError(f"{state.name} already exists in state {state.status}; choose another service_name")
        self.runner.write(*args)
        self.wait(service, "RUNNING", self.settings.wait_timeout_s)
        self.write_endpoint(service)

    def stop(self, service: Service) -> None:
        state = self.state(service)
        if state.status in {"ABSENT", "STOPPED", "FAILED"}:
            print(f"{state.name}\t{state.status}")
            return
        platform = service.platform
        self.runner.write("serving", "stop", platform.name, "--workspace", platform.workspace)
        self.wait(service, "STOPPED", self.settings.stop_timeout_s)

    def delete(self, service: Service, confirmation: str | None) -> None:
        platform = service.platform
        if confirmation != platform.name:
            raise ConfigError(f"delete requires --confirm-delete {platform.name}")
        state = self.state(service)
        if state.status == "ABSENT":
            self.remove_endpoint(service)
            print(f"{state.name}\t{state.status}")
            return
        if state.status not in {"STOPPED", "FAILED"}:
            self.runner.write("serving", "stop", platform.name, "--workspace", platform.workspace)
            self.wait(service, "STOPPED", self.settings.stop_timeout_s)
        self.runner.write("serving", "delete", platform.name, "--workspace", platform.workspace, "--yes")
        self.wait(service, "ABSENT", self.settings.delete_timeout_s)
        self.remove_endpoint(service)

    def run(self, action: str, kind: str, confirmation: str | None) -> None:
        service = self.config.get(kind)
        if action == "plan":
            self.plan(service)
        elif action == "deploy":
            self.deploy(service)
        elif action == "status":
            state = self.state(service)
            print(f"{state.name}\t{state.status}")
        elif action == "wait":
            self.wait(service, "RUNNING", self.settings.wait_timeout_s)
        elif action == "endpoint":
            self.write_endpoint(service)
        elif action == "stop":
            self.stop(service)
        elif action == "delete":
            self.delete(service, confirmation)
        else:
            raise ConfigError(f"unknown action: {action}")


# Command-line interface.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage MOSS-TTS ASR, SIM, and Judge Inspire Servings.",
    )
    parser.add_argument("action", nargs="?", default="plan", choices=ALL_ACTIONS)
    parser.add_argument("target", nargs="?", default="all", choices=(*KINDS, "all"))
    parser.add_argument("--config", type=Path, default=ROOT / "config.toml")
    parser.add_argument("--confirm-delete", metavar="SERVICE_NAME")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.target == "all" and args.action not in ALL_SAFE_ACTIONS:
        raise ConfigError(
            "target all only supports plan, status, or endpoint; "
            "manage platform state changes one service at a time"
        )
    manager = Manager(load_config(args.config), Settings.from_env())
    kinds = KINDS if args.target == "all" else (args.target,)
    for kind in kinds:
        if len(kinds) > 1:
            print(f"== {kind} {args.action} ==")
        manager.run(args.action, kind, args.confirm_delete)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CommandError, ConfigError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
