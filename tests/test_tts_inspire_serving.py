import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVING = ROOT / "examples" / "tts_grpo" / "serving"
MANAGE = SERVING / "manage.py"

sys.path.insert(0, str(SERVING))
from endpoint import _endpoint_url, remove_endpoint, save_endpoint  # noqa: E402

FAKE_HEAD = "1234567890abcdef1234567890abcdef12345678"
from tools.tts_source_fingerprint import source_fingerprint
SIM_TREE_SHA256 = source_fingerprint(ROOT / "slime/serving/tts_sim")
FAKE_LAUNCHER_BLOB = "abcabcabcabcabcabcabcabcabcabcabcabcabca"


def _quote(value: str | Path) -> str:
    return json.dumps(str(value))


def _config(
    tmp_path: Path,
    *,
    gpus: int = 4,
    custom_domain: str = "",
    model_assets: bool = True,
) -> Path:
    checkpoint = tmp_path / "wavlm.pth"
    checkpoint.touch()
    models: dict[str, Path] = {}
    for kind in ("asr", "judge"):
        model = tmp_path / f"{kind}-model"
        model.mkdir()
        if model_assets:
            (model / "config.json").touch()
            (model / "model.safetensors.index.json").touch()
        models[kind] = model

    def platform(kind: str, image: str, registry: str) -> str:
        return f"""
service_name = "tts-rm-{kind}-test123"
workspace = "test-workspace"
project = "test-project"
group = "test-group"
image = {json.dumps(image)}
replicas = 2
shm_gib = 200
priority = 4
custom_domain = {json.dumps(custom_domain)}
registry_model = {json.dumps(registry)}
registry_version = "1"
"""

    path = tmp_path / "config.toml"
    path.write_text(
        f"""
[asr]
{platform("asr", "test-vllm:1", "test-asr")}
vllm_bin = "vllm"
model_path = {_quote(models["asr"])}
generated_audio_root = {_quote(ROOT)}
served_model_name = "test-asr"
gpu_memory_utilization = 0.9
max_model_len = 4096

[asr.quota]
gpus = {gpus}
cpus = 80
memory_gib = 800

[sim]
{platform("sim", "test-sim:1", "test-sim")}
checkpoint = {_quote(checkpoint)}
reference_audio_roots = [{_quote(ROOT)}]
generated_audio_root = {_quote(SERVING)}
per_device_batch = 16
dynamic_delay_ms = 5.0
max_items_per_request = 16
audio_workers = 16
max_queue_items = 8192
max_inflight_requests = 4096
reference_cache_items = 100000
allow_tf32 = true

[sim.quota]
gpus = {gpus}
cpus = 80
memory_gib = 800

[judge]
{platform("judge", "test-vllm:1", "test-judge")}
vllm_bin = "vllm"
model_path = {_quote(models["judge"])}
generated_audio_root = {_quote(ROOT)}
served_model_name = "test-judge"
gpu_memory_utilization = 0.9
max_model_len = 4096

[judge.quota]
gpus = {gpus}
cpus = 80
memory_gib = 800
""",
        encoding="utf-8",
    )
    return path


def _fake_tools(tmp_path: Path) -> tuple[Path, Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "inspire.jsonl"
    state_path = tmp_path / "servings.json"
    state_path.write_text("[]", encoding="utf-8")

    inspire = bin_dir / "inspire"
    inspire.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
log_path = Path(os.environ["FAKE_INSPIRE_LOG"])
state_path = Path(os.environ["FAKE_INSPIRE_STATE"])
with log_path.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(args) + "\\n")
state = json.loads(state_path.read_text())

if args[:3] == ["--json", "serving", "list"]:
    print(json.dumps({"success": True, "data": {"total": len(state), "items": state}}))
elif args[:2] == ["serving", "create"]:
    name = args[args.index("--name") + 1]
    target = "asr" if "-asr-" in name else "sim" if "-sim-" in name else "judge"
    if "--dry-run" in args and os.environ.get("FAKE_DRY_RUN_FAIL_TARGET") == target:
        print(f"dry-run rejected {target}", file=sys.stderr)
        raise SystemExit(12)
    if "--dry-run" not in args:
        if os.environ.get("FAKE_CREATE_FAIL_TARGET") == target:
            print(f"create rejected {target}", file=sys.stderr)
            raise SystemExit(13)
        state.append({"name": name, "status": "RUNNING"})
        state_path.write_text(json.dumps(state))
    print("fake create")
elif args[:2] == ["serving", "stop"]:
    name = args[2]
    for item in state:
        if item["name"] == name:
            item["status"] = "STOPPED"
    state_path.write_text(json.dumps(state))
elif args[:2] == ["serving", "delete"]:
    name = args[2]
    state_path.write_text(json.dumps([item for item in state if item["name"] != name]))
else:
    print("unsupported fake inspire invocation", args, file=sys.stderr)
    raise SystemExit(2)
""",
        encoding="utf-8",
    )
    inspire.chmod(0o755)

    inspire_python = bin_dir / "python"
    inspire_python.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
name = args[args.index("--name") + 1]
api_path = args[args.index("--api-path") + 1]
state = json.loads(Path(os.environ["FAKE_INSPIRE_STATE"]).read_text())
matches = [item for item in state if item["name"] == name and item["status"] == "RUNNING"]
if len(matches) != 1:
    print(f"service {name} is not RUNNING", file=sys.stderr)
    raise SystemExit(3)
print(f"https://{name}.example{api_path}")
""",
        encoding="utf-8",
    )
    inspire_python.chmod(0o755)

    git = bin_dir / "git"
    git.write_text(
        f"""#!/usr/bin/env python3
import sys

args = sys.argv[1:]
command = args[2:] if len(args) > 2 and args[0] == "-C" else args
if command == ["rev-parse", "HEAD"]:
    print("{FAKE_HEAD}")
elif command == ["rev-parse", "HEAD:slime/serving/tts_sim"]:
    print("{SIM_TREE_SHA256}")
elif command[:2] == ["hash-object", "--"]:
    print("{FAKE_LAUNCHER_BLOB}")
elif command[:2] == ["status", "--porcelain"]:
    pass
else:
    print("unsupported fake git invocation", args, file=sys.stderr)
    raise SystemExit(2)
""",
        encoding="utf-8",
    )
    git.chmod(0o755)
    return bin_dir, log_path, state_path


def _run(
    tmp_path: Path,
    action: str,
    target: str,
    *,
    initial_state: list[dict[str, str]] | None = None,
    config: Path | None = None,
    extra_args: tuple[str, ...] = (),
    **overrides: str,
) -> subprocess.CompletedProcess[str]:
    bin_dir, log_path, state_path = _fake_tools(tmp_path)
    if initial_state is not None:
        state_path.write_text(json.dumps(initial_state), encoding="utf-8")
    config = config or _config(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_INSPIRE_LOG": str(log_path),
        "FAKE_INSPIRE_STATE": str(state_path),
        "SERVING_STATE_DIR": str(tmp_path / "state"),
        "SERVING_ENDPOINT_BUNDLE": str(tmp_path / "endpoints.env"),
        "SERVING_INSPIRE_CWD": str(tmp_path),
        "SERVING_READONLY_ATTEMPTS": "1",
        "SERVING_RETRY_DELAY_SECONDS": "0",
        "SERVING_POLL_SECONDS": "1",
        "SERVING_WAIT_TIMEOUT_SECONDS": "5",
        "SERVING_STOP_TIMEOUT_SECONDS": "5",
        "SERVING_DELETE_TIMEOUT_SECONDS": "5",
        **overrides,
    }
    return subprocess.run(
        [str(MANAGE), action, target, "--config", str(config), *extra_args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _calls(tmp_path: Path) -> list[list[str]]:
    path = tmp_path / "inspire.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_directory_has_only_the_five_documented_files() -> None:
    files = {
        path.relative_to(SERVING).as_posix()
        for path in SERVING.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert files == {
        "README.md",
        "config.toml",
        "endpoint.py",
        "manage.py",
        "runtime/launch_sim.sh",
    }
    assert os.access(MANAGE, os.X_OK)
    assert os.access(SERVING / "runtime" / "launch_sim.sh", os.X_OK)


def test_checked_in_config_keeps_independent_cq_defaults() -> None:
    with (SERVING / "config.toml").open("rb") as stream:
        config = tomllib.load(stream)

    assert set(config) == {"asr", "sim", "judge"}
    for kind in ("asr", "sim", "judge"):
        service = config[kind]
        assert service["workspace"] == "CQ-科研驾驶舱"
        assert service["project"] == "CQ项目"
        assert service["group"] == "科研驾驶舱"
        assert service["quota"] == {"gpus": 8, "cpus": 168, "memory_gib": 1800}
    assert "rm_repo" not in config["asr"]
    assert "rm_repo" not in config["judge"]


@pytest.mark.parametrize("kind", ["asr", "judge"])
@pytest.mark.parametrize("gpus", [1, 4, 8])
def test_vllm_plan_uses_quota_gpu_count(tmp_path: Path, kind: str, gpus: int) -> None:
    result = _run(tmp_path, "plan", kind, config=_config(tmp_path, gpus=gpus))

    assert result.returncode == 0, result.stderr
    create = _calls(tmp_path)[0]
    command = create[create.index("--command") + 1]
    assert command.count("--tensor-parallel-size 1") == 1
    assert f"--data-parallel-size {gpus}" in command
    assert create[create.index("--quota") + 1] == f"{gpus},80,800"
    assert create[create.index("--replicas") + 1] == "2"


def test_sim_plan_passes_typed_runtime_values(tmp_path: Path) -> None:
    result = _run(tmp_path, "plan", "sim")

    assert result.returncode == 0, result.stderr
    create = _calls(tmp_path)[0]
    command = create[create.index("--command") + 1]
    assert "EXPECTED_VISIBLE_GPUS=4" in command
    assert f"SLIME_DEPLOY_SIM_TREE={SIM_TREE_SHA256}" in command
    assert f"SLIME_DEPLOY_LAUNCHER_BLOB={FAKE_LAUNCHER_BLOB}" in command
    assert "SLIME_DEPLOY_GIT_COMMIT" not in command
    assert "WAVLM_SIM_ALLOW_TF32=1" in command
    assert "WAVLM_SIM_MAX_REQUEST_BYTES" not in command


def test_sim_config_rejects_removed_request_byte_limit(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "allow_tf32 = true",
            "max_request_bytes = 4194304\nallow_tf32 = true",
            1,
        ),
        encoding="utf-8",
    )

    result = _run(tmp_path, "plan", "sim", config=config)

    assert result.returncode != 0
    assert "sim has unknown keys: max_request_bytes" in result.stderr


def test_custom_domain_is_explicit_opt_in(tmp_path: Path) -> None:
    without_domain = _run(tmp_path, "plan", "asr")
    assert "--custom-domain" not in _calls(tmp_path)[0]
    assert without_domain.returncode == 0

    other = tmp_path / "with-domain"
    other.mkdir()
    with_domain = _run(other, "plan", "asr", config=_config(other, custom_domain="asr-test"))
    assert with_domain.returncode == 0, with_domain.stderr
    create = _calls(other)[0]
    assert create[create.index("--custom-domain") + 1] == "asr-test"


def test_invalid_gpu_count_fails_before_inspire(tmp_path: Path) -> None:
    result = _run(tmp_path, "plan", "sim", config=_config(tmp_path, gpus=2))

    assert result.returncode != 0
    assert "quota.gpus must be one of 1, 4, or 8" in result.stderr
    assert _calls(tmp_path) == []


def test_malformed_toml_fails_before_inspire(tmp_path: Path) -> None:
    config = tmp_path / "invalid.toml"
    config.write_text("[asr\n", encoding="utf-8")

    result = _run(tmp_path, "status", "asr", config=config)

    assert result.returncode != 0
    assert "invalid TOML" in result.stderr
    assert _calls(tmp_path) == []


def test_status_does_not_validate_model_assets(tmp_path: Path) -> None:
    result = _run(tmp_path, "status", "asr", config=_config(tmp_path, model_assets=False))

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("\tABSENT")


def test_deploy_writes_state_and_partial_bundle(tmp_path: Path) -> None:
    result = _run(tmp_path, "deploy", "sim")

    assert result.returncode == 0, result.stderr
    creates = [call for call in _calls(tmp_path) if call[:2] == ["serving", "create"]]
    assert ["--dry-run" in call for call in creates] == [True, False]
    state_file = tmp_path / "state" / "sim" / "tts-rm-sim-test123.env"
    assert state_file.read_text() == ("TTS_SIM_URL=https://tts-rm-sim-test123.example/v1/similarities\n")
    assert state_file.stat().st_mode & 0o777 == 0o600
    bundle = tmp_path / "endpoints.env"
    assert bundle.read_text() == ("TTS_SIM_URL=https://tts-rm-sim-test123.example/v1/similarities\n")
    assert bundle.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "endpoints.env.lock").stat().st_mode & 0o777 == 0o600


def test_endpoint_updates_only_selected_bundle_entry(tmp_path: Path) -> None:
    bundle = tmp_path / "endpoints.env"
    bundle.write_text(
        "TTS_ASR_URL=https://asr.example/v1/chat/completions\n"
        "TTS_SIM_URL=https://old-sim.example/v1/similarities\n"
        "TTS_JUDGE_URL=https://judge.example/v1/chat/completions\n"
    )
    result = _run(
        tmp_path,
        "endpoint",
        "sim",
        initial_state=[{"name": "tts-rm-sim-test123", "status": "RUNNING"}],
    )

    assert result.returncode == 0, result.stderr
    assert bundle.read_text() == (
        "TTS_ASR_URL=https://asr.example/v1/chat/completions\n"
        "TTS_SIM_URL=https://tts-rm-sim-test123.example/v1/similarities\n"
        "TTS_JUDGE_URL=https://judge.example/v1/chat/completions\n"
    )


def test_invalid_bundle_fails_without_leaking_values(tmp_path: Path) -> None:
    bundle = tmp_path / "endpoints.env"
    bundle.write_text("INSPIRE_API_KEY=must-not-leak\n")
    result = _run(
        tmp_path,
        "endpoint",
        "sim",
        initial_state=[{"name": "tts-rm-sim-test123", "status": "RUNNING"}],
    )

    assert result.returncode != 0
    assert "unsupported key: INSPIRE_API_KEY" in result.stderr
    assert "must-not-leak" not in result.stderr


def test_failed_dry_run_does_not_create(tmp_path: Path) -> None:
    result = _run(tmp_path, "deploy", "asr", FAKE_DRY_RUN_FAIL_TARGET="asr")

    assert result.returncode != 0
    creates = [call for call in _calls(tmp_path) if call[:2] == ["serving", "create"]]
    assert len(creates) == 1
    assert "--dry-run" in creates[0]


def test_failed_create_is_not_retried(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        "deploy",
        "judge",
        FAKE_CREATE_FAIL_TARGET="judge",
        SERVING_READONLY_ATTEMPTS="3",
    )

    assert result.returncode != 0
    creates = [call for call in _calls(tmp_path) if call[:2] == ["serving", "create"]]
    assert ["--dry-run" in call for call in creates] == [True, False]
    assert not any(call[:2] == ["serving", "delete"] for call in _calls(tmp_path))


def test_delete_requires_exact_confirmation(tmp_path: Path) -> None:
    result = _run(tmp_path, "delete", "asr", extra_args=("--confirm-delete", "wrong"))

    assert result.returncode != 0
    assert "--confirm-delete tts-rm-asr-test123" in result.stderr
    assert _calls(tmp_path) == []


def test_delete_stops_then_deletes_selected_service(tmp_path: Path) -> None:
    name = "tts-rm-asr-test123"
    result = _run(
        tmp_path,
        "delete",
        "asr",
        initial_state=[
            {"name": name, "status": "RUNNING"},
            {"name": "tts-rm-sim-test123", "status": "RUNNING"},
        ],
        extra_args=("--confirm-delete", name),
    )

    assert result.returncode == 0, result.stderr
    assert [call[2] for call in _calls(tmp_path) if call[:2] == ["serving", "stop"]] == [name]
    assert [call[2] for call in _calls(tmp_path) if call[:2] == ["serving", "delete"]] == [name]


def test_delete_old_service_keeps_new_service_bundle_url(tmp_path: Path) -> None:
    old_name = "tts-rm-asr-test123"
    state_file = tmp_path / "state" / "asr" / f"{old_name}.env"
    state_file.parent.mkdir(parents=True)
    state_file.write_text("TTS_ASR_URL=https://old.example/v1/chat/completions\n")
    bundle = tmp_path / "endpoints.env"
    bundle.write_text("TTS_ASR_URL=https://new.example/v1/chat/completions\n")

    result = _run(
        tmp_path,
        "delete",
        "asr",
        extra_args=("--confirm-delete", old_name),
    )

    assert result.returncode == 0, result.stderr
    assert bundle.read_text() == "TTS_ASR_URL=https://new.example/v1/chat/completions\n"
    assert not state_file.exists()
    assert "service state file has a different URL" in result.stderr


def test_stop_handles_pending_service(tmp_path: Path) -> None:
    name = "tts-rm-sim-test123"
    result = _run(
        tmp_path,
        "stop",
        "sim",
        initial_state=[{"name": name, "status": "PENDING"}],
    )

    assert result.returncode == 0, result.stderr
    assert [call[2] for call in _calls(tmp_path) if call[:2] == ["serving", "stop"]] == [name]


def test_all_rejects_platform_state_changes(tmp_path: Path) -> None:
    result = _run(tmp_path, "deploy", "all")

    assert result.returncode != 0
    assert "target all only supports plan, status, or endpoint" in result.stderr
    assert _calls(tmp_path) == []


def test_endpoint_all_refreshes_complete_bundle(tmp_path: Path) -> None:
    state = [{"name": f"tts-rm-{kind}-test123", "status": "RUNNING"} for kind in ("asr", "sim", "judge")]
    result = _run(tmp_path, "endpoint", "all", initial_state=state)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "endpoints.env").read_text() == (
        "TTS_ASR_URL=https://tts-rm-asr-test123.example/v1/chat/completions\n"
        "TTS_SIM_URL=https://tts-rm-sim-test123.example/v1/similarities\n"
        "TTS_JUDGE_URL=https://tts-rm-judge-test123.example/v1/chat/completions\n"
    )


def test_endpoint_url_validates_authenticated_detail() -> None:
    assert (
        _endpoint_url(
            {"status": "RUNNING", "extra_info": {"service": "https://serving.example/"}},
            "/v1/chat/completions",
        )
        == "https://serving.example/v1/chat/completions"
    )
    with pytest.raises(ValueError, match="must use HTTP"):
        _endpoint_url(
            {"status": "RUNNING", "extra_info": {"service": "ftp://invalid.example"}},
            "/health",
        )


def test_remove_endpoint_requires_matching_service_state(tmp_path: Path) -> None:
    bundle = tmp_path / "endpoints.env"
    state_file = tmp_path / "state.env"
    bundle.write_text("TTS_SIM_URL=https://new.example/v1/similarities\n")
    state_file.write_text("TTS_SIM_URL=https://old.example/v1/similarities\n")

    result = remove_endpoint("TTS_SIM_URL", state_file, bundle)

    assert not result.bundle_removed
    assert result.warning is not None
    assert bundle.read_text() == "TTS_SIM_URL=https://new.example/v1/similarities\n"
    assert not state_file.exists()


def test_remove_endpoint_deletes_matching_bundle_entry(tmp_path: Path) -> None:
    bundle = tmp_path / "endpoints.env"
    state_file = tmp_path / "state.env"
    url = "https://sim.example/v1/similarities"
    bundle.write_text(f"TTS_ASR_URL=https://asr.example/v1/chat/completions\nTTS_SIM_URL={url}\n")
    state_file.write_text(f"TTS_SIM_URL={url}\n")

    result = remove_endpoint("TTS_SIM_URL", state_file, bundle)

    assert result.bundle_removed
    assert result.warning is None
    assert bundle.read_text() == "TTS_ASR_URL=https://asr.example/v1/chat/completions\n"
    assert not state_file.exists()


def test_endpoint_bundle_keeps_stable_order_and_permissions(tmp_path: Path) -> None:
    bundle = tmp_path / "endpoints.env"
    save_endpoint(
        "TTS_JUDGE_URL",
        "https://judge.example/v1/chat/completions",
        tmp_path / "judge.env",
        bundle,
    )
    save_endpoint(
        "TTS_ASR_URL",
        "https://asr.example/v1/chat/completions",
        tmp_path / "asr.env",
        bundle,
    )

    assert bundle.read_text() == (
        "TTS_ASR_URL=https://asr.example/v1/chat/completions\n"
        "TTS_JUDGE_URL=https://judge.example/v1/chat/completions\n"
    )
    assert bundle.stat().st_mode & 0o777 == 0o600


def test_parallel_endpoint_updates_do_not_lose_entries(tmp_path: Path) -> None:
    bundle = tmp_path / "endpoints.env"
    entries = (
        ("TTS_ASR_URL", "https://asr.example/v1/chat/completions"),
        ("TTS_SIM_URL", "https://sim.example/v1/similarities"),
        ("TTS_JUDGE_URL", "https://judge.example/v1/chat/completions"),
    )
    script = (
        "import sys; from pathlib import Path; from endpoint import save_endpoint; "
        "save_endpoint(sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))"
    )
    env = {**os.environ, "PYTHONPATH": str(SERVING)}
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                key,
                url,
                str(tmp_path / f"state-{index}.env"),
                str(bundle),
            ],
            env=env,
        )
        for index, (key, url) in enumerate(entries * 4)
    ]

    assert [process.wait() for process in processes] == [0] * len(processes)
    assert bundle.read_text() == (
        "TTS_ASR_URL=https://asr.example/v1/chat/completions\n"
        "TTS_SIM_URL=https://sim.example/v1/similarities\n"
        "TTS_JUDGE_URL=https://judge.example/v1/chat/completions\n"
    )


def _launcher_env(tmp_path: Path) -> dict[str, str]:
    bin_dir, _, _ = _fake_tools(tmp_path)
    python = bin_dir / "python3"
    python.write_text(
        f'#!/usr/bin/env bash\nif [ "${{1:-}}" = "-m" ]; then exit 0; fi\nexec {_quote(sys.executable)} "$@"\n',
        encoding="utf-8",
    )
    python.chmod(0o755)
    checkpoint = tmp_path / "wavlm.pth"
    checkpoint.touch()
    return {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "EXPECTED_VISIBLE_GPUS": "4",
        "SLIME_REPO": str(ROOT),
        "SLIME_DEPLOY_SIM_TREE": SIM_TREE_SHA256,
        "SLIME_DEPLOY_LAUNCHER_BLOB": FAKE_LAUNCHER_BLOB,
        "WAVLM_SIM_CHECKPOINT": str(checkpoint),
        "WAVLM_SIM_ALLOWED_ROOTS": f"{ROOT}:{SERVING}",
        "SERVICE_PORT": "8000",
    }


def test_sim_launcher_validates_relevant_source_hashes(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(SERVING / "runtime" / "launch_sim.sh")],
        env=_launcher_env(tmp_path),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_sim_launcher_rejects_launcher_drift(tmp_path: Path) -> None:
    env = _launcher_env(tmp_path)
    env["SLIME_DEPLOY_LAUNCHER_BLOB"] = "wrong"
    result = subprocess.run(
        ["bash", str(SERVING / "runtime" / "launch_sim.sh")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "runtime launcher does not match" in result.stderr


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

NUM_GPUS = 0
