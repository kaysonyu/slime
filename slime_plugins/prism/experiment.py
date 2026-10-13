"""Prepare an immutable Prism attempt before entering the ordinary train.py.

Only configuration is added to args. Artifacts and reward clients keep their
runtime state on samples and their existing owners.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from slime.observability.config_snapshot import redact, redact_command

ARTIFACTS = "slime_plugins.prism.artifacts"


def path_component(value):
    """Keep readable identifiers without collisions after path sanitization."""
    raw = str(value)
    safe = re.sub(r"[^\w.-]", "_", raw, flags=re.UNICODE).strip(".")[:80]
    if not safe or safe != raw:
        safe = f"{safe or 'unnamed'}_{hashlib.sha256(raw.encode()).hexdigest()[:12]}"
    return safe


def command_options(command):
    """Read flag values without interpreting model-specific CLI arguments."""
    result = {}
    for index, arg in enumerate(command):
        if not arg.startswith("--"):
            continue
        key, separator, value = arg[2:].partition("=")
        if not separator:
            value = (
                command[index + 1] if index + 1 < len(command) and not command[index + 1].startswith("--") else True
            )
        result.setdefault(key.replace("-", "_"), []).append(value)
    return result


def prepare_run(run_dir, command, *, identity, dry_run=False):
    root = Path(run_dir).resolve()
    options = command_options(command)
    source_config = options.get("custom_config_path", [None])[-1]
    config = yaml.safe_load(Path(source_config).read_text()) if source_config else {}
    config = config or {}
    if not isinstance(config, dict):
        raise ValueError("Prism custom configuration must be a mapping")

    # Custom YAML is applied after CLI parsing by slime, so it has precedence.
    effective = {key: values[-1] for key, values in options.items()} | config
    reward_path = effective.get("custom_rm_path")
    if reward_path in {f"{ARTIFACTS}.reward", "examples.prism.artifacts.reward"}:
        reward_path = config.get("prism_reward_function_path")
    if not isinstance(reward_path, str) or not reward_path:
        raise ValueError("Prism artifact recording requires an explicit custom reward function")
    hooks = config.get("rollout_sample_hook_path", options.get("rollout_sample_hook_path", []))
    if not isinstance(hooks, list):
        raise ValueError("rollout_sample_hook_path must be a list")
    config.update(
        prism_artifact_dir=str(root),
        prism_reward_function_path=reward_path,
        custom_rm_path=reward_path,
        rollout_sample_hook_path=[
            p
            for p in hooks
            if p
            not in {
                f"{ARTIFACTS}.prepare_sample",
                "examples.prism.artifacts.prepare_sample",
            }
        ],
    )
    observers = config.get("rollout_sample_observer_path", options.get("rollout_sample_observer_path", []))
    if not isinstance(observers, list):
        raise ValueError("rollout_sample_observer_path must be a list")
    config["rollout_sample_observer_path"] = list(dict.fromkeys([f"{ARTIFACTS}.observe", *observers]))
    config.setdefault("omni_audio_path_function_path", f"{ARTIFACTS}.audio_path")
    for kind in ("rollout", "eval_rollout"):
        key = f"custom_{kind}_log_function_path"
        original = effective.get(key)
        if original == f"{ARTIFACTS}.log_{kind}_data":
            original = config.get(f"prism_{kind}_log_function_path")
        config[f"prism_{kind}_log_function_path"] = original
        config[key] = f"{ARTIFACTS}.log_{kind}_data"

    runtime_config = root / "config" / "experiment.yaml"
    resolved_command = [*command, "--custom-config-path", str(runtime_config)]
    if dry_run:
        return resolved_command

    if effective.get("use_tensorboard") and importlib.util.find_spec("tensorboard") is None:
        raise ImportError("Install tensorboard in TRAIN_PYTHON, or set USE_TENSORBOARD=0")

    # Reserve an attempt exclusively before writing anything into its layout.
    root.mkdir(parents=True, exist_ok=True)
    (root / "config").mkdir(exist_ok=True)
    launch_path = root / "config" / "launch.json"
    repo = Path(__file__).resolve().parents[2]
    source = {"directory": str(repo)}
    for key, git_args in (("commit", ["rev-parse", "HEAD"]), ("status", ["status", "--porcelain"])):
        result = subprocess.run(["git", "-C", str(repo), *git_args], capture_output=True, text=True)
        source[key] = result.stdout.strip() if result.returncode == 0 else None
    snapshot = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "run_dir": str(root),
        "source": source,
        "command": redact_command(resolved_command),
        "options": redact(options),
        "configuration": redact(config),
        "resume_from": effective.get("load"),
        "pretrained_from": effective.get("pretrained_checkpoint"),
        "checkpoint_dir": effective.get("save"),
        "assets": {},
    }
    snapshots = {}
    for key in (
        "custom_config_path",
        "omni_config",
        "omni_reward_config",
        "omni_teacher_config",
        "eval_config",
        "sglang_config",
    ):
        value = effective.get(key)
        if isinstance(value, str) and not value.lstrip().startswith(("{", "[")) and Path(value).is_file():
            content = Path(value).read_bytes()
            snapshot["assets"][key] = {
                "path": str(Path(value).resolve()),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            snapshots[key] = redact(yaml.safe_load(content))
    for key in ("prompt_data", "hf_checkpoint", "pretrained_checkpoint", "load"):
        value = effective.get(key)
        if isinstance(value, str) and Path(value).exists():
            item = Path(value).resolve()
            snapshot["assets"][key] = {
                "path": str(item),
                "size": item.stat().st_size,
                "mtime_ns": item.stat().st_mtime_ns,
            }
    try:
        with launch_path.open("x") as stream:
            json.dump(snapshot, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"Attempt already exists: {root}. Choose a new JOB_TAG or RUNNING_ROUND.") from error
    for name in ("logs", "tensorboard", "wandb", "records", "debug"):
        (root / name).mkdir(exist_ok=True)
    for name, value in snapshots.items():
        (root / "config" / f"{name}.yaml").write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False))
    # This is the actual training input, including any user-provided credentials.
    # Keep it private; the reviewable launch/config snapshots above are redacted.
    fd = os.open(runtime_config, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    return resolved_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--job-tag", required=True)
    parser.add_argument("--running-round", required=True, type=int)
    parser.add_argument("--command-file")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.running_round < 0:
        parser.error("a training command and a nonnegative running round are required")
    command = prepare_run(
        args.run_dir,
        command,
        identity={"run_name": args.run_name, "job_tag": args.job_tag, "running_round": args.running_round},
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(shlex.join(redact_command(command)))
    else:
        if not args.command_file:
            parser.error("--command-file is required outside dry-run")
        Path(args.command_file).write_bytes(b"\0".join(item.encode() for item in command) + b"\0")


if __name__ == "__main__":
    main()
