"""Commit ordinary model and dataset checkpoints at one durable boundary."""

import hashlib
import json
import os
import re
import uuid
from pathlib import Path


def _file_digest(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def write_checkpoint_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def begin_checkpoint_boundary(args, rollout_id: int) -> None:
    root = Path(args.save)
    if (root / "rollout" / f"training_commit_{rollout_id}.json").exists():
        raise FileExistsError(f"Training checkpoint {rollout_id} is already committed")
    write_checkpoint_json(root / "rollout" / "joint_checkpoint.json", {"version": 1})


def commit_checkpoint_boundary(args, rollout_id: int, model_args: list, *, weight_version=None) -> None:
    root = Path(args.save).resolve()
    source = root / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt"
    if not source.is_file():
        raise FileNotFoundError(f"Dataset checkpoint is missing: {source}")
    payloads = [source]
    for model in model_args:
        directory = Path(model.save).resolve() / f"iter_{rollout_id:07d}"
        if not directory.is_relative_to(root):
            raise ValueError("Model checkpoints must be inside the joint checkpoint root")
        files = [path for path in directory.rglob("*") if path.is_file()]
        if not files:
            raise FileNotFoundError(f"Model checkpoint is missing: {directory}")
        if getattr(model, "no_save_rng", False) or (
            getattr(model, "no_save_optim", False) and not getattr(model, "use_stateless_adam", False)
        ):
            raise ValueError("Joint checkpoints require optimizer, scheduler, and RNG state")
        payloads.extend(files)
    manifest = {}
    for path in payloads:
        if path.is_symlink():
            raise ValueError(f"Checkpoint payload must not be a symlink: {path}")
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
            entry = {"size": os.fstat(stream.fileno()).st_size}
            if path == source or path.suffix == ".json" or path.name == ".metadata":
                entry["sha256"] = _file_digest(stream)
        manifest[str(path.relative_to(root))] = entry
    directories = {root}
    for path in payloads:
        directories.update(parent for parent in path.parents if parent.is_relative_to(root))
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    write_checkpoint_json(
        root / "rollout" / f"training_commit_{rollout_id}.json",
        {"version": 1, "rollout_id": rollout_id, "weight_version": weight_version, "files": manifest},
    )


def resolve_checkpoint_boundary(args) -> None:
    if not getattr(args, "load", None) or getattr(args, "finetune", False):
        return
    root = Path(args.load).resolve()
    directory_step = re.fullmatch(r"iter_(\d{7,})", root.name)
    if directory_step and (root.parent / "rollout" / "joint_checkpoint.json").exists():
        step = int(directory_step[1])
        if getattr(args, "ckpt_step", None) not in (None, step):
            raise ValueError("--ckpt-step differs from the selected checkpoint directory")
        args.ckpt_step, args.load = step, str(root.parent)
        root = root.parent
    if not (root / "rollout" / "joint_checkpoint.json").exists():
        return
    commits = {
        int(path.stem.removeprefix("training_commit_")): path
        for path in (root / "rollout").glob("training_commit_*.json")
    }
    if not commits:
        raise ValueError("No complete model/dataset checkpoint was committed")
    step = getattr(args, "ckpt_step", None)
    step = max(commits) if step is None else step
    if step not in commits:
        raise ValueError(f"Checkpoint {step} has no complete model/dataset commit")
    value = json.loads(commits[step].read_text())
    if value.get("version") != 1 or value.get("rollout_id") != step:
        raise ValueError("Invalid training checkpoint commit")
    files = value.get("files")
    if not isinstance(files, dict) or f"rollout/global_dataset_state_dict_{step}.pt" not in files:
        raise ValueError("Training checkpoint commit has no dataset state")
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != expected["size"]:
            raise ValueError(f"Checkpoint payload missing or changed: {name}")
        if "sha256" in expected:
            with path.open("rb") as stream:
                actual = _file_digest(stream)
            if actual != expected["sha256"]:
                raise ValueError(f"Checkpoint payload changed: {name}")
    start = getattr(args, "start_rollout_id", None)
    if start is not None and start != step + 1:
        raise ValueError("--start-rollout-id differs from the committed checkpoint")
    args.ckpt_step, args.start_rollout_id = step, step + 1
    version = value.get("weight_version")
    if version is not None:
        args.update_weight_start_version = int(version)
