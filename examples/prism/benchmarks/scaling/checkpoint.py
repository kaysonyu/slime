"""Verify small checkpoint state ranges and tensor geometry without loading weights."""

import argparse
import hashlib
import importlib.util
import io
import json
import math
import pickle
import sys
import types
import zipfile
from pathlib import Path


def checkpoint_artifacts_valid(run, case):
    """Recheck the recorded native boundary; never manufacture a training commit."""
    verification = run / "checkpoint-verification.json"
    if not verification.is_file():
        return False
    result = json.loads(verification.read_text())
    if result.get("valid") is not True:
        return False
    step = case["steps"] - 1
    checkpoint = run / "checkpoint"
    if case.get("checkpoint_contract") != "native_object_store":
        return (checkpoint / "rollout" / f"training_commit_{step}.json").is_file()
    if (
        result.get("contract") != "native_object_store"
        or result.get("iteration") != step
        or not result.get("files")
    ):
        return False
    try:
        if int((checkpoint / "latest_checkpointed_iteration.txt").read_text().strip()) != step:
            return False
        for name, info in result["files"].items():
            path = checkpoint / name
            if path.is_symlink() or not path.resolve().is_relative_to(checkpoint.resolve()):
                return False
            if not path.is_file() or path.stat().st_size != info["size"]:
                return False
            if "sha256" in info and hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
                return False
    except (OSError, ValueError):
        return False
    return True


def indexed_files(checkpoint, directory, metadata):
    """Check every DCP storage range, including shards other than rank zero."""
    files = {}
    for location in metadata.storage_data.values():
        path = directory / location.relative_path
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Checkpoint storage escapes its iteration directory")
        size = path.stat().st_size
        if location.offset < 0 or location.length <= 0 or location.offset + location.length > size:
            raise ValueError("Checkpoint storage range is missing/truncated")
        files[str(path.relative_to(checkpoint))] = {"size": size}
    if not files:
        raise ValueError("Checkpoint has no tensor storage")
    for path in (directory / ".metadata", directory / "metadata.json"):
        files[str(path.relative_to(checkpoint))] = {
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return files


def verify(run, manifest, case):
    # Serialized configuration refers to MCore enums. Avoid loading CUDA extensions
    # in the CPU controller by importing just those real enum modules.
    for name in ("megatron", "megatron.core", "megatron.core.transformer"):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    for name in ("megatron.core.enums", "megatron.core.transformer.enums"):
        path = Path(manifest["megatron"]).joinpath(*name.split(".")).with_suffix(".py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    import torch

    step = case["steps"] - 1
    checkpoint = run / "checkpoint"
    native = case.get("checkpoint_contract") == "native_object_store"
    if native:
        options = json.loads((run / "resolved-args.json").read_text())["options"]
        assert options["rollout_data_transport"] == "object-store"
        assert options["save_debug_rollout_data"] is None
        assert not options["async_save"] and not options["no_save_optim"] and not options["no_save_rng"]
        assert int((checkpoint / "latest_checkpointed_iteration.txt").read_text().strip()) == step
        if case.get("resume"):
            parent = run.parent / case["resume"] / "checkpoint"
            assert Path(options["load"]).resolve() == parent.resolve()
            assert not any(options[key] for key in ("finetune", "no_load_optim", "no_load_rng"))
    else:
        commit = json.loads((checkpoint / "rollout" / f"training_commit_{step}.json").read_text())
        assert commit["rollout_id"] == step
        for name, info in commit["files"].items():
            path = checkpoint / name
            assert path.is_file() and path.stat().st_size == info["size"]
            if "sha256" in info:
                assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]
    directory = checkpoint / f"iter_{step:07d}"
    with (directory / ".metadata").open("rb") as stream:
        metadata = pickle.load(stream)
    files = indexed_files(checkpoint, directory, metadata)

    def read_bytes(key, *, allow_small_tensors=False):
        assert type(metadata.state_dict_metadata[key]).__name__ == "BytesStorageMetadata"
        locations = [location for index, location in metadata.storage_data.items() if index.fqn == key]
        assert len(locations) == 1
        location = locations[0]
        assert 0 < location.length < 1_000_000
        with (directory / location.relative_path).open("rb") as stream:
            stream.seek(location.offset)
            payload = stream.read(location.length)
        assert len(payload) == location.length
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            assert allow_small_tensors or not any("/data/" in name for name in archive.namelist())
        value = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        assert isinstance(value, list) and len(value) == 1
        return value[0]

    common = read_bytes("common_state/shard_0_1")
    optimizer = common["optimizer"].get("optimizer")
    if optimizer is None:
        keys = [
            key for key in metadata.state_dict_metadata if key.startswith("optimizer.") and ".optimizer/shard_" in key
        ]
        assert len(keys) == 1
        optimizer = read_bytes(keys[0])
    models = {
        key: value
        for key, value in metadata.state_dict_metadata.items()
        if hasattr(value, "size") and not key.startswith("optimizer.")
    }
    elements = sum(math.prod(value.size) for value in models.values())
    assert len(models) == 302
    counts = {}
    for kind in ("param", "exp_avg", "exp_avg_sq"):
        entries = [
            value
            for key, value in metadata.state_dict_metadata.items()
            if hasattr(value, "size")
            and (
                key.startswith(f"optimizer.state.{kind}.")
                or (key.startswith("optimizer.distributed.") and key.endswith("." + kind))
            )
        ]
        counts[kind] = sum(math.prod(value.size) for value in entries)
        assert counts[kind] == elements
    steps = [int(group["step"]) for group in optimizer["param_groups"]]
    assert int(common["iteration"]) == step
    assert steps == [case["steps"], case["steps"]]
    assert int(common["opt_param_scheduler"]["num_steps"]) == case["steps"] * case["batch"]
    result = {
        "valid": True,
        "iteration": step,
        "optimizer_steps": steps,
        "model_tensors": len(models),
        "model_elements": elements,
        "optimizer_elements": counts,
        "scheduler_samples": int(common["opt_param_scheduler"]["num_steps"]),
    }
    if native:
        rng_keys = [key for key in metadata.state_dict_metadata if key.startswith("rng_state/")]
        assert rng_keys
        for key in rng_keys:
            states = read_bytes(key, allow_small_tensors=True)
            assert states and all(
                all(name in state for name in ("random_rng_state", "np_rng_state", "torch_rng_state",
                                              "cuda_rng_state", "rng_tracker_states"))
                for state in states
            )
        cursor_path = checkpoint / "rollout" / f"global_dataset_state_dict_{step}.pt"
        cursor = torch.load(cursor_path, map_location="cpu", weights_only=False)
        assert cursor["sample_index"] == case["steps"] * case["batch"]
        assert cursor["sample_group_index"] == case["steps"] * case["batch"] // 8
        assert cursor["sample_offset"] >= 0 and cursor["epoch_id"] >= 0
        for path in (cursor_path, checkpoint / "latest_checkpointed_iteration.txt"):
            files[str(path.relative_to(checkpoint))] = {
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        result.update(
            contract="native_object_store", files=files, rng_entries=len(rng_keys),
            dataset_samples=cursor["sample_index"], dataset_groups=cursor["sample_group_index"],
            scope="Native synchronous model/optimizer/RNG plus dataset snapshot. Not a straw/replay joint commit; "
                  "this integrity check alone does not prove a successful restart.",
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--native-partial-steps", type=int)
    parser.add_argument("--output", type=Path)
    cli = parser.parse_args()
    run = cli.run
    root = run.parents[1]
    case = json.loads((run / "case.json").read_text())
    if cli.native_partial_steps is not None:
        if cli.output is None or cli.output.resolve() == (run / "checkpoint-verification.json").resolve():
            raise ValueError("Partial diagnostics require a separate output, never the acceptance receipt")
        if not 0 < cli.native_partial_steps < case["steps"]:
            raise ValueError("Partial step must precede the required final checkpoint")
        case.update(steps=cli.native_partial_steps, checkpoint_contract="native_object_store")
    result = verify(run, json.loads((root / "manifest.json").read_text()), case)
    if cli.native_partial_steps is not None:
        result.update(partial_diagnostic=True, lifecycle_accepted=False)
    from prepare import write_json

    write_json(cli.output or run / "checkpoint-verification.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}))
