"""Verify small checkpoint state ranges and tensor geometry without loading weights."""

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

    def read_bytes(key):
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
            assert not any("/data/" in name for name in archive.namelist())
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
    return {
        "valid": True,
        "iteration": step,
        "optimizer_steps": steps,
        "model_tensors": len(models),
        "model_elements": elements,
        "optimizer_elements": counts,
        "scheduler_samples": int(common["opt_param_scheduler"]["num_steps"]),
    }


if __name__ == "__main__":
    run = Path(sys.argv[1])
    root = run.parents[1]
    result = verify(run, json.loads((root / "manifest.json").read_text()), json.loads((run / "case.json").read_text()))
    from prepare import write_json

    write_json(run / "checkpoint-verification.json", result)
    print(json.dumps(result))
