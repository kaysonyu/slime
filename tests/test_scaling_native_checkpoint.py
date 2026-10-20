"""Native checkpoint acceptance must check real artifacts, not obsolete markers."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/prism/benchmarks/scaling"))
from checkpoint import checkpoint_artifacts_valid, indexed_files
from prepare import write_json

NUM_GPUS = 0
pytestmark = pytest.mark.unit


def native_fixture(run):
    root = run / "checkpoint"
    directory = root / "iter_0000099"
    directory.mkdir(parents=True)
    (directory / "shard.distcp").write_bytes(b"checkpoint")
    (directory / ".metadata").write_bytes(b"index")
    (directory / "metadata.json").write_text("{}")
    (root / "latest_checkpointed_iteration.txt").write_text("99")
    metadata = SimpleNamespace(storage_data={
        "tensor": SimpleNamespace(relative_path="shard.distcp", offset=0, length=10),
    })
    files = indexed_files(root, directory, metadata)
    write_json(run / "checkpoint-verification.json", {
        "valid": True, "iteration": 99, "contract": "native_object_store", "files": files,
    })
    return root, directory, metadata


@pytest.mark.parametrize("damage", [None, "truncate", "metadata", "tracker", "iteration", "contract", "empty"])
def test_native_acceptance_rechecks_every_recorded_boundary(tmp_path, damage):
    root, directory, _ = native_fixture(tmp_path)
    result = json.loads((tmp_path / "checkpoint-verification.json").read_text())
    if damage == "truncate":
        (directory / "shard.distcp").write_bytes(b"x")
    elif damage == "metadata":
        (directory / ".metadata").write_bytes(b"INDEX")  # Same size, different hash.
    elif damage == "tracker":
        (root / "latest_checkpointed_iteration.txt").write_text("98")
    elif damage == "iteration":
        result["iteration"] = 98
    elif damage == "contract":
        result["contract"] = "training_commit"
    elif damage == "empty":
        result["files"] = {}
    write_json(tmp_path / "checkpoint-verification.json", result)
    assert checkpoint_artifacts_valid(tmp_path, {
        "steps": 100, "checkpoint_contract": "native_object_store",
    }) == (damage is None)
    # Native evidence does not silently weaken another checkpoint contract.
    assert not checkpoint_artifacts_valid(tmp_path, {"steps": 100})


@pytest.mark.parametrize("offset,length", [(-1, 5), (0, 0), (9, 2)])
def test_index_rejects_invalid_storage_ranges(tmp_path, offset, length):
    root, directory, metadata = native_fixture(tmp_path)
    metadata.storage_data["tensor"].offset = offset
    metadata.storage_data["tensor"].length = length
    with pytest.raises(ValueError, match="range"):
        indexed_files(root, directory, metadata)


def test_index_rejects_escaping_or_symlink_shards(tmp_path):
    root, directory, metadata = native_fixture(tmp_path)
    (directory / "link").symlink_to(directory / "shard.distcp")
    for relative in ("link", "../latest_checkpointed_iteration.txt"):
        metadata.storage_data["tensor"].relative_path = relative
        with pytest.raises(ValueError, match="escapes"):
            indexed_files(root, directory, metadata)
