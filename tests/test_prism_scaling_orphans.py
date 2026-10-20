import json
import sys
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/prism/benchmarks/scaling"))
from reap_orphans import eligible, validate_completed


def test_orphan_selection_requires_all_scope_checks(tmp_path):
    run = tmp_path / "runs/finished"
    process = SimpleNamespace(
        ppid=lambda: 1,
        status=lambda: psutil.STATUS_SLEEPING,
        environ=lambda: {"PRISM_SCALE_RUN": str(run)},
        cmdline=lambda: [
            "python",
            "-c",
            "from multiprocessing.spawn import spawn_main; spawn_main()",
            "--multiprocessing-fork",
        ],
    )
    assert eligible(process, run)
    assert not eligible(process, tmp_path / "runs/current")
    process.ppid = lambda: 42
    assert not eligible(process, run)
    process.ppid = lambda: 1
    process.cmdline = lambda: ["python", "campaign.py"]
    assert not eligible(process, run)


def test_orphan_reaping_rejects_incomplete_and_unknown_cases(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"nodes": 16, "cases": [{"name": "done"}]}))
    run = tmp_path / "runs/done"
    run.mkdir(parents=True)
    with pytest.raises(ValueError, match="Unknown"):
        validate_completed(tmp_path, "../current")
    with pytest.raises(RuntimeError, match="not complete"):
        validate_completed(tmp_path, "done")
    (run / "summary.json").write_text("{}")
    for rank in range(15):
        (run / f"done-{rank}.json").write_text("{}")
    with pytest.raises(RuntimeError, match="All node"):
        validate_completed(tmp_path, "done")
    (run / "done-15.json").write_text("{}")
    assert validate_completed(tmp_path, "done") == run
