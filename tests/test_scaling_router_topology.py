import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/prism/benchmarks/scaling"))
import verify_router_topology as topology


@pytest.mark.parametrize("actual", [1, 8])
def test_live_router_process_count_is_checked(monkeypatch, tmp_path, actual):
    entry = "/pin/slime/backends/sglang_utils/omni_router.py"
    case = dict(
        router_python="/pin/examples/prism/benchmarks/scaling/router_python.py",
        router_entry_sha256="hash", router_processes=8, router_policy="round_robin",
    )
    launch = dict(pid=123, entry=entry, entry_sha256="hash", router_processes=8, policy="round_robin")
    children = [
        SimpleNamespace(
            pid=i, cmdline=lambda kind=kind: ["python", entry, kind],
            environ=lambda: {"PRISM_SCALE_RUN": str(tmp_path)}, status=lambda: "sleeping",
        )
        for i, kind in enumerate(["--control-plane"] + ["--data-plane"] * actual)
    ]
    parent = SimpleNamespace(
        pid=123, cmdline=lambda: ["python", entry, json.dumps([{"router_processes": 8}, False])],
        environ=lambda: {"PRISM_SCALE_RUN": str(tmp_path)}, children=lambda recursive: children,
        create_time=lambda: 456,
    )
    monkeypatch.setattr(topology.psutil, "Process", lambda pid: parent)
    observed = topology.inspect_router(tmp_path, case, launch)
    assert observed["valid"] is (actual == 8)
    launch["router_processes"] = 1
    with pytest.raises(ValueError, match="differs from pinned"):
        topology.inspect_router(tmp_path, case, launch)
