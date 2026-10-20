"""Verify actual run-scoped router children, not just planned configuration."""

import argparse
import json
import time
from pathlib import Path

import psutil

from campaign import atomic_json


def inspect_router(run, case, launch):
    expected_entry = str(Path(case["router_python"]).parents[4] / "slime/backends/sglang_utils/omni_router.py")
    if (
        launch["entry"] != expected_entry
        or launch["entry_sha256"] != case["router_entry_sha256"]
        or launch["router_processes"] != case["router_processes"]
        or launch["policy"] != case["router_policy"]
    ):
        raise ValueError("Actual router launch differs from pinned case")
    parent = psutil.Process(launch["pid"])
    command = parent.cmdline()
    if expected_entry not in command or parent.environ().get("PRISM_SCALE_RUN") != str(run):
        raise ValueError("Router PID provenance mismatch")
    config = json.loads(command[command.index(expected_entry) + 1])[0]
    if config.get("router_processes", 1) != case["router_processes"]:
        raise ValueError("Running router topology differs from case")
    children = parent.children(recursive=True)
    data, control = [], []
    for child in children:
        command = child.cmdline()
        if expected_entry not in command or child.environ().get("PRISM_SCALE_RUN") != str(run):
            continue
        if child.status() == psutil.STATUS_ZOMBIE:
            continue
        if "--data-plane" in command:
            data.append(child.pid)
        elif "--control-plane" in command:
            control.append(child.pid)
    count = case["router_processes"]
    valid = len(data) == count and len(control) == 1 if count > 1 else not data and not control
    return {
        "valid": valid, "time": time.time(), "pid": parent.pid, "created": parent.create_time(),
        "data_plane_pids": data, "control_plane_pids": control,
        "expected_data_planes": count if count > 1 else 0, "entry": expected_entry,
        "scope": "Live process topology and run provenance; not an end-to-end health or stability assertion",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    case = json.loads((run / "case.json").read_text())
    deadline = time.monotonic() + 900
    previous = None
    while time.monotonic() < deadline:
        launch = run / "router-launch.json"
        if launch.exists():
            try:
                observed = inspect_router(run, case, json.loads(launch.read_text()))
                signature = (observed["pid"], sorted(observed["data_plane_pids"]), observed["control_plane_pids"])
                if observed["valid"] and signature == previous:
                    atomic_json(run / "router-topology.json", observed)
                    return
                previous = signature if observed["valid"] else None
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                previous = None
        time.sleep(2)
    raise TimeoutError("No stable, correctly sized run-scoped router topology appeared")


if __name__ == "__main__":
    main()
