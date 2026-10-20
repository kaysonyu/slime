"""Sample a router or rollout manager during the final, unmeasured round."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from observe_processes import role


def select_router(processes, run, source, *, multiprocess=False):
    """Profile a busy data plane, never a supervisor or a previous run's router."""
    expected = "omni_router_data" if multiprocess else "omni_router"
    candidates = []
    for process in processes:
        if role(process.info["cmdline"], source) != expected:
            continue
        try:
            if process.environ().get("PRISM_SCALE_RUN") == str(run):
                usage = process.cpu_times()
                candidates.append((usage.user + usage.system, process))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if not candidates:
        raise RuntimeError(f"No {expected} belongs to this run")
    return max(candidates, key=lambda item: item[0])[1], [p.pid for _, p in candidates]


def select_manager(processes, run, source):
    """Require exactly one run-scoped manager, never another campaign's actor."""
    candidates = []
    for process in processes:
        if role(process.info["cmdline"], source) != "RolloutManager":
            continue
        try:
            if process.environ().get("PRISM_SCALE_RUN") == str(run):
                candidates.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one run-scoped RolloutManager, found {len(candidates)}")
    return candidates[0], [candidates[0].pid]


def wait_case_ready(run, timeout=7200):
    """Queue a bounded CPU sampler before the planned GPU case is prepared."""
    deadline = time.monotonic() + timeout
    while not (run / "case.json").exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Planned profiling case was never prepared")
        time.sleep(3)
    return json.loads((run / "case.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("case")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--duration", type=int, default=15)
    parser.add_argument("--component", choices=("router", "manager"), default="router")
    args = parser.parse_args()
    if not 1 <= args.duration <= 30:
        parser.error("--duration must be between 1 and 30 seconds")
    if os.environ.get("PET_NODE_RANK") != "0":
        raise RuntimeError("Profiler belongs in the campaign head pod only")
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if args.case not in [case["name"] for case in manifest["cases"]]:
        raise ValueError("Unknown campaign case")
    run = root / "runs" / args.case
    output = run / f"{args.component}-profile"
    output.mkdir(parents=True, exist_ok=True)
    if args.launch:
        if (output / "controller.json").exists():
            raise FileExistsError("Profiler already launched; inspect it before replacement")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            str(root),
            args.case,
            "--duration",
            str(args.duration),
            "--component",
            args.component,
        ]
        with (output / "controller.log").open("a") as log:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        (output / "controller.json").write_text(json.dumps({"pid": process.pid, "command": command}) + "\n")
        print(json.dumps({"pid": process.pid, "case": args.case, "output": str(output)}))
        return
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "result.json").exists():
            raise FileExistsError("Profile already attempted")
        case = wait_case_ready(run)
        marker = f"Published policy version={case['steps']} to {case['gpus']} rollout replicas"
        deadline = time.monotonic() + 7200
        while marker not in (run / "node-0/launch.log").read_text(errors="replace"):
            if time.monotonic() >= deadline or (run / "summary.json").exists():
                raise TimeoutError("Case finished or stalled before final publication")
            time.sleep(3)
        # Publication precedes the adjacent-generate timestamp ending the last
        # measured window. Leave a margin before attaching the sampler.
        time.sleep(3)
        phases = [json.loads(line) for line in (run / "phases.jsonl").open()]
        if any(row["phase"] == "generate" and row["rollout"] == case["steps"] - 1 for row in phases):
            raise RuntimeError("Final generation already completed; do not profile another phase")
        source = Path(case["router_python"]).parents[4] if "router_python" in case else Path(manifest["source"])
        multiprocess = case.get("router_processes", 1) > 1
        if args.component == "manager":
            process, candidates = select_manager(psutil.process_iter(["cmdline"]), run, source)
        else:
            process, candidates = select_router(
                psutil.process_iter(["cmdline"]), run, source, multiprocess=multiprocess
            )
        command = [
            "/opt/sglang/bin/py-spy",
            "record",
            "--pid",
            str(process.pid),
            "--duration",
            str(args.duration),
            "--rate",
            "99",
            "--gil",
            "--nonblocking",
            "--format",
            "speedscope",
            "--output",
            str(output / f"{args.component}.speedscope.json"),
        ]
        started = time.time()
        with (output / "py-spy.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.duration + 15)
        receipt = {
            "started": started,
            "ended": time.time(),
            "pid": process.pid,
            "role": (
                "RolloutManager"
                if args.component == "manager"
                else ("omni_router_data" if multiprocess else "omni_router")
            ),
            "candidate_pids": candidates,
            "selection": (
                "The unique run-scoped rollout manager"
                if args.component == "manager"
                else "Highest cumulative CPU among run-scoped eligible router processes; one representative, not all data planes"
            ),
            "command": command,
            "returncode": result.returncode,
            "excluded_from_timing": "Final rollout has no adjacent generate start and is never in steady_windows",
            "rollout": case["steps"] - 1,
        }
        (output / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt))


if __name__ == "__main__":
    main()
