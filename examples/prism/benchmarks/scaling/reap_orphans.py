"""Inspect or reap orphaned spawn workers belonging to one completed case only."""

import argparse
import fcntl
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil

from campaign import atomic_json


def eligible(process, run):
    """Require exact provenance, orphan status and the known spawn entry point."""
    try:
        return (
            process.ppid() == 1
            and process.status() != psutil.STATUS_ZOMBIE
            and process.environ().get("PRISM_SCALE_RUN") == str(run)
            and "--multiprocessing-fork" in process.cmdline()
            and any("from multiprocessing.spawn import spawn_main;" in arg for arg in process.cmdline())
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def validate_completed(root, name):
    manifest = json.loads((root / "manifest.json").read_text())
    if name not in {case["name"] for case in manifest["cases"]}:
        raise ValueError("Unknown campaign case")
    run = root / "runs" / name
    if not (run / "summary.json").exists():
        raise RuntimeError("Case is not complete")
    for rank in range(manifest["nodes"]):
        if not (run / f"done-{rank}.json").exists():
            raise RuntimeError("All node completion receipts are required")
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("case")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--wait", action="store_true", help="Wait for this exact case to finish, then audit once")
    parser.add_argument("--launch", action="store_true", help="Launch a bounded, CPU-only completion audit")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if args.case not in {case["name"] for case in manifest["cases"]}:
        raise ValueError("Unknown campaign case")
    if args.launch:
        if not args.wait:
            parser.error("--launch requires --wait")
        directory = root / "platform/orphan-cleanup" / args.case
        directory.mkdir(parents=True, exist_ok=True)
        controller = directory / f"{socket.gethostname()}-controller.json"
        if controller.exists():
            raise FileExistsError("Completion auditor already launched; inspect it before replacement")
        command = [sys.executable, str(Path(__file__).resolve()), str(root), args.case, "--wait"]
        if args.apply:
            command.append("--apply")
        with (directory / f"{socket.gethostname()}.log").open("a") as log:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True
            )
        atomic_json(controller, {"pid": process.pid, "command": command, "time": time.time()})
        print(json.dumps({"pid": process.pid, "controller": str(controller)}))
        return
    if args.wait:
        directory = root / "platform/orphan-cleanup" / args.case
        directory.mkdir(parents=True, exist_ok=True)
        lock = (directory / f"{socket.gethostname()}.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + 3 * 3600
        while True:
            try:
                validate_completed(root, args.case)
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Completion audit timed out without signaling any process")
                time.sleep(5)
    run = validate_completed(root, args.case)
    candidates = [p for p in psutil.process_iter() if eligible(p, run)]
    receipt = {
        "time": time.time(),
        "hostname": socket.gethostname(),
        "case": args.case,
        "apply": args.apply,
        "candidates": [
            {"pid": p.pid, "ppid": p.ppid(), "created": p.create_time(), "cpu_seconds": sum(p.cpu_times()[:2])}
            for p in candidates
        ],
    }
    destination = root / "platform/orphan-cleanup" / f"{socket.gethostname()}-{time.time_ns()}-{os.getpid()}.json"
    atomic_json(destination, receipt)  # Preserve provenance before any signal.
    if args.apply:
        signaled = []
        for process in candidates:
            if eligible(process, run):  # Revalidate immediately before sending a signal.
                process.terminate()
                signaled.append(process)
        _, alive = psutil.wait_procs(signaled, timeout=5)
        killed = []
        for process in alive:
            if eligible(process, run):
                process.kill()
                killed.append(process.pid)
        psutil.wait_procs(alive, timeout=5)
        receipt.update(
            signaled=[p.pid for p in signaled],
            killed=killed,
            remaining=[p.pid for p in psutil.process_iter() if eligible(p, run)],
            finished=time.time(),
        )
    gpu = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    receipt["gpu_processes"] = gpu.stdout.splitlines()
    atomic_json(destination, receipt)
    print(json.dumps({"receipt": str(destination), **receipt}))
    if receipt.get("remaining"):
        raise RuntimeError("Scoped orphan workers remain")


if __name__ == "__main__":
    main()
