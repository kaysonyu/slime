"""Sample a case's router only during its final, unmeasured generation round."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("case")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if os.environ.get("PET_NODE_RANK") != "0":
        raise RuntimeError("Profiler belongs in the campaign head pod only")
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if args.case not in [case["name"] for case in manifest["cases"]]:
        raise ValueError("Unknown campaign case")
    run = root / "runs" / args.case
    output = run / "router-profile"
    output.mkdir(exist_ok=True)
    if args.launch:
        command = [sys.executable, str(Path(__file__).resolve()), str(root), args.case]
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
        case = json.loads((run / "case.json").read_text())
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
        source = Path(manifest["source"])
        routers = [
            p
            for p in psutil.process_iter(["cmdline", "create_time"])
            if role(p.info["cmdline"], source) == "omni_router"
        ]
        process = max(routers, key=lambda p: p.info["create_time"])
        command = [
            "/opt/sglang/bin/py-spy",
            "record",
            "--pid",
            str(process.pid),
            "--duration",
            "15",
            "--rate",
            "99",
            "--gil",
            "--nonblocking",
            "--format",
            "speedscope",
            "--output",
            str(output / "router.speedscope.json"),
        ]
        started = time.time()
        with (output / "py-spy.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=30)
        receipt = {
            "started": started,
            "ended": time.time(),
            "pid": process.pid,
            "command": command,
            "returncode": result.returncode,
            "excluded_from_timing": "Final rollout has no adjacent generate start and is never in steady_windows",
            "rollout": case["steps"] - 1,
        }
        (output / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt))


if __name__ == "__main__":
    main()
