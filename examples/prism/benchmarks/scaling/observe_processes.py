"""Low-rate read-only CPU/RSS observer inside this campaign's head Job pod."""

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


def role(command, source):
    if not command:
        return None
    if str(source / "slime/backends/sglang_utils/omni_router.py") in command:
        return "omni_router"
    if str(source / "train.py") in command:
        return "training_driver"
    for name in ("GroupWorker", "RolloutManager", "ServingCluster", "OmniEngine", "MegatronTrainRayActor"):
        if command[0].startswith(f"ray::{name}"):
            return name
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    if args.interval < 2:
        raise ValueError("Keep the observer low-rate")
    if os.environ.get("PET_NODE_RANK") != "0":
        raise RuntimeError("Run only in the allocated campaign's PET rank 0 pod")
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if Path(manifest["root"]) != root:
        raise ValueError("Campaign root mismatch")
    source = Path(manifest["source"])
    if args.launch:
        command = [sys.executable, str(Path(__file__).resolve()), str(root), "--interval", str(args.interval)]
        with (root / "platform/head-process-observer.log").open("a") as log:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        receipt = {"pid": process.pid, "command": command, "host": socket.gethostname(), "started": time.time()}
        (root / "platform/head-process-observer.json").write_text(json.dumps(receipt) + "\n")
        print(json.dumps(receipt))
        return
    with (root / "platform/head-process-observer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = {}
        deadline = time.monotonic() + 10 * 3600
        with (root / "platform/head-processes.jsonl").open("a", buffering=1) as output:
            while time.monotonic() < deadline and not (root / "campaign-finished.json").exists():
                if (root / "platform/monitor-finished.json").exists():
                    break
                instant, monotonic = time.time(), time.monotonic()
                samples = []
                current = {}
                for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
                    kind = role(process.info["cmdline"], source)
                    if kind is None:
                        continue
                    try:
                        times = process.cpu_times()
                        key = (process.pid, process.info["create_time"])
                        cpu = times.user + times.system
                        before = previous.get(key)
                        sample = {
                            "pid": process.pid,
                            "role": kind,
                            "cpu_seconds": cpu,
                            "rss_bytes": process.memory_info().rss,
                            "threads": process.num_threads(),
                        }
                        if before is not None:
                            sample["cpu_cores_used"] = (cpu - before[0]) / (monotonic - before[1])
                        current[key] = (cpu, monotonic)
                        samples.append(sample)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                previous = current
                request = json.loads((root / "services/request.json").read_text())
                output.write(json.dumps({"time": instant, "case": request["case"], "processes": samples}) + "\n")
                time.sleep(args.interval)


if __name__ == "__main__":
    main()
