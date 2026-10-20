"""Sample allocated GPU and container CPU/memory counters into JSONL."""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path


def read_cgroup(name):
    path = Path("/sys/fs/cgroup") / name
    if not path.exists():
        return None
    return path.read_text().strip()


def main(args):
    deadline = time.monotonic() + args.seconds
    previous = None
    with Path(args.output).open("w") as stream:
        while time.monotonic() < deadline:
            now = time.time()
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            row = {
                "timestamp": now,
                "gpu": list(csv.reader(proc.stdout.splitlines())),
                "memory_bytes": read_cgroup("memory.current"),
            }
            raw = read_cgroup("cpu.stat")
            if raw:
                counters = dict(line.split() for line in raw.splitlines())
                used = int(counters["usage_usec"])
                if previous:
                    row["cpu_cores"] = (used - previous[1]) / 1e6 / (now - previous[0])
                previous = (now, used)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--interval", type=float, default=1)
    main(parser.parse_args())
