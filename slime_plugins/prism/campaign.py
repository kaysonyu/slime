"""Run pinned cases sequentially inside one fixed GPU allocation."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def publish(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    arguments = parser.parse_args()
    manifest = json.loads(arguments.manifest.read_text())
    root = Path(manifest["output"])
    root.mkdir(parents=True, exist_ok=True)
    rank, nodes = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if nodes != manifest["nodes"]:
        raise ValueError("Campaign allocation differs from the declared node count")
    status = 0
    for number, case in enumerate(manifest["cases"]):
        case_path = Path(case)
        configuration = json.loads(case_path.read_text())
        if configuration["nodes"] != nodes:
            raise ValueError("Every campaign case must use the same allocation")
        if rank == 0:
            publish(
                root / "current.json",
                {
                    "case": str(case_path),
                    "number": number,
                    "services": configuration["services"],
                    "ready_file": configuration["ready_file"],
                },
            )
        print(json.dumps({"rank": rank, "case": case_path.stem, "event": "start"}), flush=True)
        started = time.time()
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("cluster.py")), str(case_path)])
        if rank == 0 and result.returncode == 0 and manifest.get("verification"):
            with (Path(configuration["output"]) / "verification.log").open("w") as log:
                checked = subprocess.run(
                    [sys.executable, manifest["verification"], str(case_path)], stdout=log, stderr=subprocess.STDOUT
                )
            if checked.returncode:
                result.returncode = checked.returncode
        receipt = {
            "rank": rank,
            "case": case_path.stem,
            "returncode": result.returncode,
            "started": started,
            "elapsed_seconds": time.time() - started,
        }
        barrier = root / f"case-{number}"
        publish(barrier / f"rank-{rank}.json", receipt)
        deadline = time.monotonic() + 600
        while not all((barrier / f"rank-{peer}.json").exists() for peer in range(nodes)):
            if time.monotonic() > deadline:
                raise TimeoutError("Case cleanup did not finish on every allocated node")
            time.sleep(2)
        statuses = [json.loads((barrier / f"rank-{peer}.json").read_text()) for peer in range(nodes)]
        if any(row["returncode"] != 0 for row in statuses):
            status = 1
            break
        print(json.dumps({"rank": rank, "case": case_path.stem, "event": "complete"}), flush=True)
    if rank == 0:
        publish(root / "result.json", {"returncode": status})
    raise SystemExit(status)


if __name__ == "__main__":
    main()
