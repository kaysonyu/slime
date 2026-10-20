"""Preserve one completed case's local Ray logs before its Job pod is released."""

import argparse
import hashlib
import json
import os
import tarfile
import time
from pathlib import Path

from campaign import atomic_json


def capture_local_failure(run, rank, started):
    """Archive only Ray sessions created during this case, before pod release."""
    output = run / "failure-evidence" / f"node-{rank}"
    output.mkdir(parents=True, exist_ok=True)
    sessions = [
        path
        for path in Path("/tmp/ray").glob("session_*")
        if not path.is_symlink() and path.is_dir() and path.stat().st_mtime >= started - 1
    ]
    files = []
    with tarfile.open(output / "ray-logs.tar.gz", "w:gz") as bundle:
        for session in sessions:
            for path in sorted((session / "logs").glob("*")):
                if path.is_file() and not path.is_symlink() and path.suffix in {".out", ".err", ".log"}:
                    bundle.add(path, arcname=f"{session.name}/{path.name}", recursive=False)
                    files.append(str(path))
    atomic_json(
        output / "receipt.json",
        {
            "captured": time.time(),
            "case_started": started,
            "rank": rank,
            "files": files,
            "cgroup": {
                name: (Path("/sys/fs/cgroup") / name).read_text()
                for name in ("memory.events", "memory.peak", "cpu.stat", "cpu.max")
                if (Path("/sys/fs/cgroup") / name).exists()
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("case")
    parser.add_argument("session", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text())
    if args.case not in {case["name"] for case in manifest["cases"]}:
        raise ValueError("Unknown case")
    run = args.root / "runs" / args.case
    if not (run / "summary.json").exists():
        raise RuntimeError("Capture only a completed case")
    if args.session.is_symlink() or args.session.resolve().parent != Path("/tmp/ray"):
        raise ValueError("Use an explicit local Ray session, never session_latest")
    if not args.session.name.startswith("session_") or args.session.name == "session_latest":
        raise ValueError("Invalid session")
    rank = int(os.environ["PET_NODE_RANK"])
    if int(os.environ.get("PET_NNODES", "0")) != manifest["nodes"] or rank not in range(manifest["nodes"]):
        raise RuntimeError("Use the original campaign allocation")
    node = json.loads((args.root / "platform" / f"node-{rank}.json").read_text())
    output = run / "failure-evidence" / f"node-{rank}"
    output.mkdir(parents=True, exist_ok=False)
    logs = args.session / "logs"
    files = sorted({*logs.glob("worker-*.err"), *logs.glob("worker-*.out"), logs / "raylet.out"})
    archive = output / "ray-logs.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in files:
            bundle.add(path, arcname=path.name, recursive=False)
    memory = Path("/sys/fs/cgroup/memory.events")
    receipt = {
        "time": time.time(),
        "case": args.case,
        "node": node,
        "session": str(args.session),
        "file_count": len(files),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "memory_events_at_capture": memory.read_text() if memory.exists() else None,
        "caveat": "Post-failure cgroup counters; logs belong to the explicit completed Ray session. No runtime or source mutation.",
    }
    atomic_json(output / "receipt.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
