#!/opt/venvs/core/bin/python
"""Pinned experiment adapter: replace only the router entry, not learner/engines."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path


def main():
    python = "/opt/venvs/core/bin/python"
    arguments = sys.argv[1:]
    run = Path(os.environ["PRISM_SCALE_RUN"])
    manifest = json.loads((run.parent.parent / "manifest.json").read_text())
    if arguments and Path(arguments[0]).name == "omni_router.py":
        case = json.loads((run / "case.json").read_text())
        sources = {Path(manifest["source"]), Path(case.get("runtime_source", manifest["source"]))}
        expected = {str(source / "slime/backends/sglang_utils/omni_router.py") for source in sources}
        if arguments[0] not in expected:
            raise RuntimeError("Unrecognized router source; refusing silent single-process fallback")
        configuration, disabled = json.loads(arguments[1])
        configuration.update(
            router_processes=case["router_processes"],
            policy=case["router_policy"],
            router_state_dir=str(run / "router-state"),
        )
        entry = Path(__file__).resolve().parents[4] / "slime/backends/sglang_utils/omni_router.py"
        receipt = {
            "time": time.time(),
            "pid": os.getpid(),
            "incoming_entry": arguments[0],
            "entry": str(entry),
            "entry_sha256": hashlib.sha256(entry.read_bytes()).hexdigest(),
            "router_processes": configuration["router_processes"],
            "policy": configuration["policy"],
            "state_dir": configuration["router_state_dir"],
            "scope": "Actual exec arguments, not proof that child processes became healthy",
        }
        temporary = run / f".router-launch-{os.getpid()}.json"
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(run / "router-launch.json")
        arguments = [str(entry), json.dumps([configuration, disabled])]
    os.execv(python, [python, *arguments])


if __name__ == "__main__":
    main()
