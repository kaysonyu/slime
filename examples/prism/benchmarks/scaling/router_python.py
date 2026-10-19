#!/opt/venvs/core/bin/python
"""Pinned experiment adapter: replace only the router entry, not learner/engines."""

import json
import os
import sys
from pathlib import Path


def main():
    python = "/opt/venvs/core/bin/python"
    arguments = sys.argv[1:]
    run = Path(os.environ["PRISM_SCALE_RUN"])
    manifest = json.loads((run.parent.parent / "manifest.json").read_text())
    original = str(Path(manifest["source"]) / "slime/backends/sglang_utils/omni_router.py")
    if arguments and arguments[0] == original:
        case = json.loads((run / "case.json").read_text())
        configuration, disabled = json.loads(arguments[1])
        configuration.update(
            router_processes=case["router_processes"],
            policy=case["router_policy"],
            router_state_dir=str(run / "router-state"),
        )
        entry = Path(__file__).resolve().parents[4] / "slime/backends/sglang_utils/omni_router.py"
        arguments = [str(entry), json.dumps([configuration, disabled])]
    os.execv(python, [python, *arguments])


if __name__ == "__main__":
    main()
