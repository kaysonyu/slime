"""Replace the two exact recorded CPU monitors, preserving the existing thread."""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import psutil

from campaign import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.commit):
        raise ValueError("Expected a full source commit")
    path = args.root / "platform/controller-pids.json"
    old = json.loads(path.read_text())
    assert set(old["processes"]) == {"watch", "resume"}
    processes, commands = {}, {}
    for name, record in old["processes"].items():
        process = psutil.Process(record["pid"])
        if process.cmdline() != record["command"] or process.children(recursive=True):
            # A watch CLI query can briefly be a child: retry at a safe idle poll.
            raise RuntimeError(f"{name} identity changed or query in progress; no signals sent")
        processes[name] = process
        command = list(record["command"])
        script = "watch.py" if name == "watch" else "resume_watch.py"
        command[2] = str(Path(__file__).resolve().with_name(script))
        if not Path(command[2]).is_file() or command[3] != str(args.root):
            raise RuntimeError("Unexpected monitor command")
        commands[name] = command
    atomic_json(args.root / f"platform/controllers-before-{time.time_ns()}.json", old)
    for process in processes.values():
        process.terminate()
    for process in processes.values():
        process.wait(timeout=15)
    updated = {}
    for name, record in old["processes"].items():
        command = commands[name]
        with (args.root / "platform" / f"{name}.log").open("a") as log:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        updated[name] = {
            "pid": process.pid, "command": command, "source_commit": args.commit,
            "replaced_pid": record["pid"],
        }
    result = dict(time=time.time(), thread_id=old["thread_id"], processes=updated)
    atomic_json(path, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
