"""Run bounded GPU validation and restore the user's existing gpu-occupy process.

Run this inside the selected train instance. Only the exact named occupancy
program is stopped. Its argv/cwd/environment are retained in memory and restored
in finally; no credential-bearing environment is written to a report.
"""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

OCCUPY = "/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/bin/gpu-occupy"


def running(pid):
    try:
        return bool(Path(f"/proc/{pid}/cmdline").read_bytes())
    except OSError:
        return False


def processes():
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = entry.joinpath("cmdline").read_bytes().split(b"\0")
            argv = [os.fsdecode(x) for x in argv if x]
            if OCCUPY not in argv:
                continue
            environment = dict(
                part.split("=", 1)
                for part in os.fsdecode(entry.joinpath("environ").read_bytes()).split("\0")
                if "=" in part
            )
            result.append((int(entry.name), argv, os.readlink(entry / "cwd"), environment))
        except (OSError, ProcessLookupError):
            continue
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Provide the validation command after --")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    occupied = processes()
    if len(occupied) != 1:
        raise RuntimeError(f"Expected exactly one existing gpu-occupy supervisor; found {len(occupied)}")
    pid, argv, cwd, environment = occupied[0]
    record = {"occupy_pid_before": pid, "occupy_argv": argv, "command": command, "started": time.time()}
    child = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 90
        while running(pid):
            if time.monotonic() >= deadline:
                raise RuntimeError("gpu-occupy did not stop; refusing to start GPU validation")
            time.sleep(1)
        print("gpu-occupy stopped", flush=True)
        with (output / "validation.log").open("w") as log:
            child = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={**os.environ, "SLIME_VALIDATION_RUN": str(output.resolve())},
            )
            try:
                record["exit_code"] = child.wait(timeout=args.timeout)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
        if record["exit_code"]:
            raise RuntimeError(f"GPU validation exited {record['exit_code']}; see {output / 'validation.log'}")
    finally:
        if child is not None:
            # Stop server descendants left in the validation process group.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            # Ray daemons may start a new session. Match only descendants that
            # inherited this validation's unique environment tag.
            marker = b"SLIME_VALIDATION_RUN=" + os.fsencode(str(output.resolve()))
            leftovers = []
            for entry in Path("/proc").iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    if marker in (entry / "environ").read_bytes().split(b"\0"):
                        leftovers.append(int(entry.name))
                        os.kill(int(entry.name), signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    pass
            deadline = time.monotonic() + 30
            while leftovers and time.monotonic() < deadline:
                leftovers = [pid for pid in leftovers if running(pid)]
                if leftovers:
                    time.sleep(1)
            for leftover in leftovers:
                try:
                    os.kill(leftover, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if not processes():
            with (output / "gpu-occupy-restored.log").open("ab") as log:
                restored = subprocess.Popen(
                    argv,
                    cwd=cwd,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            record["occupy_pid_after"] = restored.pid
            print(f"gpu-occupy restored: {restored.pid}", flush=True)
        else:
            record["occupy_pid_after"] = processes()[0][0]
        record["finished"] = time.time()
        (output / "guard.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
