"""Exercise occupancy restoration with disposable CPU processes, never GPUs."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

NUM_GPUS = 0


def test_failed_command_restores_the_original_supervisor(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "tts_gpu_guard", Path(__file__).parents[1] / "tools/tts_gpu_guard.py"
    )
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    program = tmp_path / "occupy_cpu.py"
    program.write_text("import time\nwhile True: time.sleep(1)\n")
    monkeypatch.setattr(guard, "OCCUPY", str(program))
    original = subprocess.Popen([sys.executable, str(program)])
    output = tmp_path / "result"
    monkeypatch.setattr(
        sys,
        "argv",
        ["guard", "--output-dir", str(output), "--timeout", "10", "--", sys.executable, "-c", "raise SystemExit(7)"],
    )
    old_term, old_hup = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGHUP)
    try:
        with pytest.raises(RuntimeError, match="exited 7"):
            guard.main()
        original.wait(timeout=5)
        record = json.loads((output / "guard.json").read_text())
        assert record["exit_code"] == 7
        assert record["occupy_pid_after"] != original.pid
        (restored,) = guard.processes()
        assert restored[0] == record["occupy_pid_after"]
        assert restored[1] == [sys.executable, str(program)]
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGHUP, old_hup)
        for pid, *_ in guard.processes():
            os.kill(pid, signal.SIGTERM)
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        if original.poll() is None:
            original.terminate()
            original.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
