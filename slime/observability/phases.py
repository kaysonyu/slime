"""Driver phase timing at explicit training boundaries."""

import json
import time
from contextlib import contextmanager
from pathlib import Path


class PhaseRecorder:
    def __init__(self, path: str | None):
        self.path = Path(path) if path else None
        self.stream = None

    def __enter__(self):
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = self.path.open("a", buffering=1)
        return self

    def __exit__(self, exception_type, exception, traceback):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    @contextmanager
    def measure(self, phase: str, rollout_id: int):
        if self.stream is None:
            yield
            return
        started, clock = time.time(), time.perf_counter_ns()
        error_name = None
        try:
            yield
        except BaseException as error:
            error_name = type(error).__name__
            raise
        finally:
            self.stream.write(
                json.dumps(
                    {
                        "phase": phase,
                        "rollout": rollout_id,
                        "started": started,
                        "start_monotonic_ns": clock,
                        "seconds": (time.perf_counter_ns() - clock) / 1e9,
                        "error": error_name,
                    }
                )
                + "\n"
            )
