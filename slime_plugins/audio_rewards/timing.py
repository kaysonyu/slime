"""Task-local reward timing without training, logging, or serving dependencies.

Durations use a monotonic clock. Wall timestamps allow completed samples to be
ordered across rollout workers (whose host clocks must be synchronized). Stage
durations are summed work: parallel audio/reference/text requests can overlap,
so their sum is not a decomposition of component wall time.
"""

import asyncio
import time
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar

_current = ContextVar("audio_reward_timing", default=None)
STAGES = (
    "client_queue_s",
    "http_s",
    "prepare_s",
    "identity_wait_s",
    "identity_http_s",
    "cache_wait_s",
    "cache_lookup_s",
)


@contextmanager
def capture():
    """Collect one component, including its child tasks and failure duration."""
    data = {name: 0.0 for name in STAGES}
    data.update(started_at=time.time(), http_requests=0, status="success")
    state = {"data": data, "closed": False}
    token = _current.set(state)
    started = time.perf_counter()
    try:
        yield data
    except BaseException as error:
        data["status"] = "cancelled" if isinstance(error, asyncio.CancelledError) else "error"
        raise
    finally:
        data.update(total_s=time.perf_counter() - started, finished_at=time.time())
        # A shielded reference fill may outlive its initiating sample. It must
        # not mutate an already archived timing record.
        state["closed"] = True
        _current.reset(token)


@contextmanager
def phase(name):
    state = _current.get()
    if state is None or state["closed"]:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        if not state["closed"]:
            data = state["data"]
            data[name] = data.get(name, 0.0) + time.perf_counter() - started


def count(name, value=1):
    state = _current.get()
    if state is not None and not state["closed"]:
        data = state["data"]
        data[name] = data.get(name, 0) + value


@asynccontextmanager
async def acquire(lock, name):
    """Measure only admission wait, and release on success/error/cancellation."""
    with phase(name):
        await lock.acquire()
    try:
        yield
    finally:
        lock.release()
