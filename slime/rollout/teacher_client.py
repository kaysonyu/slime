"""Bounded teacher request queues owned by a rollout execution runtime."""

import asyncio
import math
import time
from dataclasses import dataclass, fields

import aiohttp
import httpx


@dataclass(frozen=True)
class TeacherClientOptions:
    batch_size: int = 8
    batch_wait_ms: float = 2.0
    max_inflight: int = 16
    max_attempts: int = 2
    request_timeout_s: float = 120.0

    def __post_init__(self):
        if (
            type(self.batch_size) is not int
            or not 1 <= self.batch_size <= 64
            or type(self.max_inflight) is not int
            or self.max_inflight < 1
            or type(self.max_attempts) is not int
            or self.max_attempts < 1
            or not isinstance(self.batch_wait_ms, (int, float))
            or not 0 <= self.batch_wait_ms <= 1000
        ):
            raise ValueError("Invalid frozen teacher batch, wait, concurrency, or retry limit")
        if (
            not isinstance(self.request_timeout_s, (int, float))
            or not math.isfinite(self.request_timeout_s)
            or self.request_timeout_s <= 0
        ):
            raise ValueError("Teacher request timeout must be positive and finite")

    @classmethod
    def from_config(cls, configuration):
        configured = configuration.get("client", {})
        if not isinstance(configured, dict) or set(configured) - {field.name for field in fields(cls)}:
            raise ValueError("Unknown teacher client options")
        return cls(**configured)


@dataclass(frozen=True)
class ScoreReceipt:
    value: object
    batch_size: int
    request_seconds: float
    attempts: int


class BatchedTeacherClient:
    """Own batching, concurrency, cancellation and retries independently of a wire protocol."""

    def __init__(self, options: TeacherClientOptions):
        self.options = options
        self.semaphore = asyncio.Semaphore(options.max_inflight)
        self.pending = {}
        self.timers = {}
        self.tasks = set()
        self.closed = False

    async def score(self, route, request):
        if self.closed:
            raise RuntimeError("Teacher client is closed")
        result = asyncio.get_running_loop().create_future()
        pending = self.pending.setdefault(route, [])
        pending.append((request, result))
        if len(pending) >= self.options.batch_size:
            self.dispatch(route)
        elif route not in self.timers:
            self.timers[route] = asyncio.get_running_loop().call_later(
                self.options.batch_wait_ms / 1000, self.dispatch, route
            )
        try:
            return await asyncio.shield(result)
        except asyncio.CancelledError:
            result.cancel()
            raise

    def dispatch(self, route):
        timer = self.timers.pop(route, None)
        if timer is not None:
            timer.cancel()
        pending = [item for item in self.pending.pop(route, []) if not item[1].done()]
        for begin in range(0, len(pending), self.options.batch_size):
            task = asyncio.create_task(self.send(route, pending[begin : begin + self.options.batch_size]))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

    async def send_batch(self, route, requests):
        """Return one result per request, in input order."""
        raise NotImplementedError

    async def send(self, route, batch):
        try:
            async with self.semaphore:
                started = time.perf_counter()
                for attempt in range(self.options.max_attempts):
                    try:
                        results = await asyncio.wait_for(
                            self.send_batch(route, [request for request, _ in batch]),
                            timeout=self.options.request_timeout_s,
                        )
                        break
                    except (
                        httpx.TransportError,
                        httpx.HTTPStatusError,
                        aiohttp.ClientError,
                        asyncio.TimeoutError,
                        TimeoutError,
                    ) as error:
                        status = (
                            error.response.status_code
                            if isinstance(error, httpx.HTTPStatusError)
                            else error.status if isinstance(error, aiohttp.ClientResponseError) else None
                        )
                        if (
                            status is not None and status not in {429, 502, 503, 504}
                        ) or attempt + 1 == self.options.max_attempts:
                            raise
                        await asyncio.sleep(min(0.1 * 2**attempt, 1.0))
            if len(results) != len(batch):
                raise ValueError("Teacher batch result count differs from request count")
            for (_, future), result in zip(batch, results, strict=True):
                if not future.done():
                    future.set_result(ScoreReceipt(result, len(batch), time.perf_counter() - started, attempt + 1))
        except BaseException as error:
            for _, future in batch:
                if not future.done():
                    if isinstance(error, asyncio.CancelledError):
                        future.cancel()
                    else:
                        future.set_exception(error)

    async def flush(self):
        for route in tuple(self.pending):
            self.dispatch(route)
        if self.tasks:
            await asyncio.gather(*tuple(self.tasks))

    async def close(self):
        self.closed = True
        for timer in self.timers.values():
            timer.cancel()
        self.timers.clear()
        for pending in self.pending.values():
            for _, future in pending:
                future.cancel()
        self.pending.clear()
        for task in tuple(self.tasks):
            task.cancel()
        if self.tasks:
            await asyncio.gather(*tuple(self.tasks), return_exceptions=True)
