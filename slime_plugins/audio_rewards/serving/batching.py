"""Oldest-first GPU batching; the HTTP server owns bounded item admission."""

import asyncio
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor


class ServiceBusy(RuntimeError):
    pass


class Batcher:
    def __init__(self, encode, *, max_batch=32, max_cost=16384, window_ms=5):
        self.encode = encode
        self.max_batch, self.max_cost = max_batch, max_cost
        self.window = window_ms / 1000
        self.queues = {}
        self.pending = 0
        self.condition = asyncio.Condition()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embedding-gpu")
        self.tasks = []
        self.closed = False
        self.stats = Counter()
        self.batch_sizes = Counter()

    def start(self):
        self.tasks = [asyncio.create_task(self._worker())]

    async def submit(self, value, key, cost):
        if cost > self.max_cost:
            raise ValueError("one input exceeds the configured batch cost budget")
        future = asyncio.get_running_loop().create_future()
        async with self.condition:
            if self.closed:
                raise ServiceBusy("embedding service is shutting down")
            self.queues.setdefault(key, deque()).append((time.monotonic(), value, future, cost))
            self.pending += 1
            self.condition.notify_all()
        return await future

    async def _take(self):
        async with self.condition:
            while True:
                if self.closed:
                    raise asyncio.CancelledError
                if not self.queues:
                    await self.condition.wait()
                    continue
                key = min(self.queues, key=lambda k: self.queues[k][0][0])
                queue = self.queues[key]
                remaining = queue[0][0] + self.window - time.monotonic()
                cost = sum(row[3] for row in list(queue)[: self.max_batch])
                if remaining > 0 and len(queue) < self.max_batch and cost < self.max_cost:
                    try:
                        await asyncio.wait_for(self.condition.wait(), remaining)
                    except asyncio.TimeoutError:
                        pass
                    continue
                selected, cost = [], 0
                while queue and len(selected) < self.max_batch:
                    row = queue[0]
                    if selected and cost + row[3] > self.max_cost:
                        break
                    queue.popleft()
                    self.pending -= 1
                    if row[2].cancelled():
                        continue
                    cost += row[3]
                    selected.append(row)
                if not queue:
                    del self.queues[key]
                if selected:
                    return selected

    async def _worker(self):
        while True:
            batch = await self._take()
            started = time.monotonic()
            self.stats["queue_ms"] += sum((started - row[0]) * 1000 for row in batch)
            try:
                results = await asyncio.get_running_loop().run_in_executor(
                    self.pool, self.encode, [x[1] for x in batch]
                )
                if len(results) != len(batch):
                    raise RuntimeError("model returned a different batch size")
                for row, result in zip(batch, results, strict=True):
                    if not row[2].done():
                        row[2].set_result(result)
                self.stats["items"] += len(batch)
            except Exception as exc:
                self.stats["failed_items"] += len(batch)
                for row in batch:
                    if not row[2].done():
                        row[2].set_exception(exc)
            finally:
                self.stats["batches"] += 1
                self.stats["compute_ms"] += (time.monotonic() - started) * 1000
                self.batch_sizes[len(batch)] += 1

    async def close(self):
        async with self.condition:
            self.closed = True
            for queue in self.queues.values():
                for row in queue:
                    if not row[2].done():
                        row[2].set_exception(ServiceBusy("service shutting down"))
            self.queues.clear()
            self.pending = 0
            self.condition.notify_all()
        # Let active forwards finish before destroying their model/executor.
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.pool.shutdown(wait=True)

    def metrics(self):
        return {**dict(self.stats), "queued": self.pending, "batch_sizes": dict(self.batch_sizes)}
