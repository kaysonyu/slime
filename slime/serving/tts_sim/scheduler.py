"""Bounded dynamic batching worker for embedding inference."""

from __future__ import annotations

import concurrent.futures
import threading
import time
from collections import deque
from dataclasses import dataclass

import torch

from .backend import EmbeddingBackend


class QueueFullError(RuntimeError):
    pass


class SchedulerClosedError(RuntimeError):
    pass


@dataclass(slots=True)
class _WorkItem:
    waveform: torch.Tensor
    length: int
    future: concurrent.futures.Future[torch.Tensor]
    enqueued_at: float


class DynamicBatchingBackend:
    """Queue same-width waveforms until a batch or latency limit is reached."""

    name = "dynamic_batch"

    def __init__(
        self,
        backend: EmbeddingBackend,
        *,
        max_batch: int = 16,
        max_delay_ms: float = 5.0,
        max_queue_items: int = 8_192,
    ) -> None:
        if max_batch <= 0 or max_queue_items <= 0 or max_delay_ms < 0:
            raise ValueError("invalid dynamic batching limits")
        self.backend = backend
        self.fingerprint = backend.fingerprint
        self.checkpoint_sha256 = getattr(backend, "checkpoint_sha256", self.fingerprint)
        self.allow_tf32 = bool(getattr(backend, "allow_tf32", False))
        self.precision = str(getattr(backend, "precision", "unknown"))
        self.details = dict(getattr(backend, "details", {}))
        self.max_batch = int(max_batch)
        self.max_delay_seconds = float(max_delay_ms) / 1_000.0
        self.max_queue_items = int(max_queue_items)
        self._queues: dict[int, deque[_WorkItem]] = {}
        self._queued_items = 0
        self._condition = threading.Condition()
        self._closed = False
        self.batches = 0
        self.items = 0
        self.max_observed_batch = 0
        self.batch_size_counts: dict[int, int] = {}
        self.shape_counts: dict[str, int] = {}
        self._worker = threading.Thread(
            target=self._run, name="embedding-dynamic-batcher", daemon=True
        )
        self._worker.start()

    @property
    def is_ready(self) -> bool:
        return not self._closed and bool(self.backend.is_ready)

    @property
    def queue_depth(self) -> int:
        with self._condition:
            return self._queued_items

    def _oldest_width(self) -> int | None:
        populated = (
            (queue[0].enqueued_at, width) for width, queue in self._queues.items() if queue
        )
        return min(populated, default=(0.0, None))[1]

    def _take_batch(self) -> list[_WorkItem]:
        width = self._oldest_width()
        if width is None:
            return []
        queue = self._queues[width]
        now = time.monotonic()
        remaining = self.max_delay_seconds - (now - queue[0].enqueued_at)
        if len(queue) < self.max_batch and remaining > 0 and not self._closed:
            self._condition.wait(timeout=remaining)
            return []
        count = min(len(queue), self.max_batch)
        result = [queue.popleft() for _ in range(count)]
        self._queued_items -= count
        if not queue:
            del self._queues[width]
        self._condition.notify_all()
        return result

    def _run(self) -> None:
        """Worker loop that completes every submitted Future on success/failure."""
        while True:
            with self._condition:
                while not self._queued_items and not self._closed:
                    self._condition.wait()
                if self._closed and not self._queued_items:
                    return
                items = self._take_batch()
            if not items:
                continue
            try:
                batch = torch.stack([item.waveform for item in items], dim=0)
                lengths = torch.tensor([item.length for item in items], dtype=torch.long)
                outputs = self.backend.embed(batch, lengths)
                if outputs.shape[0] != len(items):
                    raise RuntimeError("dynamic backend result count does not match batch")
                self.batches += 1
                self.items += len(items)
                self.max_observed_batch = max(self.max_observed_batch, len(items))
                batch_size = len(items)
                width = int(items[0].waveform.numel())
                self.batch_size_counts[batch_size] = self.batch_size_counts.get(batch_size, 0) + 1
                shape = f"{width}x{batch_size}"
                self.shape_counts[shape] = self.shape_counts.get(shape, 0) + 1
                for item, output in zip(items, outputs, strict=True):
                    item.future.set_result(output)
            # This worker boundary must complete every Future on backend failure.
            except Exception as exc:
                for item in items:
                    item.future.set_exception(exc)

    def embed(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim != 2 or lengths.ndim != 1:
            raise ValueError("dynamic backend expects rank-2 waveforms and rank-1 lengths")
        if waveforms.shape[0] != lengths.shape[0] or waveforms.shape[0] == 0:
            raise ValueError("dynamic backend received an empty or mismatched batch")
        if waveforms.shape[0] > self.max_batch:
            raise ValueError(
                f"submitted batch {waveforms.shape[0]} exceeds dynamic max {self.max_batch}"
            )
        rows = waveforms.detach().cpu().float().contiguous()
        values = lengths.detach().cpu().long().tolist()
        futures: list[concurrent.futures.Future[torch.Tensor]] = []
        now = time.monotonic()
        with self._condition:
            if self._closed:
                raise SchedulerClosedError("dynamic batcher is closed")
            incoming = int(rows.shape[0])
            if self._queued_items + incoming > self.max_queue_items:
                raise QueueFullError(f"embedding queue would exceed {self.max_queue_items} items")
            width = int(rows.shape[1])
            queue = self._queues.setdefault(width, deque())
            for row, length in zip(rows, values, strict=True):
                future: concurrent.futures.Future[torch.Tensor] = concurrent.futures.Future()
                futures.append(future)
                queue.append(_WorkItem(row, int(length), future, now))
            self._queued_items += incoming
            self._condition.notify_all()
        return torch.stack([future.result() for future in futures], dim=0)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._worker.join(timeout=30)
        if self._worker.is_alive():
            raise RuntimeError("dynamic batcher did not stop within 30 seconds")
        close = getattr(self.backend, "close", None)
        if close is not None:
            close()
