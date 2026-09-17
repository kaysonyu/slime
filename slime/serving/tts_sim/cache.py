"""Bounded, thread-safe reference embedding cache."""

from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass

import torch

from .audio import FileSignature


@dataclass(frozen=True, slots=True)
class CacheKey:
    signature: FileSignature
    backend_fingerprint: str
    preprocess_version: str
    bucket_version: str


@dataclass(frozen=True, slots=True)
class CacheValue:
    embedding: torch.Tensor
    bucket_seconds: int
    effective_samples: int
    truncated: bool


class ReferenceEmbeddingCache:
    """Bounded LRU cache with single-flight reference inference."""

    def __init__(self, max_items: int = 100_000) -> None:
        self.max_items = max(0, int(max_items))
        self._values: OrderedDict[CacheKey, CacheValue] = OrderedDict()
        self._inflight: dict[CacheKey, Future[CacheValue]] = {}
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.singleflight_waits = 0

    @staticmethod
    def _safe_value(value: CacheValue) -> CacheValue:
        return CacheValue(
            embedding=value.embedding.detach().cpu().float().clone(),
            bucket_seconds=value.bucket_seconds,
            effective_samples=value.effective_samples,
            truncated=value.truncated,
        )

    def get(self, key: CacheKey) -> CacheValue | None:
        with self._lock:
            value = self._values.pop(key, None)
            if value is None:
                self.misses += 1
                return None
            self._values[key] = value
            self.hits += 1
            return self._safe_value(value)

    def reserve(self, key: CacheKey) -> tuple[CacheValue | None, Future[CacheValue] | None, bool]:
        """Return a hit, join an in-flight fill, or reserve ownership.

        The owner must call :meth:`publish` or :meth:`fail`. Waiters never
        decode or infer the same reference while that owner is active.
        """

        with self._lock:
            value = self._values.pop(key, None)
            if value is not None:
                self._values[key] = value
                self.hits += 1
                return self._safe_value(value), None, False
            future = self._inflight.get(key)
            if future is not None:
                self.singleflight_waits += 1
                return None, future, False
            future = Future()
            self._inflight[key] = future
            self.misses += 1
            return None, future, True

    @classmethod
    def wait(cls, future: Future[CacheValue]) -> CacheValue:
        return cls._safe_value(future.result())

    def publish(self, key: CacheKey, future: Future[CacheValue], value: CacheValue) -> None:
        safe = self._safe_value(value)
        with self._lock:
            if self.max_items:
                self._values.pop(key, None)
                self._values[key] = safe
                while len(self._values) > self.max_items:
                    self._values.popitem(last=False)
                    self.evictions += 1
            current = self._inflight.get(key)
            if current is future:
                del self._inflight[key]
            else:
                current = None
        if current is not None and not future.done():
            future.set_result(safe)

    def fail(self, key: CacheKey, future: Future[CacheValue], exc: BaseException) -> None:
        with self._lock:
            current = self._inflight.get(key)
            if current is future:
                del self._inflight[key]
            else:
                current = None
        if current is not None and not future.done():
            future.set_exception(exc)

    def put(self, key: CacheKey, value: CacheValue) -> None:
        if self.max_items == 0:
            return
        safe = self._safe_value(value)
        with self._lock:
            self._values.pop(key, None)
            self._values[key] = safe
            while len(self._values) > self.max_items:
                self._values.popitem(last=False)
                self.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._values.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)
