"""Bounded retry/concurrency runtime for reward HTTP services."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import aiohttp

from slime.rollout.failures import RolloutFailure, SampleFailure

from .config import ServiceConfig


class RewardServiceError(RolloutFailure):
    def __init__(self, service: str, category: str, message: str) -> None:
        super().__init__(
            SampleFailure(
                category=f"reward.{service}.{category}",
                retryable=category == "retry_exhausted",
            ),
            message,
        )
        self.service = service
        self.category = category


@dataclass(slots=True)
class _ServiceStats:
    requests: int = 0
    attempts: int = 0
    failures: int = 0
    in_flight: int = 0
    max_in_flight: int = 0
    queue_wait_seconds: float = 0.0
    request_seconds: float = 0.0
    max_request_seconds: float = 0.0
    service_seconds: float = 0.0
    max_service_seconds: float = 0.0


class RewardHttpRuntime:
    """Rollout-scoped HTTP sessions, limits, retries, and auth ownership."""

    def __init__(self, services: Mapping[str, ServiceConfig]) -> None:
        self._services: dict[str, ServiceConfig] = {}
        self._sessions: dict[str, aiohttp.ClientSession] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._stats: dict[str, _ServiceStats] = {}
        self._closed = False
        self.add_services(services)

    async def __aenter__(self) -> RewardHttpRuntime:
        self._ensure_open()
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback) -> None:
        await self.close()

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Reward HTTP runtime is closed.")

    def add_services(self, services: Mapping[str, ServiceConfig]) -> None:
        """Register services without replacing an existing rollout-wide limit."""
        self._ensure_open()
        for name, config in services.items():
            existing = self._services.get(name)
            if existing is not None:
                if existing != config:
                    raise RewardServiceError(
                        name,
                        "configuration",
                        "service configuration changed within one rollout",
                    )
                continue
            self._services[name] = config
            self._semaphores[name] = asyncio.Semaphore(config.runtime.concurrency)
            self._stats[name] = _ServiceStats()

    def collect_metrics(self) -> dict[str, float]:
        """Return aggregate numeric metrics without request or service payload data."""
        metrics: dict[str, float] = {}
        for name, stats in sorted(self._stats.items()):
            prefix = f"reward/{name}"
            metrics[f"{prefix}/requests"] = float(stats.requests)
            metrics[f"{prefix}/attempts"] = float(stats.attempts)
            metrics[f"{prefix}/failures"] = float(stats.failures)
            metrics[f"{prefix}/max_in_flight"] = float(stats.max_in_flight)
            metrics[f"{prefix}/queue_wait_seconds"] = stats.queue_wait_seconds
            metrics[f"{prefix}/request_seconds"] = stats.request_seconds
            metrics[f"{prefix}/max_request_seconds"] = stats.max_request_seconds
            metrics[f"{prefix}/service_seconds"] = stats.service_seconds
            metrics[f"{prefix}/max_service_seconds"] = stats.max_service_seconds
        return metrics

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        sessions = tuple(self._sessions.values())
        self._sessions.clear()
        await asyncio.gather(*(session.close() for session in sessions))

    def service(self, name: str) -> ServiceConfig:
        self._ensure_open()
        try:
            return self._services[name]
        except KeyError as error:
            raise RewardServiceError(name, "configuration", "service is not configured") from error

    def headers(
        self,
        name: str,
        request_headers: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        config = self.service(name)
        headers: dict[str, str] = {}
        if config.auth_token_env is not None:
            token = os.environ.get(config.auth_token_env)
            if not token:
                raise RewardServiceError(
                    name,
                    "configuration",
                    f"environment variable {config.auth_token_env!r} is not set",
                )
            headers["Authorization"] = f"Bearer {token}"
        for key, value in (request_headers or {}).items():
            if key.casefold() == "authorization":
                raise RewardServiceError(
                    name,
                    "configuration",
                    "per-request headers cannot override Authorization",
                )
            headers[key] = value
        return headers

    def _session(self, name: str) -> aiohttp.ClientSession:
        if name not in self._sessions:
            config = self.service(name)
            self._sessions[name] = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=config.runtime.concurrency, enable_cleanup_closed=True),
                timeout=aiohttp.ClientTimeout(total=config.runtime.timeout_seconds),
                trust_env=False,
            )
        return self._sessions[name]

    async def post_json(
        self,
        name: str,
        payload: Mapping[str, Any],
        *,
        request_headers: Mapping[str, str] | None = None,
    ) -> Any:
        return await self._request(name, json_payload=payload, request_headers=request_headers)

    async def post_json_factory(
        self,
        name: str,
        payload_factory: Callable[[], Mapping[str, Any]],
        *,
        request_headers: Mapping[str, str] | None = None,
    ) -> Any:
        """Build a potentially large payload only after acquiring the service permit."""
        return await self._request(
            name,
            json_factory=payload_factory,
            request_headers=request_headers,
        )

    async def post_form(self, name: str, form_factory: Callable[[], aiohttp.FormData]) -> Any:
        return await self._request(name, form_factory=form_factory)

    async def _request(
        self,
        name: str,
        *,
        json_payload: Mapping[str, Any] | None = None,
        json_factory: Callable[[], Mapping[str, Any]] | None = None,
        form_factory: Callable[[], aiohttp.FormData] | None = None,
        request_headers: Mapping[str, str] | None = None,
    ) -> Any:
        """Issue one bounded request; retry only transient HTTP/network failures."""
        config = self.service(name)
        if sum(value is not None for value in (json_payload, json_factory, form_factory)) != 1:
            raise ValueError("Exactly one JSON payload, JSON factory, or form factory must be supplied.")
        max_attempts = config.runtime.max_retries + 1
        # Acquire the service permit before constructing large payloads or
        # opening a request, keeping concurrency and memory bounded together.
        stats = self._stats[name]
        request_started = perf_counter()
        async with self._semaphores[name]:
            stats.queue_wait_seconds += perf_counter() - request_started
            stats.requests += 1
            stats.in_flight += 1
            stats.max_in_flight = max(stats.max_in_flight, stats.in_flight)
            service_started = perf_counter()
            try:
                if json_factory is not None:
                    json_payload = json_factory()
                for attempt in range(max_attempts):
                    try:
                        data = form_factory() if form_factory is not None else None
                        headers = self.headers(name, request_headers)
                        session = self._session(name)
                        stats.attempts += 1
                        async with session.post(
                            config.endpoint,
                            json=json_payload,
                            data=data,
                            headers=headers,
                        ) as response:
                            if response.status == 429 or 500 <= response.status < 600:
                                if attempt + 1 < max_attempts:
                                    await asyncio.sleep(min(2**attempt, 8))
                                    continue
                                raise RewardServiceError(name, "retry_exhausted", f"HTTP status {response.status}")
                            if response.status >= 400:
                                category = "authentication" if response.status in {401, 403} else "request"
                                raise RewardServiceError(name, category, f"HTTP status {response.status}")
                            try:
                                return await response.json()
                            except (aiohttp.ContentTypeError, json.JSONDecodeError) as error:
                                raise RewardServiceError(name, "protocol", "response is not valid JSON") from error
                    except (aiohttp.ClientConnectionError, aiohttp.ServerTimeoutError, asyncio.TimeoutError) as error:
                        if attempt + 1 >= max_attempts:
                            raise RewardServiceError(name, "retry_exhausted", type(error).__name__) from error
                        await asyncio.sleep(min(2**attempt, 8))
            except RewardServiceError:
                stats.failures += 1
                raise
            finally:
                finished = perf_counter()
                request_elapsed = finished - request_started
                service_elapsed = finished - service_started
                stats.request_seconds += request_elapsed
                stats.max_request_seconds = max(stats.max_request_seconds, request_elapsed)
                stats.service_seconds += service_elapsed
                stats.max_service_seconds = max(stats.max_service_seconds, service_elapsed)
                stats.in_flight -= 1
        raise AssertionError("Reward request retry loop exited unexpectedly.")


_CURRENT_REWARD_HTTP_RUNTIME: ContextVar[RewardHttpRuntime | None] = ContextVar(
    "tts_reward_http_runtime",
    default=None,
)


def get_reward_http_runtime() -> RewardHttpRuntime:
    runtime = _CURRENT_REWARD_HTTP_RUNTIME.get()
    if runtime is None:
        raise RuntimeError("MOSS-TTS HTTP scoring requires the plugin rollout entry point to own a reward runtime scope.")
    return runtime


@asynccontextmanager
async def reward_http_runtime_scope() -> AsyncIterator[RewardHttpRuntime]:
    """Own one HTTP runtime shared by every task in a top-level rollout."""
    if _CURRENT_REWARD_HTTP_RUNTIME.get() is not None:
        raise RuntimeError("Reward HTTP runtime scopes cannot be nested.")
    async with RewardHttpRuntime({}) as runtime:
        token = _CURRENT_REWARD_HTTP_RUNTIME.set(runtime)
        try:
            yield runtime
        finally:
            _CURRENT_REWARD_HTTP_RUNTIME.reset(token)
