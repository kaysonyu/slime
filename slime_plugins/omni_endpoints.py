"""Send Omni generation straight to engines; the router stays the membership source.

The Omni router owns registration, health checks and admission. Relaying every
generation request through it adds a hop and a single process that saturates
at scale. Rollout workers instead read the router's routable workers and post
to an engine directly, as Prism teacher scoring already does.

Each execution worker owns a stable partition of the engines: sorted engine i
belongs to worker i mod W. The execution pool gives every worker an equal share
of the generation budget, so when W divides the engine count a worker is the
only client of its engines and its local least-request choice is exact. A
worker with no reachable engine in its partition uses every routable engine.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import random
import time
from collections import Counter

import httpx

from slime.utils.http_utils import get
from slime_plugins.prism.http_client import _request

logger = logging.getLogger(__name__)

# The router re-probes worker health every few seconds; a fresher view only
# adds router load without changing which engines are routable.
_MEMBERSHIP_TTL_S = 2.0
# Nothing reached the engine, so another engine can take the request.
_CONNECT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)

_memberships: dict[str, _Membership] = {}
_inflight: Counter[str] = Counter()
_rotation = itertools.count()


class _Membership:
    def __init__(self, router_url: str, model: str):
        self.router_url = router_url
        self.model = model
        self.engines: list[str] = []
        self.unreachable: set[str] = set()
        self.fetched_at = float("-inf")
        self.refresh_task: asyncio.Task | None = None

    async def current(self, timeout: float) -> list[str]:
        """Routable engines; refreshes in the background once the view is stale."""
        if time.monotonic() - self.fetched_at > _MEMBERSHIP_TTL_S:
            loop = asyncio.get_running_loop()
            task = self.refresh_task
            if task is None or task.done() or task.get_loop() is not loop:
                task = self.refresh_task = loop.create_task(self._refresh(timeout))
            if not self.engines:
                await asyncio.shield(task)
        return self.engines

    async def _refresh(self, timeout: float) -> None:
        try:
            body = await asyncio.wait_for(get(self.router_url + "/workers", timeout=timeout), timeout=timeout)
            if not isinstance(body, dict) or not isinstance(body.get("workers"), list):
                raise ValueError(f"Omni router {self.router_url} /workers returned an invalid worker list")
            engines = sorted(
                {
                    worker["url"].rstrip("/")
                    for worker in body["workers"]
                    if isinstance(worker, dict)
                    and worker.get("health_state") == "healthy"
                    and worker.get("routable") is True
                    and worker.get("disabled") is False
                    and worker.get("model") in (None, self.model)
                    and isinstance(worker.get("url"), str)
                    and worker["url"].startswith(("http://", "https://"))
                }
            )
            if not engines:
                raise ValueError(f"Omni router {self.router_url} has no healthy routable workers")
        except Exception:
            if not self.engines:
                raise
            logger.warning(
                "Keeping the cached Omni engines of %s after a failed refresh", self.router_url, exc_info=True
            )
            self.fetched_at = time.monotonic()
            return
        self.engines = engines
        self.unreachable.clear()
        self.fetched_at = time.monotonic()


def partition(engines: list[str], index: int | None, count: int | None) -> list[str]:
    """Engines owned by execution worker ``index`` of ``count``; all of them without a pool."""
    if index is None or not count or count <= 1:
        return engines
    if len(engines) >= count:
        return engines[index % count :: count]
    return [engines[index % len(engines)]]


def choose(engines: list[str], policy: str) -> str:
    if policy == "random":
        return random.choice(engines)
    start = next(_rotation) % len(engines)
    ordered = engines[start:] + engines[:start]
    if policy == "least_request":
        # min keeps the first of equally loaded engines, so ties rotate.
        return min(ordered, key=_inflight.__getitem__)
    return ordered[0]


async def post_to_engine(args, router_url: str, model: str, path: str, payload, *, headers=None):
    """POST ``payload`` to one engine of ``model``; fail over only if nothing was sent."""
    membership = _memberships.get(router_url)
    if membership is None:
        membership = _memberships[router_url] = _Membership(router_url, model)
    timeout = float(getattr(args, "rollout_health_check_timeout", 10.0))
    policy = getattr(args, "omni_router_policy", "round_robin")
    for attempt in range(2):
        engines = await membership.current(timeout)
        own = partition(
            engines, getattr(args, "rollout_worker_index", None), getattr(args, "rollout_worker_count", None)
        )
        candidates = [engine for engine in own if engine not in membership.unreachable]
        candidates = candidates or [engine for engine in engines if engine not in membership.unreachable] or engines
        engine = choose(candidates, policy)
        _inflight[engine] += 1
        try:
            return await _request(args, engine + path, payload, headers=headers)
        except _CONNECT_ERRORS:
            if attempt:
                raise
            logger.warning("Omni engine %s refused the connection; retrying on another engine", engine)
            membership.unreachable.add(engine)
        finally:
            _inflight[engine] -= 1
