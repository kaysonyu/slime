"""Deterministic assignment/eviction interleavings for the pinned HTTP/1 pool."""

import asyncio
import importlib.util
from pathlib import Path

import httpcore
import pytest
from httpcore._async.connection_pool import AsyncConnectionPool, AsyncPoolRequest

NUM_GPUS = 0
pytestmark = pytest.mark.unit
path = Path(__file__).resolve().parents[1] / "slime/backends/sglang_utils/omni_router.py"
spec = importlib.util.spec_from_file_location("router_pool_under_test", path)
router = importlib.util.module_from_spec(spec)
spec.loader.exec_module(router)


class Connection:
    def __init__(self, origin, *, expired=False):
        self.origin = origin
        self.expired = expired
        self.closed = False

    def is_closed(self):
        return self.closed

    def has_expired(self):
        return self.expired

    def is_idle(self):
        return not self.closed

    def is_available(self):
        return not self.closed

    def can_handle_request(self, origin):
        return origin == self.origin


def request(host):
    return AsyncPoolRequest(httpcore.Request("GET", f"http://{host}/generate"))


@pytest.mark.parametrize("guarded", [False, True])
@pytest.mark.parametrize("pressure", ["expiry", "retention", "capacity"])
def test_assigned_but_not_started_connection_survives_eviction(monkeypatch, guarded, pressure):
    async def exercise():
        monkeypatch.setattr(
            AsyncConnectionPool, "_assign_requests_to_connections", AsyncConnectionPool._assign_requests_to_connections
        )
        if guarded:
            router.install_reserved_connection_guard()
        pool = AsyncConnectionPool(max_connections=1, max_keepalive_connections=0 if pressure == "retention" else 1)
        owner = request("worker-a")
        connection = Connection(owner.request.url.origin, expired=pressure == "expiry")
        owner.assign_to_connection(connection)
        pool._connections = [connection]
        pool._requests = [owner]
        waiter = request("worker-b")
        if pressure == "capacity":
            pool._requests.append(waiter)
        closing = pool._assign_requests_to_connections()
        assert (connection in closing) is not guarded
        assert (connection in pool.connections) is guarded
        assert owner.connection is connection
        if guarded and pressure == "capacity":
            assert waiter.is_queued()
            pool._requests.remove(owner)
            closing = pool._assign_requests_to_connections()
            assert connection in closing
            assert waiter.connection is not connection
        assert len(pool.connections) <= 1

    asyncio.run(exercise())


def test_http1_connections_are_reserved_once_within_an_assignment_pass(monkeypatch):
    async def exercise():
        monkeypatch.setattr(
            AsyncConnectionPool, "_assign_requests_to_connections", AsyncConnectionPool._assign_requests_to_connections
        )
        router.install_reserved_connection_guard()
        pool = AsyncConnectionPool(max_connections=2, max_keepalive_connections=2)
        requests = [request("worker-a") for _ in range(3)]
        existing = Connection(requests[0].request.url.origin)
        pool._connections = [existing]
        pool._requests = requests
        assert pool._assign_requests_to_connections() == []
        assert requests[0].connection is existing
        assert requests[1].connection is not None and requests[1].connection is not existing
        assert requests[2].is_queued()
        assert len(pool.connections) == 2
        assert pool._assign_requests_to_connections() == []
        assert requests[2].is_queued()

    asyncio.run(exercise())


def test_unowned_expired_connections_are_still_closed(monkeypatch):
    async def exercise():
        monkeypatch.setattr(
            AsyncConnectionPool, "_assign_requests_to_connections", AsyncConnectionPool._assign_requests_to_connections
        )
        router.install_reserved_connection_guard()
        pool = AsyncConnectionPool(max_connections=2)
        waiting = request("worker-a")
        expired = Connection(waiting.request.url.origin, expired=True)
        pool._connections = [expired]
        pool._requests = [waiting]
        assert pool._assign_requests_to_connections() == [expired]
        assert waiting.connection is not expired

    asyncio.run(exercise())


def test_guard_is_idempotent_and_requires_the_validated_dependency(monkeypatch):
    monkeypatch.setattr(
        AsyncConnectionPool, "_assign_requests_to_connections", AsyncConnectionPool._assign_requests_to_connections
    )
    monkeypatch.setattr(httpcore, "__version__", "unvalidated")
    with pytest.raises(RuntimeError, match="Revalidate"):
        router.install_reserved_connection_guard()
    monkeypatch.setattr(httpcore, "__version__", "1.0.9")
    router.install_reserved_connection_guard()
    installed = AsyncConnectionPool._assign_requests_to_connections
    router.install_reserved_connection_guard()
    assert AsyncConnectionPool._assign_requests_to_connections is installed
