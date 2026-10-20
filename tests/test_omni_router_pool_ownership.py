"""Deterministic assignment/eviction interleavings for the pinned HTTP/1 pool."""

import asyncio
import importlib.util
from pathlib import Path

import httpcore
import httpx
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
@pytest.mark.parametrize("exclusive", [False, True])
def test_assigned_but_not_started_connection_survives_eviction(monkeypatch, guarded, pressure, exclusive):
    async def exercise():
        monkeypatch.setattr(
            AsyncConnectionPool, "_assign_requests_to_connections", AsyncConnectionPool._assign_requests_to_connections
        )
        if guarded:
            router.install_reserved_connection_guard(exclusive_assignment=exclusive)
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


def test_optimistic_mode_preserves_assignment_but_never_evicts_owners(monkeypatch):
    async def exercise():
        monkeypatch.setattr(
            AsyncConnectionPool, "_assign_requests_to_connections", AsyncConnectionPool._assign_requests_to_connections
        )
        router.install_reserved_connection_guard(exclusive_assignment=False)
        pool = AsyncConnectionPool(max_connections=1, max_keepalive_connections=1)
        first, second, other = request("worker-a"), request("worker-a"), request("worker-b")
        connection = Connection(first.request.url.origin)
        pool._connections = [connection]
        pool._requests = [first, second, other]
        assert pool._assign_requests_to_connections() == []
        assert first.connection is second.connection is connection
        assert other.is_queued()
        pool._requests.remove(first)
        assert pool._assign_requests_to_connections() == []
        assert other.is_queued()  # The second owner still protects it.
        pool._requests.remove(second)
        assert pool._assign_requests_to_connections() == [connection]
        assert other.connection is not connection
        with pytest.raises(RuntimeError, match="mode cannot change"):
            router.install_reserved_connection_guard(exclusive_assignment=True)

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


@pytest.mark.parametrize("exclusive", [False, True])
def test_real_http1_waves_preserve_parallelism_limits_and_exactly_once_posts(monkeypatch, exclusive):
    async def exercise():
        monkeypatch.setattr(
            AsyncConnectionPool, "_assign_requests_to_connections", AsyncConnectionPool._assign_requests_to_connections
        )
        router.install_reserved_connection_guard(exclusive_assignment=exclusive)
        active, peak, count = 0, 0, 0
        handlers = set()

        async def handle(reader, writer):
            nonlocal active, peak, count
            handlers.add(asyncio.current_task())
            try:
                while True:
                    header = await reader.readuntil(b"\r\n\r\n")
                    length = next(
                        int(line.split(b":", 1)[1])
                        for line in header.split(b"\r\n")
                        if line.lower().startswith(b"content-length:")
                    )
                    body = await reader.readexactly(length)
                    count += 1
                    active += 1
                    peak = max(peak, active)
                    await asyncio.sleep(0.01)
                    writer.write(
                        b"HTTP/1.1 200 OK\r\nContent-Length: "
                        + str(len(body)).encode() + b"\r\n\r\n" + body
                    )
                    await writer.drain()
                    active -= 1
            except asyncio.IncompleteReadError:
                pass
            finally:
                writer.close()
                await writer.wait_closed()
                handlers.discard(asyncio.current_task())

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/generate"
        try:
            async with httpx.AsyncClient(
                limits=httpx.Limits(max_connections=4, max_keepalive_connections=4),
                trust_env=False, timeout=5,
            ) as client:
                for wave in range(3):
                    payloads = [f"{wave}:{i}".encode() for i in range(12)]
                    responses = await asyncio.gather(*(client.post(url, content=p) for p in payloads))
                    assert [response.content for response in responses] == payloads
            if handlers:
                await asyncio.gather(*handlers)
        finally:
            server.close()
            await server.wait_closed()
        assert count == 36
        assert peak == 4

    asyncio.run(exercise())
