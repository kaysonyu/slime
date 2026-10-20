"""The client side must use the same ownership guard without changing limits."""

import asyncio
from types import SimpleNamespace

import httpcore
import httpx
import pytest
from httpcore._async.connection_pool import AsyncConnectionPool
from httpcore._async.http11 import AsyncHTTP11Connection

from slime.utils import http_utils

NUM_GPUS = 0
pytestmark = pytest.mark.unit


def test_sharded_client_installs_guard_and_preserves_limits(monkeypatch):
    for cls, name in (
        (AsyncConnectionPool, "_assign_requests_to_connections"),
        (AsyncHTTP11Connection, "aclose"),
        (httpx.AsyncClient, "send"),
    ):
        monkeypatch.setattr(cls, name, getattr(cls, name))
    transport = http_utils._ShardedHTTPTransport(65)
    assert transport._pool_sizes == [22, 22, 21]
    assert getattr(AsyncConnectionPool._assign_requests_to_connections, "_prism_reserved_connection_guard")
    assert getattr(AsyncHTTP11Connection.aclose, "_prism_active_close_diagnostics")
    assert getattr(httpx.AsyncClient.send, "_prism_transport_diagnostics")


def test_failed_generation_post_is_not_retried():
    calls = []
    failure = httpx.ReadError("not retryable without idempotency")

    async def post(url, **kwargs):
        calls.append((url, kwargs))
        raise failure

    with pytest.raises(httpx.ReadError) as raised:
        asyncio.run(http_utils._post(SimpleNamespace(post=post), "http://router/generate", {}, max_retries=1))
    assert raised.value is failure
    assert len(calls) == 1
