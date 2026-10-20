"""Direct Omni generation: membership, worker partitions, least-request and failover."""

import asyncio
import itertools
import json
import types
from collections import Counter

import httpx
import pytest

from slime.utils import http_utils
from slime_plugins import omni_endpoints as endpoints

NUM_GPUS = 0
pytestmark = pytest.mark.unit
ROUTER = "http://router:3000"
ENGINES = [f"http://10.0.0.{index}:8000" for index in range(8)]


def worker_list(urls, **overrides):
    return {
        "workers": [
            {"url": url, "health_state": "healthy", "routable": True, "disabled": False, "model": None}
            | overrides.get(url, {})
            for url in urls
        ]
    }


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(endpoints, "_memberships", {})
    monkeypatch.setattr(endpoints, "_inflight", Counter())
    monkeypatch.setattr(endpoints, "_rotation", itertools.count())


def fake_router(monkeypatch, *responses):
    calls = []
    pending = list(responses)

    async def get(url, *, timeout=None, headers=None):
        calls.append(url)
        response = pending.pop(0) if len(pending) > 1 else pending[0]
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(endpoints, "get", get)
    return calls


def args(**overrides):
    return types.SimpleNamespace(**({"omni_router_policy": "least_request"} | overrides))


def test_partitions_cover_every_engine_once_and_degrade_when_workers_outnumber_engines():
    for count in (1, 2, 3, 4, 8):
        owned = [endpoints.partition(ENGINES, index, count) for index in range(count)]
        assert sorted(engine for part in owned for engine in part) == ENGINES
        assert max(map(len, owned)) - min(map(len, owned)) <= 1
    assert endpoints.partition(ENGINES[:2], 3, 4) == [ENGINES[1]]
    assert endpoints.partition(ENGINES, None, None) == ENGINES


def test_membership_excludes_workers_the_router_would_not_route_to(monkeypatch):
    excluded = {
        ENGINES[1]: {"health_state": "unhealthy"},
        ENGINES[2]: {"disabled": True},
        ENGINES[3]: {"routable": False},
        ENGINES[4]: {"model": "teacher"},
    }
    fake_router(monkeypatch, worker_list(ENGINES[:6], **excluded))
    seen = []

    async def request(args, url, payload, headers=None):
        seen.append(url)
        return {"ok": True}

    monkeypatch.setattr(endpoints, "_request", request)

    async def exercise():
        for _ in range(4):
            await endpoints.post_to_engine(args(), ROUTER, "student", "/generate", {})

    asyncio.run(exercise())
    assert sorted(set(seen)) == [ENGINES[0] + "/generate", ENGINES[5] + "/generate"]


def test_least_request_balances_a_partition_exactly_and_releases_on_failure(monkeypatch):
    fake_router(monkeypatch, worker_list(ENGINES[:4]))
    release = asyncio.Event()
    peak = Counter()

    async def request(args, url, payload, headers=None):
        engine = url.removesuffix("/generate")
        peak[engine] = max(peak[engine], endpoints._inflight[engine])
        await release.wait()
        if payload.get("fail"):
            raise httpx.ReadError("upstream reset")
        return engine

    monkeypatch.setattr(endpoints, "_request", request)
    owner = args(rollout_worker_index=1, rollout_worker_count=2)

    async def exercise():
        tasks = [
            asyncio.create_task(
                endpoints.post_to_engine(owner, ROUTER, "student", "/generate", {"fail": index % 3 == 0})
            )
            for index in range(16)
        ]
        while sum(endpoints._inflight.values()) < 16:
            await asyncio.sleep(0)
        release.set()
        return await asyncio.gather(*tasks, return_exceptions=True)

    results = asyncio.run(exercise())
    assert set(peak) == {ENGINES[1], ENGINES[3]} and set(peak.values()) == {8}
    assert sum(isinstance(result, httpx.ReadError) for result in results) == 6
    assert not +endpoints._inflight


def test_refused_connections_fail_over_once_and_other_errors_are_not_retried(monkeypatch):
    fake_router(monkeypatch, worker_list(ENGINES[:2]))
    attempts = []

    async def request(args, url, payload, headers=None):
        attempts.append(url)
        if payload["mode"] == "refused" or url.startswith(ENGINES[0]) and payload["mode"] == "first":
            raise httpx.ConnectError("connection refused")
        if payload["mode"] == "reset":
            raise httpx.ReadError("upstream reset")
        return url

    monkeypatch.setattr(endpoints, "_request", request)
    policy = args(omni_router_policy="round_robin")

    async def post(mode):
        return await endpoints.post_to_engine(policy, ROUTER, "student", "/generate", {"mode": mode})

    assert asyncio.run(post("first")) == ENGINES[1] + "/generate"
    assert endpoints._memberships[ROUTER].unreachable == {ENGINES[0]}
    attempts.clear()
    with pytest.raises(httpx.ConnectError):
        asyncio.run(post("refused"))
    assert len(attempts) == 2
    attempts.clear()
    with pytest.raises(httpx.ReadError):
        asyncio.run(post("reset"))
    assert len(attempts) == 1


def test_stale_views_refresh_in_the_background_and_survive_router_failures(monkeypatch):
    calls = fake_router(
        monkeypatch, worker_list(ENGINES[:1]), worker_list(ENGINES[1:2]), RuntimeError("router restarting")
    )
    monkeypatch.setattr(endpoints, "_MEMBERSHIP_TTL_S", 0.0)

    async def request(args, url, payload, headers=None):
        return url

    monkeypatch.setattr(endpoints, "_request", request)

    async def post():
        return await endpoints.post_to_engine(args(), ROUTER, "student", "/generate", {})

    async def exercise():
        first = await post()
        stale = await post()  # served from the cached view while the refresh runs
        await asyncio.sleep(0)
        fresh = await post()
        await post()  # the failing refresh is still pending
        await asyncio.sleep(0)
        kept = await post()
        return first, stale, fresh, kept

    first, stale, fresh, kept = asyncio.run(exercise())
    assert first == stale == ENGINES[0] + "/generate"
    assert fresh == kept == ENGINES[1] + "/generate"
    # The third refresh failed; later requests kept using the second view.
    assert len(calls) >= 3 and set(calls) == {ROUTER + "/workers"}


def test_a_router_without_a_cached_view_fails_the_request(monkeypatch):
    fake_router(monkeypatch, worker_list([]))
    with pytest.raises(ValueError, match="no healthy routable workers"):
        asyncio.run(endpoints.post_to_engine(args(), ROUTER, "student", "/generate", {}))


def test_real_http_workers_only_reach_their_own_engines(monkeypatch):
    async def exercise():
        hits = Counter()

        async def serve(reader, writer):
            port = writer.get_extra_info("sockname")[1]
            try:
                while True:
                    head = await reader.readuntil(b"\r\n\r\n")
                    length = next(
                        (
                            int(line[15:])
                            for line in head.split(b"\r\n")
                            if line.lower().startswith(b"content-length:")
                        ),
                        0,
                    )
                    await reader.readexactly(length)
                    if head.startswith(b"GET /workers"):
                        body = json.dumps(worker_list(urls)).encode()
                    else:
                        hits[f"http://127.0.0.1:{port}"] += 1
                        await asyncio.sleep(0.01)
                        body = b'{"ok": true}'
                    writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body))
                    await writer.drain()
            except (asyncio.IncompleteReadError, ConnectionError):
                pass
            finally:
                writer.close()

        servers = [await asyncio.start_server(serve, "127.0.0.1", 0) for _ in range(5)]
        router, engines = servers[0], servers[1:]
        urls = sorted(f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}" for server in engines)
        base = types.SimpleNamespace(
            rollout_backend="sglang-omni",
            rollout_num_gpus=4,
            rollout_num_gpus_per_engine=1,
            sglang_server_concurrency=8,
            use_distributed_post=False,
            omni_router_policy="least_request",
            omni_request_timeout_s=10.0,
            omni_cancel_drain_timeout_s=1.0,
        )
        http_utils.init_http_client(base)
        router_url = f"http://127.0.0.1:{router.sockets[0].getsockname()[1]}"
        try:
            workers = [
                types.SimpleNamespace(**vars(base), rollout_worker_index=i, rollout_worker_count=2) for i in (0, 1)
            ]
            await asyncio.gather(
                *(
                    endpoints.post_to_engine(worker, router_url, "student", "/generate", {})
                    for worker in workers
                    for _ in range(16)
                )
            )
        finally:
            await http_utils._http_client.aclose()
            for server in servers:
                server.close()
        return urls, hits

    monkeypatch.setattr(http_utils, "_http_client", None)
    urls, hits = asyncio.run(exercise())
    assert {url: hits[url] for url in urls} == dict.fromkeys(urls, 8)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
