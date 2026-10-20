"""Real HTTP and admission invariants for the simplified reward request budgets."""

import asyncio
import json
import os
import sys

import httpx
import pytest

from slime_plugins.audio_rewards.client import ServiceClient
from slime_plugins.audio_rewards.concurrency import worker_budget
from slime_plugins.audio_rewards.transport import RewardHTTPClient

NUM_GPUS = 0


@pytest.mark.parametrize("workers", [1, 3, 8, 16])
@pytest.mark.parametrize("total", [64, 128, 256, 257])
def test_total_budget_is_conserved_when_worker_count_changes(total, workers):
    shares = [worker_budget(total, index, workers) for index in range(workers)]
    assert sum(shares) == total
    assert min(shares) > 0 and max(shares) - min(shares) <= 1


@pytest.mark.parametrize("arguments", [(0,), (True,), (None,), (8, 0, 0), (8, -1, 2), (8, 2, 2), (3, 0, 4)])
def test_invalid_or_unallocatable_budgets_fail_explicitly(arguments):
    with pytest.raises(ValueError):
        worker_budget(*arguments)


def test_old_per_worker_setting_is_not_silently_reinterpreted():
    with pytest.raises(ValueError, match="max_inflight was removed"):
        ServiceClient({"api": "openai_audio", "endpoint": "http://test", "model": "asr", "max_inflight": 32})


def test_client_defaults_match_the_single_replica_template():
    asr = ServiceClient({"api": "openai_audio", "endpoint": "http://test", "model": "asr"}, worker_count=8)
    embedding = ServiceClient({"api": "voxrl_embedding", "endpoint": "http://test"}, worker_count=8)
    assert (asr.total_concurrency, asr.concurrency) == (128, 16)
    assert (embedding.total_concurrency, embedding.concurrency) == (64, 8)


def test_worker_shares_bound_real_concurrent_calls(tmp_path):
    path = tmp_path / "one.wav"
    path.write_bytes(b"wave")

    async def run():
        release, full = asyncio.Event(), asyncio.Event()
        active = [0, 0]
        maximum = [0, 0]
        calls = 0

        async def respond(request):
            nonlocal calls
            worker = int(request.headers["X-Request-ID"].split("-")[0])
            active[worker] += 1
            maximum[worker] = max(maximum[worker], active[worker])
            calls += 1
            if sum(active) == 4:
                full.set()
            try:
                await release.wait()
                return httpx.Response(200, json={"text": "ok"})
            finally:
                active[worker] -= 1

        clients = [
            ServiceClient(
                {"api": "openai_audio", "endpoint": "http://test", "model": "asr", "concurrency": 4},
                worker_index=index,
                worker_count=2,
            )
            for index in range(2)
        ]
        for client in clients:
            client.http._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        tasks = [
            asyncio.create_task(client.transcribe(path, f"{index}-{i}"))
            for index, client in enumerate(clients)
            for i in range(6)
        ]
        try:
            await asyncio.wait_for(full.wait(), 5)
            assert calls == 4 and active == [2, 2]
        finally:
            release.set()
            await asyncio.gather(*tasks)
            await asyncio.gather(*(client.aclose() for client in clients))
        assert calls == 12 and maximum == [2, 2]
        assert all(client.http._client.is_closed for client in clients)

    asyncio.run(run())


def test_socket_concurrency_is_independent_of_generation_pool(tmp_path, monkeypatch):
    """Eight real TCP requests must start while the generation pool is capped at one."""
    from slime.utils import http_utils

    path = tmp_path / "one.wav"
    path.write_bytes(b"wave")

    async def run():
        release, full = asyncio.Event(), asyncio.Event()
        entered = 0

        async def serve(reader, writer):
            nonlocal entered
            try:
                header = (await reader.readuntil(b"\r\n\r\n")).decode()
                length = next(
                    int(line.split(":", 1)[1])
                    for line in header.splitlines()
                    if line.lower().startswith("content-length:")
                )
                await reader.readexactly(length)
                entered += 1
                if entered == 8:
                    full.set()
                await release.wait()
                body = b'{"text":"ok"}'
                writer.write(
                    f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body
                )
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(serve, "127.0.0.1", 0)
        async with server, httpx.AsyncClient(limits=httpx.Limits(max_connections=1), trust_env=False) as generation:
            monkeypatch.setattr(http_utils, "_http_client", generation)
            client = ServiceClient(
                {
                    "api": "openai_audio",
                    "endpoint": f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}",
                    "model": "asr",
                    "concurrency": 8,
                }
            )
            tasks = [asyncio.create_task(client.transcribe(path, str(i))) for i in range(8)]
            try:
                await asyncio.wait_for(full.wait(), 5)
                assert entered == 8
            finally:
                release.set()
                await asyncio.gather(*tasks)
                await client.aclose()

    asyncio.run(run())


def test_embedding_upload_is_prepared_only_after_admission(tmp_path):
    path = tmp_path / "one.wav"
    path.write_bytes(b"wave")

    async def run():
        client = ServiceClient(
            {"api": "voxrl_embedding", "endpoint": "http://test", "concurrency": 1, "audio_transport": "upload"}
        )
        client._info = {
            "protocol": "voxrl.embedding.v1",
            "embedding_version": "v1",
            "dimension": 1,
            "dtype": "float32",
            "normalized": True,
        }
        entered, release = asyncio.Event(), asyncio.Event()
        prepared = []
        original = client.audio

        def prepare(value):
            prepared.append(value)
            return original(value)

        async def respond(request):
            import json

            body = json.loads(await request.aread())
            entered.set()
            await release.wait()
            return httpx.Response(
                200,
                json={
                    **client._info,
                    "request_id": body["request_id"],
                    "results": [{"id": body["inputs"][0]["id"], "status": "success", "embedding": [1.0]}],
                },
            )

        client.audio = prepare
        client.http._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        first = asyncio.create_task(client.embed_audio(path, "first"))
        await entered.wait()
        queued = asyncio.create_task(client.embed_audio(path, "queued"))
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert prepared == [str(path)]
        release.set()
        await first
        await client.aclose()

    asyncio.run(run())


def test_ambiguous_audio_transport_is_not_silently_rewritten():
    async def run():
        client = ServiceClient({"api": "voxrl_embedding", "endpoint": "http://test"})
        client._info = {
            "protocol": "voxrl.embedding.v1",
            "embedding_version": "v1",
            "dimension": 1,
            "dtype": "float32",
            "normalized": True,
        }
        audio = {"path": "/server-only/path.wav", "data_b64": "invalid"}

        async def respond(request):
            import json

            assert json.loads(await request.aread())["inputs"][0]["audio"] == audio
            return httpx.Response(400, json={"error": "ambiguous audio transport"})

        client.http._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await client.embeddings("audio", [{"id": "a", "audio": audio}], "a")
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("status", [400, 401, 409, 413, 422])
def test_permanent_http_failures_are_not_retried(status):
    async def run():
        calls = []

        async def respond(request):
            calls.append(request)
            return httpx.Response(status, json={"error": "invalid"})

        client = RewardHTTPClient(4)
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await client.post("http://test/embed_audio", {}, attempts=3)
            assert len(calls) == 1
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("model", "overrides", "batch", "pending", "processes", "compile_model"),
    [
        ("redimnet2", [], 32, 256, 1, True),
        ("judge_clap", [], 16, 128, 4, False),
        ("redimnet2", ["--no-compile", "--processes", "2", "--max-batch", "8"], 8, 64, 2, False),
        (
            "judge_clap",
            ["--compile", "--processes", "1", "--max-batch", "12", "--max-pending", "17"],
            12,
            17,
            1,
            True,
        ),
    ],
)
def test_serving_cli_defaults_and_overrides(monkeypatch, model, overrides, batch, pending, processes, compile_model):
    from slime_plugins.audio_rewards.serving.server import main

    launches = []

    def run(app, **options):
        launches.append((json.loads(os.environ["PRISM_EMBEDDING_CONFIG"]), options))

    monkeypatch.setattr("uvicorn.run", run)
    monkeypatch.setenv("PRISM_EMBEDDING_CONFIG", "{}")
    paths = (
        ["--checkpoint", "unused.pt"]
        if model == "redimnet2"
        else ["--base", "unused-base", "--adapter", "unused-adapter", "--atom-stats", "unused.npz"]
    )
    monkeypatch.setattr(sys, "argv", ["server", "--model", model, *paths, *overrides])
    main()

    assert len(launches) == 1
    config, options = launches[0]
    assert config["max_batch"] == batch
    assert config["max_pending"] == pending
    assert config["processes"] == options["workers"] == processes
    assert config["compile"] is compile_model
    assert config["decode_workers"] == 4
    assert config["batching"] == "masked"
    assert config["preprocessing"] == "per_audio"
    assert config["max_cost"] == 16384
    assert config["batch_window_ms"] == (5 if model == "redimnet2" else 10)


def test_server_keeps_single_admission_reservation_until_cancelled_work_drains():
    import threading

    from tests.test_prism_embedding_services import Stub, request, wav_payload

    from slime_plugins.audio_rewards.serving.server import create_app

    async def run():
        release = threading.Event()
        started = threading.Event()
        backend = Stub()
        app = create_app(lambda: backend, max_pending=2, max_batch=2, window_ms=0)
        async with app.router.lifespan_context(app):
            original = backend.encode_audio

            def blocked(waves):
                started.set()
                release.wait(timeout=5)
                return original(waves)

            app.state.audio.encode = blocked
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                pending = asyncio.create_task(
                    client.post(
                        "/embed_audio", json=request([{"id": str(i), "audio": wav_payload()} for i in range(2)])
                    )
                )
                try:
                    assert await asyncio.to_thread(started.wait, 5)
                    pending.cancel()
                    await asyncio.sleep(0)
                    rejected = await client.post("/embed_text", json=request([{"id": "text", "text": "Speed: fast"}]))
                    assert rejected.status_code == 503
                    assert app.state.active == app.state.peak_active == 2
                    assert app.state.rejected_inputs == 1
                finally:
                    release.set()
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                assert app.state.active == 0
                response = await client.post("/embed_text", json=request([{"id": "text", "text": "Speed: fast"}]))
                assert response.status_code == 200

    asyncio.run(run())
