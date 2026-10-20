"""Cancellation must not outlive file ownership or release admission early."""

import asyncio
import json
import threading
from pathlib import Path

import httpx
import pytest

from slime_plugins.audio_rewards.client import AudioRewardClient, ServiceClient
from slime_plugins.audio_rewards.protocol import PROTOCOL

NUM_GPUS = 0

IDENTITY = {
    "protocol": PROTOCOL,
    "embedding_version": "test-v1",
    "dimension": 1,
    "dtype": "float32",
    "normalized": True,
}


def service_options(api):
    return {
        "api": api,
        "endpoint": "http://test",
        "model": "asr",
        "audio_transport": "upload",
        "concurrency": 1,
        "timeout_s": 5,
    }


async def request(client, path, name):
    if client.api == "openai_audio":
        return await client.transcribe(path, name)
    return await client.embed_audio(path, name)


async def settle():
    # Let cancellation and shield/wait_for callbacks run without releasing I/O.
    for _ in range(10):
        await asyncio.sleep(0)


class BlockedRead:
    def __init__(self, monkeypatch, path, *, fail=False):
        self.entered = asyncio.Event()
        self.release = threading.Event()
        self.active = self.peak = self.calls = 0
        loop = asyncio.get_running_loop()
        original = Path.read_bytes

        def read(candidate):
            if candidate != path:
                return original(candidate)
            self.calls += 1
            first = self.calls == 1
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                if first:
                    loop.call_soon_threadsafe(self.entered.set)
                    if not self.release.wait(5):
                        raise TimeoutError("test did not release the file reader")
                    if fail:
                        raise OSError("read failed after cancellation")
                return original(candidate)
            finally:
                self.active -= 1

        monkeypatch.setattr(Path, "read_bytes", read)


def attach_http(client, calls):
    async def respond(request):
        calls.append(request.url.path)
        if request.method == "GET":
            return httpx.Response(200, json=IDENTITY)
        if client.api == "openai_audio":
            return httpx.Response(200, json={"text": "ok"})
        body = json.loads(await request.aread())
        return httpx.Response(
            200,
            json={
                **IDENTITY,
                "request_id": body["request_id"],
                "results": [{"id": item["id"], "status": "success", "embedding": [1.0]} for item in body["inputs"]],
            },
        )

    client._info = dict(IDENTITY)
    client.http._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))


@pytest.mark.parametrize("api", ["openai_audio", "voxrl_embedding"])
@pytest.mark.parametrize("interrupt", ["cancel", "timeout"])
@pytest.mark.parametrize("read_fails", [False, True])
def test_interrupted_upload_drains_reader_before_returning_or_admitting_next(
    tmp_path, monkeypatch, api, interrupt, read_fails
):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"audio")

    async def run():
        client = ServiceClient(service_options(api))
        calls = []
        attach_http(client, calls)
        reader = BlockedRead(monkeypatch, path, fail=read_fails)
        if interrupt == "timeout":
            client.timeout = 0.05
        first = asyncio.create_task(request(client, path, "first"))
        tasks = [first]
        try:
            await asyncio.wait_for(reader.entered.wait(), 2)
            if interrupt == "cancel":
                first.cancel()
                await settle()
                first.cancel()  # Repeated cancellation must also wait for I/O.
                await settle()
            else:
                await asyncio.wait({first}, timeout=0.1)
            assert not first.done()
            assert reader.active == 1 and client._slots.locked()
            client.timeout = 5
            second = asyncio.create_task(request(client, path, "second"))
            tasks.append(second)
            await settle()
            assert not second.done()
            assert reader.calls == 1 and not calls
            reader.release.set()
            with pytest.raises(asyncio.CancelledError if interrupt == "cancel" else TimeoutError):
                await first
            await second
            assert reader.peak == 1 and reader.active == 0
            assert len(calls) == 1  # The interrupted request never reaches HTTP.
            assert not client._slots.locked()
        finally:
            reader.release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            await client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("api", ["openai_audio", "voxrl_embedding"])
@pytest.mark.parametrize("owner", ["service", "reward"])
def test_close_drains_active_upload_cancels_waiters_and_rejects_new_work(tmp_path, monkeypatch, api, owner):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"audio")

    async def run():
        name = "wer" if api == "openai_audio" else "sim"
        reward = AudioRewardClient({name: service_options(api)})
        client = reward.service(name)
        close = client.aclose if owner == "service" else reward.aclose
        calls = []
        attach_http(client, calls)
        reader = BlockedRead(monkeypatch, path)
        active = asyncio.create_task(request(client, path, "active"))
        tasks = [active]
        try:
            await asyncio.wait_for(reader.entered.wait(), 2)
            queued = asyncio.create_task(request(client, path, "queued"))
            tasks.append(queued)
            await settle()
            closing = asyncio.create_task(close())
            tasks.append(closing)
            await settle()
            closing.cancel()
            await settle()
            closing.cancel()
            again = asyncio.create_task(close())
            tasks.append(again)
            await settle()
            assert not closing.done() and not again.done()
            assert reader.active == 1 and client._slots.locked()
            assert not client.http._client.is_closed
            with pytest.raises(RuntimeError, match="closed"):
                await request(client, path, "late")
            if owner == "reward":
                with pytest.raises(RuntimeError, match="closed"):
                    reward.service(name)
            reader.release.set()
            with pytest.raises(asyncio.CancelledError):
                await closing
            await again
            for task in (active, queued):
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert reader.calls == 1 and reader.active == 0 and not calls
            assert client.http._client.is_closed
            assert not client._slots.locked()
            await close()
        finally:
            reader.release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            await reward.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("route", ["info", "transcribe", "embeddings"])
def test_close_cancels_http_including_identity_queries(tmp_path, route):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"audio")

    async def run():
        api = "openai_audio" if route == "transcribe" else "voxrl_embedding"
        client = ServiceClient(service_options(api))
        if route == "embeddings":
            client._info = dict(IDENTITY)
        entered = asyncio.Event()
        finished = asyncio.Event()

        async def respond(request):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                finished.set()

        client.http._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        pending = asyncio.create_task(client.info() if route == "info" else request(client, path, "http"))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            await client.aclose()
            assert finished.is_set()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert client.http._client.is_closed
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            await client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("api", ["openai_audio", "voxrl_embedding"])
def test_read_error_releases_admission_and_preserves_original_error(tmp_path, monkeypatch, api):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"audio")

    async def run():
        client = ServiceClient(service_options(api))
        calls = []
        attach_http(client, calls)
        reader = BlockedRead(monkeypatch, path, fail=True)
        reader.release.set()
        try:
            with pytest.raises(OSError, match="read failed"):
                await request(client, path, "failed")
            assert not calls and reader.active == 0
            assert not client._slots.locked()
            await request(client, path, "next")
            assert len(calls) == 1
        finally:
            await client.aclose()

    asyncio.run(run())


def test_reward_close_drains_shared_reference_upload(tmp_path, monkeypatch):
    candidate, reference = tmp_path / "candidate.wav", tmp_path / "reference.wav"
    candidate.write_bytes(b"candidate")
    reference.write_bytes(b"reference")

    async def run():
        reward = AudioRewardClient({"sim": service_options("voxrl_embedding")})
        client = reward.service("sim")
        attach_http(client, [])
        client._info["service_task"] = "speaker_embedding"
        reader = BlockedRead(monkeypatch, reference)
        score = asyncio.create_task(reward.sim(candidate, reference, "sim"))
        closing = None
        try:
            await asyncio.wait_for(reader.entered.wait(), 2)
            closing = asyncio.create_task(reward.aclose())
            await settle()
            assert not closing.done() and reader.active == 1
            reader.release.set()
            await closing
            with pytest.raises(asyncio.CancelledError):
                await score
            assert reader.active == 0
            assert not reward._reference_pending
            assert client.http._client.is_closed
        finally:
            reader.release.set()
            await asyncio.gather(score, *([closing] if closing else []), return_exceptions=True)
            await reward.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("stage", ["reference_digest", "reference_cache", "text_cache", "text_cache_without_waiter"])
def test_reward_close_drains_local_cache_io_even_when_close_is_cancelled(tmp_path, monkeypatch, stage):
    from test_audio_reward_shared_texts import attach_judge, judge_options, score

    from slime_plugins.audio_rewards import client as client_module

    candidate, reference = tmp_path / "candidate.wav", tmp_path / "reference.wav"
    candidate.write_bytes(b"candidate")
    reference.write_bytes(b"reference")

    async def run():
        entered, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()
        active = 0

        def block():
            nonlocal active
            active += 1
            try:
                loop.call_soon_threadsafe(entered.set)
                if not release.wait(5):
                    raise TimeoutError("test did not release local cache I/O")
            finally:
                active -= 1

        if stage == "reference_digest":
            original = client_module.hashlib.sha256

            def digest(*args, **kwargs):
                block()  # The reference file is already open in the reader thread.
                return original(*args, **kwargs)

            monkeypatch.setattr(client_module.hashlib, "sha256", digest)
        else:

            def read_cache(path, identity, keys):
                block()
                return {key: [1.0] for key in keys}

            monkeypatch.setattr(client_module, "read_embeddings", read_cache)

        if stage.startswith("text_cache"):
            reward = AudioRewardClient({"judge_clap": judge_options(embedding_cache="cache.sqlite")})

            async def on_text(texts):
                pytest.fail("all texts should be read from SQLite")

            client = attach_judge(reward, on_text)
            operation = score(reward, candidate, ["Speed: slow"], "judge")
        else:
            options = service_options("voxrl_embedding")
            if stage == "reference_cache":
                options["embedding_cache"] = "cache.sqlite"
            reward = AudioRewardClient({"sim": options})
            client = reward.service("sim")
            attach_http(client, [])
            client._info["service_task"] = "speaker_embedding"
            operation = reward.sim(candidate, reference, "sim")
        pending = asyncio.create_task(operation)
        tasks = [pending]
        try:
            await asyncio.wait_for(entered.wait(), 2)
            if stage == "text_cache_without_waiter":
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            closing = asyncio.create_task(reward.aclose())
            tasks.append(closing)
            await asyncio.wait({closing}, timeout=0.05)
            assert not closing.done() and active == 1
            closing.cancel()
            await settle()
            closing.cancel()
            again = asyncio.create_task(reward.aclose())
            tasks.append(again)
            await settle()
            assert not closing.done() and not again.done()
            with pytest.raises(RuntimeError, match="closed"):
                reward.service("judge_clap" if stage.startswith("text_cache") else "sim")
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await closing
            await again
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert active == 0
            assert client.http._client.is_closed
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            await reward.aclose()

    asyncio.run(run())
