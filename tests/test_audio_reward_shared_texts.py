"""Concurrent JudgeCLAP callers share fills without serializing unrelated queries."""

import asyncio
import json
from collections import Counter

import httpx
import pytest

from slime_plugins.audio_rewards.client import AudioRewardClient
from slime_plugins.audio_rewards.protocol import PROTOCOL

NUM_GPUS = 0

JUDGE_IDENTITY = {
    "protocol": PROTOCOL,
    "embedding_version": "judge-v1",
    "dimension": 1,
    "dtype": "float32",
    "normalized": False,
    "service_task": "judge_clap",
    "supported_categories": ["Speed"],
    "query_versions": ["v1"],
    "scoring": {"type": "siglip_margin_v1", "scale": 1.0, "bias": 0.0},
}


def attach_judge(reward, on_text):
    client = reward.service("judge_clap")
    client._info = dict(JUDGE_IDENTITY)

    async def respond(request):
        body = json.loads(await request.aread())
        if request.url.path == "/embed_text":
            await on_text([item["text"] for item in body["inputs"]])
        return httpx.Response(
            200,
            json={
                **JUDGE_IDENTITY,
                "request_id": body["request_id"],
                "results": [{"id": item["id"], "status": "success", "embedding": [1.0]} for item in body["inputs"]],
            },
        )

    client.http._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    return client


def judge_options(**kwargs):
    return {"api": "voxrl_embedding", "endpoint": "http://test", "concurrency": 32, "timeout_s": 5, **kwargs}


def score(reward, path, queries, request_id):
    return reward.judge_clap(path, queries, request_id, query_version="v1")


@pytest.mark.parametrize("failure", ["timeout", "error"])
def test_concurrent_text_failure_is_shared_and_later_call_can_retry(tmp_path, failure):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"audio")

    async def run():
        reward = AudioRewardClient({"judge_clap": judge_options(timeout_s=0.1)})
        calls = []
        release = asyncio.Event()
        failing = True

        async def on_text(texts):
            calls.append(texts)
            if failing:
                await release.wait()
                raise ValueError("text backend failed")

        attach_judge(reward, on_text)
        tasks = [asyncio.create_task(score(reward, path, ["Speed: slow"], str(i))) for i in range(8)]
        try:
            if failure == "error":
                # All callers have entered the cache before the backend fails.
                for _ in range(30):
                    await asyncio.sleep(0)
                release.set()
            outcomes = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 3)
            error = TimeoutError if failure == "timeout" else ValueError
            assert all(isinstance(outcome, error) for outcome in outcomes)
            assert calls == [["Speed: slow"]]
            failing = False
            result = await score(reward, path, ["Speed: slow"], "retry")
            assert result["status"] == "success"
            assert calls == [["Speed: slow"], ["Speed: slow"]]
        finally:
            await reward.aclose()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run())


def test_blocked_text_fill_does_not_block_cached_or_unrelated_queries(tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"audio")

    async def run():
        reward = AudioRewardClient({"judge_clap": judge_options()})
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def on_text(texts):
            calls.extend(texts)
            if "Speed: blocked" in texts:
                entered.set()
                await release.wait()

        attach_judge(reward, on_text)
        tasks = []
        try:
            await score(reward, path, ["Speed: cached"], "warmup")
            tasks.append(asyncio.create_task(score(reward, path, ["Speed: blocked"], "blocked")))
            await asyncio.wait_for(entered.wait(), 2)
            for query in ["Speed: cached", "Speed: unrelated"]:
                result = await asyncio.wait_for(score(reward, path, [query], query), 1)
                assert result["status"] == "success"
            assert not tasks[0].done()
            assert Counter(calls) == Counter(["Speed: cached", "Speed: blocked", "Speed: unrelated"])
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            await reward.aclose()

    asyncio.run(run())


def test_cancelled_waiter_preserves_shared_fill_and_overlapping_queries_are_deduplicated(tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"audio")

    async def run():
        reward = AudioRewardClient({"judge_clap": judge_options()})
        entered, release, unique_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls, cancelled = [], []

        async def on_text(texts):
            calls.extend(texts)
            if "Speed: unique" in texts:
                unique_entered.set()
            if "Speed: shared" in texts:
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancelled.extend(texts)
                    raise

        attach_judge(reward, on_text)
        first = asyncio.create_task(score(reward, path, ["Speed: shared"], "first"))
        tasks = [first]
        try:
            await asyncio.wait_for(entered.wait(), 2)
            second = asyncio.create_task(
                score(reward, path, ["Speed: shared", "Speed: unique", "Speed: shared"], "second")
            )
            tasks.append(second)
            await asyncio.wait_for(unique_entered.wait(), 1)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert not cancelled and not second.done()
            release.set()
            result = await second
            assert result["details"]["attribute_count"] == 3
            assert Counter(calls) == Counter(["Speed: shared", "Speed: unique"])
            await score(reward, path, ["Speed: shared", "Speed: unique"], "cached")
            assert len(calls) == 2
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            await reward.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("query_count", [130, 8193])
def test_text_batches_preserve_values_and_order_across_cache_eviction(tmp_path, query_count):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"audio")
    queries = [f"Speed: {i}" for i in range(query_count)]

    async def run():
        reward = AudioRewardClient({"judge_clap": judge_options()})
        calls = []

        async def on_text(texts):
            calls.append(texts)

        attach_judge(reward, on_text)
        try:
            result = await score(reward, path, queries + queries[:2], "many")
            assert result["details"]["queries"] == queries + queries[:2]
            assert result["details"]["attribute_count"] == query_count + 2
            assert result["raw"] == 0.5
            assert all(len(batch) <= 64 for batch in calls)
            assert Counter(text for batch in calls for text in batch) == Counter(queries)
            assert len(reward._text_vectors) == min(query_count, 8192)
        finally:
            await reward.aclose()

    asyncio.run(run())
