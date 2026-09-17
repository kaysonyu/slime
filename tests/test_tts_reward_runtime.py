import asyncio
import hashlib
import os
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from aiohttp import web

from slime.utils.types import Sample
from slime.rollout.rm_hub.config import RuntimeConfig, ServiceConfig
from slime.rollout.rm_hub.config import ComponentReward
from slime.rollout.rm_hub import sim
from slime.rollout.rm_hub import runtime as runtime_module
from slime.rollout.rm_hub.runtime import (
    RewardHttpRuntime,
    RewardServiceError,
    get_reward_http_runtime,
    reward_http_runtime_scope,
)

NUM_GPUS = 0


class NativeSimRuntime:
    def __init__(
        self,
        response: object | Callable[[Mapping[str, object]], object],
    ) -> None:
        self.config = ServiceConfig(
            endpoint="https://reward.example/v1/similarities",
            auth_token_env="INSPIRE_API_KEY",
            model=None,
            runtime=RuntimeConfig(timeout_seconds=1.0, concurrency=4, max_retries=0),
        )
        self.response = response
        self.requests: list[tuple[Mapping[str, object], Mapping[str, str] | None]] = []

    def service(self, name: str) -> ServiceConfig:
        assert name == "timbre_sim"
        return self.config

    async def post_json(
        self,
        name: str,
        payload: Mapping[str, object],
        *,
        request_headers: Mapping[str, str] | None = None,
    ) -> object:
        assert name == "timbre_sim"
        self.requests.append((payload, request_headers))
        if callable(self.response):
            return self.response(payload)
        return self.response


class PartiallyFailingRuntime(NativeSimRuntime):
    async def post_json(
        self,
        name: str,
        payload: Mapping[str, object],
        *,
        request_headers: Mapping[str, str] | None = None,
    ) -> object:
        self.requests.append((payload, request_headers))
        items = payload["items"]
        assert isinstance(items, list)
        if len(items) < sim.MAX_SIM_ITEMS_PER_REQUEST:
            raise RewardServiceError("timbre_sim", "retry_exhausted", "network unavailable")
        return _success_response(payload)


def _native_sample(
    tmp_path: Path,
    index: int,
    *,
    reference_name: str = "reference.wav",
) -> Sample:
    reference_path = tmp_path / reference_name
    candidate_path = tmp_path / f"candidate-{index}.wav"
    reference_path.touch(exist_ok=True)
    candidate_path.touch()
    return Sample(
        audio_path=str(candidate_path),
        index=index,
        metadata={
            "reference_audios": [
                {
                    "id": f"reference-{index}",
                    "path": str(reference_path),
                    "uses": ["timbre"],
                }
            ],
        },
    )


def _success_response(payload: Mapping[str, object]) -> dict[str, object]:
    items = payload["items"]
    assert isinstance(items, list)
    return {
        "results": [
            {
                "id": item["id"],
                "similarity": -0.25,
                "reward_similarity": 0.0,
                "error": None,
            }
            for item in items
            if isinstance(item, Mapping)
        ]
    }


def test_sim_batches_sixteen_items_and_reuses_reference_affinity_key(
    tmp_path: Path,
) -> None:
    samples = [_native_sample(tmp_path, index) for index in range(18)]
    runtime = NativeSimRuntime(_success_response)

    results = asyncio.run(sim.score_batch(samples, runtime))

    assert len(results) == 18
    assert all(isinstance(result, ComponentReward) for result in results)
    assert all(result.name == "timbre" for result in results if isinstance(result, ComponentReward))
    assert [len(request[0]["items"]) for request in runtime.requests] == [16, 2]
    headers = [request_headers for _, request_headers in runtime.requests]
    assert headers[0] == headers[1]
    reference_path = (tmp_path / "reference.wav").resolve()
    expected_key = hashlib.sha256(b"mosstts-sim-ref-v1\0" + os.fsencode(reference_path)).hexdigest()
    assert headers == [
        {"x-inspire-inference-key": expected_key},
        {"x-inspire-inference-key": expected_key},
    ]
    first_item = runtime.requests[0][0]["items"][0]
    assert first_item == {
        "id": "slime-0-0",
        "reference_path": str(reference_path),
        "candidate_path": str((tmp_path / "candidate-0.wav").resolve()),
    }


def test_sim_uses_distinct_affinity_keys_for_distinct_references(
    tmp_path: Path,
) -> None:
    samples = [
        _native_sample(tmp_path, 0, reference_name="reference-a.wav"),
        _native_sample(tmp_path, 1, reference_name="reference-b.wav"),
    ]
    runtime = NativeSimRuntime(_success_response)

    asyncio.run(sim.score_batch(samples, runtime))

    assert len(runtime.requests) == 2
    assert runtime.requests[0][1] != runtime.requests[1][1]


def test_sim_selects_timbre_reference_by_use_instead_of_array_position(tmp_path: Path) -> None:
    emotion_path = tmp_path / "emotion.wav"
    timbre_path = tmp_path / "timbre.wav"
    candidate_path = tmp_path / "candidate.wav"
    for path in (emotion_path, timbre_path, candidate_path):
        path.write_bytes(b"audio")
    sample = Sample(
        audio_path=str(candidate_path),
        index=7,
        metadata={
            "reference_audios": [
                {"id": "emotion", "path": str(emotion_path), "uses": ["emotion"]},
                {"id": "voice", "path": str(timbre_path), "uses": ["timbre"]},
            ],
        },
    )
    runtime = NativeSimRuntime(_success_response)

    asyncio.run(sim.score_batch([sample], runtime))

    item = runtime.requests[0][0]["items"][0]
    assert item["reference_path"] == str(timbre_path.resolve())


def test_sim_retry_exhaustion_invalidates_only_the_failed_chunk(
    tmp_path: Path,
) -> None:
    samples = [_native_sample(tmp_path, index) for index in range(18)]
    runtime = PartiallyFailingRuntime(_success_response)

    results = asyncio.run(sim.score_batch(samples, runtime))

    assert all(isinstance(result, ComponentReward) for result in results[:16])
    assert all(isinstance(result, sim.SimBatchFailure) for result in results[16:])
    assert runtime.requests[0][1] == runtime.requests[1][1]


def test_sim_maps_too_short_candidate_to_zero_reward(tmp_path: Path) -> None:
    sample = _native_sample(tmp_path, 7)
    response = {
        "results": [
            {
                "id": "slime-0-7",
                "similarity": None,
                "reward_similarity": None,
                "error": {
                    "code": "audio_too_short",
                    "message": "audio is shorter than the minimum duration",
                },
            }
        ]
    }

    results = asyncio.run(sim.score_batch([sample], NativeSimRuntime(response)))

    assert len(results) == 1
    assert results[0] == ComponentReward("timbre", 0.0, 0.0)


@pytest.mark.parametrize(
    "response,message",
    [
        ({}, "results list"),
        ({"results": []}, "ids do not match"),
        (
            {
                "results": [
                    {
                        "id": "slime-0-7",
                        "error": {
                            "code": "file_not_found",
                            "message": "/private/path.wav",
                        },
                    }
                ]
            },
            "file_not_found",
        ),
        (
            {
                "results": [
                    {
                        "id": "slime-0-7",
                        "similarity": float("nan"),
                        "reward_similarity": 0.5,
                        "error": None,
                    }
                ]
            },
            "finite number",
        ),
        (
            {
                "results": [
                    {
                        "id": "slime-0-7",
                        "similarity": 0.5,
                        "reward_similarity": 1.25,
                        "error": None,
                    }
                ]
            },
            r"within \[0, 1\]",
        ),
    ],
)
def test_sim_rejects_invalid_service_response(tmp_path: Path, response: object, message: str) -> None:
    sample = _native_sample(tmp_path, 7)

    with pytest.raises(RewardServiceError, match=message) as caught:
        asyncio.run(sim.score_batch([sample], NativeSimRuntime(response)))

    assert "/private/path.wav" not in str(caught.value)


def test_runtime_merges_affinity_header_without_allowing_auth_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INSPIRE_API_KEY", "test-token")
    config = ServiceConfig(
        endpoint="https://reward.example/v1/similarities",
        auth_token_env="INSPIRE_API_KEY",
        model=None,
        runtime=RuntimeConfig(timeout_seconds=1.0, concurrency=1, max_retries=0),
    )
    runtime = RewardHttpRuntime({"timbre_sim": config})

    assert runtime.headers("timbre_sim", {"x-inspire-inference-key": "stable-key"}) == {
        "Authorization": "Bearer test-token",
        "x-inspire-inference-key": "stable-key",
    }
    with pytest.raises(RewardServiceError, match="cannot override Authorization"):
        runtime.headers("timbre_sim", {"authorization": "Bearer attacker"})


def test_rollout_runtime_shares_service_limit_and_connection_pool_across_groups() -> None:
    async def scenario() -> tuple[int, int, int, dict[str, float], RewardHttpRuntime]:
        in_flight = 0
        max_in_flight = 0
        request_count = 0
        connection_ids: set[int] = set()

        async def handle(request: web.Request) -> web.Response:
            nonlocal in_flight, max_in_flight, request_count
            request_count += 1
            if request.transport is not None:
                connection_ids.add(id(request.transport))
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            try:
                await asyncio.sleep(0.02)
                return web.json_response({"ok": True})
            finally:
                in_flight -= 1

        app = web.Application()
        app.router.add_post("/score", handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        assert site._server is not None
        port = site._server.sockets[0].getsockname()[1]
        config = ServiceConfig(
            endpoint=f"http://127.0.0.1:{port}/score",
            auth_token_env=None,
            model=None,
            runtime=RuntimeConfig(timeout_seconds=1.0, concurrency=2, max_retries=0),
        )
        scoped_runtime: RewardHttpRuntime | None = None
        try:
            async with reward_http_runtime_scope() as runtime:
                scoped_runtime = runtime

                async def score_group(group_index: int) -> None:
                    group_runtime = get_reward_http_runtime()
                    assert group_runtime is runtime
                    group_runtime.add_services({"shared": config})
                    responses = await asyncio.gather(
                        *(group_runtime.post_json("shared", {"group": group_index}) for _ in range(4))
                    )
                    assert responses == [{"ok": True}] * 4

                await asyncio.gather(*(score_group(group_index) for group_index in range(3)))
                assert not runtime.closed
                metrics = runtime.collect_metrics()
        finally:
            await runner.cleanup()
        assert scoped_runtime is not None
        return max_in_flight, request_count, len(connection_ids), metrics, scoped_runtime

    max_in_flight, request_count, connection_count, metrics, runtime = asyncio.run(scenario())

    assert max_in_flight == 2
    assert request_count == 12
    assert 1 <= connection_count <= 2
    assert runtime.closed
    assert metrics["reward/shared/requests"] == 12.0
    assert metrics["reward/shared/attempts"] == 12.0
    assert metrics["reward/shared/failures"] == 0.0
    assert metrics["reward/shared/max_in_flight"] == 2.0
    assert metrics["reward/shared/queue_wait_seconds"] > 0
    assert metrics["reward/shared/request_seconds"] > metrics["reward/shared/service_seconds"] > 0
    assert 0 < metrics["reward/shared/max_request_seconds"] <= metrics["reward/shared/request_seconds"]
    assert 0 < metrics["reward/shared/max_service_seconds"] <= metrics["reward/shared/service_seconds"]
    assert set(metrics) == {
        "reward/shared/requests",
        "reward/shared/attempts",
        "reward/shared/failures",
        "reward/shared/max_in_flight",
        "reward/shared/queue_wait_seconds",
        "reward/shared/request_seconds",
        "reward/shared/max_request_seconds",
        "reward/shared/service_seconds",
        "reward/shared/max_service_seconds",
    }


def test_runtime_metrics_count_retries_and_logical_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> dict[str, float]:
        retry_attempts = 0

        async def handle(request: web.Request) -> web.Response:
            nonlocal retry_attempts
            payload = await request.json()
            if payload["kind"] == "retry":
                retry_attempts += 1
                if retry_attempts == 1:
                    return web.Response(status=503)
                return web.json_response({"ok": True})
            return web.Response(status=400)

        app = web.Application()
        app.router.add_post("/score", handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        assert site._server is not None
        port = site._server.sockets[0].getsockname()[1]
        config = ServiceConfig(
            endpoint=f"http://127.0.0.1:{port}/score",
            auth_token_env=None,
            model=None,
            runtime=RuntimeConfig(timeout_seconds=1.0, concurrency=1, max_retries=1),
        )
        try:
            async with RewardHttpRuntime({"judge": config}) as runtime:
                assert await runtime.post_json("judge", {"kind": "retry"}) == {"ok": True}
                monkeypatch.setattr(runtime_module.asyncio, "sleep", real_sleep)
                with pytest.raises(RewardServiceError, match="HTTP status 400") as caught:
                    await runtime.post_json("judge", {"kind": "fail"})
                assert caught.value.category == "request"
                metrics = runtime.collect_metrics()
        finally:
            await runner.cleanup()
        return metrics

    real_sleep = asyncio.sleep

    async def skip_retry_backoff(delay: float) -> None:
        assert delay == 1
        await real_sleep(0)

    monkeypatch.setattr(runtime_module.asyncio, "sleep", skip_retry_backoff)
    metrics = asyncio.run(scenario())

    assert metrics["reward/judge/requests"] == 2.0
    assert metrics["reward/judge/attempts"] == 3.0
    assert metrics["reward/judge/failures"] == 1.0
    assert metrics["reward/judge/max_in_flight"] == 1.0
    assert metrics["reward/judge/request_seconds"] >= metrics["reward/judge/service_seconds"] > 0


def test_json_payload_factory_waits_for_service_permit() -> None:
    async def scenario() -> tuple[int, int]:
        first_request_started = asyncio.Event()
        release_first_request = asyncio.Event()
        second_task_started = asyncio.Event()
        payload_builds = 0

        async def handle(request: web.Request) -> web.Response:
            payload = await request.json()
            if payload["request"] == 1:
                first_request_started.set()
                await release_first_request.wait()
            return web.json_response({"ok": True})

        app = web.Application()
        app.router.add_post("/score", handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        assert site._server is not None
        port = site._server.sockets[0].getsockname()[1]
        config = ServiceConfig(
            endpoint=f"http://127.0.0.1:{port}/score",
            auth_token_env=None,
            model=None,
            runtime=RuntimeConfig(timeout_seconds=1.0, concurrency=1, max_retries=0),
        )

        def build_payload(request_index: int) -> Mapping[str, object]:
            nonlocal payload_builds
            payload_builds += 1
            return {"request": request_index}

        try:
            async with RewardHttpRuntime({"shared": config}) as runtime:
                first = asyncio.create_task(runtime.post_json_factory("shared", lambda: build_payload(1)))
                await first_request_started.wait()

                async def send_second() -> object:
                    second_task_started.set()
                    return await runtime.post_json_factory("shared", lambda: build_payload(2))

                second = asyncio.create_task(send_second())
                await second_task_started.wait()
                await asyncio.sleep(0)
                builds_while_first_held_permit = payload_builds
                release_first_request.set()
                assert await asyncio.gather(first, second) == [{"ok": True}, {"ok": True}]
        finally:
            release_first_request.set()
            await runner.cleanup()
        return builds_while_first_held_permit, payload_builds

    builds_while_first_held_permit, total_builds = asyncio.run(scenario())

    assert builds_while_first_held_permit == 1
    assert total_builds == 2


def test_rollout_runtime_scope_closes_on_cancellation() -> None:
    runtimes: list[RewardHttpRuntime] = []

    async def scenario() -> None:
        async with reward_http_runtime_scope() as runtime:
            runtimes.append(runtime)
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(scenario())

    assert len(runtimes) == 1
    assert runtimes[0].closed
    assert runtimes[0].collect_metrics() == {}
    with pytest.raises(RuntimeError, match="rollout entry point"):
        get_reward_http_runtime()


def test_rollout_runtime_rejects_service_configuration_changes() -> None:
    config = ServiceConfig(
        endpoint="https://reward.example/score",
        auth_token_env=None,
        model=None,
        runtime=RuntimeConfig(timeout_seconds=1.0, concurrency=2, max_retries=0),
    )

    async def scenario() -> None:
        async with reward_http_runtime_scope() as runtime:
            runtime.add_services({"shared": config})
            with pytest.raises(RewardServiceError, match="changed within one rollout") as caught:
                runtime.add_services({"shared": replace(config, endpoint="https://other.example/score")})
            assert caught.value.category == "configuration"
            assert not caught.value.failure.retryable

    asyncio.run(scenario())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
