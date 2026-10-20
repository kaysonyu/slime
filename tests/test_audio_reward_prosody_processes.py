"""Actual spawned CPU workers, shared references, admission and shutdown."""

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from slime_plugins.audio_rewards import prosody, timing
from slime_plugins.audio_rewards.scoring import ComponentScorers

NUM_GPUS = 0


def _probe(directory):
    """Hold both workers until the parent observes concurrent child processes."""
    from threadpoolctl import threadpool_info

    directory = Path(directory)
    (directory / f"entered-{os.getpid()}").touch()
    deadline = time.monotonic() + 30
    while not (directory / "release").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("parent did not release process probe")
        time.sleep(0.01)
    return {
        "pid": os.getpid(),
        "threads": [library["num_threads"] for library in threadpool_info()],
        "omp": os.environ["OMP_NUM_THREADS"],
        "openblas": os.environ["OPENBLAS_NUM_THREADS"],
    }


def _blocked_extract(audio):
    directory = audio.decode() if isinstance(audio, bytes) else Path(audio).read_text()
    _probe(directory)
    return dict.fromkeys(prosody.FEATURES, 0.0), None


def _crash_worker(audio):
    os._exit(17)


async def _wait_entered(directory, count):
    async def wait():
        while len(list(directory.glob("entered-*"))) < count:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), timeout=30)


def test_default_pool_is_lazy_and_legacy_option_is_a_process_alias():
    scorer = prosody.ProsodyScorer()
    assert scorer.processes == 8 and scorer._executor is None
    asyncio.run(scorer.aclose())
    assert scorer._executor is None
    assert prosody.ProsodyScorer({"workers": 3}).processes == 3


def test_spawned_workers_use_separate_pids_one_native_thread_and_are_joined(tmp_path):
    pytest.importorskip("threadpoolctl")

    async def run():
        scorer = prosody.ProsodyScorer({"processes": 2})
        pool = scorer._pool()
        assert pool._mp_context.get_start_method() == "spawn"
        futures = [asyncio.get_running_loop().run_in_executor(pool, _probe, str(tmp_path)) for _ in range(2)]
        children = []
        try:
            await _wait_entered(tmp_path, 2)
            children = list(pool._processes.values())
            assert len(children) == 2 and all(p.is_alive() for p in children)
            (tmp_path / "release").touch()
            results = await asyncio.gather(*futures)
            assert len({result["pid"] for result in results}) == 2
            assert all(result["pid"] != os.getpid() for result in results)
            assert all(result["threads"] and set(result["threads"]) == {1} for result in results)
            assert all(result["omp"] == result["openblas"] == "1" for result in results)
        finally:
            (tmp_path / "release").touch()
            await asyncio.gather(*futures, return_exceptions=True)
            await scorer.aclose()
        assert all(not child.is_alive() for child in children)

    asyncio.run(run())


def test_real_batch_matches_golden_and_computes_one_reference_across_processes(tmp_path):
    pytest.importorskip("librosa")
    from test_audio_reward_prosody import synthetic_audio

    candidate, reference, reference_copy = [tmp_path / name for name in ("candidate.wav", "ref.wav", "copy.wav")]
    candidate.write_bytes(synthetic_audio(0))
    reference.write_bytes(synthetic_audio(1))
    reference_copy.write_bytes(reference.read_bytes())
    expected = json.loads((Path(__file__).parent / "fixtures/prosody_golden.json").read_text())["en"]

    async def run():
        scorer = prosody.ProsodyScorer({"processes": 2})
        heartbeats = 0
        peak_submitted = 0

        async def heartbeat():
            nonlocal heartbeats, peak_submitted
            while True:
                heartbeats += 1
                if scorer._executor is not None:
                    peak_submitted = max(peak_submitted, len(scorer._executor._pending_work_items))
                await asyncio.sleep(0.01)

        ticker = asyncio.create_task(heartbeat())
        try:
            with timing.capture() as profile:
                records = await asyncio.gather(
                    *(
                        scorer.score(candidate, ref, "en", str(i), scorers=ComponentScorers())
                        for i, ref in enumerate([reference, reference_copy] * 8)
                    )
                )
            assert [r["details"]["request_id"] for r in records] == [str(i) for i in range(16)]
            assert all(r["raw"] == pytest.approx(expected["prosody_diff"], abs=1e-6) for r in records)
            assert all(r["details"]["groups"] == pytest.approx(expected["groups"], abs=1e-6) for r in records)
            assert profile["reference_cache_miss"] == 1
            assert profile["reference_cache_hit"] == 15
            assert len(scorer._executor._processes) == 2
            assert 0 < peak_submitted <= 2  # A large batch cannot flood the process executor's queue.
            assert heartbeats > 10  # CPU extraction leaves the event loop responsive.
        finally:
            ticker.cancel()
            await asyncio.gather(ticker, return_exceptions=True)
            await scorer.aclose()

    asyncio.run(run())


def test_different_references_extract_concurrently_without_a_global_lock(tmp_path, monkeypatch):
    pytest.importorskip("threadpoolctl")
    monkeypatch.setattr(prosody, "_extract_audio", _blocked_extract)
    directories = [tmp_path / str(i) for i in range(2)]
    paths = []
    for directory in directories:
        directory.mkdir()
        path = directory / "audio"
        path.write_text(str(directory))
        paths.append(path)

    async def run():
        scorer = prosody.ProsodyScorer({"processes": 2})
        tasks = [
            asyncio.create_task(scorer.score(path, path, "en", str(i), scorers=ComponentScorers()))
            for i, path in enumerate(paths)
        ]
        try:
            # Both reference jobs must enter before either is allowed to finish.
            await asyncio.gather(*(_wait_entered(directory, 1) for directory in directories))
        finally:
            for directory in directories:
                (directory / "release").touch()
            try:
                records = await asyncio.gather(*tasks)
            finally:
                await scorer.aclose()
        assert all(record["raw"] == 0 and record["transformed"] == 1 for record in records)

    asyncio.run(run())


@pytest.mark.parametrize("close", [False, True])
def test_running_process_readers_drain_on_cancel_and_close(tmp_path, monkeypatch, close):
    pytest.importorskip("threadpoolctl")
    monkeypatch.setattr(prosody, "_extract_audio", _blocked_extract)
    path = tmp_path / "audio"
    path.write_text(str(tmp_path))

    async def run():
        scorer = prosody.ProsodyScorer({"processes": 1})
        task = asyncio.create_task(scorer.score(path, path, "en", "one", scorers=ComponentScorers()))
        closing = None
        children = []
        try:
            await _wait_entered(tmp_path, 1)
            children = list(scorer._executor._processes.values())
            if close:
                closing = asyncio.create_task(scorer.aclose())
            else:
                task.cancel()
            for _ in range(10):
                await asyncio.sleep(0)
            assert not task.done()
            if closing is not None:
                closing.cancel()
                for _ in range(10):
                    await asyncio.sleep(0)
                assert not closing.done()
        finally:
            (tmp_path / "release").touch()
            if closing is not None:
                with pytest.raises(asyncio.CancelledError):
                    await closing
            with pytest.raises(asyncio.CancelledError):
                await task
            await scorer.aclose()
        assert not scorer._operations and not scorer._reference_pending
        assert all(not child.is_alive() for child in children)

    asyncio.run(run())


def test_worker_crash_propagates_and_pool_can_close(tmp_path, monkeypatch):
    from concurrent.futures.process import BrokenProcessPool

    pytest.importorskip("threadpoolctl")
    monkeypatch.setattr(prosody, "_extract_audio", _crash_worker)
    path = tmp_path / "audio"
    path.write_bytes(b"audio")

    async def run():
        scorer = prosody.ProsodyScorer({"processes": 2})
        try:
            with pytest.raises(BrokenProcessPool):
                await scorer.score(path, path, "en", "one", scorers=ComponentScorers())
        finally:
            await scorer.aclose()

    asyncio.run(run())
