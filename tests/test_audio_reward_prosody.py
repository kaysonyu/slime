"""Numerical parity, waveform edge cases and owned CPU scoring work."""

import asyncio
import io
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from slime_plugins.audio_rewards import prosody
from slime_plugins.audio_rewards.scoring import ComponentScorers

NUM_GPUS = 0


@pytest.fixture
def local_prosody_pool(monkeypatch):
    """Keep monkeypatched feature functions in-process for orchestration unit tests.

    Actual spawn/IPC/process lifecycle is tested in test_audio_reward_prosody_processes.
    """
    monkeypatch.setattr(
        prosody, "ProcessPoolExecutor", lambda *, max_workers, **kwargs: ThreadPoolExecutor(max_workers)
    )


def vector(z=0, language="en"):
    return {
        name: value["median"] + z * value["scale_iqr_div_1_349"]
        for name, value in prosody._stats()[0][language].items()
    }


def test_identity_symmetry_and_delta_clipping():
    a, b = vector(100), vector(102)
    assert prosody.pair_diff(a, a, "English")[0] == 0
    assert prosody.pair_diff(a, b, "en")[0] == pytest.approx(2)
    assert prosody.pair_diff(b, a, "en")[0] == pytest.approx(2)
    assert prosody.pair_diff(vector(0), vector(50), "en")[0] == 10


def test_equal_group_and_block_weights_not_flat_feature_rmse():
    a, b = vector(), vector()
    scales = prosody._stats()[0]["en"]
    b["f0_std_st"] += 6 * scales["f0_std_st"]["scale_iqr_div_1_349"]
    raw, groups = prosody.pair_diff(a, b, "en")
    # One dimension in a two-feature block; three f0 blocks; four final groups.
    assert groups == pytest.approx({"f0": math.sqrt(6), "energy": 0, "rate": 0, "pause": 0})
    assert raw == pytest.approx(math.sqrt(1.5))
    with pytest.raises(ValueError, match="finite"):
        prosody.pair_diff(a, {**b, "f0_std_st": float("nan")}, "en")


def synthetic_audio(variation=0, sr=16000, stereo=False):
    sf = pytest.importorskip("soundfile")
    t = np.arange(sr * 4) / sr
    f0 = 150 + (20 + variation * 15) * np.sin(2 * np.pi * (0.6 + 0.2 * variation) * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    envelope = 0.1 + 0.2 * (1 + np.sin(2 * np.pi * (3 + variation) * t)) / 2
    envelope[((t > 1.1) & (t < 1.4)) | ((t > 2.5) & (t < 2.8 + 0.1 * variation))] = 0
    wav = (envelope * (np.sin(phase) + 0.3 * np.sin(2 * phase))).astype(np.float32)
    if stereo:
        wav = np.stack([wav, wav], axis=1)
    stream = io.BytesIO()
    sf.write(stream, wav, sr, format="WAV", subtype="FLOAT")
    return stream.getvalue()


@pytest.mark.parametrize("language", ["zh", "en"])
def test_waveform_distance_matches_frozen_benchmark(language):
    pytest.importorskip("librosa")
    # Computed independently with the supplied bench_triplet prosody_diff.py,
    # including its original prosody_operator.py and original robust statistics.
    expected = json.loads((Path(__file__).parent / "fixtures/prosody_golden.json").read_text())[language]
    a, b = [prosody.extract_features(synthetic_audio(i)) for i in (0, 1)]
    raw, groups = prosody.pair_diff(a, b, language)
    assert raw == pytest.approx(expected["prosody_diff"], abs=1e-6)
    assert groups == pytest.approx(expected["groups"], abs=1e-6)


def test_stereo_resampling_and_decoder_fallback(monkeypatch):
    pytest.importorskip("librosa")
    from slime_plugins.audio_rewards import _prosody_features

    raw = synthetic_audio(sr=24000, stereo=True)
    primary = prosody.extract_features(raw)
    assert all(math.isfinite(x) for x in primary.values())
    monkeypatch.setattr(_prosody_features, "decode_audio", lambda *_: (_ for _ in ()).throw(ValueError("fallback")))
    fallback = prosody.extract_features(raw)
    assert all(math.isfinite(x) for x in fallback.values())
    # Both paths preserve the same signal, but use different resamplers.
    assert prosody.pair_diff(primary, fallback, "en")[0] < 0.1


@pytest.mark.parametrize(
    "samples,reason",
    [
        (np.zeros(100, dtype=np.float32), "too_short"),
        (np.zeros(16000, dtype=np.float32), "near_silent"),
        (np.random.default_rng(0).normal(0, 0.1, 32000).astype(np.float32), "nonfinite_features"),
    ],
)
def test_unscorable_waveforms(samples, reason):
    pytest.importorskip("librosa")
    sf = pytest.importorskip("soundfile")
    stream = io.BytesIO()
    sf.write(stream, samples, 16000, format="WAV", subtype="FLOAT")
    with pytest.raises(prosody.InvalidProsodyAudio, match=reason):
        prosody.extract_features(stream.getvalue())


def test_broken_audio_is_not_silently_a_zero_reward():
    pytest.importorskip("librosa")
    with pytest.raises(ValueError, match="decode"):
        prosody.extract_features(b"not audio")


@pytest.mark.parametrize("bad_side", ["candidate", "reference"])
def test_invalid_candidate_zero_but_invalid_reference_is_error(tmp_path, monkeypatch, bad_side, local_prosody_pool):
    candidate, reference = tmp_path / "candidate", tmp_path / "reference"
    candidate.write_bytes(b"candidate")
    reference.write_bytes(b"reference")

    def extract(raw):
        if raw.decode() == bad_side:
            raise prosody.InvalidProsodyAudio("nonfinite_features: f0_std_st")
        return vector()

    monkeypatch.setattr(prosody, "extract_features", extract)

    async def run():
        scorer = prosody.ProsodyScorer()
        try:
            return await scorer.score(candidate, reference, "en", "one", scorers=ComponentScorers())
        finally:
            await scorer.aclose()

    if bad_side == "reference":
        with pytest.raises(ValueError, match="unscorable prosody reference"):
            asyncio.run(run())
    else:
        result = asyncio.run(run())
        assert result["status"] == "invalid_audio" and result["raw"] is None and result["transformed"] == 0
        json.dumps(result, allow_nan=False)


def test_cache_invalidates_when_reference_contents_change(tmp_path, monkeypatch, local_prosody_pool):
    candidate, reference = tmp_path / "candidate", tmp_path / "reference"
    candidate.write_bytes(b"0")
    reference.write_bytes(b"1")
    calls = []

    def extract(raw):
        calls.append(raw)
        return vector(int(raw))

    monkeypatch.setattr(prosody, "extract_features", extract)

    async def run():
        scorer = prosody.ProsodyScorer()
        try:
            values = []
            for payload in (b"1", b"1", b"2"):
                reference.write_bytes(payload)
                values.append(await scorer.score(candidate, reference, "en", "one", scorers=ComponentScorers()))
            assert [r["raw"] for r in values] == pytest.approx([1, 1, 2])
            assert [r["transformed"] for r in values] == pytest.approx([0.5, 0.5, 1 / 3])
        finally:
            await scorer.aclose()

    asyncio.run(run())
    assert calls.count(b"1") == calls.count(b"2") == 1


@pytest.mark.parametrize("close", [False, True])
def test_cancellation_and_shutdown_drain_cpu_readers(tmp_path, monkeypatch, close, local_prosody_pool):
    path = tmp_path / "audio"
    path.write_bytes(b"audio")
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def extract(raw):
        started.set()
        assert release.wait(5), "test did not release CPU reader"
        finished.set()
        return vector()

    monkeypatch.setattr(prosody, "extract_features", extract)

    async def run():
        scorer = prosody.ProsodyScorer({"processes": 1})
        task = asyncio.create_task(scorer.score(path, path, "en", "one", scorers=ComponentScorers()))
        closing = None
        try:
            assert await asyncio.to_thread(started.wait, 5)
            if close:
                closing = asyncio.create_task(scorer.aclose())
            else:
                task.cancel()
            for _ in range(10):
                await asyncio.sleep(0)
            assert not task.done() and not finished.is_set()
        finally:
            release.set()
            if closing is not None:
                await closing
            with pytest.raises(asyncio.CancelledError):
                await task
            await scorer.aclose()
        assert finished.is_set() and not scorer._operations
        with pytest.raises(RuntimeError, match="closed"):
            await scorer.score(path, path, "en", "two", scorers=ComponentScorers())

    asyncio.run(run())


@pytest.mark.parametrize(
    "options",
    [
        {"processes": 0},
        {"processes": True},
        {"processes": 1.5},
        {"workers": 0},
        {"workers": 1.5},
        {"workers": True},
        {"processes": 8, "workers": 2},
        {"endpoint": "unused"},
        [],
    ],
)
def test_invalid_local_configuration(options):
    with pytest.raises(ValueError, match="prosody"):
        prosody.ProsodyScorer(options)
