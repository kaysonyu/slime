"""Local CPU prosody distance, matching the frozen 18-dimensional benchmark."""

import asyncio
import hashlib
import io
import json
import math
import multiprocessing
import os
from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

from . import timing
from .client import _drain_on_cancel, _owned_component
from .scoring import component_result

SAMPLE_RATE = 16_000
MAX_AUDIO_SEC = 300.0
GROUPS = {
    "f0": (("f0_std_st", "f0_range_st"), ("f0_mad_delta_st",), ("f0_tail_delta_st", "f0_tail_level_st")),
    "energy": (("energy_std_db", "energy_range_db"), ("energy_mad_delta_db",), ("energy_tail_delta_db",)),
    "rate": (("nucleus_rate_speech",), ("nucleus_rate_quarter_std", "nucleus_rate_quarter_range")),
    "pause": (
        ("internal_pause_fraction",),
        ("pause_rate_per_sec",),
        ("pause_mean_sec", "pause_p90_sec"),
        ("speech_run_mean_sec", "speech_run_std_sec"),
    ),
}
FEATURES = tuple(name for blocks in GROUPS.values() for block in blocks for name in block)
LANG_ALIASES = {
    "zh": "zh",
    "chinese": "zh",
    "zh-cn": "zh",
    "mandarin": "zh",
    "en": "en",
    "english": "en",
}


def normalize_language(value):
    code = LANG_ALIASES.get(str(value or "").strip().lower())
    if code is None:
        raise ValueError(f"unsupported prosody language {value!r}: only zh/en statistics exist")
    return code


@lru_cache(maxsize=1)
def _stats():
    raw = Path(__file__).with_name("prosody_stats.json").read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def pair_diff(candidate, reference, language):
    """Robust z-score -> clipped pair delta -> block/group/equal-group RMSE."""
    import numpy as np

    stats = _stats()[0][normalize_language(language)]
    median = np.asarray([stats[name]["median"] for name in FEATURES], dtype=np.float64)
    scale = np.asarray([max(stats[name]["scale_iqr_div_1_349"], 1e-6) for name in FEATURES], dtype=np.float64)
    a = np.asarray([candidate[name] for name in FEATURES], dtype=np.float64)
    b = np.asarray([reference[name] for name in FEATURES], dtype=np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("prosody distance requires 18 finite features on both sides")
    # Clip the pair delta, never the individual z scores.
    delta = np.minimum(np.abs((a - median) / scale - (b - median) / scale), 10.0)
    index = {name: i for i, name in enumerate(FEATURES)}
    groups = {}
    for group, blocks in GROUPS.items():
        scores = [float(np.sqrt(np.mean(np.square(delta[[index[name] for name in block]])))) for block in blocks]
        groups[group] = float(np.sqrt(np.mean(np.square(scores))))
    return float(np.sqrt(np.mean(np.square(list(groups.values()))))), groups


class InvalidProsodyAudio(ValueError):
    """An audio signal with insufficient finite features for this metric."""


def extract_features(raw):
    """Decode bytes using the benchmark's PyAV path and soundfile fallback."""
    import numpy as np

    from ._prosody_features import decode_audio, waveform_features

    if not raw:
        raise ValueError("empty prosody audio file")
    try:
        wav = decode_audio(raw, MAX_AUDIO_SEC)
    except Exception as av_error:
        import librosa
        import soundfile as sf

        try:
            data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
            wav = data.mean(axis=1)
            if sr != SAMPLE_RATE:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
            wav = wav[: int(MAX_AUDIO_SEC * SAMPLE_RATE)].astype(np.float32, copy=False)
        except Exception as error:
            raise ValueError(f"failed to decode prosody audio: {av_error}; {error}") from error
    if len(wav) < 400:
        raise InvalidProsodyAudio("too_short")
    peak = float(np.max(np.abs(wav)))
    if not np.isfinite(wav).all():
        raise InvalidProsodyAudio("nonfinite_waveform")
    if peak < 1e-4:
        raise InvalidProsodyAudio("near_silent")
    try:
        features = waveform_features(wav)
    except ValueError as error:
        raise InvalidProsodyAudio(str(error)) from error
    bad = [name for name in FEATURES if not math.isfinite(float(features.get(name, math.nan)))]
    if bad:
        raise InvalidProsodyAudio("nonfinite_features: " + ",".join(bad))
    return {name: float(features[name]) for name in FEATURES}


def _initialize_cpu_worker():
    # Spawn avoids inheriting a training process's CUDA state and native locks.
    # Limit both libraries loaded by __main__ and libraries imported later.
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[name] = "1"
    from threadpoolctl import threadpool_limits

    from . import _prosody_features  # noqa: F401

    global _CPU_LIMITER
    _CPU_LIMITER = threadpool_limits(limits=1)


def _extract_audio(audio):
    """Picklable CPU job; candidates travel as paths, references as cached bytes."""
    raw = audio if isinstance(audio, bytes) else Path(audio).read_bytes()
    try:
        return extract_features(raw), None
    except InvalidProsodyAudio as error:
        return None, str(error)


def _read_reference(path):
    raw = Path(path).read_bytes()
    return raw, hashlib.sha256(raw).hexdigest()


class ProsodyScorer:
    """Own a bounded process pool and share reference features across its workers."""

    def __init__(self, options=None):
        options = {} if options is None else options
        if not isinstance(options, Mapping) or set(options) - {"processes", "workers"}:
            raise ValueError("prosody options accept processes (per rollout worker)")
        if "processes" in options and "workers" in options:
            raise ValueError("set only prosody.processes; workers is its legacy alias")
        self.processes = options.get("processes", options.get("workers", 8))
        if isinstance(self.processes, bool) or not isinstance(self.processes, int) or self.processes < 1:
            raise ValueError("prosody processes must be a positive integer")
        # Bound submitted CPU jobs and reference byte buffers independently.
        self._slots = asyncio.Semaphore(self.processes)
        self._reference_slots = asyncio.Semaphore(self.processes)
        self._references = OrderedDict()
        self._reference_pending = {}
        self._executor = None
        self._operations = set()
        self._close_task = None

    def _ensure_open(self):
        if self._close_task is not None:
            raise RuntimeError("prosody scorer is closed")

    def _pool(self):
        self._ensure_open()
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.processes,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_cpu_worker,
            )
        return self._executor

    async def aclose(self):
        if self._close_task is None:

            async def close():
                pending = self._operations | set(self._reference_pending.values())
                for task in pending:
                    task.cancel()
                try:
                    await asyncio.gather(*pending, return_exceptions=True)
                finally:
                    if self._executor is not None:
                        await _drain_on_cancel(
                            asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)
                        )
                    self._references.clear()

            self._close_task = asyncio.create_task(close())
        await _drain_on_cancel(self._close_task)

    async def _extract(self, audio):
        async with timing.acquire(self._slots, "client_queue_s"):
            with timing.phase("prepare_s"):
                # Shield/drain the actual process future: cancelling an asyncio
                # waiter cannot stop a child that is still reading its audio.
                return await _drain_on_cancel(
                    asyncio.get_running_loop().run_in_executor(self._pool(), _extract_audio, audio)
                )

    async def _fill_reference(self, raw, checksum, reference):
        features, reason = await self._extract(raw)
        if reason is not None:
            raise ValueError(f"unscorable prosody reference {reference}: {reason}")
        self._references[checksum] = features
        self._references.move_to_end(checksum)
        while len(self._references) > 256:
            self._references.popitem(last=False)
        return features

    async def _reference(self, reference):
        async with timing.acquire(self._reference_slots, "client_queue_s"):
            with timing.phase("cache_lookup_s"):
                raw, checksum = await _drain_on_cancel(asyncio.to_thread(_read_reference, reference))
            cached = self._references.get(checksum)
            if cached is not None:
                timing.count("reference_cache_hit")
                self._references.move_to_end(checksum)
                return cached, checksum
            pending = self._reference_pending.get(checksum)
            if pending is None:
                timing.count("reference_cache_miss")
                pending = asyncio.create_task(self._fill_reference(raw, checksum, reference))
                self._reference_pending[checksum] = pending

                def finished(task):
                    self._reference_pending.pop(checksum, None)
                    if not task.cancelled():
                        task.exception()

                pending.add_done_callback(finished)
            else:
                timing.count("reference_cache_hit")
            # No global lock around extraction: different references use different
            # processes, while all candidates sharing a checksum await one fill.
            with timing.phase("cache_wait_s"):
                features = await _drain_on_cancel(pending)
            return features, checksum

    @_owned_component
    async def score(self, audio_path, reference, language, request_id, *, scorers):
        language = normalize_language(language)
        target, checksum = await self._reference(reference)
        candidate, reason = await self._extract(str(audio_path))
        identity = {
            "service_task": "prosody_distance",
            "pipeline_version": "prosody-18d-cpu-v3",
            "stats_sha256": _stats()[1],
        }
        if reason is not None:
            record = component_result(None, 0.0, {"identity": identity}, reason=reason)
            record["status"] = "invalid_audio"
        else:
            raw, groups = pair_diff(candidate, target, language)
            record = await scorers.score("prosody", distance=raw, groups=groups, identity=identity)
        record["details"].update(
            language=language,
            reference_sha256=checksum,
            request_id=request_id,
        )
        return record
