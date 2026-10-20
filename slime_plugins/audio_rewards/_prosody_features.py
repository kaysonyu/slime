"""Frozen numerical core from MOSS-TTS-Data's prosody-18d-cpu-v3 operator.

Only decoding and waveform feature functions are included; see PROSODY_SOURCES.md.
Imported lazily by prosody.py so other reward components need no DSP dependencies.
"""

import io
import math

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter1d, median_filter
from scipy.signal import find_peaks

SAMPLE_RATE = 16_000
FRAME_LENGTH = 400
HOP_LENGTH = 160


def decode_audio(raw: bytes, max_audio_sec: float) -> np.ndarray:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required for audio decoding") from exc
    max_samples = int(max_audio_sec * SAMPLE_RATE)
    if max_samples <= 0:
        raise ValueError("max_audio_sec must be positive")
    chunks: list[np.ndarray] = []
    sample_count = 0
    container = av.open(io.BytesIO(raw), mode="r")
    try:
        streams = [stream for stream in container.streams if stream.type == "audio"]
        if not streams:
            raise ValueError("audio stream not found")
        resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
        for frame in container.decode(streams[0]):
            converted = resampler.resample(frame)
            converted_frames = converted if isinstance(converted, list) else [converted]
            for item in converted_frames:
                if item is None:
                    continue
                array = np.asarray(item.to_ndarray(), dtype=np.float32)
                array = array[0] if array.ndim == 2 else array.reshape(-1)
                if len(array):
                    chunks.append(array)
                    sample_count += len(array)
            if sample_count >= max_samples:
                break
    finally:
        container.close()
    if not chunks:
        raise ValueError("decoded audio is empty")
    waveform = np.concatenate(chunks)[:max_samples].astype(np.float32, copy=False)
    if not np.isfinite(waveform).all():
        raise ValueError("decoded waveform contains NaN/Inf")
    return waveform


def _runs(mask: np.ndarray, value: bool = True) -> list[tuple[int, int]]:
    target = np.asarray(mask, dtype=bool) == value
    changes = np.diff(np.pad(target.astype(np.int8), (1, 1)))
    return list(
        zip(
            np.flatnonzero(changes == 1).tolist(),
            np.flatnonzero(changes == -1).tolist(),
            strict=True,
        )
    )


def _finite_stats(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan, math.nan, math.nan
    return (
        float(np.median(values)),
        float(np.std(values)),
        float(np.quantile(values, 0.9) - np.quantile(values, 0.1)),
        float(np.mean(np.abs(np.diff(values)))) if len(values) > 1 else 0.0,
    )


def _tail_track_delta(track: np.ndarray, valid: np.ndarray, *, max_frames: int = 100) -> float:
    indices = np.flatnonzero(valid & np.isfinite(track))
    if not len(indices):
        return math.nan
    end = int(indices[-1])
    indices = indices[indices >= max(0, end - max_frames + 1)]
    span_sec = (indices[-1] - indices[0]) * HOP_LENGTH / SAMPLE_RATE
    if len(indices) < 20 or span_sec < 0.30:
        return math.nan
    values = track[indices]
    edge = max(5, len(values) // 5)
    return float(np.median(values[-edge:]) - np.median(values[:edge]))


def waveform_features(waveform: np.ndarray) -> dict[str, float | int]:
    """Extract the frozen 18-d vector and lightweight diagnostics."""
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa is required for pitch extraction") from exc

    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    duration = float(len(waveform) / SAMPLE_RATE)
    if len(waveform) < FRAME_LENGTH:
        raise ValueError(f"audio too short: {duration:.3f}s")
    if not np.isfinite(waveform).all():
        raise ValueError("waveform contains NaN/Inf")

    frames = np.lib.stride_tricks.sliding_window_view(waveform, FRAME_LENGTH)[::HOP_LENGTH]
    energy_db = 20.0 * np.log10(np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12) + 1e-10)
    n_frames = len(energy_db)
    p20, p95 = np.quantile(energy_db, [0.20, 0.95])
    vad_threshold = float(min(p95 - 12.0, max(p95 - 35.0, p20 + 6.0)))
    speech = binary_closing(energy_db >= vad_threshold, structure=np.ones(5, dtype=bool))
    speech = binary_opening(speech, structure=np.ones(3, dtype=bool))
    if int(speech.sum()) < 10:
        raise ValueError("insufficient speech frames")

    spectrum = np.abs(np.fft.rfft(frames * np.hanning(FRAME_LENGTH), axis=1)).T + 1e-10
    flatness = np.exp(np.mean(np.log(spectrum), axis=0)) / np.mean(spectrum, axis=0)
    f0 = librosa.yin(
        waveform,
        fmin=55.0,
        fmax=500.0,
        sr=SAMPLE_RATE,
        frame_length=2048,
        hop_length=HOP_LENGTH,
    )
    if len(f0) != n_frames:
        f0 = np.interp(
            np.linspace(0.0, 1.0, n_frames),
            np.linspace(0.0, 1.0, len(f0)),
            f0,
        )
    pitch_valid = speech & (flatness < 0.25) & np.isfinite(f0) & (f0 >= 55.0) & (f0 <= 500.0)
    if int(pitch_valid.sum()) >= 9:
        log_f0 = np.log2(np.clip(f0, 55.0, 500.0)) * 12.0
        filled = log_f0.copy()
        filled[~pitch_valid] = np.nanmedian(log_f0[pitch_valid])
        local_median = median_filter(filled, size=9, mode="nearest")
        pitch_valid &= np.abs(log_f0 - local_median) <= 7.0
    else:
        log_f0 = np.full(n_frames, np.nan)

    if int(pitch_valid.sum()) >= 20:
        f0_st = log_f0[pitch_valid]
        f0_centered = f0_st - np.median(f0_st)
        _, f0_std_st, f0_range_st, f0_mad_delta_st = _finite_stats(f0_centered)
        pitch_track_centered = log_f0 - np.median(f0_st)
        f0_tail_delta_st = _tail_track_delta(pitch_track_centered, pitch_valid)
        f0_tail_level_st = float(np.median(f0_st[-min(len(f0_st), 20) :]) - np.median(f0_st))
        if np.isfinite(f0_tail_delta_st):
            f0_tail_delta_st = float(np.clip(f0_tail_delta_st, -18.0, 18.0))
    else:
        f0_std_st = f0_range_st = f0_mad_delta_st = math.nan
        f0_tail_delta_st = f0_tail_level_st = math.nan

    speech_energy = energy_db[speech]
    energy_median_db, energy_std_db, energy_range_db, energy_mad_delta_db = _finite_stats(speech_energy)
    energy_centered = energy_db - energy_median_db
    energy_tail_delta_db = _tail_track_delta(energy_centered, speech)

    first_speech = int(np.flatnonzero(speech)[0])
    last_speech = int(np.flatnonzero(speech)[-1])
    internal = speech[first_speech : last_speech + 1]
    pause_lengths = np.asarray(
        [(end - start) * HOP_LENGTH / SAMPLE_RATE for start, end in _runs(internal, False) if end - start >= 20],
        dtype=np.float64,
    )
    speech_runs = np.asarray(
        [(end - start) * HOP_LENGTH / SAMPLE_RATE for start, end in _runs(speech, True)],
        dtype=np.float64,
    )
    internal_duration = max(len(internal) * HOP_LENGTH / SAMPLE_RATE, 1e-6)

    smoothed_energy = gaussian_filter1d(energy_db.astype(np.float64), sigma=4.0)
    peaks, _ = find_peaks(
        smoothed_energy,
        distance=12,
        prominence=1.5,
        height=vad_threshold + 2.0,
    )
    peaks = peaks[speech[np.clip(peaks, 0, n_frames - 1)]]
    boundaries = np.linspace(0, n_frames, 5, dtype=int)
    quarter_rates = []
    for low, high in zip(boundaries[:-1], boundaries[1:], strict=True):
        seconds = max((high - low) * HOP_LENGTH / SAMPLE_RATE, 1e-6)
        quarter_rates.append(float(np.sum((peaks >= low) & (peaks < high)) / seconds))

    return {
        "f0_std_st": f0_std_st,
        "f0_range_st": f0_range_st,
        "f0_mad_delta_st": f0_mad_delta_st,
        "f0_tail_delta_st": f0_tail_delta_st,
        "f0_tail_level_st": f0_tail_level_st,
        "energy_std_db": energy_std_db,
        "energy_range_db": energy_range_db,
        "energy_mad_delta_db": energy_mad_delta_db,
        "energy_tail_delta_db": energy_tail_delta_db,
        "nucleus_rate_speech": float(len(peaks) / max(float(speech.sum()) * HOP_LENGTH / SAMPLE_RATE, 1e-6)),
        "nucleus_rate_quarter_std": float(np.std(quarter_rates)),
        "nucleus_rate_quarter_range": float(max(quarter_rates) - min(quarter_rates)),
        "internal_pause_fraction": (float(pause_lengths.sum() / internal_duration) if len(pause_lengths) else 0.0),
        "pause_rate_per_sec": float(len(pause_lengths) / max(duration, 1e-6)),
        "pause_mean_sec": float(np.mean(pause_lengths)) if len(pause_lengths) else 0.0,
        "pause_p90_sec": (float(np.quantile(pause_lengths, 0.9)) if len(pause_lengths) else 0.0),
        "speech_run_mean_sec": float(np.mean(speech_runs)) if len(speech_runs) else 0.0,
        "speech_run_std_sec": float(np.std(speech_runs)) if len(speech_runs) else 0.0,
        "speech_fraction": float(np.mean(speech)),
        "f0_voiced_fraction": float(np.mean(pitch_valid)),
        "pause_count": int(len(pause_lengths)),
        "nucleus_count": int(len(peaks)),
        "frame_count": int(n_frames),
        "vad_threshold_dbfs": vad_threshold,
        "decoded_peak_abs": float(np.max(np.abs(waveform))),
    }
