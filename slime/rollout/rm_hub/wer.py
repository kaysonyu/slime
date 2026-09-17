"""Multilingual WER reward backed by a Qwen-ASR endpoint.

The ASR backend remains Qwen-ASR. Text normalization and scoring follow the
normalization and tokenization used by X-Voice's ``run_wer.py``.
"""

from __future__ import annotations

import math
import unicodedata
import wave
from pathlib import Path

import aiohttp
import regex
import zhconv

from slime.utils.types import Sample

from .config import ComponentReward
from .language import LanguageSpec, resolve_language
from .runtime import RewardHttpRuntime, RewardServiceError

MIN_ASR_AUDIO_SECONDS = 0.1


def _clean_special_chars(text: str) -> str:
    cleaned = "".join(character for character in text if not unicodedata.category(character).startswith(("P", "S")))
    return " ".join(cleaned.split())


def _normalize_for_metric(text: str, language: LanguageSpec) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = " ".join(normalized.split()).lower()
    if language.canonical in {"cantonese", "chinese"}:
        normalized = zhconv.convert(normalized, "zh-cn")
    normalized = _clean_special_chars(normalized)
    if language.token_mode == "character":
        normalized = " ".join(regex.findall(r"\X", normalized.replace(" ", "")))
    return normalized


def normalize_text(text: str) -> str:
    """Keep the historical helper behavior for callers outside the scorer."""

    return _clean_special_chars(unicodedata.normalize("NFKC", text).casefold()).replace(" ", "")


def graphemes(text: str) -> list[str]:
    return regex.findall(r"\X", normalize_text(text))


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_item in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def _tokens(text: str, language: LanguageSpec) -> list[str]:
    return _normalize_for_metric(text, language).split()


def wer(reference: str, hypothesis: str, language: str | LanguageSpec) -> float:
    """Compute token-level WER after the shared language normalization path."""
    spec = resolve_language(language) if isinstance(language, str) else language
    reference_tokens = _tokens(reference, spec)
    hypothesis_tokens = _tokens(hypothesis, spec)
    if not reference_tokens:
        return math.inf
    return edit_distance(reference_tokens, hypothesis_tokens) / len(reference_tokens)


def cer(reference: str, hypothesis: str) -> float:
    """Backward-compatible character error helper used by older tests/callers."""

    reference_items = graphemes(reference)
    return edit_distance(reference_items, graphemes(hypothesis)) / max(len(reference_items), 1)


def _reference_text(sample: Sample) -> str:
    if isinstance(sample.label, str) and sample.label.strip():
        return sample.label
    raise ValueError("WER reward requires a non-empty sample label.")


def _generated_audio(sample: Sample) -> Path:
    value = sample.audio_path
    if not isinstance(value, str) or not value:
        raise ValueError("WER reward requires generated audio metadata.")
    path = Path(value)
    if not path.is_file():
        raise ValueError("WER generated audio is not a readable file.")
    return path


def _audio_is_too_short_for_asr(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as stream:
            frame_count = stream.getnframes()
            sample_rate = stream.getframerate()
    except (EOFError, wave.Error) as error:
        raise ValueError("WER generated audio must be a readable PCM WAV file.") from error
    if frame_count <= 0 or sample_rate <= 0:
        raise ValueError("WER generated audio must contain valid WAV frames.")
    return frame_count < math.ceil(sample_rate * MIN_ASR_AUDIO_SECONDS)


def _record_score(
    sample: Sample,
    language: LanguageSpec,
    raw_wer: float,
    *,
    skipped_reason: str | None = None,
) -> ComponentReward:
    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata.update(
        {
            "wer_raw_wer": raw_wer,
            "wer": raw_wer,
            "wer_language": language.canonical,
            "wer_mosstts_language": language.mosstts,
            "wer_qwen_asr_language": language.qwen_asr,
            "wer_metric_type": "wer",
            "wer_metric_unit": language.token_mode,
        }
    )
    if skipped_reason is not None:
        sample.metadata["wer_skipped_reason"] = skipped_reason
    return ComponentReward("wer", 1.0 - min(raw_wer, 1.0), raw_wer)


def _chat_transcription(response: object) -> str:
    if not isinstance(response, dict):
        raise RewardServiceError("wer", "protocol", "ASR chat response is missing message content")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RewardServiceError("wer", "protocol", "ASR chat response is missing message content")
    message = choices[0].get("message")
    if not isinstance(message, dict) or "content" not in message:
        raise RewardServiceError("wer", "protocol", "ASR chat response is missing message content")
    content = message["content"]
    if not isinstance(content, str):
        raise RewardServiceError("wer", "protocol", "ASR chat response requires string message content")
    if "<asr_text>" in content:
        content = content.rsplit("<asr_text>", 1)[1]
    return content.strip()


async def score(sample: Sample, runtime: RewardHttpRuntime) -> ComponentReward:
    """Transcribe generated audio and map WER to a bounded reward."""
    config = runtime.service("wer")
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    language = resolve_language(metadata.get("language"))
    reference_text = _reference_text(sample)
    if sample.audio_path is None and sample.trajectory is not None and sample.trajectory.num_frames == 0:
        return _record_score(sample, language, 1.0, skipped_reason="empty_generation")
    audio_path = _generated_audio(sample)

    if _audio_is_too_short_for_asr(audio_path):
        return _record_score(
            sample,
            language,
            1.0,
            skipped_reason="audio_too_short",
        )

    if config.protocol == "qwen3_asr_chat_path":
        response = await runtime.post_json(
            "wer",
            {
                "model": config.model or "qwen3-asr-1.7b",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "audio_url",
                                "audio_url": {"url": audio_path.resolve().as_uri()},
                            }
                        ],
                    }
                ],
                "temperature": 0,
            },
        )
        hypothesis = _chat_transcription(response)
    else:

        def form_factory() -> aiohttp.FormData:
            form = aiohttp.FormData()
            form.add_field("model", config.model or "qwen-asr")
            form.add_field("file", audio_path.read_bytes(), filename="generated.wav", content_type="audio/wav")
            form.add_field("temperature", "0")
            form.add_field("language", language.qwen_asr)
            return form

        response = await runtime.post_form("wer", form_factory)
        if not isinstance(response, dict) or not isinstance(response.get("text"), str):
            raise RewardServiceError("wer", "protocol", "ASR response requires string field 'text'")
        hypothesis = response["text"]
    raw_wer = wer(reference_text, hypothesis, language)
    expected, actual = _tokens(reference_text, language), _tokens(hypothesis, language)
    sample.metadata.update(
        errors=edit_distance(expected, actual), reference_tokens=len(expected), hypothesis_tokens=len(actual)
    )
    return _record_score(sample, language, raw_wer)


def word_error_rate(reference: str, hypothesis: str, language: str = "en") -> dict:
    """Public metric helper using the migrated Delay normalization and definition."""
    spec = resolve_language(language)
    expected, actual = _tokens(reference, spec), _tokens(hypothesis, spec)
    errors = edit_distance(expected, actual)
    return {
        "wer": errors / len(expected) if expected else math.inf,
        "errors": errors,
        "reference_tokens": len(expected),
        "hypothesis_tokens": len(actual),
        "tokenization": "characters" if spec.token_mode == "character" else "words",
    }


async def reward_func(args, sample: Sample, **_kwargs) -> float:
    """Single-sample extension entry point; built-in rollout shares a batch runtime."""
    from .config import get_reward_config

    sample.metadata.setdefault("language", getattr(args, "wer_language", "en"))
    async with RewardHttpRuntime(get_reward_config(args).services) as runtime:
        return (await score(sample, runtime)).reward
