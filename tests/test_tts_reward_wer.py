import asyncio
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from slime.rollout.rm_hub import wer as wer_reward
from slime.rollout.rm_hub.language import resolve_language
from slime.rollout.rm_hub.runtime import RewardServiceError
from slime.rollout.rm_hub.wer import cer, edit_distance, graphemes, normalize_text, wer
from slime.utils.types import Sample

NUM_GPUS = 0


class _RecordingRuntime:
    def __init__(self, response: object) -> None:
        self.response = response
        self.payloads: list[dict[str, object]] = []

    def service(self, name: str):
        assert name == "wer"
        return SimpleNamespace(model="qwen3-asr-1.7b", protocol="qwen3_asr_chat_path")

    async def post_json(self, name: str, payload: dict[str, object]):
        assert name == "wer"
        self.payloads.append(payload)
        return self.response

    async def post_form(self, *_args, **_kwargs):
        raise AssertionError("path transport must not upload multipart audio")


def _write_wav(path: Path, *, frame_count: int = 9_600) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(48_000)
        stream.writeframes(b"\x00" * frame_count * 4)


def test_legacy_cer_normalization_helpers():
    assert normalize_text("Ａ， B! 你 好") == "ab你好"
    assert graphemes("e\u0301") == ["é"]


def test_legacy_cer_exact_and_substitution():
    assert cer("hello", "hello") == 0.0
    assert cer("abc", "axc") == pytest.approx(1 / 3)


def test_legacy_raw_cer_can_exceed_one():
    assert cer("a", "abcd") == 3.0


def test_legacy_cer_empty_reference_uses_denominator_one():
    assert cer("", "xy") == 2.0


def test_language_maps_mosstts_and_qwen_asr_names():
    language = resolve_language("chinese")
    assert language.mosstts == "chinese"
    assert language.qwen_asr == "Chinese"


def test_x_voice_style_uses_words_for_english_and_characters_for_chinese():
    assert wer("hello world", "hello there", "english") == pytest.approx(0.5)
    assert wer("你好世界", "你好", "chinese") == pytest.approx(0.5)


def test_empty_reference_matches_x_voice_whole_semantics():
    assert wer("", "", "english") == float("inf")


def test_edit_distance_insert_delete():
    assert edit_distance(list("abc"), list("abdc")) == 1
    assert edit_distance(list("abc"), list("ac")) == 1


def test_qwen3_asr_path_protocol_sends_only_file_uri(tmp_path: Path):
    audio_path = tmp_path / "generated audio.wav"
    _write_wav(audio_path)
    runtime = _RecordingRuntime({"choices": [{"message": {"content": "language English<asr_text>hello world"}}]})
    sample = Sample(
        index=7,
        label="hello world",
        audio_path=str(audio_path),
        metadata={"language": "english"},
    )

    result = asyncio.run(wer_reward.score(sample, runtime))

    assert result.reward == 1.0
    assert len(runtime.payloads) == 1
    content = runtime.payloads[0]["messages"][0]["content"]
    assert content == [
        {
            "type": "audio_url",
            "audio_url": {"url": audio_path.resolve().as_uri()},
        }
    ]
    assert "data:" not in repr(runtime.payloads[0])
    assert "RIFF-test" not in repr(runtime.payloads[0])


def test_qwen3_asr_path_protocol_rejects_malformed_chat_response(tmp_path: Path):
    audio_path = tmp_path / "generated.wav"
    _write_wav(audio_path)
    runtime = _RecordingRuntime({"choices": [{"message": {"content": None}}]})
    sample = Sample(label="hello", audio_path=str(audio_path), metadata={"language": "english"})

    with pytest.raises(RewardServiceError, match="message content"):
        asyncio.run(wer_reward.score(sample, runtime))


def test_qwen3_asr_path_protocol_scores_too_short_audio_without_request(tmp_path: Path):
    audio_path = tmp_path / "generated.wav"
    _write_wav(audio_path, frame_count=1)
    runtime = _RecordingRuntime({"choices": [{"message": {"content": "must not be used"}}]})
    sample = Sample(
        label="hello",
        audio_path=str(audio_path),
        metadata={"language": "english"},
    )

    result = asyncio.run(wer_reward.score(sample, runtime))

    assert result.reward == 0.0
    assert result.raw_value == 1.0
    assert runtime.payloads == []
    assert sample.metadata["wer_skipped_reason"] == "audio_too_short"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
