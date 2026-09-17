import asyncio
import math
from pathlib import Path
from types import SimpleNamespace
from typing import TypeAlias
import wave

import pytest
from aiohttp import web

from slime.utils.types import Sample
from slime.rollout.rm_hub.config import RuntimeConfig, ServiceConfig
from slime.rollout.rm_hub.config import ComponentReward
from slime.rollout.rm_hub import judge as judge_reward
from slime.rollout.rm_hub.judge import (
    parse_yes_no_logprobs,
    parse_yes_no_logits,
    yes_probability,
)
from slime.rollout.rm_hub.runtime import RewardHttpRuntime, RewardServiceError

NUM_GPUS = 0

_YES_TOKEN_IDS = (101, 102, 103, 104, 105, 106)
_NO_TOKEN_IDS = (201, 202, 203, 204, 205, 206)

_RubricFixture: TypeAlias = dict[str, str | int | list[str]]
_MetadataFixture: TypeAlias = dict[str, str | int | list[_RubricFixture | str]]


class _FakeTokenizer:
    _ENCODED = {
        "yes": [101],
        " yes": [102],
        "Yes": [103],
        " Yes": [104],
        "YES": [105, 999],
        " YES": [106],
        "no": [201],
        " no": [202],
        "No": [203],
        " No": [204],
        "NO": [205, 998],
        " NO": [206],
    }
    _DECODED = {
        **{
            token_id: label
            for token_id, label in zip(
                _YES_TOKEN_IDS,
                ("yes", " yes", "Yes", " Yes", "YES", " YES"),
                strict=True,
            )
        },
        **{
            token_id: label
            for token_id, label in zip(
                _NO_TOKEN_IDS,
                ("no", " no", "No", " No", "NO", " NO"),
                strict=True,
            )
        },
        998: "##",
        999: "##",
    }

    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return self._ENCODED[value]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self._DECODED[token_id] for token_id in token_ids)


@pytest.fixture(autouse=True)
def _fake_judge_tokenizer(monkeypatch: pytest.MonkeyPatch):
    judge_reward._load_judge_label_token_ids.cache_clear()
    from transformers import AutoTokenizer
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda _path, **_kwargs: _FakeTokenizer())
    yield
    judge_reward._load_judge_label_token_ids.cache_clear()


class _RecordingRuntime:
    def __init__(self, protocol: str | None) -> None:
        self.protocol = protocol
        self.payloads: list[dict[str, object]] = []

    def service(self, name: str):
        assert name == "judge"
        return SimpleNamespace(model="judge", protocol=self.protocol, tokenizer_path="/models/judge")

    async def post_json(self, name: str, payload: dict[str, object]):
        assert name == "judge"
        self.payloads.append(payload)
        return _response(
            [
                {"token": "token_id:101", "logprob": -0.1},
                {"token": "token_id:201", "logprob": -2.0},
            ]
        )


class _ConcurrentRuntime(_RecordingRuntime):
    def __init__(self, expected_requests: int) -> None:
        super().__init__("openai_chat_path")
        self.expected_requests = expected_requests
        self.all_started = asyncio.Event()

    async def post_json(self, name: str, payload: dict[str, object]):
        assert name == "judge"
        self.payloads.append(payload)
        if len(self.payloads) == self.expected_requests:
            self.all_started.set()
        await asyncio.wait_for(self.all_started.wait(), timeout=1.0)
        return _response(
            [
                {"token": "token_id:101", "logprob": math.log(3)},
                {"token": "token_id:201", "logprob": 0.0},
            ]
        )


class _ProtocolFailingRuntime(_RecordingRuntime):
    async def post_json(self, name: str, payload: dict[str, object]):
        assert name == "judge"
        if "BROKEN_RUBRIC" in repr(payload):
            self.payloads.append(payload)
            return {"choices": []}
        return await super().post_json(name, payload)


def _response(entries):
    return {"choices": [{"logprobs": {"content": [{"top_logprobs": entries}]}}]}


def _sample_with_audio(
    tmp_path: Path, rubric: list[_RubricFixture], *, frame_count: int = 4_800
) -> Sample:
    audio_path = tmp_path / "generated.wav"
    with wave.open(str(audio_path), "wb") as stream:
        stream.setparams((2, 2, 48_000, frame_count, "NONE", "not compressed"))
        stream.writeframes(b"\x00" * frame_count * 4)
    return Sample(
        audio_path=str(audio_path),
        prompt="TEXT_SENTINEL",
        label="REFERENCE_SENTINEL",
        metadata={
            "instructions": "INSTRUCTION_SENTINEL",
            "rubric": rubric,
        },
    )


def test_yes_probability_uses_stable_two_way_softmax():
    assert yes_probability(1000.0, 1000.0) == pytest.approx(0.5)
    assert yes_probability(math.log(3), 0.0) == pytest.approx(0.75)


def test_parse_uses_max_variant_per_class():
    yes, no = parse_yes_no_logprobs(
        _response(
            [
                {"token": "token_id:101", "logprob": -0.2},
                {"token": "token_id:103", "logprob": -0.1},
                {"token": "token_id:202", "logprob": -2.0},
            ]
        ),
        yes_token_ids=_YES_TOKEN_IDS,
        no_token_ids=_NO_TOKEN_IDS,
    )
    assert yes == -0.1
    assert no == -2.0


def test_parse_requires_both_classes():
    with pytest.raises(RewardServiceError, match="both yes and no"):
        parse_yes_no_logits(
            _response([{"token": "token_id:101", "logprob": -0.1}]),
            yes_token_ids=_YES_TOKEN_IDS,
            no_token_ids=_NO_TOKEN_IDS,
        )


def test_label_token_ids_match_all_single_token_variants():
    labels = judge_reward.judge_label_token_ids("/models/judge")

    assert labels.yes == _YES_TOKEN_IDS
    assert labels.no == _NO_TOKEN_IDS
    assert labels.all == (*_YES_TOKEN_IDS, *_NO_TOKEN_IDS)


def test_label_token_ids_reject_empty_class_encoding(monkeypatch: pytest.MonkeyPatch):
    class MissingNoTokenizer(_FakeTokenizer):
        def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
            if value.casefold().strip() == "no":
                return []
            return super().encode(value, add_special_tokens=add_special_tokens)

    from transformers import AutoTokenizer
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda _path, **_kwargs: MissingNoTokenizer())
    judge_reward._load_judge_label_token_ids.cache_clear()

    with pytest.raises(ValueError, match="at least one yes and one no"):
        judge_reward.judge_label_token_ids("/models/missing")


def test_sample_rubric_parser_accepts_provenance_and_ignores_polarity(tmp_path: Path):
    sample = _sample_with_audio(
        tmp_path,
        [
            {
                "dimension": "  delivery  ",
                "statement": "  rubric statement  ",
                "polarity": "negative",
                "ids": ["L1.01"],
                "source_category": "explicit_core",
            }
        ],
    )

    parsed = judge_reward.parse_sample_rubrics(sample)

    assert [(item.dimension, item.statement) for item in parsed] == [("delivery", "rubric statement")]


@pytest.mark.parametrize(
    ("metadata", "error_type", "message"),
    [
        ({"rubric": [{"dimension": "quality", "statement": "ok"}]}, ValueError, "metadata.instructions"),
        (
            {"instructions": 7, "rubric": [{"dimension": "quality", "statement": "ok"}]},
            TypeError,
            "instructions to be a string",
        ),
        ({"instructions": "instructions"}, ValueError, "metadata.rubric"),
        ({"instructions": "instructions", "rubric": "not-an-array"}, TypeError, "must be an array"),
        ({"instructions": "instructions", "rubric": []}, ValueError, "between 1 and 32"),
        ({"instructions": "instructions", "rubric": ["not-an-object"]}, TypeError, "must be an object"),
        (
            {
                "instructions": "instructions",
                "rubric": [{"dimension": "quality", "statement": "ok"}] * 33,
            },
            ValueError,
            "between 1 and 32",
        ),
        (
            {"instructions": "instructions", "rubric": [{"statement": "ok"}]},
            TypeError,
            "dimension to be a string",
        ),
        (
            {"instructions": "instructions", "rubric": [{"dimension": " ", "statement": "ok"}]},
            ValueError,
            "dimension must be non-empty",
        ),
        (
            {"instructions": "instructions", "rubric": [{"dimension": 7, "statement": "ok"}]},
            TypeError,
            "dimension to be a string",
        ),
        (
            {"instructions": "instructions", "rubric": [{"dimension": "x" * 129, "statement": "ok"}]},
            ValueError,
            "at most 128",
        ),
        (
            {"instructions": "instructions", "rubric": [{"dimension": "quality", "statement": " "}]},
            ValueError,
            "statement must be non-empty",
        ),
        (
            {"instructions": "instructions", "rubric": [{"dimension": "quality", "statement": 7}]},
            TypeError,
            "statement to be a string",
        ),
        (
            {"instructions": "instructions", "rubric": [{"dimension": "quality", "statement": "x" * 513}]},
            ValueError,
            "at most 512",
        ),
    ],
)
def test_sample_rubric_parser_rejects_invalid_contract(
    metadata: _MetadataFixture, error_type: type[Exception], message: str
) -> None:
    sample = Sample(metadata=metadata)
    sample.metadata["judge_rubric_scores"] = []

    with pytest.raises(error_type, match=message):
        asyncio.run(judge_reward.score(sample, _RecordingRuntime(None)))

    assert "judge_rubric_scores" not in sample.metadata


def test_judge_path_protocol_sends_only_file_uri(tmp_path: Path):
    sample = _sample_with_audio(
        tmp_path,
        [{"dimension": "DIMENSION_SENTINEL", "statement": "RUBRIC_SENTINEL", "polarity": "legacy"}],
    )
    audio_path = Path(sample.audio_path)
    runtime = _RecordingRuntime("openai_chat_path")

    asyncio.run(judge_reward.score(sample, runtime))

    messages = runtime.payloads[0]["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    content = messages[1]["content"]
    assert content[0] == {
        "type": "audio_url",
        "audio_url": {"url": audio_path.resolve().as_uri()},
    }
    rendered_payload = repr(runtime.payloads[0])
    assert "请判断：【DIMENSION_SENTINEL】RUBRIC_SENTINEL" in rendered_payload
    assert "legacy" not in rendered_payload
    assert "TEXT_SENTINEL" not in rendered_payload
    assert "REFERENCE_SENTINEL" not in rendered_payload
    assert "INSTRUCTION_SENTINEL" not in rendered_payload
    assert "data:" not in rendered_payload
    assert "RIFF-test" not in rendered_payload
    assert runtime.payloads[0]["logprobs"] is True
    assert runtime.payloads[0]["logprob_token_ids"] == [*range(101, 107), *range(201, 207)]
    assert runtime.payloads[0]["return_tokens_as_token_ids"] is True
    assert "top_logprobs" not in runtime.payloads[0]


def test_judge_default_protocol_preserves_data_url_transport(tmp_path: Path):
    sample = _sample_with_audio(tmp_path, [{"dimension": "quality", "statement": "Is it natural?"}])
    runtime = _RecordingRuntime(None)

    asyncio.run(judge_reward.score(sample, runtime))

    audio_url = runtime.payloads[0]["messages"][1]["content"][0]["audio_url"]["url"]
    assert audio_url.startswith("data:audio/wav;base64,")


def test_judge_scores_too_short_audio_without_request(tmp_path: Path):
    sample = _sample_with_audio(
        tmp_path,
        [{"dimension": "quality", "statement": "Is it natural?"}],
        frame_count=1,
    )
    sample.metadata["judge_rubric_scores"] = [{"stale": True}]
    runtime = _RecordingRuntime("openai_chat_path")

    result = asyncio.run(judge_reward.score(sample, runtime))

    assert result.reward == 0.0
    assert result.raw_value == 0.0
    assert runtime.payloads == []
    assert sample.metadata["judge_skipped_reason"] == "audio_too_short"
    assert "judge_rubric_scores" not in sample.metadata


def test_judge_scores_rubrics_concurrently_and_averages_p_yes(tmp_path: Path):
    sample = _sample_with_audio(
        tmp_path,
        [
            {"dimension": "quality", "statement": "desired trait", "polarity": "positive"},
            {"dimension": "style", "statement": "another desired trait", "polarity": "negative"},
        ],
    )
    runtime = _ConcurrentRuntime(expected_requests=2)

    result = asyncio.run(judge_reward.score(sample, runtime))

    assert result.reward == pytest.approx(0.75)
    assert sample.metadata["judge_rubric_scores"] == [
        {"index": 0, "dimension": "quality", "p_yes": pytest.approx(0.75)},
        {"index": 1, "dimension": "style", "p_yes": pytest.approx(0.75)},
    ]


def test_judge_does_not_keep_partial_or_stale_scores_when_one_rubric_fails(tmp_path: Path):
    sample = _sample_with_audio(
        tmp_path,
        [
            {"dimension": "quality", "statement": "valid rubric"},
            {"dimension": "style", "statement": "BROKEN_RUBRIC"},
        ],
    )
    sample.metadata["judge_rubric_scores"] = [{"stale": True}]
    runtime = _ProtocolFailingRuntime("openai_chat_path")

    with pytest.raises(RewardServiceError, match="missing targeted OpenAI logprobs"):
        asyncio.run(judge_reward.score(sample, runtime))

    assert len(runtime.payloads) == 2
    assert "judge_rubric_scores" not in sample.metadata


def test_judge_scores_through_real_http_runtime(tmp_path: Path):
    sample = _sample_with_audio(tmp_path, [{"dimension": "quality", "statement": "audible trait"}])

    async def scenario() -> tuple[ComponentReward, list[str], list[list[str]]]:
        seen_models: list[str] = []
        seen_roles: list[list[str]] = []

        async def handle(request: web.Request) -> web.Response:
            payload = await request.json()
            seen_models.append(payload["model"])
            seen_roles.append([message["role"] for message in payload["messages"]])
            return web.json_response(
                _response(
                    [
                        {"token": "token_id:101", "logprob": math.log(3)},
                        {"token": "token_id:201", "logprob": 0.0},
                    ]
                )
            )

        app = web.Application()
        app.router.add_post("/v1/chat/completions", handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        assert site._server is not None
        port = site._server.sockets[0].getsockname()[1]
        config = ServiceConfig(
            endpoint=f"http://127.0.0.1:{port}/v1/chat/completions",
            auth_token_env=None,
            model="judge-model",
            runtime=RuntimeConfig(timeout_seconds=1.0, concurrency=2, max_retries=0),
            protocol="openai_chat_path",
            tokenizer_path="/models/judge",
        )
        try:
            async with RewardHttpRuntime({"judge": config}) as runtime:
                result = await judge_reward.score(sample, runtime)
        finally:
            await runner.cleanup()
        return result, seen_models, seen_roles

    result, seen_models, seen_roles = asyncio.run(scenario())

    assert result.reward == pytest.approx(0.75)
    assert seen_models == ["judge-model"]
    assert seen_roles == [["system", "user"]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
