"""Custom component math over the real client/adapter/cache pipeline."""

import asyncio
import copy
import json
import math
import sys
from types import SimpleNamespace

import pytest
from test_audio_reward_clients import config, sample, score_all
from test_audio_reward_clients import service as service

from slime_plugins.audio_rewards.client import AudioRewardClient
from slime_plugins.audio_rewards.protocol import ScoringError
from slime_plugins.audio_rewards.scoring import ComponentScorers, default_judge_clap, default_wer
from slime_plugins.prism.reward_adapter import QUERY_VERSION

NUM_GPUS = 0


def hook(monkeypatch, function, **kwargs):
    monkeypatch.setattr(sys.modules[__name__], "custom_scorer", function, raising=False)
    return {"path": f"{__name__}.custom_scorer", "kwargs": kwargs}


def test_custom_wer_reuses_defaults_and_other_components_through_yaml(tmp_path, service, monkeypatch):
    async def exponential(*, reference, hypothesis, language, identity, alpha):
        await asyncio.sleep(0)
        assert identity["backend"] == "openai_audio"
        assert (reference, hypothesis, language) == ("你好，世界！", "你好，世。", "zh-CN")
        result = default_wer(reference=reference, hypothesis=hypothesis, language=language, identity=identity)
        result["transformed"] = math.exp(-alpha * result["raw"])
        result["details"]["reward_transform"] = "exponential"
        return result

    import yaml

    settings = config(wer=1, sim=1, judge_clap=1)
    settings["scorers"] = {"wer": hook(monkeypatch, exponential, alpha=2)}
    path = tmp_path / "rewards.yaml"
    path.write_text(yaml.safe_dump(settings))
    item = sample(tmp_path)
    total = asyncio.run(score_all(SimpleNamespace(omni_reward_config=path), item))
    assert total == pytest.approx(math.exp(-0.5) + 0.6 + 0.6)
    record = item.metadata["omni_reward"]["components"]["wer"]
    assert record["raw"] == 0.25
    assert record["details"]["scorer"] == settings["scorers"]["wer"]
    assert record["details"]["request_id"]
    assert record["status"] == "success"
    json.dumps(item.metadata["omni_reward"], allow_nan=False)


def test_custom_wer_owns_tokenization(tmp_path, service, monkeypatch):
    def punctuation(*, reference, hypothesis, language, identity):
        assert reference == "!!!"
        return {"raw": len(reference), "transformed": -len(reference)}

    settings = config(wer=1)
    settings["scorers"] = {"wer": hook(monkeypatch, punctuation)}
    item = sample(tmp_path, transcript_target="!!!")
    assert asyncio.run(score_all(SimpleNamespace(omni_reward_config=settings), item)) == -3


def test_custom_sim_gets_vectors_and_keeps_reference_cache(tmp_path, service, monkeypatch):
    seen = []

    def affine(*, candidate, reference, identity, offset):
        seen.append((candidate, reference))
        assert identity["service_task"] == "speaker_embedding"
        raw = sum(a * b for a, b in zip(candidate, reference, strict=True))
        identity["embedding_version"] = "mutated"
        return {"raw": raw, "transformed": raw + offset}

    service.candidate_vector = [-1.0, 0.0]
    item = sample(tmp_path)

    async def run():
        async with AudioRewardClient(
            config(sim=1)["services"], scorers={"sim": hook(monkeypatch, affine, offset=1)}
        ) as client:
            results = [
                await client.sim(item.metadata["omni_audio_path"], tmp_path / "reference.wav", str(i))
                for i in range(2)
            ]
            assert all(r["raw"] == -1 and r["transformed"] == 0 for r in results)
            assert all(r["identity"]["embedding_version"] != "mutated" for r in results)
            assert results[0]["details"]["reference_sha256"] == results[1]["details"]["reference_sha256"]

    asyncio.run(run())
    assert seen == [((-1.0, 0.0), (1.0, 0.0))] * 2
    posts = [call for call in service.calls if call[0] == "post"]
    assert len(posts) == 3  # Two candidates, one shared reference.


def test_custom_judge_changes_attribute_reduction_without_mutating_caches(tmp_path, service, monkeypatch):
    snapshots = []

    def minimum(*, audio, texts, queries, identity, options):
        snapshots.append((copy.deepcopy(audio), [v.tolist() for v in texts], list(queries), dict(options)))
        result = default_judge_clap(audio=audio, texts=texts, queries=queries, identity=identity)
        result["transformed"] = min(result["details"]["attribute_scores"])
        result["details"].update(request_id="spoofed", queries=["spoofed"], scorer="spoofed")
        # User code may modify its working copies without corrupting later samples.
        texts[0][:] = 100
        audio[0] = 100
        queries.clear()
        identity["scoring"]["scale"] = 100
        options["minimum"] = 100
        return result

    settings = config(judge_clap=1)
    settings["scorers"] = {"judge_clap": hook(monkeypatch, minimum, options={"minimum": 0})}
    original = copy.deepcopy(settings)
    first, second = sample(tmp_path), sample(tmp_path)

    async def run():
        args = SimpleNamespace(omni_reward_config=settings)
        assert await score_all(args, first) == pytest.approx(0.4)
        assert await score_all(args, second) == pytest.approx(0.4)

    asyncio.run(run())
    assert snapshots[0] == snapshots[1]
    assert settings == original
    details = second.metadata["omni_reward"]["components"]["judge_clap"]["details"]
    assert details["request_id"] != "spoofed"
    assert details["queries"] == ["Gender: female", "Speed: slow"]
    assert details["scorer"] == settings["scorers"]["judge_clap"]
    assert len([c for c in service.calls if c[0] == "post" and c[1].endswith("/embed_text")]) == 1


def test_custom_judge_owns_formula_but_service_contract_still_applies(tmp_path, service, monkeypatch):
    service.identity["judge_clap"]["scoring"] = {"type": "custom"}
    calls = []

    def score(**inputs):
        calls.append(inputs)
        return {"raw": 7, "transformed": 2}

    options = {"judge_clap": hook(monkeypatch, score)}
    item = sample(tmp_path)

    async def run():
        async with AudioRewardClient(config(judge_clap=1)["services"], scorers=options) as client:
            record = await client.judge_clap(
                item.metadata["omni_audio_path"], ["Gender: female"], "one", query_version=QUERY_VERSION
            )
            assert record["transformed"] == 2
        service.identity["judge_clap"]["normalized"] = True
        async with AudioRewardClient(config(judge_clap=1)["services"], scorers=options) as client:
            with pytest.raises(ScoringError, match="unnormalized"):
                await client.judge_clap(
                    item.metadata["omni_audio_path"], ["Gender: female"], "two", query_version=QUERY_VERSION
                )

    asyncio.run(run())
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["audio", "text"])
def test_custom_judge_cannot_bypass_fp32_embedding_validation(tmp_path, service, monkeypatch, kind):
    original_post = service.post
    called = []

    async def overflow(url, payload, **kwargs):
        body = await original_post(url, payload, **kwargs)
        if url.endswith(f"/embed_{kind}"):
            body["results"][0]["embedding"][0] = 1e100
        return body

    def scorer(**inputs):
        called.append(inputs)
        return {"raw": 1, "transformed": 1}

    from slime_plugins.audio_rewards.transport import RewardHTTPClient

    monkeypatch.setattr(RewardHTTPClient, "post", staticmethod(overflow))
    settings = config(judge_clap=1)
    settings["scorers"] = {"judge_clap": hook(monkeypatch, scorer)}
    with pytest.raises(ScoringError, match="FP32"):
        asyncio.run(score_all(SimpleNamespace(omni_reward_config=settings), sample(tmp_path)))
    assert called == []


@pytest.mark.parametrize(
    "result",
    [
        1,
        {"transformed": 1},
        {"raw": 0, "transformed": float("nan")},
        {"raw": float("inf"), "transformed": 0},
        {"raw": None, "transformed": 0},
        {"raw": True, "transformed": 1},
        {"raw": 0, "transformed": False},
        {"raw": 0, "transformed": 1, "identity": {}},
        {"raw": 0, "transformed": 1, "details": []},
        {"raw": 0, "transformed": 1, "details": {"bad": float("nan")}},
        {"raw": 0, "transformed": 1, "details": {"bad": object()}},
    ],
)
def test_invalid_custom_result_propagates_as_scoring_error(tmp_path, service, monkeypatch, result):
    settings = config(wer=1)
    settings["scorers"] = {"wer": hook(monkeypatch, lambda **inputs: result)}
    item = sample(tmp_path)
    with pytest.raises(ScoringError, match="scorer"):
        asyncio.run(score_all(SimpleNamespace(omni_reward_config=settings), item))
    outcome = item.metadata["omni_reward"]
    assert outcome["status"] == "error"
    assert outcome["components"]["wer"]["error"] == "ScoringError"
    assert "total" not in outcome


def test_callback_exception_is_not_swallowed(tmp_path, service, monkeypatch):
    def broken(**inputs):
        raise RuntimeError("custom math failed")

    settings = config(wer=1)
    settings["scorers"] = {"wer": hook(monkeypatch, broken)}
    item = sample(tmp_path)
    with pytest.raises(RuntimeError, match="custom math failed"):
        asyncio.run(score_all(SimpleNamespace(omni_reward_config=settings), item))
    assert item.metadata["omni_reward"]["status"] == "error"


@pytest.mark.parametrize(
    "scorers",
    [
        [],
        {"unknown": {}},
        {"wer": "module.function"},
        {"wer": {}},
        {"wer": {"path": "no_dot"}},
        {"wer": {"path": "nonexistent_audio_scorer.function"}},
        {"wer": {"path": "math.missing"}},
        {"wer": {"path": "math.pi"}},
        {"wer": {"path": "math.exp", "extra": 1}},
        {"wer": {"path": "math.exp", "kwargs": []}},
        {"wer": {"path": "math.exp", "kwargs": {"reference": "override"}}},
        {"wer": {"path": "math.exp", "kwargs": {"alpha": float("nan")}}},
    ],
)
def test_invalid_scorer_configuration_fails_before_http(service, scorers):
    with pytest.raises(ValueError, match="scorer"):
        AudioRewardClient(config(wer=1)["services"], scorers=scorers)
    assert service.calls == []


def test_scorer_config_changes_are_isolated_in_shared_reward_state(tmp_path, service, monkeypatch):
    def constant(**inputs):
        return {"raw": 0, "transformed": inputs["value"]}

    settings = config(wer=1)
    settings["scorers"] = {"wer": hook(monkeypatch, constant, value=0.2)}

    async def run():
        args = SimpleNamespace(omni_reward_config=settings)
        assert await score_all(args, sample(tmp_path)) == 0.2
        settings["scorers"]["wer"]["kwargs"]["value"] = 0.7
        assert await score_all(args, sample(tmp_path)) == 0.7

    asyncio.run(run())


@pytest.mark.parametrize("invalid", ["zero_frame", "too_short"])
def test_invalid_audio_does_not_call_custom_scorer(tmp_path, service, monkeypatch, invalid):
    def forbidden(**inputs):
        pytest.fail("invalid audio must not call a scorer")

    def short(body):
        if body["request_id"].endswith("candidate"):
            body["results"][0] = {
                "id": body["results"][0]["id"],
                "status": "invalid_audio",
                "reason": "too_short",
                "num_samples": 200,
                "min_samples": 1600,
                "sample_rate": 16000,
            }
        return body

    service.mutate = short
    settings = config(sim=1)
    settings["scorers"] = {"sim": hook(monkeypatch, forbidden)}
    item = sample(tmp_path, omni_frames=0 if invalid == "zero_frame" else 3)
    assert asyncio.run(score_all(SimpleNamespace(omni_reward_config=settings), item)) == 0
    record = item.metadata["omni_reward"]["components"]["sim"]
    assert record["raw"] is None and record["status"] == "invalid_audio"
    assert record["details"]["reason"] == invalid
    if invalid == "zero_frame":
        assert service.calls == []


def test_default_judge_rejects_unknown_calibration():
    async def run():
        with pytest.raises(ScoringError, match="scoring rule"):
            await ComponentScorers().score(
                "judge_clap", audio=[1, 0], texts=[[1, 0]], queries=["Gender: female"], identity={"scoring": {}}
            )

    asyncio.run(run())
