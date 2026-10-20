"""Reference-attribute routing, actual component math, and shared SIM calls."""

import asyncio
import json
import math
from types import SimpleNamespace

import pytest
from examples.prism import rewards
from examples.prism.rewards import common
from test_audio_reward_clients import config, sample
from test_audio_reward_clients import service as _service
from test_audio_reward_prosody import local_prosody_pool as _local_prosody_pool

from slime_plugins.audio_rewards import prosody
from slime_plugins.prism.metrics import compute_metrics
from slime_plugins.prism.reward_adapter import PrismRewardAdapter

NUM_GPUS = 0
service = _service
local_prosody_pool = _local_prosody_pool


def settings():
    result = config(wer=1, sim=1)
    result["weights"].update(accent=1, prosody=1)
    return SimpleNamespace(omni_reward_config=result)


@pytest.fixture
def features(monkeypatch, local_prosody_pool):
    seen = []
    stats = prosody._stats()[0]["zh"]

    def extract(raw):
        seen.append(raw)
        offset = {b"candidate": 0, b"reference": 1, b"style": 2}[raw]
        return {name: row["median"] + offset * row["scale_iqr_div_1_349"] for name, row in stats.items()}

    monkeypatch.setattr(prosody, "extract_features", extract)
    return seen


@pytest.mark.parametrize(
    "task,accent,style,emotion,expected",
    [
        ("sing clone", "audio1", "none", "none", 2.0),
        ("Base VoiceClone", "audio1", "none", "none", 2.0),
        ("Crosslingual VoiceClone", "none", "none", "none", 1.4),
        ("Instruction Voice Clone", "none", "none", "none", 1.4),
        ("Full VoiceClone", "audio1", "audio1", "audio1", 2.5),
        ("Attribute Clone Pair", "audio1", "audio1", "none", 2.5),
        ("Attribute Clone Triplet", "audio1", "audio2", "audio2", 2 + 1 / 3),
    ],
)
def test_task_combinations_use_explicit_slots(tmp_path, service, features, task, accent, style, emotion, expected):
    item = sample(tmp_path, reward_category="voiceclone", task_type=task)
    style_path = tmp_path / "style.wav"
    style_path.write_bytes(b"style")
    prompt = item.metadata["omni_prompt"]
    # Deliberately reverse dictionary order; task_type is metadata, not routing.
    prompt["references"] = {"audio2": style_path.as_uri(), **prompt["references"]}
    prompt["reference_attributes"] = {"timbre": "audio1", "accent": accent, "prosody": style, "emotion": emotion}
    args = settings()

    async def run():
        try:
            return await rewards.reward(args, item)
        finally:
            await common.reward_states.close(args)

    assert asyncio.run(run()) == pytest.approx(expected)
    records = item.metadata["omni_reward"]["components"]
    assert "emotion" not in records
    assert ("accent" in records) == (accent != "none")
    assert ("prosody" in records) == (style != "none")
    if accent != "none":
        assert records["accent"]["raw"] == records["sim"]["raw"]
        assert records["accent"]["transformed"] == records["sim"]["transformed"]
        assert records["accent"]["details"]["reused_from"] == "sim"
        assert "timing" not in records["accent"]
    if style != "none":
        assert records["prosody"]["details"]["reference_id"] == style
        assert records["prosody"]["raw"] == pytest.approx(2 if style == "audio2" else 1)
        assert set(features) == {b"candidate", b"style" if style == "audio2" else b"reference"}
    else:
        assert features == []
    posts = [call for call in service.calls if call[0] == "post"]
    assert len(posts) == 2  # One candidate SIM embedding and one reference, even with accent.
    json.dumps(item.metadata["omni_reward"], allow_nan=False)


def test_batch_selects_attributes_per_sample_and_reuses_reference_features(tmp_path, service, features):
    items = [sample(tmp_path) for _ in range(3)]
    for item in items[1:]:
        item.metadata["omni_prompt"]["reference_attributes"].update(accent="audio1", prosody="audio1")
    args = settings()

    async def run():
        try:
            return await rewards.reward_voiceclone(args, items)
        finally:
            await common.reward_states.close(args)

    assert asyncio.run(run()) == pytest.approx([1.4, 2.5, 2.5])
    assert features.count(b"reference") == 1
    assert features.count(b"candidate") == 2


@pytest.mark.parametrize("component", ["accent", "prosody"])
@pytest.mark.parametrize("slot", ["none", "audio9", None])
def test_explicit_component_rejects_missing_assignment(tmp_path, service, component, slot):
    item = sample(tmp_path)
    item.metadata["omni_prompt"]["reference_attributes"][component] = slot
    adapter = PrismRewardAdapter(config(sim=1)["services"])
    with pytest.raises(ValueError, match=component):
        asyncio.run(adapter.score_components(item, [component]))
    assert service.calls == []


def test_different_accent_and_timbre_slots_are_not_silently_aliased(tmp_path, service):
    item = sample(tmp_path)
    prompt = item.metadata["omni_prompt"]
    prompt["references"]["audio2"] = prompt["references"]["audio1"]
    prompt["reference_attributes"]["accent"] = "audio2"
    with pytest.raises(ValueError, match="share the timbre"):
        asyncio.run(rewards.reward_voiceclone(settings(), item))
    assert service.calls == []


def test_accent_only_reuses_custom_sim_without_exposing_hidden_component(tmp_path, service):
    item = sample(tmp_path)
    item.metadata["omni_prompt"]["reference_attributes"]["accent"] = "audio1"
    adapter = PrismRewardAdapter(
        config(sim=1)["services"],
        scorers={"sim": {"path": f"{__name__}.custom_sim"}},
    )
    records = asyncio.run(adapter.score_components(item, ["accent"]))
    assert set(records) == {"accent"}
    assert records["accent"]["raw"] == 0.6 and records["accent"]["transformed"] == 0.36
    assert len([call for call in service.calls if call[0] == "post"]) == 2


def custom_sim(**inputs):
    return {"raw": 0.6, "transformed": 0.36}


def custom_prosody(*, distance, groups, identity, alpha):
    assert identity["pipeline_version"] == "prosody-18d-cpu-v3"
    return {"raw": distance, "transformed": math.exp(-alpha * distance), "details": {"groups": groups}}


def test_prosody_only_supports_custom_transform_without_remote_service(tmp_path, service, features):
    item = sample(tmp_path)
    item.metadata["omni_prompt"]["reference_attributes"]["prosody"] = "audio1"
    adapter = PrismRewardAdapter(
        {}, scorers={"prosody": {"path": f"{__name__}.custom_prosody", "kwargs": {"alpha": 2}}}
    )

    async def run():
        try:
            return await adapter.score_components(item, ["prosody"])
        finally:
            await adapter.aclose()

    records = asyncio.run(run())
    assert records["prosody"]["raw"] == pytest.approx(1)
    assert records["prosody"]["transformed"] == pytest.approx(math.exp(-2))
    assert records["prosody"]["details"]["reference_id"] == "audio1"
    assert service.calls == []


def test_zero_frames_do_not_read_audio_or_extract_features(tmp_path, service, monkeypatch):
    item = sample(tmp_path, omni_frames=0, omni_audio_path=None)
    prompt = item.metadata["omni_prompt"]
    prompt["reference_attributes"].update(accent="audio1", prosody="audio1", emotion="audio1")
    (tmp_path / "reference.wav").unlink()
    monkeypatch.setattr(prosody, "extract_features", lambda *_: pytest.fail("must not extract"))
    assert asyncio.run(rewards.reward_voiceclone(settings(), item)) == 0
    assert service.calls == []
    assert set(item.metadata["omni_reward"]["components"]) == {"wer", "sim", "accent", "prosody"}
    assert all(r["status"] == "invalid_audio" for r in item.metadata["omni_reward"]["components"].values())


def test_unavailable_emotion_does_not_change_reward_or_require_its_audio(tmp_path, service):
    item = sample(tmp_path)
    item.metadata["omni_prompt"]["reference_attributes"]["emotion"] = "audio2"
    item.metadata["omni_prompt"]["references"]["audio2"] = "/missing/emotion.wav"
    assert asyncio.run(rewards.reward_voiceclone(settings(), item)) == pytest.approx(1.4)


def test_unsupported_prosody_language_is_explicit_before_http(tmp_path, service):
    item = sample(tmp_path, language="French")
    item.metadata["omni_prompt"]["reference_attributes"]["prosody"] = "audio1"
    with pytest.raises(ValueError, match="only zh/en"):
        asyncio.run(rewards.reward_voiceclone(settings(), item))
    assert service.calls == []


def test_prosody_failure_preserves_completed_accent_and_timbre_scores(
    tmp_path, service, monkeypatch, local_prosody_pool
):
    item = sample(tmp_path)
    item.metadata["omni_prompt"]["reference_attributes"].update(accent="audio1", prosody="audio1")

    def broken(raw):
        raise ValueError("broken reference audio")

    monkeypatch.setattr(prosody, "extract_features", broken)
    with pytest.raises(ValueError, match="broken reference"):
        asyncio.run(rewards.reward_voiceclone(settings(), item))
    outcome = item.metadata["omni_reward"]
    assert outcome["status"] == "error" and "total" not in outcome
    assert outcome["components"]["sim"]["raw"] == outcome["components"]["accent"]["raw"] == 0.6


def test_attribute_metrics_use_only_active_samples_and_report_group_distances():
    rows = [
        dict(
            dataset="eval",
            prompt_index=i,
            frames=1,
            generation_status="completed",
            reward_status="success",
            reward=1,
            reward_details={"components": components},
        )
        for i, components in enumerate(
            [
                {},
                {"prosody": {"status": "invalid_audio", "raw": None, "transformed": 0}},
                {
                    "accent": {"status": "success", "raw": 0.6, "transformed": 0.6},
                    "prosody": {
                        "status": "success",
                        "raw": 1,
                        "transformed": 0.5,
                        "details": {"groups": {group: 1 for group in prosody.GROUPS}},
                    },
                },
            ]
        )
    ]
    metrics = compute_metrics(rows, group_size=1)
    assert metrics["component/prosody/active_count"] == 2
    assert metrics["component/prosody/valid_count"] == 1
    assert metrics["component/prosody/raw/mean"] == 1
    assert metrics["component/prosody/group/rate/mean"] == 1
    assert metrics["component/accent/coverage_ratio"] == pytest.approx(1 / 3)
    assert not any("emotion" in key for key in metrics)
