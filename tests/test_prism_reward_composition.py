"""Category-owned nonlinear formulas over shared frozen TTS components."""

import asyncio
import copy
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.prism import rewards
from examples.prism.rewards import common
from test_audio_reward_clients import Service, config
from test_grouped_rewards import builder, segment
from test_prism_reward_categories import categorized_sample

from slime.utils.misc import load_function
from slime_plugins.audio_rewards import client as audio_client
from slime_plugins.prism import reward_adapter

NUM_GPUS = 0


@pytest.fixture
def service(monkeypatch):
    backend = Service()
    monkeypatch.setattr(audio_client.RewardHTTPClient, "get", staticmethod(backend.get))
    monkeypatch.setattr(audio_client.RewardHTTPClient, "post", staticmethod(backend.post))
    monkeypatch.setattr(audio_client.RewardHTTPClient, "post_file", staticmethod(backend.post_file))
    return backend


def arguments(**weights):
    return SimpleNamespace(omni_reward_config=config(**(weights or {"wer": 1, "sim": 1, "judge_clap": 1})))


def sample(tmp_path, category, *, group=1, index=0, frames=3):
    item = segment(group, index, None, mask=(1,) * frames)
    item.metadata = categorized_sample(
        tmp_path, category, omni_frames=frames, **({"omni_audio_path": None} if frames == 0 else {})
    ).metadata
    return item


@pytest.mark.parametrize(
    "category,expected", [("base", 0.8), ("voiceclone", 1.4), ("instruction", 1.4), ("voiceclone_gated", 0)]
)
def test_category_functions_are_direct_single_and_batch_reward_hooks(tmp_path, service, category, expected):
    function = load_function(f"examples.prism.rewards.{category}.reward")
    items = [sample(tmp_path, category, index=index) for index in range(2)]

    async def run():
        args = arguments()
        return await function(args, items[0]), await function(args, items)

    single, batch = asyncio.run(run())
    assert single == pytest.approx(expected)
    assert batch == pytest.approx([expected, expected])
    assert all(item.metadata["omni_reward"]["category"] == category for item in items)


def test_component_scorer_exposes_raw_scores_and_attribute_details_without_aggregation(tmp_path, service):
    scorer = reward_adapter.PrismRewardAdapter(config(wer=1, sim=1, judge_clap=1)["services"])
    result = asyncio.run(scorer.score_components(sample(tmp_path, "unused"), ["judge_clap"]))
    assert set(result) == {"judge_clap"}
    assert result["judge_clap"]["details"]["attribute_scores"] == pytest.approx([0.4, 0.8])
    assert {call[1].split("/")[2] for call in service.calls} == {"judge_clap"}


def test_category_formula_can_use_raw_scores_and_individual_attributes(tmp_path, service):
    item = sample(tmp_path, "custom")

    def aggregate(sample, components):
        assert sample is item
        return math.exp(-2 * components["wer"]["raw"]) * min(components["judge_clap"]["details"]["attribute_scores"])

    value = asyncio.run(
        common.score(arguments(), item, category="custom", components=("wer", "judge_clap"), aggregate=aggregate)
    )
    assert value == pytest.approx(math.exp(-0.5) * 0.4)
    assert item.metadata["omni_reward"]["total"] == value


@pytest.mark.parametrize("result", [True, float("nan"), float("inf"), [1.0]])
def test_invalid_category_result_preserves_component_error_record(tmp_path, service, result):
    item = sample(tmp_path, "custom")
    with pytest.raises(ValueError, match="finite scalar"):
        asyncio.run(
            common.score(arguments(), item, category="custom", components=("wer",), aggregate=lambda *_: result)
        )
    outcome = item.metadata["omni_reward"]
    assert outcome["status"] == "error" and "total" not in outcome
    assert outcome["errors"] == {"aggregation": "ValueError"}
    assert outcome["components"]["wer"]["raw"] == 0.25


def test_services_only_configuration_and_nonlinear_categories_do_not_need_coefficients(tmp_path, service):
    args = SimpleNamespace(omni_reward_config={"services": config(wer=1, sim=1)["services"]})
    settings = copy.deepcopy(args.omni_reward_config)
    items = [sample(tmp_path, "base", group=1), sample(tmp_path, "voiceclone_gated", group=2)]
    assert asyncio.run(rewards.reward(args, items)) == pytest.approx([0.8, 0])
    assert args.omni_reward_config == settings
    assert "weights" not in items[1].metadata["omni_reward"]


def test_gated_category_ignores_coefficients_and_records_gate_decision(tmp_path, service):
    args = arguments(wer=0, sim=0)
    service.text = "你好，世界！"
    item = sample(tmp_path, "voiceclone_gated")
    assert asyncio.run(rewards.reward(args, item)) == pytest.approx(0.6)
    assert item.metadata["omni_reward"]["gate"] == {"max_error_rate": 0.15, "passed": True}
    assert set(item.metadata["omni_reward"]["components"]) == {"wer", "sim"}


def test_reference_and_service_caches_are_shared_between_categories(tmp_path, service):
    args = arguments()

    async def run():
        await rewards.reward(args, sample(tmp_path, "voiceclone", group=1))
        await rewards.reward(args, sample(tmp_path, "voiceclone_gated", group=2))

    asyncio.run(run())
    assert sum(call[0] == "get" and "/sim/" in call[1] for call in service.calls) == 1
    assert sum(call[0] == "post" and call[2]["request_id"].endswith("reference") for call in service.calls) == 1


def test_immediate_stop_stays_zero_even_for_a_category_with_an_additive_bonus(tmp_path, service):
    item = sample(tmp_path, "custom", frames=0)
    value = asyncio.run(
        common.score(arguments(), item, category="custom", components=("wer",), aggregate=lambda *_: 10.0)
    )
    assert value == 0.0 and service.calls == []
    assert item.metadata["omni_reward"]["components"]["wer"]["details"] == {"reason": "zero_frame"}


@pytest.mark.parametrize("std", [False, True])
@pytest.mark.parametrize("batch_rm", [False, True])
def test_gate_changes_ranking_before_native_grpo_and_preserves_mixed_groups(
    tmp_path, service, std, batch_rm, monkeypatch
):
    service.texts_by_file = {"a.wav": "你好，世。", "b.wav": "你好，世界！"}
    service.vectors_by_file = {"a.wav": [0.99, math.sqrt(1 - 0.99**2)], "b.wav": [0.7, math.sqrt(1 - 0.7**2)]}
    args = arguments()
    args.custom_rm_path = "examples.prism.rewards.reward"
    items = [
        sample(tmp_path, "voiceclone_gated", group=20, index=0),
        sample(tmp_path, "voiceclone", group=10, index=1),
        sample(tmp_path, "voiceclone_gated", group=20, index=2),
        sample(tmp_path, "voiceclone", group=10, index=3),
    ]
    for item, filename in zip(items, ("a.wav", "a.wav", "b.wav", "b.wav"), strict=True):
        path = tmp_path / filename
        path.write_bytes(b"audio")
        item.metadata["omni_audio_path"] = str(path)

    async def run():
        from slime.rollout.rm_hub import async_rm, batched_async_rm

        return (
            await batched_async_rm(args, items)
            if batch_rm
            else await asyncio.gather(*(async_rm(args, item) for item in items))
        )

    values = asyncio.run(run())
    assert values == pytest.approx([0, 1.79, 0.7, 1.7])
    for item, value in zip(items, values, strict=True):
        item.reward = value
    data = builder(grpo_std_normalization=std).convert(items)
    assert data["raw_reward"] == pytest.approx(values)
    if std:
        gated = 0.35 / (math.sqrt(0.245) + 1e-6)
        linear = 0.045 / (math.sqrt(0.00405) + 1e-6)
        expected = [-gated, linear, gated, -linear]
    else:
        expected = [-0.35, 0.045, 0.35, -0.045]
    assert data["rewards"] == pytest.approx(expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
