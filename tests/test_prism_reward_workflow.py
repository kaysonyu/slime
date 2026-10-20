"""Service outputs -> component math -> composition -> native filters -> training."""

import asyncio
import copy
import math
from types import SimpleNamespace

import pytest
import torch
import yaml
from examples.prism.rewards import common, filters, scorers
from test_audio_reward_clients import config
from test_audio_reward_clients import service as service
from test_grouped_rewards import builder, segment
from test_prism_reward_composition import sample

from slime.rollout.base_types import finalize_rollout_groups, iter_samples
from slime.rollout.filter_hub.base_types import call_dynamic_filter
from slime.rollout.rm_hub import batched_async_rm
from slime.utils.misc import load_function
from slime_plugins.audio_rewards.protocol import ScoringError

NUM_GPUS = 0


@pytest.mark.parametrize("action_stream", [False, True])
def test_documented_workflow_uses_native_reward_filter_and_batch_contracts(tmp_path, service, action_stream):
    settings = config(wer=1, sim=1, judge_clap=1)
    settings["scorers"] = {
        "wer": {"path": "examples.prism.rewards.scorers.exponential_wer", "kwargs": {"alpha": 2}},
        "judge_clap": {"path": "examples.prism.rewards.scorers.minimum_judge_clap"},
    }
    path = tmp_path / "rewards.yaml"
    path.write_text(yaml.safe_dump(settings))
    args = SimpleNamespace(
        omni_reward_config=path,
        custom_rm_path="examples.prism.rewards.reward",
        dynamic_sampling_filter_path="examples.prism.rewards.filters.keep_group_with_wer_pass",
        rollout_sample_filter_path="examples.prism.rewards.filters.mask_high_wer",
        rollout_data_transport="object-store",
    )
    groups = [
        [sample(tmp_path, category, group=group, index=2 * group + i) for i in range(2)]
        for group, category in enumerate(("voiceclone", "instruction", "instruction"))
    ]
    items = list(iter_samples(groups))
    for item, correct in zip(items, (True, False, False, False, True, False), strict=True):
        audio = tmp_path / f"{item.index}.wav"
        audio.write_bytes(b"audio")
        item.metadata["omni_audio_path"] = str(audio)
        service.texts_by_file[audio.name] = "你好，世界！" if correct else "你好，世。"
        if action_stream:
            valid = torch.tensor([False, True, True, True])
            item.policy_actions = {
                "stop": {"actions": torch.zeros(4, dtype=torch.long), "valid_mask": valid, "loss_mask": valid.clone()}
            }

    async def score():
        try:
            return await batched_async_rm(args, items)
        finally:
            await common.reward_states.close(args)

    values = asyncio.run(score())
    transformed = math.exp(-0.5)
    assert values == pytest.approx(
        [1.6, transformed + 0.6, transformed + 0.4, transformed + 0.4, 1.4, transformed + 0.4]
    )
    for item, value in zip(items, values, strict=True):
        item.reward = value  # The production rollout assigns the returned scalar here.
    details = items[2].metadata["omni_reward"]["components"]
    assert details["judge_clap"]["details"]["attribute_scores"] == pytest.approx([0.4, 0.8])
    assert details["judge_clap"]["raw"] == pytest.approx(0.4)
    assert details["wer"]["raw"] == 0.25
    assert details["wer"]["details"]["scorer"]["path"] == settings["scorers"]["wer"]["path"]
    http_count = len(service.calls)
    original_rewards = [item.reward for item in items]

    group_filter = load_function(args.dynamic_sampling_filter_path)
    verdicts = [call_dynamic_filter(group_filter, args, group) for group in groups]
    assert [verdict.keep for verdict in verdicts] == [True, False, True]
    assert verdicts[1].reason == "wer_no_passing_candidate"
    selected = [group for group, verdict in zip(groups, verdicts, strict=True) if verdict.keep]
    output = finalize_rollout_groups(args, 0, selected)
    selected_items = list(iter_samples(output.samples))
    assert [item.index for item in selected_items] == [0, 1, 4, 5]
    assert [item.remove_sample for item in selected_items] == [False, True, False, True]
    assert [item.reward for item in items] == original_rewards
    assert len(service.calls) == http_count  # Filtering never calls an inference service.
    assert selected_items[1].metadata["omni_reward"]["filters"]["wer_loss_mask"] == {
        "max_error_rate": 0.15,
        "passed": False,
    }

    data = builder().convert(selected_items)
    assert data["raw_reward"] == pytest.approx([1.6, transformed + 0.6, 1.4, transformed + 0.4])
    delta = (1 - transformed) / 2
    # Masked low rewards still affect each surviving group's normalization.
    assert data["rewards"] == pytest.approx([delta, -delta, delta, -delta])
    assert data["loss_masks"] == [[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]]
    if action_stream:
        assert [mask.item() for mask in data["policy_sample_masks"]] == [True, False, True, False]


def test_reward_gate_keeps_zero_reward_candidate_until_filter_is_selected(tmp_path, service):
    args = SimpleNamespace(omni_reward_config=config(wer=1, sim=1), custom_rm_path="examples.prism.rewards.reward")
    item = sample(tmp_path, "voiceclone_gated")

    async def run():
        try:
            [item.reward] = await batched_async_rm(args, [item])
        finally:
            await common.reward_states.close(args)

    asyncio.run(run())
    assert item.reward == 0
    assert item.metadata["omni_reward"]["gate"]["passed"] is False
    assert item.remove_sample is False
    assert item.loss_mask == [1, 1, 1]
    assert filters.keep_group_with_wer_pass(args, [item]).keep is False
    filters.mask_high_wer(args, [[item]])
    assert item.remove_sample is True and item.reward == 0


def scored(raw=0.0):
    item = segment(1, 0, 0.7)
    item.metadata = {
        "omni_reward": {
            "status": "success",
            "total": item.reward,
            "components": {"wer": {"status": "success", "raw": raw, "transformed": 1 / (1 + raw)}},
        }
    }
    return item


@pytest.mark.parametrize("raw,keep", [(0.0, True), (0.15, True), (0.15001, False), (2.0, False)])
def test_filter_threshold_uses_raw_error_rate(raw, keep):
    item = scored(raw)
    item.metadata["omni_reward"]["components"]["wer"]["transformed"] = 123.0
    assert filters.keep_group_with_wer_pass(None, [item]).keep is keep
    filters.mask_high_wer(None, [[item]])
    assert item.remove_sample is (not keep)
    assert item.reward == 0.7


def test_mask_handles_nested_groups_and_preserves_existing_removals():
    prior, good, bad = scored(), scored(), scored(0.25)
    prior.remove_sample = True
    groups = [[[prior], good], [[bad]]]
    filters.mask_high_wer(None, groups)
    filters.mask_high_wer(None, groups)
    assert [item.remove_sample for item in iter_samples(groups)] == [True, False, True]
    assert groups[0][0][0] is prior and groups[1][0][0] is bad


@pytest.mark.parametrize("case", ["missing_wer", "outcome_error", "component_error", "nan", "negative"])
def test_filter_does_not_swallow_scoring_faults_or_partially_mask_batch(case):
    first, broken = scored(0.25), scored()
    outcome = broken.metadata["omni_reward"]
    if case == "missing_wer":
        outcome["components"].clear()
    elif case == "outcome_error":
        outcome["status"] = "error"
    elif case == "component_error":
        outcome["components"]["wer"]["status"] = "error"
    else:
        outcome["components"]["wer"]["raw"] = float("nan") if case == "nan" else -0.1
    before = copy.deepcopy(first.metadata)
    with pytest.raises((ValueError, ScoringError)):
        filters.mask_high_wer(None, [[first, broken]])
    assert first.remove_sample is False and first.metadata == before
    # A valid passing candidate must not short-circuit validation of a broken peer.
    with pytest.raises((ValueError, ScoringError)):
        filters.keep_group_with_wer_pass(None, [scored(), broken])


def test_zero_frame_reward_is_filtered_without_requesting_service(tmp_path, service):
    args = SimpleNamespace(omni_reward_config=config(wer=1), custom_rm_path="examples.prism.rewards.base.reward")
    item = sample(tmp_path, "base", frames=0)

    async def run():
        try:
            [item.reward] = await batched_async_rm(args, [item])
        finally:
            await common.reward_states.close(args)

    asyncio.run(run())
    assert item.reward == 0 and service.calls == []
    assert filters.keep_group_with_wer_pass(args, [item]).keep is False
    assert filters.keep_group_with_wer_pass(args, [item, scored()]).keep is True
    filters.mask_high_wer(args, [[item]])
    assert item.remove_sample is True


def test_empty_group_is_a_configuration_error():
    with pytest.raises(ValueError, match="nonempty"):
        filters.keep_group_with_wer_pass(None, [])


@pytest.mark.parametrize("alpha", [0, -1, True, float("nan"), float("inf")])
def test_exponential_example_rejects_invalid_alpha(alpha):
    with pytest.raises((ValueError, ScoringError)):
        scorers.exponential_wer(reference="hello", hypothesis="hello", language="English", identity={}, alpha=alpha)
