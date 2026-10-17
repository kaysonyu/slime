"""Public multi-domain OPD scoring, independent of task reward and model protocol."""

import asyncio
from argparse import Namespace

import pytest
import torch

from slime.rollout import on_policy_distillation as opd
from slime.rollout.rm_hub import async_rm
from slime.utils.types import Sample

NUM_GPUS = 0


def arguments(**overrides):
    return Namespace(
        **{
            "use_opd": True,
            "opd_type": "sglang",
            "custom_rm_path": None,
            "rm_type": None,
            "reward_key": None,
            "opd_eval_teacher_scoring": True,
            "opd_teacher_config": {
                "models": {"zh": {"endpoint": "http://zh/generate"}, "en": {"endpoint": "http://en/generate"}},
                "by_domain": {"chinese": "zh", "english": "en"},
            },
            **overrides,
        }
    )


def response(value=-0.3):
    return {"meta_info": {"input_token_logprobs": [[None, 0], [-9.0, 1], [value, 2], [value - 0.1, 3]]}}


def test_text_teachers_route_each_sample_and_leave_task_reward_independent(monkeypatch):
    calls = []

    async def score(args, sample, **kwargs):
        calls.append(kwargs["endpoint"])
        return response(-0.2 if sample.policy_teacher == "zh" else -0.6)

    monkeypatch.setattr(opd, "reward_func", score)
    args = arguments()
    samples = [
        Sample(tokens=[0, 1, 2, 3], response_length=2, metadata={"domain": domain})
        for domain in ("chinese", "english")
    ]

    async def run():
        for sample in samples:
            assert await opd.score_teacher(args, sample) is sample
            assert sample.reward is None
            sample.reward = await async_rm(args, sample)

    asyncio.run(run())
    assert calls == ["http://zh/generate", "http://en/generate"]
    assert [sample.policy_teacher for sample in samples] == ["zh", "en"]
    assert [sample.reward for sample in samples] == [0.0, 0.0]
    torch.testing.assert_close(samples[0].teacher_log_probs, torch.tensor([-0.2, -0.3]))
    torch.testing.assert_close(samples[1].teacher_log_probs, torch.tensor([-0.6, -0.7]))


def test_model_score_function_receives_binding_and_named_actions(monkeypatch):
    args = arguments()
    args.opd_teacher_config["score_function"] = "custom.score"
    sample = Sample(
        tokens=[0, 1, 2],
        metadata={"domain": "english"},
        reward=0.8,
        policy_actions={
            "audio": {
                "valid_mask": torch.ones(3, 3, dtype=torch.bool),
                "loss_mask": (torch.ones(3, 3, dtype=torch.bool)).clone(),
                "actions": torch.zeros_like(torch.ones(3, 3, dtype=torch.bool), dtype=torch.long),
            },
            "control": {
                "valid_mask": torch.ones(3, dtype=torch.bool),
                "loss_mask": (torch.ones(3, dtype=torch.bool)).clone(),
                "actions": torch.zeros_like(torch.ones(3, dtype=torch.bool), dtype=torch.long),
            },
        },
    )

    async def score(args, item, *, binding, evaluation):
        assert binding["endpoint"] == "http://en/generate"
        assert item.policy_teacher == "en" and evaluation is False
        for stream in item.policy_actions.values():
            stream["teacher_log_probs"] = torch.full(stream["valid_mask"].shape, -0.2)
        return item

    monkeypatch.setattr(opd, "load_function", lambda path: score)
    assert asyncio.run(opd.score_teacher(args, sample)) is sample
    assert sample.reward == 0.8
    with pytest.raises(ValueError, match="only one teacher"):
        asyncio.run(opd.score_teacher(args, sample))


def test_unknown_domain_fails_before_request_and_default_is_explicit(monkeypatch):
    args = arguments()
    calls = []

    async def score(*a, **kw):
        calls.append(True)
        return response()

    monkeypatch.setattr(opd, "reward_func", score)
    for metadata in ({"domain": "unknown"}, {}):
        with pytest.raises(ValueError, match="Unmapped"):
            asyncio.run(opd.score_teacher(args, Sample(response_length=2, metadata=metadata)))
    assert not calls
    args.opd_teacher_config["default_teacher"] = "en"
    sample = Sample(response_length=2)
    asyncio.run(opd.score_teacher(args, sample))
    assert sample.policy_teacher == "en" and calls == [True]


def test_independent_eval_can_skip_teacher_and_use_dataset_reward(monkeypatch):
    from slime.rollout import rm_hub

    args = arguments(opd_eval_teacher_scoring=False)
    sample = Sample(custom_rm_path="quality.reward")

    async def quality(args, sample):
        return 0.9

    monkeypatch.setattr(rm_hub, "load_function", lambda path: quality)

    async def run():
        assert await opd.score_teacher(args, sample, evaluation=True) is sample
        return await async_rm(args, sample)

    assert asyncio.run(run()) == 0.9
    assert sample.teacher_log_probs is None


def test_teacher_scorer_lifecycle_is_owned_by_runtime(monkeypatch):
    args = arguments()
    args.opd_teacher_config["score_function"] = "custom.score"
    calls = []

    async def scorer(*args, **kwargs):
        return None

    async def flush(args):
        calls.append("flush")

    async def close(args):
        calls.append("close")

    scorer.flush = flush
    scorer.close = close
    monkeypatch.setattr(opd, "load_function", lambda path: scorer)

    async def run():
        runtime = opd.TeacherRuntime(args)
        await runtime.flush()
        await runtime.close()
        await runtime.close()

    asyncio.run(run())
    assert calls == ["flush", "close"]


@pytest.mark.parametrize("length", [0, 2])
def test_official_reward_postprocess_populates_teacher_scores(length):
    sample = Sample(response_length=length, reward=response())
    raw, rewards = opd.post_process_rewards(Namespace(reward_key=None), [sample])
    assert raw == rewards == [0.0]
    assert sample.teacher_log_probs.shape == (length,)


def test_incomplete_teacher_scores_fail_at_the_scoring_boundary(monkeypatch):
    args = arguments()
    args.opd_teacher_config["score_function"] = "custom.score"
    monkeypatch.setattr(opd, "load_function", lambda path: lambda *a, **kw: None)
    with pytest.raises(ValueError, match="did not attach"):
        asyncio.run(opd.score_teacher(args, Sample(metadata={"domain": "chinese"})))


def test_teacher_client_configuration_has_one_effective_value_for_cache_and_execution():
    config = {"client": {"max_inflight": 7, "batch_size": 3}}
    options = opd.TeacherClientOptions.from_config(config)
    assert options.max_inflight == 7 and options.batch_size == 3
    assert options.request_timeout_s == opd.TeacherClientOptions.request_timeout_s
    assert opd.TeacherClientOptions.from_config({}) == opd.TeacherClientOptions()
    with pytest.raises(ValueError, match="Unknown"):
        opd.TeacherClientOptions.from_config({"client": {"misspelled_limit": 7}})


@pytest.mark.parametrize("configuration", [{}, "", {"models": []}])
def test_configured_empty_or_invalid_teacher_routes_are_not_silently_disabled(configuration):
    with pytest.raises(ValueError):
        asyncio.run(opd.score_teacher(arguments(opd_teacher_config=configuration), Sample()))


def test_replacing_a_scored_sample_preserves_the_public_route_labels(monkeypatch):
    args = arguments()
    args.opd_teacher_config["score_function"] = "custom.score"
    sample = Sample(tokens=[0, 1], response_length=1, metadata={"domain": "english"})

    def score(args, sample, **kwargs):
        return Sample(tokens=sample.tokens, response_length=1, teacher_log_probs=[-0.2])

    monkeypatch.setattr(opd, "load_function", lambda path: score)
    replacement = asyncio.run(opd.score_teacher(args, sample))
    assert replacement is not sample
    assert replacement.policy_domain == "english" and replacement.policy_teacher == "en"


def test_aborted_sample_skips_teacher_configuration(monkeypatch):
    item = Sample(status=Sample.Status.ABORTED)
    monkeypatch.setattr(opd, "load_teacher_config", lambda *args: pytest.fail("aborted sample created a teacher"))
    assert asyncio.run(opd.score_teacher(arguments(), item)) is item


@pytest.mark.parametrize("stream_name", ["rvq", "text"])
def test_default_teacher_requires_a_model_scorer_for_explicit_actions(monkeypatch, stream_name):
    item = Sample(
        tokens=[0, 1],
        response_length=1,
        metadata={"domain": "chinese"},
        policy_actions={
            stream_name: {
                "actions": torch.tensor([1, 0]),
                "valid_mask": torch.tensor([True, False]),
                "loss_mask": torch.tensor([True, False]),
            }
        },
    )

    async def unexpected(*args, **kwargs):
        pytest.fail("Explicit actions reached the ordinary text teacher")

    monkeypatch.setattr(opd, "reward_func", unexpected)
    with pytest.raises(ValueError, match="model-specific teacher score_function"):
        asyncio.run(opd.score_teacher(arguments(), item))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
