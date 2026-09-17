"""Complete-group recovery, version binding, and composite reward contracts."""

import asyncio
import wave
from types import SimpleNamespace

import httpx
import pytest

from slime.rollout import sglang_omni_rollout as rollout
from slime.rollout.failures import FatalRolloutError
from slime.rollout.rm_hub import judge, wer
from slime.rollout.rm_hub.composite import reward_batch
from slime.rollout.rm_hub.config import RewardComponentConfig, RewardConfig, RuntimeConfig, ServiceConfig
from slime.rollout.rm_hub.runtime import RewardServiceError, reward_http_runtime_scope
from slime.utils.types import Sample

NUM_GPUS = 0


def rollout_args(**overrides):
    values = dict(
        omni_endpoints=["http://omni.invalid"],
        omni_stage="tts_engine",
        omni_timeout=1,
        objective="grpo",
        omni_concurrency=2,
        rollout_group_max_retries=2,
        max_recoverable_rollout_failures=4,
        expected_weight_version="policy:1",
        rollout_seed=42,
        rollout_temperature=1,
        rollout_max_response_len=128,
        model_family="moss_tts_local",
        custom_rm_path=None,
        reward_configuration=RewardConfig((RewardComponentConfig("noop", 1),)),
    )
    return SimpleNamespace(**(values | overrides))


def install_generation(monkeypatch, *, version_change=False, fatal=False):
    clients, calls = [], []

    class Client:
        def __init__(self, *args):
            self.closed = False
            clients.append(self)

        async def generate(self, payload):
            seed = payload["sampling_params"]["seed"]
            calls.append(seed)
            if seed == 42:
                request = httpx.Request("POST", "http://omni.invalid/generate")
                response = httpx.Response(400 if fatal else 503, request=request)
                response.raise_for_status()
            return {"version": "policy:2" if version_change and seed >= 1000000 else "policy:1"}

        async def close(self):
            self.closed = True

    def response(args, sample, data, rollout_id):
        sample.status = Sample.Status.COMPLETED
        sample.trajectory = SimpleNamespace(weight_version=data["version"], num_frames=1)

    monkeypatch.setattr(rollout, "OmniClient", Client)
    # Replace only GPU generation/trajectory decoding; retry, grouping, reward and version logic stay real.
    monkeypatch.setattr(rollout, "apply_response", response)
    return clients, calls


def groups():
    return [
        [
            Sample(
                index=index, group_index=index // 2, prompt="hello", label="hello", metadata={"language": "english"}
            )
            for index in range(start, start + 2)
        ]
        for start in (0, 2)
    ]


def test_transient_generation_retries_whole_group_under_same_policy(monkeypatch):
    clients, calls = install_generation(monkeypatch)
    original = groups()
    complete, metrics = asyncio.run(rollout._collect(rollout_args(), 0, original))
    assert [s.metadata["generation_attempt"] for s in complete[0]] == [1, 1]
    assert [s.metadata["generation_attempt"] for s in complete[1]] == [0, 0]
    assert all(s.status == Sample.Status.COMPLETED and not s.remove_sample for group in complete for s in group)
    assert all(s.status == Sample.Status.PENDING for group in original for s in group)
    assert metrics["rollout/recoverable_failures/total"] == 1
    assert metrics["rollout/retried_groups"] == 1
    assert len(calls) == 6 and all(client.closed for client in clients)


def test_policy_change_during_retry_is_fatal(monkeypatch):
    clients, _ = install_generation(monkeypatch, version_change=True)
    with pytest.raises(ValueError, match="weight version"):
        asyncio.run(rollout._collect(rollout_args(), 0, groups()))
    assert all(client.closed for client in clients)


def test_failure_budget_and_nonretryable_http_errors(monkeypatch):
    clients, _ = install_generation(monkeypatch)
    with pytest.raises(FatalRolloutError, match="budget"):
        asyncio.run(rollout._collect(rollout_args(max_recoverable_rollout_failures=0), 0, groups()))
    assert all(client.closed for client in clients)
    clients, calls = install_generation(monkeypatch, fatal=True)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(rollout._collect(rollout_args(), 0, groups()))
    assert all(seed < 1000000 for seed in calls) and all(client.closed for client in clients)


def test_zero_weight_components_need_no_services():
    config = RewardConfig(
        (RewardComponentConfig("noop", 1), RewardComponentConfig("judge", 0), RewardComponentConfig("wer", 0))
    )
    samples = [Sample(index=0, group_index=0)]
    scores = asyncio.run(reward_batch(SimpleNamespace(reward_configuration=config), samples))
    assert scores == [0]
    assert samples[0].metadata["reward_components"] == {"noop": {"reward": 0.0, "raw": 0.0}}


def test_wer_bounded_reward_retains_raw_insertions_and_old_normalization(tmp_path):
    audio = tmp_path / "speech.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0" * 19200)
    sample = Sample(label="one", audio_path=str(audio), metadata={"language": "english"})

    class ASR:
        def service(self, name):
            return ServiceConfig("http://asr.invalid", protocol="qwen3_asr_chat_path")

        async def post_json(self, name, payload):
            return {"choices": [{"message": {"content": "one two three"}}]}

    score = asyncio.run(wer.score(sample, ASR()))
    assert score.reward == 0 and score.raw_value == 2 and sample.metadata["wer"] == 2
    assert wer.wer("繁體中文", "繁体中文", "chinese") == 0
    assert wer.wer("don't", "dont", "english") == 0


def test_composite_does_not_renormalize_or_hide_service_failures(monkeypatch):
    service = ServiceConfig(
        "http://reward.invalid", model="judge", tokenizer_path="/tokenizer", runtime=RuntimeConfig(max_retries=0)
    )
    config = RewardConfig(
        (RewardComponentConfig("wer", 0.4), RewardComponentConfig("judge", 0.6)), {"wer": service, "judge": service}
    )
    sample = Sample(metadata={"instructions": "calm", "rubric": [{"dimension": "voice", "statement": "calm voice"}]})

    async def wer_score(*_args):
        from slime.rollout.rm_hub.config import ComponentReward

        return ComponentReward("wer", 0.5, 0.5)

    async def judge_score(*_args):
        from slime.rollout.rm_hub.config import ComponentReward

        return ComponentReward("judge", 0.8, 0.8)

    monkeypatch.setattr(wer, "score", wer_score)
    monkeypatch.setattr(judge, "score", judge_score)

    async def run():
        async with reward_http_runtime_scope():
            return await reward_batch(SimpleNamespace(reward_configuration=config), [sample])

    assert asyncio.run(run()) == pytest.approx([0.68])

    async def failure(*_args):
        raise RewardServiceError("judge", "retry_exhausted", "unavailable")

    monkeypatch.setattr(judge, "score", failure)
    assert asyncio.run(run()) == [0]
    assert sample.remove_sample and sample.status == Sample.Status.FAILED

    async def invalid_protocol(*_args):
        raise RewardServiceError("judge", "protocol", "invalid result")

    monkeypatch.setattr(judge, "score", invalid_protocol)
    with pytest.raises(RewardServiceError, match="protocol"):
        asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
