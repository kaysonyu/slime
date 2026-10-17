"""CPU-only TTS reward business and frozen HTTP protocol contracts."""

import asyncio
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cloudpickle
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.rollout.sample_hooks import close_sample_hooks  # noqa: E402
from slime.utils.misc import load_function  # noqa: E402
from slime_plugins import tts_reward  # noqa: E402

NUM_GPUS = 0

REWARD_PATHS = (
    "slime_plugins.tts_reward.reward",
    "examples.prism.rewards.reward",
    "examples.prism.rewards.base.reward",
    "examples.prism.rewards.voiceclone.reward",
    "examples.prism.rewards.instruction.reward",
    "examples.prism.rewards.voiceclone_gated.reward",
)


def test_authenticated_service_sends_token_to_identity_and_scoring(monkeypatch):
    monkeypatch.setenv("TEST_REWARD_TOKEN", "test-secret")
    observed = []
    identity = {"fingerprint": "fixed"}

    async def get(url, **kwargs):
        observed.append((url, kwargs))
        return {"protocol": tts_reward.PROTOCOL, "identity": identity}

    async def post(url, payload, **kwargs):
        observed.append((url, kwargs))
        return {"protocol": tts_reward.PROTOCOL, "identity": identity, "request_id": "one", "result": {"text": "hi"}}

    monkeypatch.setattr(tts_reward, "get", get)
    monkeypatch.setattr(tts_reward, "post", post)
    client = tts_reward._Service({"endpoint": "https://reward.example", "auth_token_env": "TEST_REWARD_TOKEN"})

    async def run():
        await client.info()
        await client.evaluate("score", {"request_id": "one"})

    asyncio.run(run())
    assert [url for url, _ in observed] == ["https://reward.example/info", "https://reward.example/score"]
    assert all(kwargs["headers"] == {"Authorization": "Bearer test-secret"} for _, kwargs in observed)


@pytest.mark.parametrize("name", ["", "not a name", 42, "MISSING_REWARD_TOKEN"])
def test_reward_auth_configuration_fails_before_http(monkeypatch, name):
    monkeypatch.delenv("MISSING_REWARD_TOKEN", raising=False)
    with pytest.raises(ValueError):
        tts_reward._Service({"endpoint": "https://reward.example", "auth_token_env": name})


def config(**weights):
    return {
        "services": {name: {"endpoint": f"http://{name}/prefix"} for name in weights},
        "weights": weights,
    }


def sample(tmp_path, **overrides):
    candidate, reference = tmp_path / "candidate.wav", tmp_path / "reference.wav"
    candidate.write_bytes(b"candidate")
    reference.write_bytes(b"reference")
    metadata = {
        "omni_frames": 3,
        "omni_audio_path": str(candidate),
        "transcript_target": "你好，世界！",
        "language": "zh-CN",
        "omni_prompt": {
            "script": "你好，世界！",
            "references": {"audio1": str(reference)},
            "reference_attributes": {"timbre": "audio1"},
            "global_instruction": {"Speed": "slow", "Gender": "female"},
        },
    }
    metadata.update(overrides)
    return SimpleNamespace(metadata=metadata)


class Service:
    def __init__(self):
        self.calls = []
        self.identity = {
            "wer": {"fingerprint": "wer", "model": "Qwen3-ASR-1.7B"},
            "sim": {"fingerprint": "sim", "service_task": "speaker_embedding", "embedding_dimension": 2},
            "judge_clap": {
                "fingerprint": "judge_clap",
                "supported_categories": ["Gender", "Speed"],
                "query_versions": [tts_reward.QUERY_VERSION],
            },
        }
        self.text = "你好，世。"
        self.candidate_vector = [0.6, 0.8]
        self.reference_vector = [1.0, 0.0]
        self.mutate = lambda body: body

    async def get(self, url, *, timeout=None):
        name = url.split("/")[2]
        self.calls.append(("get", url, timeout))
        return {"protocol": tts_reward.PROTOCOL, "identity": copy.deepcopy(self.identity[name])}

    async def post(self, url, payload, *, max_retries, timeout):
        name = url.split("/")[2]
        self.calls.append(("post", url, copy.deepcopy(payload), max_retries, timeout))
        if name == "wer":
            result = {"text": self.text}
        elif name == "sim":
            is_reference = payload["request_id"].endswith("reference")
            result = {
                "embedding": self.reference_vector if is_reference else self.candidate_vector,
                "dimension": 2,
                "normalized": True,
            }
        else:
            result = {"scores": [0.4, 0.8]}
        return self.mutate(
            {
                "protocol": tts_reward.PROTOCOL,
                "identity": copy.deepcopy(self.identity[name]),
                "request_id": payload["request_id"],
                "result": result,
            }
        )


@pytest.fixture
def service(monkeypatch):
    backend = Service()
    monkeypatch.setattr(tts_reward, "get", backend.get)
    monkeypatch.setattr(tts_reward, "post", backend.post)
    return backend


@pytest.mark.parametrize("reward_path", REWARD_PATHS)
def test_concurrent_rewards_keep_worker_args_serializable_across_event_loops(
    tmp_path, service, monkeypatch, reward_path
):
    reward = load_function(reward_path)
    settings = config(wer=1, sim=1, judge_clap=1)
    for options in settings["services"].values():
        options["max_inflight"] = 1
    args = SimpleNamespace(omni_reward_config=settings)
    original_args = copy.deepcopy(vars(args))
    clients = []

    class TrackedService(tts_reward._Service):
        def __init__(self, options):
            super().__init__(options)
            clients.append(self)

    async def post(url, payload, **kwargs):
        await asyncio.sleep(0)
        return await service.post(url, payload, **kwargs)

    monkeypatch.setattr(tts_reward, "_Service", TrackedService)
    monkeypatch.setattr(tts_reward, "post", post)

    async def evaluate():
        items = [sample(tmp_path, reward_category="base") for _ in range(2)]
        result = await asyncio.gather(*(reward(args, item) for item in items))
        assert result[0] == pytest.approx(result[1])
        # Real contention must bind a semaphore; an unused lock would hide the regression.
        assert any(client._slots._loop is asyncio.get_running_loop() for client in clients)
        worker_args = cloudpickle.loads(cloudpickle.dumps(copy.copy(args)))
        assert vars(worker_args) == original_args
        assert vars(args) == original_args

    asyncio.run(evaluate())
    count = len(clients)
    # The same configuration and args must create local clients on a fresh loop.
    asyncio.run(evaluate())
    assert len(clients) == 2 * count


@pytest.mark.parametrize("reward_path", REWARD_PATHS)
def test_reward_cache_reuses_clients_until_configured_hook_cleanup(tmp_path, service, monkeypatch, reward_path):
    clients = []

    class TrackedService(tts_reward._Service):
        def __init__(self, options):
            super().__init__(options)
            clients.append(self)

    monkeypatch.setattr(tts_reward, "_Service", TrackedService)
    reward = load_function(reward_path)
    args = SimpleNamespace(
        omni_reward_config=config(wer=1, sim=1, judge_clap=1),
        # MoPD trains with zero reward but evaluates with independent reward hooks.
        custom_rm_path=None,
        eval_datasets=[SimpleNamespace(custom_rm_path=reward_path)],
    )

    async def evaluate_and_close():
        await reward(args, sample(tmp_path, reward_category="base"))
        count = len(clients)
        assert count > 0
        await reward(copy.copy(args), sample(tmp_path, reward_category="base"))
        assert len(clients) == count
        await close_sample_hooks(args)
        await close_sample_hooks(args)
        await reward(args, sample(tmp_path, reward_category="base"))
        assert len(clients) == 2 * count
        await close_sample_hooks(args)

    asyncio.run(evaluate_and_close())


@pytest.mark.parametrize("reward_path", REWARD_PATHS[:3])
def test_reward_cache_separates_configurations_on_the_same_loop(tmp_path, service, reward_path):
    reward = load_function(reward_path)
    args = SimpleNamespace(omni_reward_config=config(wer=1))

    async def evaluate():
        first = await reward(args, sample(tmp_path, reward_category="base"))
        args.omni_reward_config = config(wer=2)
        second = await reward(args, sample(tmp_path, reward_category="base"))
        assert second == pytest.approx(2 * first)

    asyncio.run(evaluate())


def test_all_rewards_preserve_raw_and_use_frozen_protocol(tmp_path, service):
    item = sample(tmp_path)
    args = SimpleNamespace(omni_reward_config=json.dumps(config(wer=0.6, sim=0.4, judge_clap=0.25)))
    total = asyncio.run(tts_reward.reward(args, item))
    assert total == pytest.approx(0.6 * 0.8 + 0.4 * 0.6 + 0.25 * 0.6)
    outcome = item.metadata["omni_reward"]
    assert outcome["components"]["wer"]["raw"] == 0.25
    assert outcome["components"]["wer"]["details"]["reference_tokens"] == 4
    assert outcome["components"]["sim"]["details"]["purpose"] == "timbre"
    assert outcome["components"]["judge_clap"]["details"]["queries"] == ["Gender: female", "Speed: slow"]
    posts = [call for call in service.calls if call[0] == "post"]
    assert all(call[3:] == (1, 120) for call in posts)
    asr_payload = next(call[2] for call in posts if "wer/prefix/score" in call[1])
    assert set(asr_payload) == {"request_id", "audio"}
    judge_payload = next(call[2] for call in posts if "judge_clap/prefix/score" in call[1])
    assert judge_payload["query_version"] == "moss_tts_v2_revision6_instruction_v1"
    assert sum(call[0] == "get" for call in service.calls) == 2


@pytest.mark.parametrize(
    "language,target,hypothesis,raw",
    [
        ("en", "Ａ dog, runs!", "a dog", 1 / 3),
        ("Japanese", "今日は", "今日", 1 / 3),
        ("ko", "안녕", "안", 0.5),
        ("zh", "你", "你好世界", 3),
    ],
)
def test_inverse_error_multilingual_and_unclipped_raw(tmp_path, service, language, target, hypothesis, raw):
    item = sample(tmp_path)
    item.metadata.update(language=language, transcript_target=target)
    service.text = hypothesis
    value = asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=config(wer=1)), item))
    assert item.metadata["omni_reward"]["components"]["wer"]["raw"] == pytest.approx(raw)
    assert value == pytest.approx(1 / (1 + raw))


def test_batch_preserves_sample_order_and_sim_reference_cache(tmp_path, service):
    first, second = sample(tmp_path), sample(tmp_path)
    args = SimpleNamespace(omni_reward_config=config(sim=1))

    async def run():
        assert await tts_reward.reward(args, [first]) == pytest.approx([0.6])
        return await tts_reward.reward(args, [second, first])

    assert asyncio.run(run()) == pytest.approx([0.6, 0.6])
    references = [call for call in service.calls if call[0] == "post" and call[2]["request_id"].endswith("reference")]
    assert len(references) == 1
    assert sum(call[0] == "get" for call in service.calls) == 1


def test_reward_metadata_is_separate_from_model_instructions(tmp_path, service):
    item = sample(tmp_path)
    prompt = item.metadata["omni_prompt"]
    prompt["global_instruction"]["Language"] = "English"
    result = asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=config(wer=0.6, sim=0.4)), item))
    assert result == pytest.approx(0.72)


def test_sim_never_guesses_reference_from_order(tmp_path, service):
    item = sample(tmp_path)
    del item.metadata["omni_prompt"]["reference_attributes"]
    with pytest.raises(ValueError, match="explicit reference_attributes"):
        asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=config(sim=1)), item))
    assert service.calls == []


def test_sim_negative_cosine_zero_reward_and_invalid_reference_errors(tmp_path, service):
    item = sample(tmp_path)
    service.candidate_vector = [-1.0, 0.0]
    assert asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=config(sim=1)), item)) == 0
    assert item.metadata["omni_reward"]["components"]["sim"]["raw"] == -1
    service.reference_vector = [0.0, 0.0]
    with pytest.raises(tts_reward.ScoringError, match="not normalized"):
        asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=config(sim=1)), item))


def test_sim_short_candidate_is_explicit_valid_low_reward(tmp_path, service):
    item = sample(tmp_path)

    def short(body):
        if body["request_id"].endswith("candidate"):
            body["result"] = {
                "status": "invalid_audio",
                "reason": "too_short",
                "num_samples": 200,
                "min_samples": 1600,
                "sample_rate": 16000,
            }
        return body

    service.mutate = short
    assert asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=config(sim=1)), item)) == 0
    component = item.metadata["omni_reward"]["components"]["sim"]
    assert component["status"] == "invalid_audio"
    assert component["raw"] is None
    assert component["details"]["reason"] == "too_short"


def test_zero_frame_has_no_http_and_keeps_raw_none(tmp_path, service):
    item = sample(tmp_path, omni_frames=0, omni_audio_path=None)
    assert asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=config(wer=0.6, sim=0.4)), item)) == 0
    assert service.calls == []
    assert all(component["raw"] is None for component in item.metadata["omni_reward"]["components"].values())
    assert item.metadata["omni_reward"]["components"]["wer"]["details"] == {"reason": "zero_frame"}


def test_nonzero_frames_without_waveform_is_error(tmp_path, service):
    item = sample(tmp_path, omni_audio_path=None)
    with pytest.raises(ValueError, match="nonzero TTS frames"):
        asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=config(wer=1)), item))
    assert "omni_reward" not in item.metadata


@pytest.mark.parametrize("instruction", [None, {}, {"Speed": None}, {"Unknown": "fast"}, '{"Speed":"slow"}'])
def test_judge_requires_rendered_instruction_even_for_zero_frame(tmp_path, service, instruction):
    item = sample(tmp_path, omni_frames=0)
    item.metadata["omni_prompt"]["global_instruction"] = instruction
    with pytest.raises(ValueError):
        asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=config(judge_clap=1)), item))
    assert service.calls == []


def test_instruction_normalization_rejects_unsupported_shapes():
    instruction = {"speed": "“slow”", "Gender": " female\n "}
    assert tts_reward.instruction_queries(instruction) == ["Gender: female", "Speed: slow"]
    for invalid in (
        {"Speed": "slow", " speed ": "fast"},
        {"Speed": "<|reserved|>"},
        {"Speed": 2},
        {"global instruction": {"Speed": "slow"}, "local instruction": []},
    ):
        with pytest.raises(ValueError):
            tts_reward.instruction_queries(invalid)


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda body: body.update(protocol="wrong"), "protocol"),
        (lambda body: body.update(request_id="wrong"), "request_id"),
        (lambda body: body.update(result=None), "result object"),
    ],
)
def test_response_contract_errors_never_become_rewards(tmp_path, service, mutation, error):
    item = sample(tmp_path)

    def corrupt(body):
        mutation(body)
        return body

    service.mutate = corrupt
    with pytest.raises(tts_reward.ScoringError, match=error):
        asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=config(wer=1)), item))
    assert item.metadata["omni_reward"]["status"] == "error"
    assert "total" not in item.metadata["omni_reward"]


def test_rewards_and_sim_cache_work_without_fingerprint_fields(tmp_path, service):
    for identity in service.identity.values():
        identity.pop("fingerprint")
    args = SimpleNamespace(omni_reward_config=config(wer=1, sim=1, judge_clap=1))

    async def run():
        first = await tts_reward.reward(args, sample(tmp_path))
        second = await tts_reward.reward(args, sample(tmp_path))
        return first, second

    assert asyncio.run(run()) == pytest.approx((2.0, 2.0))
    references = [call for call in service.calls if call[0] == "post" and call[2]["request_id"].endswith("reference")]
    assert len(references) == 1


@pytest.mark.parametrize("reported_identity", [None, {"fingerprint": "changed", "model": "other"}])
def test_reward_response_identity_is_metadata_not_a_validation_gate(tmp_path, service, reported_identity):
    service.mutate = lambda body: {**body, "identity": reported_identity}
    item = sample(tmp_path)
    args = SimpleNamespace(omni_reward_config=config(wer=1, sim=1, judge_clap=1))
    assert asyncio.run(tts_reward.reward(args, item)) == pytest.approx(2.0)
    assert all(
        component["identity"] == reported_identity for component in item.metadata["omni_reward"]["components"].values()
    )


def test_component_failure_waits_for_other_audio_readers(tmp_path, service, monkeypatch):
    finished = []
    original = service.post

    async def post(url, payload, **kwargs):
        if "wer/prefix" in url:
            raise RuntimeError("ASR unavailable")
        await asyncio.sleep(0.01)
        result = await original(url, payload, **kwargs)
        finished.append(payload["request_id"])
        return result

    monkeypatch.setattr(tts_reward, "post", post)
    item = sample(tmp_path)
    with pytest.raises(RuntimeError, match="ASR unavailable"):
        asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=config(wer=0.6, sim=0.4)), item))
    assert len(finished) == 2
    assert item.metadata["omni_reward"]["components"]["sim"]["raw"] == pytest.approx(0.6)


def test_http_timeout_is_bounded(tmp_path, service, monkeypatch):
    async def post(url, payload, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(tts_reward, "post", post)
    settings = config(wer=1)
    settings["services"]["wer"]["timeout_s"] = 0.01
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=settings), sample(tmp_path)))


def test_json_file_configuration_and_explicit_upload(tmp_path, service):
    settings = config(wer=1)
    settings["services"]["wer"].update(audio_transport="upload", timeout_s=15, max_retries=2)
    path = tmp_path / "rewards.json"
    path.write_text(json.dumps(settings))
    item = sample(tmp_path)
    assert asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=str(path)), item)) == pytest.approx(0.8)
    call = next(call for call in service.calls if call[0] == "post")
    assert call[3:] == (3, 15)
    assert call[2]["audio"] == {"data_b64": "Y2FuZGlkYXRl", "suffix": ".wav"}


def test_yaml_configuration(tmp_path, service):
    pytest.importorskip("yaml")
    path = tmp_path / "rewards.yaml"
    path.write_text("services:\n  wer:\n    endpoint: http://wer/prefix\nweights:\n  wer: 1\n")
    assert asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=path), sample(tmp_path))) == pytest.approx(
        0.8
    )


def test_unified_prompt_uses_named_timbre_and_keeps_reward_language_out_of_asr(tmp_path, service):
    item = sample(tmp_path)
    original = item.metadata["omni_prompt"]
    reference = original["references"]["audio1"]
    item.metadata["omni_prompt"] = {
        "modality": "speech",
        "script": [{"text": "你好，"}, {"text": "世界！"}],
        "references": {"audio3": reference},
        "reference_attributes": {"timbre": "audio3"},
        "global_instruction": {"Language": "Chinese", "Gender": "female", "Speed": "slow"},
    }
    scorer = tts_reward.RewardScorer(config(wer=1, sim=1, judge_clap=1)["services"])
    result = asyncio.run(scorer.score_components(item, ["wer", "sim", "judge_clap"]))
    assert result["wer"]["raw"] == pytest.approx(0.25)
    assert result["sim"]["details"]["reference_id"] == "audio3"
    calls = [call for call in service.calls if call[0] == "post"]
    asr = next(call[2] for call in calls if "/wer/" in call[1])
    assert set(asr) == {"request_id", "audio"}
    judge = next(call[2] for call in calls if "/judge_clap/" in call[1])
    assert judge["queries"] == ["Gender: female", "Speed: slow"]


def test_unified_sim_requires_a_real_timbre_assignment(tmp_path):
    item = sample(tmp_path)
    reference = item.metadata["omni_prompt"]["references"]["audio1"]
    for mapping in ({}, {"timbre": "none"}, {"timbre": "audio2"}):
        with pytest.raises(ValueError, match="timbre"):
            tts_reward._reference({"references": {"audio1": reference}, "reference_attributes": mapping})


def test_unified_judge_keeps_supported_fields_and_rejects_unscorable_music():
    assert tts_reward.instruction_queries({"Language": "Chinese", "Gender": "female"}) == ["Gender: female"]
    with pytest.raises(ValueError, match="at least one"):
        tts_reward.instruction_queries({"Language": "Chinese"})
    with pytest.raises(ValueError, match="unsupported.*Genre"):
        tts_reward.instruction_queries({"Genre": "Pop"}, modality="sing")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
