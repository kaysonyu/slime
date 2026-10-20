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

from examples.prism.rewards import common

from slime.rollout.sample_hooks import close_sample_hooks  # noqa: E402
from slime.utils.misc import load_function  # noqa: E402
from slime_plugins.audio_rewards import client as audio_client
from slime_plugins.audio_rewards import timing
from slime_plugins.audio_rewards.protocol import PROTOCOL, ScoringError
from slime_plugins.prism import reward_adapter

NUM_GPUS = 0

REWARD_PATHS = (
    "examples.prism.rewards.reward",
    "examples.prism.rewards.base.reward",
    "examples.prism.rewards.voiceclone.reward",
    "examples.prism.rewards.instruction.reward",
    "examples.prism.rewards.voiceclone_gated.reward",
)


def test_authenticated_service_sends_token_to_identity_and_embeddings(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_REWARD_TOKEN", "test-secret")
    observed = []
    backend = Service()

    async def get(url, **kwargs):
        observed.append(kwargs)
        return await backend.get(url, **kwargs)

    async def post(url, payload, **kwargs):
        observed.append(kwargs)
        return await backend.post(url, payload, **kwargs)

    monkeypatch.setattr(audio_client.RewardHTTPClient, "get", staticmethod(get))
    monkeypatch.setattr(audio_client.RewardHTTPClient, "post", staticmethod(post))
    client = audio_client.ServiceClient(
        {"endpoint": "https://sim/prefix", "api": "voxrl_embedding", "auth_token_env": "TEST_REWARD_TOKEN"}
    )

    path = tmp_path / "audio.wav"
    path.write_bytes(b"audio")

    async def run():
        await client.embeddings("audio", [{"id": "one", "audio": {"path": str(path)}}], "one")

    asyncio.run(run())
    assert len(observed) == 2 and all(x["headers"] == {"Authorization": "Bearer test-secret"} for x in observed)


@pytest.mark.parametrize("name", ["", "not a name", 42, "MISSING_REWARD_TOKEN"])
def test_reward_auth_configuration_fails_before_http(monkeypatch, name):
    monkeypatch.delenv("MISSING_REWARD_TOKEN", raising=False)
    with pytest.raises(ValueError):
        audio_client.ServiceClient(
            {"endpoint": "https://reward.example", "api": "voxrl_embedding", "auth_token_env": name}
        )


def config(**weights):
    services = {
        name: {"endpoint": f"http://{name}/prefix", "api": "openai_audio" if name == "wer" else "voxrl_embedding"}
        for name in weights
    }
    if "wer" in services:
        services["wer"]["model"] = "Qwen3-ASR-1.7B"
    return {"services": services, "weights": weights}


async def score_all(args, item):
    """Exercise arbitrary formulas through the existing example-owned hook."""
    settings = reward_adapter.load_reward_config(args.omni_reward_config)
    weights = settings["weights"]
    return await common.score(
        args,
        item,
        category="test",
        components=tuple(weights),
        aggregate=lambda sample, records: sum(weights[k] * records[k]["transformed"] for k in weights),
        weights=weights,
    )


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
    """A native transcription / embedding protocol fixture, with real score math."""

    def __init__(self):
        self.calls = []
        self.identity = {
            "sim": {
                "service_task": "speaker_embedding",
                "dimension": 2,
                "normalized": True,
                "embedding_version": "sim-v1",
                "dtype": "float32",
            },
            "judge_clap": {
                "service_task": "judge_clap",
                "dimension": 2,
                "normalized": False,
                "embedding_version": "judge-v1",
                "dtype": "float32",
                "supported_categories": ["Gender", "Speed"],
                "query_versions": [reward_adapter.QUERY_VERSION],
                "scoring": {"type": "siglip_margin_v1", "scale": 1.0, "bias": 0.0},
            },
        }
        self.text = "你好，世。"
        self.candidate_vector = [0.6, 0.8]
        self.reference_vector = [1.0, 0.0]
        self.texts_by_file = {}
        self.vectors_by_file = {}
        self.mutate = lambda body: body

    async def get(self, url, *, timeout=None, **kwargs):
        name = url.split("/")[2]
        self.calls.append(("get", url, timeout))
        return {"protocol": PROTOCOL, **copy.deepcopy(self.identity[name])}

    async def post_file(self, url, *, fields, filename, content, attempts, timeout, **kwargs):
        await asyncio.sleep(0)
        self.calls.append(
            (
                "post_file",
                url,
                {"fields": copy.deepcopy(fields), "filename": filename, "content": content},
                attempts,
                timeout,
            )
        )
        return self.mutate({"text": self.texts_by_file.get(filename, self.text)})

    async def post(self, url, payload, *, attempts, timeout, **kwargs):
        import math

        name = url.split("/")[2]
        self.calls.append(("post", url, copy.deepcopy(payload), attempts, timeout))
        identity = self.identity[name]
        results = []
        for item in payload["inputs"]:
            if name == "sim":
                reference = payload["request_id"].endswith("reference")
                filename = Path(item["audio"].get("path", "upload.wav")).name
                vector = (
                    self.reference_vector if reference else self.vectors_by_file.get(filename, self.candidate_vector)
                )
            elif url.endswith("embed_text"):
                vector = [1.0, 0.0] if item["text"].startswith("Gender:") else [0.0, 1.0]
            else:
                vector = [1 + math.log(0.4 / 0.6), 1 + math.log(0.8 / 0.2)]
            results.append({"id": item["id"], "status": "success", "embedding": list(vector)})
        return self.mutate(
            {
                "protocol": PROTOCOL,
                "request_id": payload["request_id"],
                **{k: identity[k] for k in ["embedding_version", "dimension", "normalized", "dtype"]},
                "results": results,
            }
        )


@pytest.fixture
def service(monkeypatch):
    backend = Service()
    monkeypatch.setattr(audio_client.RewardHTTPClient, "get", staticmethod(backend.get))
    monkeypatch.setattr(audio_client.RewardHTTPClient, "post", staticmethod(backend.post))
    monkeypatch.setattr(audio_client.RewardHTTPClient, "post_file", staticmethod(backend.post_file))
    return backend


@pytest.mark.parametrize("reward_path", REWARD_PATHS)
def test_concurrent_rewards_keep_worker_args_serializable_across_event_loops(
    tmp_path, service, monkeypatch, reward_path
):
    reward = load_function(reward_path)
    settings = config(wer=1, sim=1, judge_clap=1)
    for options in settings["services"].values():
        options["concurrency"] = 1
    args = SimpleNamespace(omni_reward_config=settings)
    original_args = copy.deepcopy(vars(args))
    clients = []

    class TrackedService(audio_client.ServiceClient):
        def __init__(self, options, **kwargs):
            super().__init__(options, **kwargs)
            clients.append(self)

    async def post(url, payload, **kwargs):
        await asyncio.sleep(0)
        return await service.post(url, payload, **kwargs)

    monkeypatch.setattr(audio_client, "ServiceClient", TrackedService)
    monkeypatch.setattr(audio_client.RewardHTTPClient, "post", staticmethod(post))

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

    class TrackedService(audio_client.ServiceClient):
        def __init__(self, options, **kwargs):
            super().__init__(options, **kwargs)
            clients.append(self)

    monkeypatch.setattr(audio_client, "ServiceClient", TrackedService)
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


@pytest.mark.parametrize("reward_path", REWARD_PATHS[:2])
def test_reward_cache_separates_configurations_on_the_same_loop(tmp_path, service, reward_path):
    reward = load_function(reward_path)
    args = SimpleNamespace(omni_reward_config=config(wer=1))

    async def evaluate():
        first = await reward(args, sample(tmp_path, reward_category="base"))
        args.omni_reward_config = config(wer=2)
        second = await reward(args, sample(tmp_path, reward_category="base"))
        assert second == pytest.approx(2 * first)

    asyncio.run(evaluate())


def test_total_budget_follows_actual_worker_context_and_is_archived(tmp_path, service):
    settings = config(wer=1)
    settings["services"]["wer"]["concurrency"] = 10

    async def run():
        for index, expected in enumerate([4, 3, 3]):
            args = SimpleNamespace(omni_reward_config=settings, rollout_worker_index=index, rollout_worker_count=3)
            item = sample(tmp_path)
            await score_all(args, item)
            profile = item.metadata["omni_reward"]["components"]["wer"]["timing"]
            assert (
                profile["concurrency"],
                profile["worker_concurrency"],
                profile["worker_index"],
                profile["worker_count"],
            ) == (10, expected, index, 3)
        await common.reward_states.close(args)

    asyncio.run(run())


def test_all_rewards_preserve_raw_and_use_frozen_protocol(tmp_path, service):
    item = sample(tmp_path)
    args = SimpleNamespace(omni_reward_config=json.dumps(config(wer=0.6, sim=0.4, judge_clap=0.25)))
    total = asyncio.run(score_all(args, item))
    assert total == pytest.approx(0.6 * 0.8 + 0.4 * 0.6 + 0.25 * 0.6)
    outcome = item.metadata["omni_reward"]
    assert outcome["components"]["wer"]["raw"] == 0.25
    assert outcome["components"]["wer"]["details"]["reference_tokens"] == 4
    assert outcome["components"]["sim"]["details"]["purpose"] == "timbre"
    assert outcome["components"]["judge_clap"]["details"]["queries"] == ["Gender: female", "Speed: slow"]
    posts = [call for call in service.calls if call[0] == "post"]
    assert all(call[3:] == (1, 120) for call in posts)
    upload = next(call for call in service.calls if call[0] == "post_file")
    assert set(upload[2]["fields"]) == {"model", "response_format", "temperature", "max_completion_tokens"}
    assert upload[2]["content"] == b"candidate"
    assert all(call[2]["expected_embedding_version"] for call in posts)
    assert sum(call[0] == "get" for call in service.calls) == 2


@pytest.mark.parametrize(
    "language,target,hypothesis,raw",
    [
        ("en", "Ａ dog, runs!", "a dog", 1 / 3),
        ("Japanese", "今日は", "今日", 1 / 3),
        ("ko", "안녕", "안", 0.5),
        ("zh", "你", "你好世界", 3),
        ("English", "Ａ dog, runs!", "a dog", 1 / 3),
        (" Chinese ", "你好", "你", 0.5),
        (" Japanese ", "今日は", "今日", 1 / 3),
        (" Korean ", "안녕", "안", 0.5),
    ],
)
def test_inverse_error_multilingual_and_unclipped_raw(tmp_path, service, language, target, hypothesis, raw):
    item = sample(tmp_path)
    item.metadata.update(language=language, transcript_target=target)
    service.text = hypothesis
    value = asyncio.run(score_all(SimpleNamespace(omni_reward_config=config(wer=1)), item))
    assert item.metadata["omni_reward"]["components"]["wer"]["raw"] == pytest.approx(raw)
    assert value == pytest.approx(1 / (1 + raw))


def test_batch_preserves_sample_order_and_sim_reference_cache(tmp_path, service):
    first, second = sample(tmp_path), sample(tmp_path)
    args = SimpleNamespace(omni_reward_config=config(sim=1))

    async def run():
        assert await score_all(args, [first]) == pytest.approx([0.6])
        return await score_all(args, [second, first])

    assert asyncio.run(run()) == pytest.approx([0.6, 0.6])
    references = [call for call in service.calls if call[0] == "post" and call[2]["request_id"].endswith("reference")]
    assert len(references) == 1
    assert sum(call[0] == "get" for call in service.calls) == 1


def test_reward_metadata_is_separate_from_model_instructions(tmp_path, service):
    item = sample(tmp_path)
    prompt = item.metadata["omni_prompt"]
    prompt["global_instruction"]["Language"] = "English"
    result = asyncio.run(score_all(SimpleNamespace(omni_reward_config=config(wer=0.6, sim=0.4)), item))
    assert result == pytest.approx(0.72)


def test_sim_never_guesses_reference_from_order(tmp_path, service):
    item = sample(tmp_path)
    del item.metadata["omni_prompt"]["reference_attributes"]
    with pytest.raises(ValueError, match="explicit reference_attributes"):
        asyncio.run(score_all(SimpleNamespace(omni_reward_config=config(sim=1)), item))
    assert service.calls == []


def test_sim_negative_cosine_zero_reward_and_invalid_reference_errors(tmp_path, service):
    item = sample(tmp_path)
    service.candidate_vector = [-1.0, 0.0]
    assert asyncio.run(score_all(SimpleNamespace(omni_reward_config=config(sim=1)), item)) == 0
    assert item.metadata["omni_reward"]["components"]["sim"]["raw"] == -1
    service.reference_vector = [0.0, 0.0]
    with pytest.raises(ScoringError, match="not normalized"):
        asyncio.run(score_all(SimpleNamespace(omni_reward_config=config(sim=1)), item))


def test_sim_short_candidate_is_explicit_valid_low_reward(tmp_path, service):
    item = sample(tmp_path)

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
    assert asyncio.run(score_all(SimpleNamespace(omni_reward_config=config(sim=1)), item)) == 0
    component = item.metadata["omni_reward"]["components"]["sim"]
    assert component["status"] == "invalid_audio"
    assert component["raw"] is None
    assert component["details"]["reason"] == "too_short"


def test_zero_frame_has_no_http_and_keeps_raw_none(tmp_path, service):
    item = sample(tmp_path, omni_frames=0, omni_audio_path=None)
    assert asyncio.run(score_all(SimpleNamespace(omni_reward_config=config(wer=0.6, sim=0.4)), item)) == 0
    assert service.calls == []
    assert all(component["raw"] is None for component in item.metadata["omni_reward"]["components"].values())
    assert item.metadata["omni_reward"]["components"]["wer"]["details"] == {"reason": "zero_frame"}


def test_nonzero_frames_without_waveform_is_error(tmp_path, service):
    item = sample(tmp_path, omni_audio_path=None)
    with pytest.raises(ValueError, match="nonzero TTS frames"):
        asyncio.run(score_all(SimpleNamespace(omni_reward_config=config(wer=1)), item))
    assert "omni_reward" not in item.metadata


@pytest.mark.parametrize("instruction", [None, {}, {"Speed": None}, {"Unknown": "fast"}, '{"Speed":"slow"}'])
def test_judge_requires_rendered_instruction_even_for_zero_frame(tmp_path, service, instruction):
    item = sample(tmp_path, omni_frames=0)
    item.metadata["omni_prompt"]["global_instruction"] = instruction
    with pytest.raises(ValueError):
        asyncio.run(score_all(SimpleNamespace(omni_reward_config=config(judge_clap=1)), item))
    assert service.calls == []


def test_instruction_normalization_rejects_unsupported_shapes():
    instruction = {"speed": "“slow”", "Gender": " female\n "}
    assert reward_adapter.instruction_queries(instruction) == ["Gender: female", "Speed: slow"]
    for invalid in (
        {"Speed": "slow", " speed ": "fast"},
        {"Speed": "<|reserved|>"},
        {"Speed": 2},
        {"global instruction": {"Speed": "slow"}, "local instruction": []},
    ):
        with pytest.raises(ValueError):
            reward_adapter.instruction_queries(invalid)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("protocol", "wrong", "envelope"),
        ("request_id", "wrong", "envelope"),
        ("embedding_version", "changed", "version"),
        ("dimension", 3, "contract"),
        ("normalized", False, "contract"),
        ("results", None, "ids/order"),
    ],
)
def test_response_contract_errors_never_become_rewards(tmp_path, service, field, value, error):
    service.mutate = lambda body: {**body, field: value}
    item = sample(tmp_path)
    with pytest.raises(ScoringError, match=error):
        asyncio.run(score_all(SimpleNamespace(omni_reward_config=config(sim=1)), item))
    assert item.metadata["omni_reward"]["status"] == "error"
    assert "total" not in item.metadata["omni_reward"]


def test_embedding_identity_is_required_before_scoring(tmp_path, service):
    service.identity["sim"].pop("embedding_version")
    with pytest.raises(ScoringError, match="identity"):
        asyncio.run(score_all(SimpleNamespace(omni_reward_config=config(sim=1)), sample(tmp_path)))


def test_removed_scoring_protocol_is_rejected():
    with pytest.raises(ValueError, match="explicitly select"):
        audio_client.ServiceClient({"endpoint": "http://old", "api": "voxrl_scoring"})


def test_component_failure_waits_for_other_audio_readers(tmp_path, service, monkeypatch):
    finished = []
    original = service.post

    async def post(url, payload, **kwargs):
        await asyncio.sleep(0.01)
        result = await original(url, payload, **kwargs)
        finished.append(payload["request_id"])
        return result

    async def failed_upload(*args, **kwargs):
        raise RuntimeError("ASR unavailable")

    monkeypatch.setattr(audio_client.RewardHTTPClient, "post_file", staticmethod(failed_upload))
    monkeypatch.setattr(audio_client.RewardHTTPClient, "post", staticmethod(post))
    item = sample(tmp_path)
    with pytest.raises(RuntimeError, match="ASR unavailable"):
        asyncio.run(score_all(SimpleNamespace(omni_reward_config=config(wer=0.6, sim=0.4)), item))
    assert len(finished) == 2
    assert item.metadata["omni_reward"]["components"]["sim"]["raw"] == pytest.approx(0.6)
    profiles = item.metadata["omni_reward"]["components"]
    assert profiles["wer"]["timing"]["status"] == "error"
    assert profiles["wer"]["timing"]["http_requests"] == 1
    assert profiles["sim"]["timing"]["status"] == "success"
    assert profiles["sim"]["timing"]["http_requests"] == 2
    assert item.metadata["omni_reward"]["timing"]["total_s"] >= profiles["sim"]["timing"]["total_s"]


def test_http_timeout_is_bounded(tmp_path, service, monkeypatch):
    async def post(url, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(audio_client.RewardHTTPClient, "post_file", staticmethod(post))
    settings = config(wer=1)
    settings["services"]["wer"]["timeout_s"] = 0.01
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(score_all(SimpleNamespace(omni_reward_config=settings), sample(tmp_path)))


def test_json_file_configuration_and_explicit_upload(tmp_path, service):
    settings = config(wer=1)
    settings["services"]["wer"].update(audio_transport="upload", timeout_s=15, max_retries=2)
    path = tmp_path / "rewards.json"
    path.write_text(json.dumps(settings))
    item = sample(tmp_path)
    assert asyncio.run(score_all(SimpleNamespace(omni_reward_config=str(path)), item)) == pytest.approx(0.8)
    call = next(call for call in service.calls if call[0] == "post_file")
    assert call[3:] == (3, 15)
    assert call[2]["content"] == b"candidate"


def test_yaml_configuration(tmp_path, service):
    pytest.importorskip("yaml")
    path = tmp_path / "rewards.yaml"
    path.write_text(
        "services:\n  wer:\n    endpoint: http://wer/prefix\n    api: openai_audio\n    model: Qwen3-ASR-1.7B\nweights:\n  wer: 1\n"
    )
    assert asyncio.run(score_all(SimpleNamespace(omni_reward_config=path), sample(tmp_path))) == pytest.approx(0.8)


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
    scorer = reward_adapter.PrismRewardAdapter(config(wer=1, sim=1, judge_clap=1)["services"])
    result = asyncio.run(scorer.score_components(item, ["wer", "sim", "judge_clap"]))
    assert result["wer"]["raw"] == pytest.approx(0.25)
    assert result["sim"]["details"]["reference_id"] == "audio3"
    asr = next(call[2] for call in service.calls if call[0] == "post_file")
    assert set(asr["fields"]) == {"model", "response_format", "temperature", "max_completion_tokens"}
    judge = next(call[2] for call in service.calls if call[0] == "post" and call[1].endswith("embed_text"))
    assert [x["text"] for x in judge["inputs"]] == ["Gender: female", "Speed: slow"]


def test_unified_sim_requires_a_real_timbre_assignment(tmp_path):
    item = sample(tmp_path)
    reference = item.metadata["omni_prompt"]["references"]["audio1"]
    for mapping in ({}, {"timbre": "none"}, {"timbre": "audio2"}):
        with pytest.raises(ValueError, match="timbre"):
            reward_adapter.reference_audio({"references": {"audio1": reference}, "reference_attributes": mapping})


def test_unified_judge_keeps_supported_fields_and_rejects_unscorable_music():
    assert reward_adapter.instruction_queries({"Language": "Chinese", "Gender": "female"}) == ["Gender: female"]
    with pytest.raises(ValueError, match="at least one"):
        reward_adapter.instruction_queries({"Language": "Chinese"})
    with pytest.raises(ValueError, match="unsupported.*Genre"):
        reward_adapter.instruction_queries({"Genre": "Pop"}, modality="sing")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


def test_native_asr_never_receives_reward_targets_or_language(tmp_path, monkeypatch):

    path = tmp_path / "candidate.wav"
    path.write_bytes(b"waveform")
    observed = []

    async def upload(url, **kwargs):
        observed.append((url, kwargs))
        return {"text": "hello"}

    monkeypatch.setattr(audio_client.RewardHTTPClient, "post_file", staticmethod(upload))
    scorer = audio_client.AudioRewardClient(
        {"wer": {"endpoint": "http://test", "api": "openai_audio", "model": "asr"}}
    )
    result = asyncio.run(scorer.wer(path, "hello", "en", "rid"))
    assert result["raw"] == 0
    assert observed[0][0] == "http://test/v1/audio/transcriptions"
    fields = observed[0][1]["fields"]
    assert fields == {"model": "asr", "response_format": "json", "temperature": "0", "max_completion_tokens": "1024"}
    assert observed[0][1]["content"] == b"waveform"


def test_multipart_retry_rebuilds_the_file_and_preserves_auth(monkeypatch):
    import httpx

    from slime_plugins.audio_rewards.transport import RewardHTTPClient

    calls = []

    async def respond(request):
        content = await request.aread()
        calls.append((content, request.headers["authorization"]))
        return httpx.Response(503 if len(calls) == 1 else 200, json={"text": "ok"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            owner = RewardHTTPClient(4)
            owner._client = client
            result = await owner.post_file(
                "http://asr/v1/audio/transcriptions",
                fields={"model": "asr"},
                filename="a.wav",
                content=b"wave-bytes",
                headers={"Authorization": "Bearer test"},
                attempts=2,
            )
            assert result == {"text": "ok"}

    asyncio.run(run())
    assert len(calls) == 2
    assert all(b"wave-bytes" in content and auth == "Bearer test" for content, auth in calls)


def test_timing_separates_client_admission_wait_from_http_per_sample(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(timing, "time", SimpleNamespace(time=lambda: 1000 + clock[0], perf_counter=lambda: clock[0]))
    path = tmp_path / "audio.wav"
    path.write_bytes(b"audio")

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        client = audio_client.ServiceClient(
            {"endpoint": "http://asr", "api": "openai_audio", "model": "asr", "concurrency": 1}
        )

        async def post_file(url, **kwargs):
            if kwargs["headers"]["X-Request-ID"] == "first":
                entered.set()
                await release.wait()
            return {"text": "hello"}

        monkeypatch.setattr(audio_client.RewardHTTPClient, "post_file", staticmethod(post_file))
        profiles = {}

        async def transcribe(name):
            with timing.capture() as profile:
                profiles[name] = profile
                await client.transcribe(path, name)

        first = asyncio.create_task(transcribe("first"))
        await entered.wait()
        second = asyncio.create_task(transcribe("second"))
        # Wait until the second request is actually blocked on admission.
        while not client._slots._waiters:
            await asyncio.sleep(0)
        clock[0] = 5.0
        release.set()
        await asyncio.gather(first, second)
        assert profiles["first"]["client_queue_s"] == 0
        assert profiles["first"]["http_s"] == 5
        assert profiles["second"]["client_queue_s"] == 5
        assert profiles["second"]["http_s"] == 0
        assert all(p["http_requests"] == 1 and p["total_s"] == 5 for p in profiles.values())

    asyncio.run(run())


def test_timing_keeps_queue_timeout_and_releases_only_acquired_slots(tmp_path, monkeypatch):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"audio")

    async def run():
        client = audio_client.ServiceClient(
            {"endpoint": "http://asr", "api": "openai_audio", "model": "asr", "concurrency": 1, "timeout_s": 0.02}
        )
        await client._slots.acquire()
        with pytest.raises(asyncio.TimeoutError), timing.capture() as profile:
            await client.transcribe(path, "waiting")
        assert profile["status"] == "error"
        assert profile["client_queue_s"] > 0
        assert profile["http_s"] == 0 and profile["http_requests"] == 0
        assert client._slots.locked()
        client._slots.release()
        assert not client._slots.locked()

    asyncio.run(run())


def test_joint_timing_caches_and_concurrent_components(tmp_path, service, monkeypatch):
    async def run():
        args = SimpleNamespace(omni_reward_config=config(wer=1, sim=1, judge_clap=1))
        cold = sample(tmp_path)
        await score_all(args, cold)
        # Reuse files and the same loop-local scorer, not a second sample()
        # invocation which deliberately rewrites the reference fixture.
        warm = SimpleNamespace(metadata=copy.deepcopy(cold.metadata))
        await score_all(args, warm)
        a = cold.metadata["omni_reward"]
        b = warm.metadata["omni_reward"]
        assert b["total"] == a["total"]
        cp, wp = (v["components"] for v in (a, b))
        assert cp["sim"]["timing"]["reference_cache_miss"] == 1
        assert wp["sim"]["timing"]["reference_cache_hit"] == 1
        assert cp["sim"]["timing"]["http_requests"] == 2
        assert wp["sim"]["timing"]["http_requests"] == 1
        assert cp["judge_clap"]["timing"]["text_cache_miss"] == 2
        assert wp["judge_clap"]["timing"]["text_cache_hit"] == 2
        assert wp["judge_clap"]["timing"]["http_requests"] == 1
        for outcome in (a, b):
            assert outcome["timing"]["total_s"] >= max(x["timing"]["total_s"] for x in outcome["components"].values())
            assert all(x["timing"]["status"] == "success" for x in outcome["components"].values())
            assert all(x["timing"]["request_id"].endswith(name) for name, x in outcome["components"].items())
            json.dumps(outcome, allow_nan=False)

    asyncio.run(run())


def test_cancellation_keeps_component_timing_in_metadata(tmp_path, service, monkeypatch):
    async def run():
        entered = asyncio.Event()

        async def never_completes(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(audio_client.RewardHTTPClient, "post_file", staticmethod(never_completes))
        item = sample(tmp_path)
        task = asyncio.create_task(score_all(SimpleNamespace(omni_reward_config=config(wer=1)), item))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        outcome = item.metadata["omni_reward"]
        assert outcome["status"] == "cancelled"
        assert outcome["components"]["wer"]["timing"]["status"] == "cancelled"
        assert outcome["components"]["wer"]["timing"]["http_requests"] == 1
        assert outcome["timing"]["total_s"] > 0

    asyncio.run(run())


def test_joint_scores_remain_parallel_and_wall_time_is_not_summed(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(timing, "time", SimpleNamespace(time=lambda: 1000 + clock[0], perf_counter=lambda: clock[0]))

    async def run():
        entered = {name: asyncio.Event() for name in ("wer", "sim", "judge_clap")}
        release = asyncio.Event()

        def method(name):
            async def score(*args, **kwargs):
                with timing.phase("http_s"):
                    timing.count("http_requests")
                    entered[name].set()
                    await release.wait()
                return {"status": "success", "raw": 1.0, "transformed": 1.0, "details": {}}

            return score

        for name in entered:
            monkeypatch.setattr(audio_client.AudioRewardClient, name, method(name))
        item = sample(tmp_path)
        task = asyncio.create_task(
            score_all(SimpleNamespace(omni_reward_config=config(wer=1, sim=1, judge_clap=1)), item)
        )
        # All three must start before any can finish; sequential scoring fails.
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered.values())), 2)
        clock[0] = 4
        release.set()
        assert await task == 3
        outcome = item.metadata["omni_reward"]
        assert outcome["timing"]["total_s"] == 4
        assert all(c["timing"]["total_s"] == 4 for c in outcome["components"].values())
        assert all(
            c["timing"]["http_s"] == 4 and c["timing"]["http_requests"] == 1 for c in outcome["components"].values()
        )

    # The recipe uses its own clock, so patch that module with the same clock.
    monkeypatch.setattr(common, "time", timing.time)
    asyncio.run(run())
