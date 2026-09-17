"""Control protocol, source resume and Higgs sampled-action contracts."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
import torch

from slime.backends.sglang_omni_utils.client import OmniClient
from slime.rollout.data_source import RolloutDataSource
from slime_plugins.models.higgs_tts.data import HiggsTrajectory

NUM_GPUS = 0


def test_model_conditioning_preserves_reference_audio_and_neutral_sampling():
    from slime.rollout.sglang_omni_rollout import build_request
    from slime.utils.types import Sample

    sample = Sample(
        index=2, prompt="Read this sentence.", metadata={"ref_audio": "/shared/voice.wav", "ref_text": "Reference."}
    )
    args = SimpleNamespace(
        model_family="higgs_tts",
        rollout_temperature=0.8,
        rollout_seed=42,
        rollout_max_response_len=100,
        omni_stage="tts_engine",
    )
    request = build_request(args, sample)
    assert request["prompt"] == {
        "text": sample.prompt,
        "reference_audio": "/shared/voice.wav",
        "reference_text": "Reference.",
    }
    assert request["sampling_params"] == {
        "temperature": 0.8,
        "top_p": 1,
        "top_k": -1,
        "max_new_tokens": 100,
        "seed": 44,
    }
    args.model_family = "moss_tts_local"
    request = build_request(args, sample)
    assert request["prompt"] == sample.prompt
    assert request["stage_params"]["tts_engine"]["ref_audio"] == "/shared/voice.wav"
    assert request["stage_params"]["tts_engine"]["audio_repetition_penalty"] == 1


def test_teacher_routes_are_frozen_and_bind_scores_to_exact_student_actions():
    import hashlib

    from slime.rollout.on_policy_distillation import TeacherScorer
    from slime.utils.types import Sample

    trace = HiggsTrajectory(
        torch.tensor([1, 2]),
        torch.empty((0, 2), dtype=torch.long),
        torch.tensor([[1, 4], [2, 3]]),
        torch.full((2, 2), -2.0),
        torch.tensor([[True, False], [True, True]]),
        torch.ones((2, 2), dtype=torch.bool),
        {"temperature": 1.0},
        "student:1",
        "length",
        "s",
        {"config_sha256": "same-base"},
    )
    args = SimpleNamespace(
        mopd_teachers=["a=http://teacher-a.invalid", "b=http://teacher-b.invalid"],
        omni_stage="tts_engine",
        omni_timeout=10,
        teacher_versions={},
        rollout_temperature=1.0,
        model_family="higgs_tts",
        policy_config={},
    )
    corrupt = None

    def handler(request):
        (data,) = json.loads(request.content)["samples"]
        digest = hashlib.sha256(
            json.dumps(
                {k: v for k, v in data.items() if k != "sample_id"}, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        score = dict(
            sample_id=data["sample_id"],
            input_sha256="wrong" if corrupt == "actions" else digest,
            temperature=1.0,
            logprob_semantics="temperature_scaled_full_vocab_v1",
            teacher_weight_sha256="changed" if corrupt == "weights" else request.url.host,
            weight_version="fixed",
            model_identity=trace.model_identity,
            code_logprobs=[[-1.0, -2.0], [-3.0, -4.0]],
        )
        return httpx.Response(200, json={"version": 1, "results": [score]})

    async def run():
        nonlocal corrupt
        scorer = TeacherScorer(args)
        for client in scorer.clients.values():
            await client.http.aclose()
            client.http = httpx.AsyncClient(base_url=client.endpoint, transport=httpx.MockTransport(handler))
        try:
            for domain in ("a", "b"):
                sample = Sample(index=3, trajectory=trace, metadata={"domain": domain})
                await scorer.score(sample)
                assert sample.metadata["teacher_domain"] == domain
                assert sample.teacher_scores.tolist() == [[-1, -2], [-3, -4]]
            assert args.teacher_versions["a"]["weights"] != args.teacher_versions["b"]["weights"]
            corrupt = "actions"
            with pytest.raises(ValueError, match="exact student"):
                await scorer.score(sample)
            corrupt = "weights"
            with pytest.raises(ValueError, match="identity changed"):
                await scorer.score(sample)
        finally:
            await scorer.close()

    asyncio.run(run())


def test_control_rejects_success_from_the_wrong_stage():
    async def run():
        client = OmniClient("http://omni.invalid")
        await client.http.aclose()
        client.http = httpx.AsyncClient(
            base_url=client.endpoint,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"success": True, "stages": [{"stage": "vocoder", "data": {}}]}
                )
            ),
        )
        try:
            with pytest.raises(RuntimeError, match="did not execute"):
                await client.get_model_info()
        finally:
            await client.close()

    asyncio.run(run())


def test_weight_metadata_uses_json_dtype_and_exact_stage():
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"success": True, "stages": [{"stage": "tts_engine", "data": {}}]})

    async def run():
        client = OmniClient("http://omni.invalid")
        await client.http.aclose()
        client.http = httpx.AsyncClient(base_url=client.endpoint, transport=httpx.MockTransport(handler))
        try:
            await client.update_weights_from_distributed(
                ["head.weight"], [torch.bfloat16], [torch.Size([2, 8])], "sync", 7
            )
        finally:
            await client.close()

    asyncio.run(run())
    assert captured == [
        {
            "names": ["head.weight"],
            "dtypes": ["bfloat16"],
            "shapes": [[2, 8]],
            "group_name": "sync",
            "weight_version": "7",
            "stages": ["tts_engine"],
        }
    ]


def test_small_dataset_wrap_and_resume_preserve_sample_identity(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"text":"first"}\n{"text":"second"}\n')
    args = SimpleNamespace(
        model_family="moss_tts_local",
        prompt_data=str(path),
        rollout_seed=3,
        rollout_shuffle=True,
        rollout_global_dataset=True,
        n_samples_per_prompt=2,
        save=str(tmp_path),
        load=str(tmp_path),
    )
    source = RolloutDataSource(args)
    first = source.get_samples(5)
    assert len(first) == 5 and all(len(group) == 2 for group in first)
    assert [sample.index for group in first for sample in group] == list(range(10))
    source.save(0)
    resumed = RolloutDataSource(args)
    resumed.load(0)
    expected, actual = source.get_samples(4), resumed.get_samples(4)

    def identity(groups):
        return [(s.prompt, s.index, s.group_index) for group in groups for s in group]

    assert identity(actual) == identity(expected)


def test_higgs_keeps_sampled_eoc_and_original_delayed_history():
    codes = torch.tensor([[0, 4, 4], [1, 2, 4], [5, 3, 0], [2, 5, 1]])
    sampled = torch.arange(4)[:, None] >= torch.arange(3)[None]
    audio = sampled & (torch.arange(4)[:, None] < (2 + torch.arange(3))[None])
    trace = HiggsTrajectory(
        torch.tensor([1, -100, 2]),
        torch.tensor([[0, 1, 2]]),
        codes,
        torch.full((4, 3), -2.0),
        sampled,
        audio,
        {"temperature": 1, "top_p": 1, "top_k": -1},
        "1",
        "stop",
        "sample",
        {},
    )
    config = {"audio_encoder_config": {"vocab_size": 6}}
    rows, positions, labels, mask, _ = trace.training_tensors(config)
    assert mask[2, 0] and not trace.audio_mask[2, 0]
    assert labels[2, 0] == 5
    assert positions.tolist() == [2, 3, 4, 5]
    torch.testing.assert_close(rows[3:, 1:], codes[:-1])
    assert trace.num_frames == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
