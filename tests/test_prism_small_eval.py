"""Small-eval business contracts with real embedding math and mocked HTTP."""

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from examples.prism.eval import build, metrics, runtime
from test_audio_reward_clients import Service, config

from slime.utils.types import Sample
from slime_plugins.audio_rewards import client as audio_client

NUM_GPUS = 0


@pytest.fixture
def backend(monkeypatch):
    service = Service()
    service.identity["sim"]["model"] = "redimnet2"
    service.identity["judge_clap"]["model"] = "judge-clap"
    monkeypatch.setattr(audio_client.RewardHTTPClient, "get", staticmethod(service.get))
    monkeypatch.setattr(audio_client.RewardHTTPClient, "post", staticmethod(service.post))
    monkeypatch.setattr(audio_client.RewardHTTPClient, "post_file", staticmethod(service.post_file))
    return service


def metadata(tmp_path):
    for name in ("candidate", "reference", "gt"):
        (tmp_path / f"{name}.wav").write_bytes(name.encode())
    return {
        "prism_eval": {
            "id": "test/one",
            "domain": "instruction",
            "benchmark": "instruction-tts-mossv2",
            "source_id": "one",
            "task": "test-zh",
            "arm": "clone",
            "language": "zh",
            "metrics": ["wer", "sim", "judge_clap"],
            "normalization": "instruction-v1",
            "queries": ["Gender: female", "Speed: slow"],
            "gt_audio": str(tmp_path / "gt.wav"),
            "generation_seed": 123,
        },
        "omni_prompt": {
            "script": "你好世界",
            "global_instruction": {"Speed": "slow"},
            "references": {"audio1": str(tmp_path / "reference.wav")},
            "reference_attributes": {"timbre": "audio1"},
        },
        "omni_frames": 20,
        "omni_audio_path": str(tmp_path / "candidate.wav"),
        "transcript_target": "你好世界",
        "prism_eval_generation": "success",
        "eval_run_id": "run-a",
        "eval_sample_index": 0,
        "language": "Chinese",
    }


def args(tmp_path):
    return SimpleNamespace(
        omni_reward_config=config(wer=1, sim=1, judge_clap=1),
        omni_audio_dir=str(tmp_path),
        custom_eval_rollout_log_function_path="examples.prism.eval.runtime.log_eval",
        omni_return_audio=False,
        omni_eval_return_audio=False,
        reward_key=None,
        eval_reward_key=None,
    )


@pytest.mark.parametrize(
    "profile,lang,text,expected",
    [
        ("instruction-v1", "en", "Don’t [pause 12] STOP!", ["don't", "stop"]),
        ("instruction-v1", "zh", "審訊Ａ。", ["审", "讯", "a"]),
        ("dialect-t2s-v1", "zh", "係冇權利。", ["系", "冇", "权", "利"]),
        ("pair-v2", "ja", "こんにちは。", list("こんにちは")),
        ("taxonomy-v1", "zh", "幺1，A。", ["一", "一", "a"]),
        ("taxonomy-v1", "en", "Hello,世界!", ["hello", "世", "界"]),
    ],
)
def test_normalization_profiles(profile, lang, text, expected):
    assert metrics.units(text, lang, profile) == expected


def test_empty_asr_is_full_deletion_but_empty_target_is_invalid():
    value = metrics.wer_record("one two", "", "en", "pair-v2")
    assert value["value"] == 1 and value["errors"] == 2 and value["empty_transcript"]
    assert metrics.wer_record("one", "one two three", "en", "pair-v2")["value"] == 2
    with pytest.raises(ValueError, match="empty"):
        metrics.wer_record("!", "word", "en", "pair-v2")


def test_judge_threshold_and_zero_gt_denominator():
    result = metrics.judge_record(["a", "b", "c"], [0.5, 0.51, 0.9], [0.6, 0.9, 0.5])
    assert result["kept_indices"] == [0, 1]
    assert result["passed"] == 1 and result["total"] == 2 and result["value"] == 0.5
    result = metrics.judge_record(["a"], [1], [0.5])
    assert result["status"] == "not_applicable" and result["value"] is None
    with pytest.raises(ValueError):
        metrics.judge_record(["a"], [float("nan")])


def test_selection_determinism_duration_quotas_and_sparse_failure():
    rows = [
        {"id": f"{i}", "text": "hello", "language": "en", "meta": {"est_secs": s}}
        for i, s in enumerate([3] * 10 + [20] * 10 + [50] * 10 + [120] * 10)
    ]
    first = build.choose(rows, 12)
    assert first == build.choose(list(reversed(rows)), 12)
    assert sum(build.duration(r) > 30 for r in first) == 1
    assert any(10 < build.duration(r) <= 30 for r in first)
    assert all(build.duration(r) <= 60 for r in first)
    with pytest.raises(ValueError, match="Insufficient"):
        build.choose(rows, 30)


def test_same_service_gt_gate_real_score_math_and_training_scorer_isolation(tmp_path, backend):
    settings = config(wer=99, sim=1, judge_clap=0)
    settings["scorers"] = {"wer": {"path": "does.not.exist"}}

    async def run():
        state = runtime.EvalState(settings)
        try:
            await state.preflight()
            results = await asyncio.gather(*(runtime.evaluate(state, metadata(tmp_path)) for _ in range(2)))
            for result in results:
                assert result["wer"]["value"] == 0.25
                assert result["sim"]["value"] == pytest.approx(0.6)
                # Fixture returns Gender=.4 and Speed=.8; only Speed survives GT gate.
                assert result["judge_clap"]["kept_indices"] == [1]
                assert result["judge_clap"]["value"] == 1
            gt_requests = [
                c
                for c in backend.calls
                if c[0] == "post" and c[1].endswith("embed_audio") and "-gt" in c[2]["request_id"]
            ]
            assert len(gt_requests) == 1
        finally:
            await state.aclose()

    asyncio.run(run())


def test_component_failure_preserves_other_scores_and_is_not_zero(tmp_path, backend, monkeypatch):
    async def run():
        state = runtime.EvalState(config(wer=1, sim=1, judge_clap=1))

        async def fail(*args, **kwargs):
            raise TimeoutError("test timeout")

        monkeypatch.setattr(state.client.service("wer"), "transcribe", fail)
        try:
            result = await runtime.evaluate(state, metadata(tmp_path))
            assert result["wer"]["status"] == "error" and "value" not in result["wer"]
            assert result["sim"]["status"] == result["judge_clap"]["status"] == "success"
        finally:
            await state.aclose()

    asyncio.run(run())


def test_zero_frames_and_missing_audio_do_not_call_services(tmp_path, backend):
    async def run():
        state = runtime.EvalState(config(wer=1, sim=1, judge_clap=1))
        try:
            meta = metadata(tmp_path)
            meta["omni_frames"] = 0
            result = await runtime.evaluate(state, meta)
            assert {r["status"] for r in result.values()} == {"invalid_audio"}
            meta["omni_frames"] = 20
            Path(meta["omni_audio_path"]).unlink()
            result = await runtime.evaluate(state, meta)
            assert {r["status"] for r in result.values()} == {"missing_audio"}
            assert not backend.calls
        finally:
            await state.aclose()

    asyncio.run(run())


def test_cancellation_propagates(tmp_path, backend, monkeypatch):
    async def run():
        state = runtime.EvalState(config(wer=1, sim=1, judge_clap=1))

        async def cancel(*args, **kwargs):
            raise asyncio.CancelledError()

        monkeypatch.setattr(state.client.service("wer"), "transcribe", cancel)
        try:
            with pytest.raises(asyncio.CancelledError):
                await runtime.evaluate(state, metadata(tmp_path))
        finally:
            await state.aclose()

    asyncio.run(run())


def test_aggregate_micro_macro_failure_denominators_and_version_isolation(tmp_path):
    recipe = metadata(tmp_path)["prism_eval"]

    def row(result):
        return {"recipe": recipe, "metrics": result, "generation_status": "success"}

    one = metrics.judge_record(["a"], [0.9])
    two = metrics.judge_record(["a", "b", "c"], [0, 0, 0])
    rows = [row({"judge_clap": one}), row({"judge_clap": two}), row({"judge_clap": {"status": "error"}})]
    summary = metrics.summarize(rows)
    assert summary["instruction/judge_clap/pass_micro"] == 0.25
    assert summary["instruction/judge_clap/pass_macro"] == 0.5
    assert summary["instruction/judge_clap/coverage"] == pytest.approx(2 / 3)
    assert summary["instruction/judge_clap/expected"] == 3
    assert summary["instruction/judge_clap/failed"] == 1
    one["identity"], two["identity"] = {"embedding_version": "v1"}, {"embedding_version": "v2"}
    with pytest.raises(ValueError, match="Mixed judge_clap"):
        runtime.export_results(tmp_path / "bad", rows)


def test_generator_preserves_training_args_and_uses_id_seed(tmp_path, backend, monkeypatch):
    import slime_plugins.omni_rollout as omni

    observed = []

    async def fake_generate(effective_args, sample, params, evaluation=False):
        observed.append((effective_args, params, evaluation))
        sample.status = Sample.Status.COMPLETED
        sample.weight_versions = ["step-10"]
        return sample

    monkeypatch.setattr(omni, "generate", fake_generate)
    settings = args(tmp_path)
    sample = Sample(prompt="你好世界", metadata=metadata(tmp_path))
    params = {"temperature": 1, "max_new_tokens": 4096}

    async def run():
        try:
            await runtime.generate(settings, sample, params, evaluation=True)
            await runtime.score(settings, sample)
        finally:
            await runtime.states.close(settings)

    asyncio.run(run())
    assert settings.omni_eval_return_audio is False and settings.omni_return_audio is False
    assert observed[0][0].omni_eval_return_audio is True
    assert observed[0][1]["sampling_seed"] == 123 and "sampling_seed" not in params
    saved = list(tmp_path.glob("small-eval/*/records/*.json"))
    assert len(saved) == 1
    result = json.loads(saved[0].read_text())
    assert result["weight_versions"] == ["step-10"]
    assert result["metrics"]["judge_clap"]["total"] == 1


def test_generation_failure_is_archived_and_missing_config_fails_early(tmp_path, backend, monkeypatch):
    import slime_plugins.omni_rollout as omni

    async def fail(*args, **kwargs):
        raise TimeoutError("offline fake generator")

    monkeypatch.setattr(omni, "generate", fail)
    settings = args(tmp_path)
    sample = Sample(prompt="hello", metadata=metadata(tmp_path))

    async def run():
        try:
            result = await runtime.generate(settings, sample, {}, evaluation=True)
            assert result.status == Sample.Status.ABORTED
            assert result.metadata["prism_eval_generation"] == "error"
        finally:
            await runtime.states.close(settings)

    asyncio.run(run())
    assert list(tmp_path.glob("small-eval/*/records/*.json"))
    bad = copy.deepcopy(settings.omni_reward_config)
    bad["services"]["wer"]["endpoint"] = "https://replace-wer.invalid"
    with pytest.raises(ValueError, match="Configure wer"):
        runtime.EvalState(bad)


def test_no_training_use_and_artifact_run_isolation(tmp_path):
    settings = args(tmp_path)
    sample = Sample(prompt="hello", metadata=metadata(tmp_path))
    with pytest.raises(ValueError, match="cannot be used for training"):
        asyncio.run(runtime.generate(settings, sample, {}, evaluation=False))
    a = runtime.audio_path(settings, sample, evaluation=True)
    sample.metadata["eval_run_id"] = "different-run"
    assert a != runtime.audio_path(settings, sample, evaluation=True)


def test_filtered_samples_remain_in_denominator_and_duplicates_rejected(tmp_path):
    settings = args(tmp_path)
    row = {"text": "你好世界", "metadata": metadata(tmp_path)}
    source = tmp_path / "eval.jsonl"
    source.write_text(json.dumps(row) + "\n")
    settings.eval_datasets = [SimpleNamespace(name="test", path=str(source), n_samples_per_eval_prompt=1)]
    records = runtime.collect_records(settings, {})
    assert records[0]["generation_status"] == "missing"
    values = metrics.summarize(records)
    assert values["instruction/wer/expected"] == 1
    assert values["instruction/wer/coverage"] == 0
    assert "instruction/wer/instruction-v1/micro" not in values
    sample = Sample(prompt="你好世界", metadata=metadata(tmp_path))
    with pytest.raises(ValueError, match="Duplicate"):
        runtime.collect_records(settings, {"test": {"samples": [sample, sample]}})


def test_real_dataset_config_and_loader_preserve_omni_metadata(tmp_path):
    from slime.utils.data import Dataset
    from slime.utils.eval_config import build_eval_dataset_configs

    row = {"text": "你好世界", "metadata": metadata(tmp_path)}
    source = tmp_path / "dataset.jsonl"
    source.write_text(json.dumps(row) + "\n")
    configs = build_eval_dataset_configs(
        SimpleNamespace(),
        [{"name": "small", "path": str(source)}],
        {
            "input_key": "text",
            "label_key": None,
            "metadata_key": "metadata",
            "apply_chat_template": False,
            "multimodal_keys": {},
            "n_samples_per_eval_prompt": 1,
            "custom_generate_function_path": "examples.prism.eval.runtime.generate",
            "custom_rm_path": "examples.prism.eval.runtime.score",
        },
    )
    config = configs[0]
    dataset = Dataset(
        config.path,
        tokenizer=None,
        processor=None,
        max_length=None,
        prompt_key=config.input_key,
        metadata_key=config.metadata_key,
        multimodal_keys=config.multimodal_keys,
    )
    assert len(dataset.samples) == 1
    assert dataset.samples[0].metadata == row["metadata"]
    assert config.inject_metadata(dataset.samples[0].metadata)["prism_eval"]["id"] == "test/one"


def test_rescore_cli_writes_replayable_records_and_listening_index(tmp_path, backend):
    import csv

    from examples.prism.eval.cli import rescore

    settings = args(tmp_path)
    sample = Sample(prompt="hello", metadata=metadata(tmp_path), status=Sample.Status.COMPLETED)
    archived = runtime.record(settings, sample)
    source = tmp_path / "records.jsonl"
    source.write_text(json.dumps(archived) + "\n")
    services = tmp_path / "services.json"
    services.write_text(json.dumps(settings.omni_reward_config))
    output = tmp_path / "rescored"
    asyncio.run(rescore(SimpleNamespace(services=str(services), records=source, output=output, concurrency=2)))
    summary = json.loads((output / "summary.json").read_text())
    assert summary["scalars"]["instruction/judge_clap/pass_micro"] == 1
    assert summary["scalars"]["instruction/judge_clap/category/speed/pass"] == 1
    rows = list(csv.DictReader((output / "listening.csv").open()))
    assert rows[0]["audio_path"] == str(tmp_path / "candidate.wav")
    assert rows[0]["gt_path"] == str(tmp_path / "gt.wav")
    replay = json.loads((output / "samples.jsonl").read_text())
    assert replay["metrics"]["wer"]["transcript"] == "你好，世。"
    assert archived["metrics"] == {}  # Input archive was not overwritten.


def test_generator_requires_quality_logger_before_starting_services(tmp_path):
    settings = args(tmp_path)
    settings.custom_eval_rollout_log_function_path = None
    sample = Sample(prompt="hello", metadata=metadata(tmp_path))
    with pytest.raises(ValueError, match="custom-eval-rollout-log-function-path"):
        asyncio.run(runtime.generate(settings, sample, {}, evaluation=True))
