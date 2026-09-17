"""Manifest validation, artifact publication, evaluation isolation, and resume identity."""

import base64
import io
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from slime.rollout.audio_artifacts import persist_audio
from slime.rollout.data_source import RolloutDataSource
from slime.rollout.evaluation import evaluation_source
from slime.rollout.failures import RecoverableRolloutError
from slime.rollout.rm_hub.config import RewardComponentConfig, RewardConfig, load_reward_config
from slime.rollout.tts_data import load_speech_manifest
from slime.utils.eval_config import resolve_eval_datasets

NUM_GPUS = 0


def wav_bytes(channels=1, frames=4800):
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setparams((channels, 2, 48000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0" * frames * channels * 2)
    return output.getvalue()


def source_args(tmp_path, **overrides):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"text":"one two"}\n{"text":"three four"}\n')
    values = dict(
        prompt_data=str(dataset),
        rollout_seed=42,
        rollout_shuffle=True,
        rollout_global_dataset=True,
        objective="grpo",
        n_samples_per_prompt=4,
        save=str(tmp_path),
        load=str(tmp_path),
        wer_language="en",
        custom_rm_path=None,
        reward_configuration=RewardConfig((RewardComponentConfig("noop", 1),)),
        rollout_temperature=1.0,
        rollout_max_response_len=512,
        audio_output_dir=str(tmp_path / "audio"),
    )
    return SimpleNamespace(**(values | overrides))


def test_manifest_separates_target_language_and_local_instructions(tmp_path):
    reference = tmp_path / "ref.wav"
    reference.write_bytes(wav_bytes())
    source = tmp_path / "manifest.jsonl"
    source.write_text(
        json.dumps(
            dict(
                id="a",
                script="read me",
                target_text="clean spoken target",
                global_instruction="Calm voice",
                language="chinese",
                rubric=[],
                reference_audios=[dict(id="voice", path=str(reference), uses=["timbre"])],
            )
        )
    )
    samples, _, _, has_references = load_speech_manifest(source)
    (sample,) = samples
    assert sample.prompt == "read me" and sample.label == "clean spoken target"
    assert sample.metadata["language"] == "chinese"
    assert sample.metadata["instructions"] == "Calm voice"
    assert sample.metadata["ref_audio"] == str(reference) and has_references


@pytest.mark.parametrize(
    "row",
    [
        {"text": "hello", "unknown_typo": 1},
        {"text": "hello", "target_text": ""},
        {"text": "hello", "language": "not-a-language"},
        {"id": "x", "script": [{"text": "hello", "local_instruction": "whisper"}], "target_text": "hello"},
    ],
)
def test_manifest_rejects_unsupported_or_malformed_inputs(tmp_path, row):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        load_speech_manifest(path)


def test_duplicate_ids_and_missing_references_fail_before_generation(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id":"x","text":"a"}\n{"id":"x","text":"b"}\n')
    with pytest.raises(ValueError, match="Duplicate"):
        load_speech_manifest(path)
    path.write_text(json.dumps(dict(text="hello", ref_audio=str(tmp_path / "missing.wav"))))
    with pytest.raises(ValueError, match="Reference"):
        load_speech_manifest(path)


def test_resume_identity_detects_reference_and_reward_changes(tmp_path):
    args = source_args(tmp_path)
    audio = tmp_path / "ref.wav"
    audio.write_bytes(wav_bytes())
    Path(args.prompt_data).write_text(json.dumps(dict(text="hello", ref_audio=str(audio))))
    source = RolloutDataSource(args)
    source.get_samples(3)
    source.save(0)
    assert not list((tmp_path / "rollout").glob("*.tmp"))
    unchanged = RolloutDataSource(args)
    unchanged.load(0)
    assert unchanged.sample_group_index == 3
    audio.write_bytes(wav_bytes(frames=9600))
    with pytest.raises(ValueError, match="same prompt"):
        RolloutDataSource(args).load(0)
    audio.write_bytes(wav_bytes())
    args.reward_configuration = RewardConfig((RewardComponentConfig("noop", 2),))
    with pytest.raises(ValueError, match="same prompt"):
        RolloutDataSource(args).load(0)


def test_resume_identity_detects_tokenizer_change(tmp_path):
    args = source_args(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer_config.json").write_text('{"chat_template":"a"}')
    args.hf_checkpoint = str(model)
    source = RolloutDataSource(args)
    source.save(0)
    (model / "tokenizer_config.json").write_text('{"chat_template":"b"}')
    with pytest.raises(ValueError, match="same prompt"):
        RolloutDataSource(args).load(0)


@pytest.mark.parametrize("channels", [1, 2])
def test_audio_atomic_publication_and_dataset_namespaces(tmp_path, channels):
    audio = {"data": base64.b64encode(wav_bytes(channels)).decode(), "sample_rate": 48000}
    first, details = persist_audio(audio, root=tmp_path, namespace="eval/a", rollout_id=0, sample_id=0)
    second, _ = persist_audio(audio, root=tmp_path, namespace="eval/b", rollout_id=0, sample_id=0)
    retry, _ = persist_audio(audio, root=tmp_path, namespace="eval/a", rollout_id=0, sample_id=0, attempt=1)
    assert len({first, second, retry}) == 3
    assert Path(first).read_bytes() == wav_bytes(channels)
    assert details["audio_channels"] == channels
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("data,rate", [(wav_bytes()[:-4], 48000), (b"not a wav", 48000), (wav_bytes(), 24000)])
def test_invalid_audio_never_publishes_a_wav(tmp_path, data, rate):
    with pytest.raises(RecoverableRolloutError):
        persist_audio(
            {"data": base64.b64encode(data).decode(), "sample_rate": rate},
            root=tmp_path,
            namespace="train",
            rollout_id=0,
            sample_id=0,
        )
    assert not list(tmp_path.rglob("*.wav"))


def test_evaluation_defaults_overrides_and_training_sampler_isolation(tmp_path):
    args = source_args(tmp_path)
    config = tmp_path / "eval.yaml"
    config.write_text(
        "eval:\n  defaults:\n    temperature: 0.7\n    n_samples_per_eval_prompt: 2\n  datasets:\n    - name: english\n      path: train.jsonl\n    - name: chinese\n      path: train.jsonl\n      language: zh\n      n_samples_per_eval_prompt: 1\n      max_response_len: 100\n"
    )
    args.eval_config = str(config)
    datasets = resolve_eval_datasets(args)
    train_source = RolloutDataSource(args)
    first = train_source.get_samples(1)
    eval_args, eval_source = evaluation_source(args, datasets[1])
    groups = eval_source.get_samples(eval_args.rollout_batch_size)
    assert len(groups) == 2 and all(len(group) == 1 for group in groups)
    assert all(group[0].metadata["language"] == "chinese" for group in groups)
    assert eval_args.rollout_temperature == 0.7 and eval_args.rollout_max_response_len == 100
    assert eval_args.artifact_namespace == "eval/chinese"
    assert args.rollout_temperature == 1 and args.n_samples_per_prompt == 4
    assert train_source.get_samples(1)[0][0].index == first[0][-1].index + 1


@pytest.mark.parametrize(
    "body",
    [
        "eval: {datasets: [{name: a, path: train.jsonl}, {name: a, path: train.jsonl}]}",
        "eval: {datasets: [{name: a, path: train.jsonl, top_p: 0.8}]}",
        "eval: {defaults: {n_samples_per_eval_prompt: 0}, datasets: [{name: a, path: train.jsonl}]}",
        "eval: {datasets: [{name: a, path: train.jsonl, typo: 1}]}",
    ],
)
def test_invalid_eval_configs_fail_at_boundary(tmp_path, body):
    args = source_args(tmp_path)
    config = tmp_path / "eval.yaml"
    config.write_text(body)
    args.eval_config = str(config)
    with pytest.raises(ValueError):
        resolve_eval_datasets(args)


def test_reward_configuration_validates_active_services_and_environment(tmp_path, monkeypatch):
    config = tmp_path / "reward.yaml"
    config.write_text(
        "reward:\n  components: [{name: wer, weight: 1}]\nservices:\n  wer:\n    endpoint: ${TTS_TEST_ASR_URL}\n"
    )
    monkeypatch.delenv("TTS_TEST_ASR_URL", raising=False)
    with pytest.raises(ValueError, match="unset"):
        load_reward_config(config)
    monkeypatch.setenv("TTS_TEST_ASR_URL", "http://asr.invalid/v1/audio/transcriptions")
    assert load_reward_config(config).weight("wer") == 1


def test_resume_reward_identity_ignores_transport_tuning():
    from dataclasses import replace

    from slime.rollout.rm_hub.config import RuntimeConfig, ServiceConfig

    service = ServiceConfig("http://asr-a.invalid", model="qwen3-asr", protocol="qwen3_asr_chat_path")
    initial = RewardConfig((RewardComponentConfig("wer", 1),), {"wer": service})
    tuned = RewardConfig(
        initial.components,
        {"wer": replace(service, endpoint="http://asr-b.invalid", runtime=RuntimeConfig(concurrency=32))},
    )
    assert initial.identity == tuned.identity
    changed = RewardConfig(initial.components, {"wer": replace(service, model="different-model")})
    assert initial.identity != changed.identity


def test_zero_eval_temperature_does_not_silently_fall_back(tmp_path):
    args = source_args(tmp_path)
    args.eval_data = args.prompt_data
    args.eval_temperature = 0
    with pytest.raises(ValueError, match="temperature"):
        resolve_eval_datasets(args)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
