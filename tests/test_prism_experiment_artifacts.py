"""CPU contracts for attempt isolation, full audio archives and reward metrics."""

import asyncio
import base64
import io
import json
import os
import subprocess
import sys
import wave
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from examples.prism import artifacts, experiment

from slime.rollout.sample_hooks import apply_rollout_sample_hooks, rollout_context
from slime.utils.types import Sample

NUM_GPUS = 0


async def score(args, sample, **kwargs):
    if isinstance(sample, list):
        return [await score(args, item, **kwargs) for item in sample]
    if getattr(args, "scoring_error", False):
        sample.metadata["omni_reward"] = {"status": "error", "components": {}, "errors": {"wer": "RuntimeError"}}
        raise RuntimeError("score service unavailable")
    value = {0: 0.2, 1: 0.6}.get(sample.index, 1.0)
    sample.metadata["omni_reward"] = {
        "status": "success",
        "total": value,
        "components": {
            "wer": {"raw": 1 - value, "transformed": value, "status": "success", "identity": {"fingerprint": "wer-v1"}}
        },
    }
    return value


async def alternate_score(args, sample, **kwargs):
    assert kwargs == {"test_flag": 42}
    return 0.75


@pytest.fixture
def args(tmp_path):
    return SimpleNamespace(
        prism_artifact_dir=str(tmp_path / "running_round_0"),
        omni_audio_dir=str(tmp_path / "audio"),
        prism_reward_function_path=f"{__name__}.score",
        rollout_sample_hook_path=["examples.prism.artifacts.prepare_sample"],
        prompt_data="training.jsonl",
        n_samples_per_prompt=2,
        prism_rollout_log_function_path=None,
        prism_eval_rollout_log_function_path=None,
        use_tensorboard=False,
        use_wandb=False,
    )


def sample(args, index=0, *, frames=4, **metadata):
    root = Path(args.omni_audio_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"original-{index}.wav"
    # Archival preserves bytes. WAV validation is owned by the Omni plugin and
    # covered by its protocol tests, not duplicated in the filesystem hook.
    path.write_bytes(b"original waveform")
    reference = root.parent / "reference.wav"
    reference.write_bytes(b"reference waveform")
    return Sample(
        index=index,
        group_index=index // 2,
        prompt="你好",
        status=Sample.Status.COMPLETED,
        response_length=frames,
        metadata={
            "id": "data/中文",
            "domain": "chinese",
            "reward_category": "base",
            "omni_frames": frames,
            "omni_finish_reason": "stop",
            "omni_policy_version": 17,
            "omni_audio_path": str(path) if frames else None,
            "omni_prompt": {
                "transcript_target": "你好",
                "language": "zh",
                "references": [{"id": "speaker", "uri": str(reference)}],
            },
            "omni_rollout": {
                "model_identity": {"config_sha256": "student"},
                "sampling": {"max_frames": 12},
                "request_id": f"request-{index}",
                "replay_inputs": {"huge_tensor": [1, 2, 3]},
            },
            **metadata,
        },
    )


async def archive_and_score(args, item, *, rollout_id=7, evaluation=False, **kwargs):
    with rollout_context(rollout_id):
        await apply_rollout_sample_hooks(args, item, evaluation=evaluation)
    item.reward = await artifacts.reward(args, item, **kwargs)
    return item


def test_full_archive_keeps_audio_reference_identity_and_nonselected_candidates(args):
    candidates = [sample(args, index) for index in range(4)]

    async def run():
        return await asyncio.gather(*(archive_and_score(args, item) for item in candidates))

    asyncio.run(run())
    metrics = artifacts._summarize(args, 7, candidates[:2])
    record_file = Path(args.prism_artifact_dir) / "records/train/rollout_0000007.jsonl"
    records = [json.loads(line) for line in record_file.read_text().splitlines()]
    assert len(records) == 4
    assert [r["selected_for_training"] for r in records] == [True, True, False, False]
    for record in records:
        audio = Path(record["audio_path"])
        assert audio.read_bytes() == b"original waveform"
        assert "train/rollout_0000007/group_" in str(audio)
        assert (audio.parent / "prompt_text.txt").read_text() == "你好"
        assert (audio.parent / "ref_speaker.wav").resolve().read_bytes() == b"reference waveform"
        assert record["policy_version"] == 17 and record["generation_rollout_id"] == 7
        assert record["reward_details"]["components"]["wer"]["identity"]["fingerprint"] == "wer-v1"
        assert "replay_inputs" not in json.dumps(record)
    assert metrics["rollout/prism/all/group_reward/total/zero_std_ratio"] == 0.5
    assert metrics["rollout/prism/selected/group_reward/total/std_mean"] == pytest.approx(0.2)
    assert metrics["rollout/prism/not_selected/group_reward/total/zero_std_ratio"] == 1
    assert metrics["rollout/prism/reward_category/base/scored_count"] == 2


def test_scoring_failure_preserves_waveform_and_partial_result_and_raises(args):
    item = sample(args)
    args.scoring_error = True
    with pytest.raises(RuntimeError, match="score service unavailable"):
        asyncio.run(archive_and_score(args, item))
    record = json.loads(Path(item.metadata["prism_artifact"]["record_path"]).read_text())
    assert record["reward_status"] == "error" and record["reward"] is None
    assert record["reward_details"]["errors"] == {"wer": "RuntimeError"}
    assert Path(record["audio_path"]).exists()


def test_per_dataset_reward_selection_and_kwargs_survive_archival(args):
    item = sample(args)
    item.custom_rm_path = f"{__name__}.alternate_score"
    asyncio.run(archive_and_score(args, item, test_flag=42))
    assert item.reward == 0.75
    assert item.metadata["prism_artifact"]["reward_function_path"] == f"{__name__}.alternate_score"


def test_batched_rewards_keep_original_shape(args):
    candidates = [sample(args, i) for i in range(2)]

    async def run():
        with rollout_context(7):
            await apply_rollout_sample_hooks(args, candidates)
        assert await artifacts.reward(args, candidates) == [0.2, 0.6]

    asyncio.run(run())
    assert all(
        json.loads(Path(item.metadata["prism_artifact"]["record_path"]).read_text())["reward_status"] == "success"
        for item in candidates
    )


def test_zero_frames_have_records_without_waveform(args):
    item = sample(args, frames=0)
    asyncio.run(archive_and_score(args, item))
    record = json.loads(Path(item.metadata["prism_artifact"]["record_path"]).read_text())
    assert record["frames"] == 0 and record["audio_path"] is None


def test_eval_datasets_and_repeated_evaluation_at_same_step_do_not_collide(args):
    for dataset, event in (("test/a", "before"), ("test_a", "before"), ("test/a", "after")):
        item = sample(args, eval_dataset_name=dataset, eval_prompt_index=0, eval_sample_index=0, eval_run_id=event)
        asyncio.run(archive_and_score(args, item, evaluation=True))
        artifacts._summarize(args, 7, [item], dataset=dataset, group_size=1)
    records = list((Path(args.prism_artifact_dir) / "records/eval").glob("*.jsonl"))
    assert len(records) == 3
    paths = [json.loads(path.read_text())["audio_path"] for path in records]
    assert len(set(paths)) == 3 and all(Path(path).exists() for path in paths)


def test_summary_of_restored_samples_preserves_generation_version_and_adds_consumption_step(args):
    item = sample(args)
    asyncio.run(archive_and_score(args, item, rollout_id=6))
    artifacts._summarize(args, 7, [item])
    record = json.loads((Path(args.prism_artifact_dir) / "records/train/rollout_0000007.jsonl").read_text())
    assert record["generation_rollout_id"] == 6 and record["training_rollout_id"] == 7
    assert record["policy_version"] == 17


def test_component_metrics_ignore_inactive_components_and_incomplete_groups(args):
    items = [sample(args, index) for index in range(3)]
    for item in items:
        asyncio.run(archive_and_score(args, item))
    records = [json.loads(Path(item.metadata["prism_artifact"]["record_path"]).read_text()) for item in items]
    records[0]["reward_details"]["components"]["sim"] = {"raw": 0.9, "transformed": 0.9, "status": "success"}
    metrics = artifacts.compute_metrics(records, 2)
    assert metrics["component/sim/raw/mean"] == 0.9
    assert metrics["component/sim/active_ratio"] == pytest.approx(1 / 3)
    assert metrics["complete_group_count"] == 1 and metrics["incomplete_group_count"] == 1
    assert "group_reward/sim/std_mean" not in metrics


def test_corpus_error_rate_and_coverage_keep_languages_and_failures_separate():
    def row(index, language, errors, tokens, status="success"):
        return dict(
            dataset="eval",
            prompt_index=index,
            frames=10,
            audio_seconds=1.0,
            generation_status="completed",
            reward_status=status,
            reward=0.5,
            reward_details={
                "components": {
                    "wer": dict(
                        raw=errors / tokens,
                        transformed=0.5,
                        status="success",
                        details=dict(language=language, errors=errors, reference_tokens=tokens),
                    )
                }
            },
        )

    rows = [row(0, "en", 1, 2), row(1, "en", 1, 8), row(2, "zh", 3, 3), row(3, "en", 0, 4, "error")]
    # A completed WER component still counts when another component or the aggregate failed.
    rows.append(
        dict(
            dataset="eval",
            prompt_index=4,
            frames=0,
            audio_seconds=None,
            generation_status="completed",
            reward_status="success",
            reward=0,
            reward_details={"components": {"wer": dict(raw=None, transformed=0, status="invalid_audio")}},
        )
    )
    metrics = artifacts.compute_metrics(rows, group_size=1)
    assert metrics["component/wer/language/en/corpus_error_rate"] == pytest.approx(2 / 14)
    assert metrics["component/wer/language/zh/corpus_error_rate"] == 1
    assert metrics["component/wer/coverage_ratio"] == 0.8
    assert metrics["scored_ratio"] == 0.8
    assert metrics["audio_seconds/coverage_ratio"] == 0.8
    assert metrics["zero_frame_ratio"] == 0.2
    assert not any(key.startswith("group_reward/") for key in metrics)


def test_quality_metrics_keep_attribute_coverage_and_service_errors():
    row = dict(
        dataset="eval",
        prompt_index=0,
        frames=10,
        generation_status="completed",
        reward_status="error",
        reward=None,
        reward_details={
            "components": {
                "judge_clap": dict(
                    raw=0.5,
                    transformed=0.5,
                    status="success",
                    details=dict(queries=["Speed: slow", "Gender: female"], attribute_scores=[0.2, 0.8]),
                )
            },
            "errors": {"wer": "TimeoutError"},
        },
    )
    metrics = artifacts.compute_metrics([row], group_size=1)
    assert metrics["component/judge_clap/attribute/Speed/mean"] == 0.2
    assert metrics["component/judge_clap/attribute/Gender/mean"] == 0.8
    assert metrics["component/judge_clap/coverage_ratio"] == 1
    assert metrics["component/wer/coverage_ratio"] == 0
    assert metrics["component/wer/error_ratio"] == 1
    assert "component/wer/raw/mean" not in metrics


@pytest.mark.parametrize("category", [None, "base"])
def test_fixed_reward_archives_the_executed_category(args, monkeypatch, category):
    from test_tts_reward_plugin import Service, config

    from slime_plugins import tts_reward

    service = Service()
    monkeypatch.setattr(tts_reward, "get", service.get)
    monkeypatch.setattr(tts_reward, "post", service.post)
    args.prism_reward_function_path = "examples.prism.rewards.voiceclone.reward"
    args.omni_reward_config = config(wer=1, sim=1)
    item = sample(args)
    item.metadata["omni_prompt"]["reference_targets"] = {"timbre": "speaker"}
    if category is None:
        item.metadata.pop("reward_category")
    else:
        item.metadata["reward_category"] = category
    asyncio.run(archive_and_score(args, item))
    record = json.loads(Path(item.metadata["prism_artifact"]["record_path"]).read_text())
    assert record["reward_category"] == "voiceclone"
    assert record["requested_reward_category"] == category
    assert set(record["reward_details"]["components"]) == {"wer", "sim"}
    metrics = artifacts._summarize(args, 7, [item], group_size=1)
    assert metrics["rollout/prism/reward_category/voiceclone/scored_count"] == 1


def test_shared_audio_storage_keeps_attempts_and_statistics_separate(args, tmp_path):
    first = sample(args)
    asyncio.run(archive_and_score(args, first))
    first_audio = Path(first.metadata["omni_audio_path"])
    first_prompt = (first_audio.parent / "prompt_text.txt").read_text()
    second_args = SimpleNamespace(**vars(args))
    second_args.prism_artifact_dir = str(tmp_path / "another-experiment/job/running_round_0")
    second = sample(second_args)
    second.metadata["omni_prompt"]["transcript_target"] = "另一次实验使用不同文本。"
    asyncio.run(archive_and_score(second_args, second))
    metrics = artifacts._summarize(second_args, 7, [second], group_size=1)
    assert metrics["rollout/prism/all/sample_count"] == 1
    assert first_audio.parent != Path(second.metadata["omni_audio_path"]).parent
    assert (first_audio.parent / "prompt_text.txt").read_text() == first_prompt
    assert first_audio.read_bytes() == b"original waveform"


def _template_groups(args, filename, group_size):
    from slime.data.data_source import RolloutDataSource
    from slime.utils.data import Dataset

    path = REPO / "examples/prism/data" / filename
    args.prompt_data = str(path)
    args.n_samples_per_prompt = group_size
    dataset = Dataset(str(path), tokenizer=None, processor=None, max_length=None, prompt_key="script")
    assert len({item.metadata["id"] for item in dataset}) == len(dataset)
    # Use the real grouping method without loading HF assets or starting Ray.
    source = object.__new__(RolloutDataSource)
    source.args = SimpleNamespace(n_samples_per_prompt=group_size, rollout_shuffle=False)
    source.dataset = dataset
    source.sample_offset = source.epoch_id = source.sample_group_index = source.sample_index = 0
    return source.get_samples(len(dataset))


@pytest.mark.parametrize(
    "filename,reward_path,categories",
    [
        ("grpo-base.example.jsonl", "examples.prism.rewards.base.reward", ["base", "base"]),
        ("grpo-voiceclone.example.jsonl", "examples.prism.rewards.voiceclone.reward", ["voiceclone", "voiceclone"]),
        (
            "grpo-instruction.example.jsonl",
            "examples.prism.rewards.instruction.reward",
            ["instruction", "instruction"],
        ),
        (
            "mixrl.example.jsonl",
            "examples.prism.rewards.reward",
            ["base", "voiceclone", "instruction", "voiceclone_gated"],
        ),
    ],
)
def test_grpo_templates_route_native_groups_and_archive_actual_rewards(
    args, tmp_path, monkeypatch, filename, reward_path, categories
):
    from test_omni_rollout_plugin import geometry, output
    from test_tts_reward_plugin import Service, config

    from slime_plugins import omni_rollout, tts_reward

    service = Service()
    monkeypatch.setattr(tts_reward, "get", service.get)
    monkeypatch.setattr(tts_reward, "post", service.post)
    args.prism_reward_function_path = reward_path
    args.omni_reward_config = config(wer=1, sim=1, judge_clap=1)
    groups = _template_groups(args, filename, 2)
    audio = Path(args.omni_audio_dir)
    audio.mkdir()
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference waveform")
    candidates = [item for group in groups for item in group]
    for item in candidates:
        prompt = item.metadata["omni_prompt"]
        assert "reference_targets" not in prompt
        references = prompt.get("references", {})
        for slot in references:
            references[slot] = str(reference)
        payload = omni_rollout.build_generate_payload(args, item, {"max_new_tokens": 12})
        assert payload["prompt"]["script"] == item.prompt
        candidate = audio / f"sample-{item.index}.wav"
        candidate.write_bytes(b"generated waveform")
        omni_rollout.parse_rollout(output(), item, geometry())
        item.metadata["omni_audio_path"] = str(candidate)

    async def run():
        with rollout_context(7):
            await apply_rollout_sample_hooks(args, groups)
        values = await artifacts.reward(args, candidates)
        for item, value in zip(candidates, values, strict=True):
            item.reward = value

    asyncio.run(run())
    components = {
        "base": {"wer"},
        "voiceclone": {"wer", "sim"},
        "instruction": {"wer", "judge_clap"},
        "voiceclone_gated": {"wer", "sim"},
    }
    for group_index, (group, category) in enumerate(zip(groups, categories, strict=True)):
        assert {item.group_index for item in group} == {group_index}
        for item in group:
            record = json.loads(Path(item.metadata["prism_artifact"]["record_path"]).read_text())
            assert record["reward_category"] == category
            assert record["reward_function_path"] == reward_path
            assert set(record["reward_details"]["components"]) == components[category]
            assert record["data_id"] == item.metadata["id"]
            if "sim" in components[category]:
                assert record["reward_details"]["components"]["sim"]["details"]["reference_id"] == "audio1"
                assert (Path(record["audio_path"]).parent / "ref_audio1.wav").resolve() == reference
    metrics = artifacts._summarize(args, 7, candidates)
    assert metrics["rollout/prism/all/sample_count"] == len(candidates)
    assert metrics["rollout/prism/selected/complete_group_count"] == len(groups)


def test_mopd_template_routes_teachers_and_archives_without_waveform(args, monkeypatch):
    from test_omni_rollout_plugin import geometry, output, teacher_binding, teacher_output

    from slime_plugins import omni_rollout

    groups = _template_groups(args, "mopd.example.jsonl", 1)
    candidates = [group[0] for group in groups]
    args.prism_reward_function_path = "examples.prism.hooks.zero_reward"
    args.rollout_sample_hook_path.append("slime_plugins.omni_rollout.score_teacher")
    models = {key: teacher_binding() | {"endpoint": f"http://{key}"} for key in ("zh_teacher", "en_teacher")}
    routes = {"chinese": "zh_teacher", "english": "en_teacher"}
    args.omni_teacher_config = {"models": models, "by_domain": routes}
    requests = []

    async def score_teacher(_args, url, payload, **kwargs):
        binding = next(binding for binding in models.values() if url == binding["endpoint"] + "/score_actions")
        requests.append(url)
        return {"version": 1, "results": [teacher_output(payload["samples"][0], binding)]}

    monkeypatch.setattr(omni_rollout, "_request", score_teacher)
    for item in candidates:
        omni_rollout.build_generate_payload(args, item, {"max_new_tokens": 12})
        omni_rollout.parse_rollout(output(), item, geometry())
        item.metadata["omni_audio_path"] = None

    async def run():
        with rollout_context(7):
            await apply_rollout_sample_hooks(args, groups)
        values = await artifacts.reward(args, candidates)
        assert values == [0.0, 0.0]
        for item, value in zip(candidates, values, strict=True):
            item.reward = value

    asyncio.run(run())
    assert requests == ["http://zh_teacher/score_actions", "http://en_teacher/score_actions"]
    for item in candidates:
        assert "policy_teacher_code_logprobs" in item.multimodal_train_inputs
        assert "policy_teacher_stop_logprobs" in item.multimodal_train_inputs
        record = json.loads(Path(item.metadata["prism_artifact"]["record_path"]).read_text())
        assert record["audio_path"] is None and record["reward"] == 0
        assert record["teacher"]["key"] == routes[item.metadata["domain"]]
        assert record["teacher"]["weight_sha256"] == models[record["teacher"]["key"]]["weight_sha256"]
    metrics = artifacts._summarize(args, 7, candidates)
    assert metrics["rollout/prism/domain/chinese/scored_count"] == 1
    assert metrics["rollout/prism/domain/english/scored_count"] == 1
    assert not list(artifacts.audio_archive_root(args).rglob("*.wav"))


def test_attempt_preparation_snapshots_and_refuses_to_overwrite(tmp_path):
    root = tmp_path / "attempt"
    config = tmp_path / "custom.yaml"
    config.write_text(
        yaml.safe_dump({"omni_return_audio": True, "custom_rm_path": "user.config.reward", "api_key": "test-secret"})
    )
    command = [
        "python",
        "train.py",
        "--custom-config-path",
        str(config),
        "--custom-rm-path",
        "user.cli.reward",
        "--save",
        "explicit-checkpoints",
        "--load",
        "previous",
        "--rollout-sample-hook-path",
        "user.teacher.hook",
    ]
    resolved = experiment.prepare_run(root, command, identity={"running_round": 1})
    assert resolved[-2:] == ["--custom-config-path", str(root / "config/experiment.yaml")]
    runtime = yaml.safe_load((root / "config/experiment.yaml").read_text())
    assert runtime["prism_reward_function_path"] == "user.config.reward"
    assert runtime["rollout_sample_hook_path"] == ["examples.prism.artifacts.prepare_sample", "user.teacher.hook"]
    assert runtime["omni_return_audio"] is True
    assert (root / "config/experiment.yaml").stat().st_mode & 0o777 == 0o600
    manifest = json.loads((root / "config/launch.json").read_text())
    assert manifest["resume_from"] == "previous" and manifest["checkpoint_dir"] == "explicit-checkpoints"
    assert manifest["configuration"]["api_key"] == "<redacted>"
    before = (root / "config/launch.json").read_bytes()
    with pytest.raises(FileExistsError, match="Attempt already exists"):
        experiment.prepare_run(root, command, identity={})
    assert (root / "config/launch.json").read_bytes() == before


def test_dry_run_has_no_filesystem_side_effects(tmp_path):
    root = tmp_path / "dry"
    experiment.prepare_run(root, ["python", "train.py", "--custom-rm-path", "user.reward"], identity={}, dry_run=True)
    assert not root.exists()


def test_native_generation_reward_and_tensorboard_pipeline(args, tmp_path, monkeypatch):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from test_omni_rollout_plugin import geometry, output
    from test_omni_rollout_plugin import sample as omni_sample
    from test_tts_reward_plugin import Service, config

    from slime.rollout import sglang_rollout
    from slime.utils.misc import SingletonMeta
    from slime_plugins import omni_rollout, tts_reward

    service = Service()
    monkeypatch.setattr(tts_reward, "get", service.get)
    monkeypatch.setattr(tts_reward, "post", service.post)
    monkeypatch.setattr(omni_rollout, "_config", lambda path: geometry())
    response = output()
    blob = io.BytesIO()
    with wave.open(blob, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(b"\0\0" * 32)
    response["audio"] = {"data": base64.b64encode(blob.getvalue()).decode(), "format": "wav", "sample_rate": 24000}

    async def generate_response(*_args, **kwargs):
        return response

    async def pace():
        pass

    monkeypatch.setattr(omni_rollout, "_request", generate_response)
    state = SimpleNamespace(
        semaphore=asyncio.Semaphore(1),
        generation_pacer=SimpleNamespace(wait=pace),
        aborted=False,
        dp_rank_context=nullcontext,
        cancellable_tasks=set(),
    )
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
    monkeypatch.setattr(SingletonMeta, "_instances", {})
    args.partial_rollout = args.group_rm = False
    args.custom_generate_function_path = "slime_plugins.omni_rollout.generate"
    args.custom_rm_path = artifacts.REWARD_HOOK
    args.prism_reward_function_path = "examples.prism.rewards.reward"
    args.omni_reward_config = config(wer=1)
    args.hf_checkpoint = "fixture"
    args.sglang_router_ip, args.sglang_router_port = "127.0.0.1", 12345
    args.n_samples_per_prompt = 1
    args.wandb_always_use_train_step = False
    args.use_tensorboard = True
    args.tensorboard_dir = str(tmp_path / "tb")
    args.tb_project_name = args.tb_experiment_name = None
    item = omni_sample()
    reference = tmp_path / "reference.wav"
    reference.write_bytes(blob.getvalue())
    item.metadata["references"][0]["uri"] = str(reference)
    item.metadata["reward_category"] = "base"

    async def run():
        with rollout_context(7):
            return await sglang_rollout.generate_and_rm(args, item, {"max_new_tokens": 12})

    actual = asyncio.run(run())
    assert actual.reward == pytest.approx(2 / 3)
    assert actual.metadata["omni_reward"]["category"] == "base"
    assert Path(actual.metadata["omni_audio_path"]).read_bytes() == blob.getvalue()
    assert artifacts.log_rollout_data(7, args, [actual], {}, 0.1) is False
    accumulator = EventAccumulator(args.tensorboard_dir).Reload()
    event = accumulator.Scalars("rollout/prism/selected/reward/mean")[0]
    assert event.step == 7 and event.value == pytest.approx(actual.reward)
    assert accumulator.Scalars("rollout/prism/reward_category/base/component/wer/raw/mean")[0].value == pytest.approx(
        0.5
    )


def test_tensorboard_explicit_directory_beats_worker_environment_and_flushes(tmp_path, monkeypatch):
    from slime.observability import tensorboard_utils
    from slime.utils.misc import SingletonMeta

    calls = []

    class Writer:
        def __init__(self, directory):
            calls.append(("directory", directory))

        def add_scalar(self, name, value, step):
            calls.append((name, value, step))

        def flush(self):
            calls.append("flush")

        def close(self):
            pass

    monkeypatch.setattr(tensorboard_utils, "SummaryWriter", Writer)
    monkeypatch.setenv("TENSORBOARD_DIR", str(tmp_path / "old-worker-env"))
    monkeypatch.setattr(SingletonMeta, "_instances", {})
    writer = tensorboard_utils._TensorboardAdapter(
        SimpleNamespace(
            use_tensorboard=True,
            tensorboard_dir=str(tmp_path / "round1/tensorboard"),
            tb_project_name=None,
            tb_experiment_name=None,
        )
    )
    writer.log({"reward": 0.7}, step=17)
    assert calls == [("directory", str(tmp_path / "round1/tensorboard")), ("reward", 0.7, 17), "flush"]


@pytest.mark.parametrize(
    "mode,colocate,diagnostic", [("grpo", "0", "0"), ("grpo", "1", "0"), ("mopd", "0", "0"), ("grpo", "0", "1")]
)
def test_actual_launcher_isolates_attempt_and_preserves_checkpoint_override(tmp_path, mode, colocate, diagnostic):
    from test_prism_argument_configuration import _synthetic_prism_config

    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text(json.dumps(_synthetic_prism_config()))
    native = tmp_path / "native"
    native.mkdir()
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("{}\n")
    config = tmp_path / "services.yaml"
    config.write_text("{}\n")
    capture = tmp_path / "command.json"
    python = tmp_path / "training-python"
    python.write_text(
        f"#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\nif len(sys.argv)>1 and sys.argv[1].endswith('/train.py'):\n    Path(os.environ['TEST_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\nelse:\n    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
    )
    python.chmod(0o755)
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "TRAIN_PYTHON": str(python),
        "HF_CHECKPOINT": str(hf),
        "PRISM_PRETRAINED_CHECKPOINT": str(native),
        "PROMPT_DATA": str(prompts),
        "OMNI_CONFIG": str(config),
        "REWARD_CONFIG": str(config),
        "TEACHER_CONFIG": str(config),
        "TEST_CAPTURE": str(capture),
        "OUTPUT_ROOT": str(tmp_path / "runs"),
        "RUN_NAME": "experiment",
        "JOB_TAG": "job",
        "RUNNING_ROUND": "2",
        "OUTPUT_DIR": str(tmp_path / "explicit-checkpoints"),
        "USE_WANDB": "1",
        "COLOCATE": colocate,
        "DIAGNOSTIC_LOGS": diagnostic,
    }
    for key in (
        "RUN_DIR",
        "RL_LOAD",
        "PRISM_DRY_RUN",
        "AUDIO_DIR",
        "SLIME_LOG_LEVEL",
        "NCCL_DEBUG",
        "RAY_DEDUP_LOGS",
        "SGLANG_LOG_LEVEL",
        "SGLANG_LOG_LEVEL_HTTP",
        "ROUTER_LOG_LEVEL",
    ):
        environment.pop(key, None)
    result = subprocess.run(
        ["bash", str(REPO / f"examples/prism/run-{mode}.sh")],
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    command = json.loads(capture.read_text())
    options = experiment.command_options(command)
    root = tmp_path / "runs/experiment/job/running_round_2"
    assert options["save"] == [str(tmp_path / "explicit-checkpoints")]
    assert ("colocate" in options) == (colocate == "1")
    assert options["sglang_log_level"] == ["debug" if diagnostic == "1" else "warning"]
    assert options["sglang_log_level_http"] == ["info" if diagnostic == "1" else "warning"]
    assert options["router_log_level"] == ["debug" if diagnostic == "1" else "warn"]
    train_environment = json.loads(options["train_env_vars"][-1])
    assert train_environment["SLIME_LOG_LEVEL"] == ("DEBUG" if diagnostic == "1" else "INFO")
    assert train_environment["NCCL_DEBUG"] == ("INFO" if diagnostic == "1" else "WARN")
    assert train_environment["RAY_DEDUP_LOGS"] == ("0" if diagnostic == "1" else "1")
    assert options["tensorboard_dir"] == [str(root / "tensorboard")]
    assert options["wandb_dir"] == [str(root / "wandb")]
    assert options["omni_audio_dir"] == [str(root / "rollout_audio")]
    runtime = yaml.safe_load(Path(options["custom_config_path"][-1]).read_text())
    assert runtime["prism_artifact_dir"] == str(root)
    assert runtime["prism_reward_function_path"] == (
        "slime_plugins.tts_reward.reward" if mode == "grpo" else "examples.prism.hooks.zero_reward"
    )
    assert runtime["omni_return_audio"] == (mode == "grpo")
    if mode == "mopd":
        assert runtime["rollout_sample_hook_path"][-1] == "slime_plugins.omni_rollout.score_teacher"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
