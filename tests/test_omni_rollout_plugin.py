"""CPU contracts for the real Omni trace/action-score wire schemas."""

from __future__ import annotations

import asyncio
import base64
import copy
import importlib
import io
import json
import math
import os
import sys
import time
import types
import wave
from pathlib import Path

import pytest
import torch

from slime.rollout import on_policy_distillation as opd
from slime.utils.types import Sample
from slime_plugins import omni_rollout as plugin
from slime_plugins.prism import http_client as prism_http
from slime_plugins.prism import protocol as prism_protocol
from slime_plugins.prism import teacher as teacher_plugin
from slime_plugins.prism.language import SUPPORTED_LANGUAGES

NUM_GPUS = 0


def geometry():
    identity = dict(
        policy_family="moss_tts_prism",
        architecture="MOSS-TTS-Prism",
        config_sha256="student",
        modeling_sha256="model",
        processor_sha256="processor",
        prompt_protocol_sha256="prompt",
        n_vq=2,
        audio_vocab_size=8,
        sample_rate=24000,
        stop_semantics="pre_frame_threshold_v1",
    )
    return prism_protocol.PrismConfig(2, 8, 32, 9, 0, 4, 24000, 128, (1,), identity)


def sample(index=7):
    return Sample(
        index=index,
        group_index=2,
        prompt="你好",
        metadata={
            "domain": "chinese",
            "omni_prompt": {
                "modality": "speech",
                "references": {"audio1": "/shared/ref.wav"},
                "reference_attributes": {key: "audio1" for key in ("timbre", "accent", "prosody", "emotion")},
            },
            "transcript_target": "你好",
            "language": "zh",
        },
    )


def output(frames=2, finish="stop"):
    config = geometry()
    codes = [[1, 2], [3, 4]][:frames]
    logprobs = [[-0.4, -0.5], [-0.6, -0.7]][:frames]
    stops = frames + int(finish == "stop")
    actions = [0] * frames + ([1] if finish == "stop" else [])
    probabilities = [0.2, 0.3][:frames] + ([0.9] if finish == "stop" else [])
    stop_logprobs = [math.log(p if action else 1 - p) for p, action in zip(probabilities, actions, strict=True)]
    trace = dict(
        version=2,
        model_family="moss_tts_prism",
        stages=["tts_engine"],
        request_id="request-1",
        model_identity=config.identity,
        admission_weight_version="step_1",
        logprob_semantics=prism_protocol.LOGPROB_SEMANTICS,
        stop_semantics=prism_protocol.STOP_SEMANTICS,
        sampling=dict(
            audio_temperature=1.0,
            audio_top_p=1.0,
            audio_top_k=-1,
            audio_repetition_penalty=1.0,
            stop_sampling=True,
            max_new_tokens=10,
        ),
        replay_inputs=dict(
            layout=prism_protocol.LAYOUT,
            prompt_rows=[[3, 0, 0], [4, 0, 0]],
            item_kind=[0, 0],
            audio_role=[0, 0],
            successor_audio_mask=[False, False],
            successor_audio_codes=[[0, 0], [0, 0]],
            target_history_retention_mask=[[False, False]] * 2,
        ),
        action_streams=[
            dict(
                name="codes",
                stage="tts_engine",
                modality="audio",
                action_type="discrete",
                layout="time_depth_autoregressive",
                flatten_order="time_major",
                shape=[frames, 2],
                vocab_size=8,
                actions=codes,
                logprobs=logprobs,
                action_mask=[[1, 1]] * frames,
                deterministic_mask=None,
                channel_ids=[0, 1],
                channel_roles=["rvq_0", "rvq_1"],
            ),
            dict(
                name="stop_decisions",
                stage="tts_engine",
                modality="control",
                action_type="discrete",
                layout="pre_frame_autoregressive",
                flatten_order="time_major",
                shape=[stops],
                vocab_size=2,
                actions=actions,
                logprobs=stop_logprobs,
                action_mask=[1] * stops,
                deterministic_mask=None,
            ),
        ],
        non_action_outputs=[dict(name="stop_probabilities", stage="tts_engine", values=probabilities)],
        finish_reason=finish,
        total_action_count=frames * 2 + stops,
    )
    return dict(
        meta_info=dict(
            omni_rollout=trace,
            weight_version="step_1",
            finish_reason={"type": finish},
            prompt_tokens=2,
            completion_tokens=frames,
        ),
        audio=None,
    )


@pytest.mark.parametrize("frames,finish", [(2, "stop"), (2, "length"), (0, "stop")])
def test_trace_keeps_time_axis_and_exact_terminal_stop(frames, finish):
    item = sample()
    result = plugin.parse_rollout(output(frames, finish), item, geometry(), expected_weight_version="step_1")
    assert result is item and item.index == 7 and item.group_index == 2
    assert item.tokens == [3, 4] + [9] * frames
    assert item.response_length == frames
    assert item.status == (Sample.Status.COMPLETED if finish == "stop" else Sample.Status.TRUNCATED)
    assert item.rollout_log_probs is None
    mm = item.multimodal_train_inputs
    streams = item.policy_actions
    assert item.multimodal_train_inputs["audio_codes"].shape == (2 + frames, 2)
    assert streams["rvq"]["valid_mask"].sum().item() == frames * 2
    assert streams["stop"]["valid_mask"].sum().item() == frames + int(finish == "stop")
    assert streams["stop"]["actions"][streams["stop"]["valid_mask"]].tolist() == [0] * frames + (
        [1] if finish == "stop" else []
    )
    assert streams["rvq"]["valid_mask"][-1].tolist() == [False, False]
    assert streams["stop"]["valid_mask"][-1].item() == (finish == "stop")
    if frames:
        assert mm["target_history_retention_mask"][-1].tolist() == [True, False]
        torch.testing.assert_close(mm["successor_audio_codes"][1 : 1 + frames], mm["audio_codes"][2:])
        assert mm["audio_role"][2:].tolist() == [prism_protocol.TARGET] * frames
    item._validate_response_metadata_lengths()
    restored = Sample.from_dict(item.to_dict())
    assert restored.response_length == frames
    torch.testing.assert_close(restored.policy_actions["stop"]["valid_mask"], streams["stop"]["valid_mask"])


def test_trace_and_teacher_request_keep_independent_exact_snapshots():
    response, item = output(), sample()
    plugin.parse_rollout(response, item, geometry())
    snapshot = copy.deepcopy(item.metadata["omni_rollout"])
    request = teacher_plugin.teacher_request(item)
    expected = dict(
        version=2,
        sample_id="2:7:request-1",
        **{name: copy.deepcopy(snapshot["replay_inputs"][name]) for name in prism_protocol.REPLAY_FIELDS},
        codes=copy.deepcopy(snapshot["action_streams"][0]["actions"]),
        finish_reason="stop",
        temperature=1.0,
    )
    assert request == expected
    assert teacher_plugin.teacher_request_hash(request) == teacher_plugin.teacher_request_hash(expected)
    response["meta_info"]["omni_rollout"]["replay_inputs"]["prompt_rows"][0][0] = 17
    response["meta_info"]["omni_rollout"]["action_streams"][0]["actions"][0][0] = 6
    assert item.metadata["omni_rollout"] == snapshot
    request["prompt_rows"][0][0] = 20
    request["codes"][0][0] = 7
    assert item.metadata["omni_rollout"] == snapshot


@pytest.mark.parametrize(
    "mutation",
    [
        lambda response: response["meta_info"]["omni_rollout"].update(version=1),
        lambda response: response["meta_info"].update(weight_version="different"),
        lambda response: response["meta_info"]["omni_rollout"]["sampling"].update(audio_temperature=0.7),
        lambda response: response["meta_info"]["omni_rollout"]["action_streams"][0].update(channel_ids=[1, 0]),
        lambda response: response["meta_info"]["omni_rollout"]["action_streams"][0].update(
            logprobs=[[float("nan")] * 2] * 2
        ),
        lambda response: response["meta_info"]["omni_rollout"]["action_streams"][1].update(actions=[0, 1, 1]),
        lambda response: response["meta_info"]["omni_rollout"]["action_streams"][1].update(logprobs=[-5.0] * 3),
        lambda response: response["meta_info"]["omni_rollout"]["replay_inputs"].update(
            successor_audio_mask=[True, False]
        ),
        lambda response: response["meta_info"]["omni_rollout"]["replay_inputs"].update(audio_role=[0, 2]),
        lambda response: response["meta_info"]["omni_rollout"]["replay_inputs"].update(
            prompt_rows=[[3, 0, 0], [5, 0, 0]]
        ),
        lambda response: response["meta_info"].update(completion_tokens=3),
    ],
)
def test_trace_rejects_wrong_identity_distribution_carrier_and_scores(mutation):
    response = output()
    mutation(response)
    with pytest.raises(ValueError):
        plugin.parse_rollout(response, sample(), geometry())


def test_prompt_payload_uses_structured_business_fields_not_tokenization():
    item = sample()
    item.metadata["omni_prompt"]["global_instruction"] = {"Gender": "female"}
    args = types.SimpleNamespace(context_parallel_size=1, omni_return_audio=True, omni_model_name="prism")
    payload = plugin.build_generate_payload(args, item, {"max_new_tokens": 12})
    assert "input_ids" not in payload
    assert payload["prompt"]["script"] == "你好"
    assert payload["prompt"]["modality"] == "speech"
    assert "Language" not in payload["prompt"]["global_instruction"]
    assert "language" not in payload["prompt"]
    assert payload["prompt"]["references"] == item.metadata["omni_prompt"]["references"]
    assert payload["prompt"]["global_instruction"] == {"Gender": "female"}
    assert "transcript_target" not in payload["prompt"]
    assert payload["sampling_params"]["stop_sampling"] is True
    assert payload["return_omni_rollout"] and payload["return_audio"]
    with pytest.raises(ValueError, match="temperature"):
        plugin.build_generate_payload(args, item, {"max_new_tokens": 12, "temperature": 0.7})


@pytest.mark.parametrize(
    "references,attributes",
    [
        ({}, {}),
        ({"audio1": "/shared/speaker.wav"}, {"timbre": "audio1"}),
        ({"audio2": "/shared/style.wav", "audio1": "/shared/speaker.wav"}, {"timbre": "audio1", "prosody": "audio2"}),
        (
            {"audio3": "/shared/speaker.wav", "audio10": "/shared/style.wav"},
            {"timbre": "audio3", "emotion": "audio10"},
        ),
    ],
)
def test_reference_slots_keep_explicit_names_and_reward_mapping(references, attributes):
    from pathlib import Path
    from slime_plugins.prism import reward_adapter

    item = sample()
    item.metadata["omni_prompt"].update(references=references, reference_attributes=attributes)
    original = copy.deepcopy(item.metadata)
    payload = plugin.build_generate_payload(types.SimpleNamespace(), item, {"max_new_tokens": 12})
    assert payload["prompt"]["references"] == references
    assert payload["prompt"]["reference_attributes"] == attributes
    assert item.metadata == original
    plugin.parse_rollout(output(), item, geometry())
    assert item.metadata["omni_prompt"]["references"] == references
    assert "language" not in item.metadata["omni_prompt"]
    assert item.metadata["language"] == "zh"
    if references:
        assert reward_adapter.reference_audio(item.metadata["omni_prompt"], must_exist=False) == (
            attributes["timbre"],
            Path("/shared/speaker.wav"),
        )


@pytest.mark.parametrize(
    "references",
    [
        {"audio0": "/shared/ref.wav"},
        {"speaker": "/shared/ref.wav"},
        {"audio1": ""},
        {"audio1": "  "},
        {"audio1": None},
        {"audio1": {"uri": "/shared/ref.wav"}},
        [{"id": "audio1", "uri": "/shared/ref.wav"}],
    ],
)
def test_reference_slots_reject_invalid_names_and_uris(references):
    item = sample()
    item.metadata["omni_prompt"]["references"] = references
    with pytest.raises(ValueError, match="Prism references"):
        plugin.build_generate_payload(types.SimpleNamespace(), item, {"max_new_tokens": 12})


def test_unified_prompt_keeps_segments_music_attributes_and_reward_language_separate():
    item = sample()
    item.metadata["omni_prompt"].update(
        modality="sing",
        script=[{"text": "你好", "local_instruction": {"Emotion": "happy"}}, {"text": "。"}],
        global_instruction={"Genre": "Pop", "Mood": "happy"},
        global_tokens=77,
    )
    original = copy.deepcopy(item.metadata)
    payload = plugin.build_generate_payload(types.SimpleNamespace(), item, {"max_new_tokens": 120})
    assert payload["prompt"]["script"] == original["omni_prompt"]["script"]
    assert payload["prompt"]["global_instruction"] == {"Genre": "Pop", "Mood": "happy"}
    assert payload["prompt"]["global_tokens"] == 77
    assert "language" not in payload["prompt"]
    payload["prompt"]["script"][0]["text"] = "modified"
    assert item.metadata == original
    item.metadata["omni_prompt"]["global_instruction"]["Language"] = "Chinese"
    assert (
        plugin.build_generate_payload(types.SimpleNamespace(), item, {"max_new_tokens": 120})["prompt"][
            "global_instruction"
        ]["Language"]
        == "Chinese"
    )


@pytest.mark.parametrize("task_type", ["TTS", "Instruction Voice Clone", "Local Instruction", "sing", "custom-task"])
def test_task_type_is_metadata_and_does_not_select_model_conditions_teacher_or_reward(task_type):
    from examples.prism import rewards

    item = sample()
    item.metadata["reward_category"] = "base"
    args = types.SimpleNamespace()
    baseline = plugin.build_generate_payload(args, item, {"max_new_tokens": 12})
    item.metadata["task_type"] = task_type
    original = copy.deepcopy(item.metadata)
    payload = plugin.build_generate_payload(args, item, {"max_new_tokens": 12})
    assert payload == baseline
    assert item.metadata == original
    assert rewards._category(item) == "base"
    config = {
        "models": {"domain_teacher": {"endpoint": "http://teacher"}},
        "by_domain": {"chinese": "domain_teacher"},
    }
    assert opd.select_teacher(config, item)[0] == "domain_teacher"
    plugin.parse_rollout(output(), item, geometry())
    assert item.metadata["task_type"] == task_type
    assert "task_type" not in item.metadata["omni_prompt"]
    assert item.metadata["reward_category"] == "base"
    assert item.policy_teacher == "domain_teacher"


@pytest.mark.parametrize("task_type", [None, "", " ", False, 1, [], {}])
def test_task_type_rejects_empty_or_non_string_labels(task_type):
    item = sample()
    item.metadata["task_type"] = task_type
    with pytest.raises(ValueError, match="metadata.task_type must be a nonempty string"):
        plugin.build_generate_payload(types.SimpleNamespace(), item, {"max_new_tokens": 12})


@pytest.mark.parametrize("language", SUPPORTED_LANGUAGES)
def test_show_language_uses_canonical_metadata_without_mutating_source(language):
    item = sample()
    item.metadata.update(language=f" {language.swapcase()} ", show_language=True)
    item.metadata["omni_prompt"]["global_instruction"] = {
        "Language": "English",
        " language ": "French",
        "Gender": "female",
    }
    original = copy.deepcopy(item.metadata)
    payload = plugin.build_generate_payload(types.SimpleNamespace(), item, {"max_new_tokens": 12})
    assert payload["prompt"]["global_instruction"] == {"Language": language, "Gender": "female"}
    assert "show_language" not in payload["prompt"]
    assert "language" not in payload["prompt"]
    assert item.metadata == original


@pytest.mark.parametrize("language", [None, "zh", ""])
def test_hide_language_removes_explicit_headers_without_requiring_a_renderer_language(language):
    item = sample()
    item.metadata.update(language=language, show_language=False)
    item.metadata["omni_prompt"]["global_instruction"] = {
        "Language": "Chinese",
        " LANGUAGE ": "English",
        "Gender": "female",
    }
    original = copy.deepcopy(item.metadata)
    payload = plugin.build_generate_payload(types.SimpleNamespace(), item, {"max_new_tokens": 12})
    assert payload["prompt"]["global_instruction"] == {"Gender": "female"}
    assert item.metadata == original


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}])
def test_language_display_switch_requires_a_json_boolean(value):
    item = sample()
    item.metadata.update(language="Chinese", show_language=value)
    with pytest.raises(ValueError, match="metadata.show_language must be a Boolean"):
        plugin.build_generate_payload(types.SimpleNamespace(), item, {"max_new_tokens": 12})


@pytest.mark.parametrize("language", [None, "", " ", "zh", "en", "Portuguese", "Spanish", "unknown", 1, False, []])
def test_show_language_rejects_missing_names_codes_and_unsupported_languages(language):
    item = sample()
    item.metadata.pop("language")
    item.metadata["show_language"] = True
    if language is not None:
        item.metadata["language"] = language
    with pytest.raises(ValueError, match="metadata.language to be a supported Prism language name"):
        plugin.build_generate_payload(types.SimpleNamespace(), item, {"max_new_tokens": 12})


@pytest.mark.parametrize("direct_request", [False, True])
@pytest.mark.parametrize("show_language", [False, True])
def test_language_display_keeps_local_segments_and_complete_request_inputs(direct_request, show_language):
    item = sample()
    segments = [{"text": "你好"}, {"text": "。", "local_instruction": {"Emotion": "happy"}}]
    instruction = {"Language": "English", "Gender": "female", "Appearance": "tall"}
    item.metadata.update(language="Chinese", show_language=show_language)
    if direct_request:
        item.prompt = {"script": segments, "global_instruction": instruction}
    else:
        item.prompt = segments
        item.metadata["omni_prompt"]["global_instruction"] = instruction
    original = copy.deepcopy((item.prompt, item.metadata))
    payload = plugin.build_generate_payload(types.SimpleNamespace(), item, {"max_new_tokens": 12})
    expected = {"Gender": "female", "Appearance": "tall"}
    if show_language:
        expected["Language"] = "Chinese"
    assert payload["prompt"]["global_instruction"] == expected
    assert payload["prompt"]["script"] == segments
    assert (item.prompt, item.metadata) == original
    plugin.parse_rollout(output(), item, geometry())
    assert item.metadata["omni_prompt"] == payload["prompt"]
    assert item.metadata["show_language"] is show_language
    assert item.metadata["language"] == "Chinese"
    assert plugin._prompt(item) == payload["prompt"]


def test_example_datasets_load_with_explicit_language_controls():
    from slime.utils.data import Dataset

    directory = Path(__file__).resolve().parents[1] / "examples/prism/data"
    for path in sorted(directory.glob("*.jsonl")):
        dataset = Dataset(
            str(path), tokenizer=None, processor=None, max_length=None, prompt_key="script", metadata_key="metadata"
        )
        for item in dataset.samples:
            assert isinstance(item.metadata["task_type"], str) and item.metadata["task_type"].strip(), path
            assert item.metadata["language"] in SUPPORTED_LANGUAGES, path
            assert type(item.metadata["show_language"]) is bool, path
            assert "script" not in item.metadata["omni_prompt"], path
            assert "Language" not in item.metadata["omni_prompt"].get("global_instruction", {}), path
            payload = plugin.build_generate_payload(types.SimpleNamespace(), item, {"max_new_tokens": 12})
            assert "task_type" not in payload["prompt"], path
            actual = payload["prompt"]["global_instruction"].get("Language")
            expected = item.metadata["language"] if item.metadata["show_language"] else None
            assert actual == expected, path
            script = payload["prompt"]["script"]
            text = script if isinstance(script, str) else "".join(segment["text"] for segment in script)
            assert text == item.metadata["transcript_target"], path


@pytest.mark.parametrize("legacy", [{"task_type": "TTS"}, {"show_language": False}, {"tokens_control": False}])
def test_legacy_prompt_requires_explicit_migration(legacy):
    item = sample()
    item.metadata["omni_prompt"].update(legacy)
    with pytest.raises(ValueError, match="Unknown Prism prompt fields"):
        plugin.build_generate_payload(types.SimpleNamespace(), item, {"max_new_tokens": 12})


@pytest.mark.parametrize(
    "params,expected", [({"seed": 0}, 0), ({"sampling_seed": 321}, 321), ({"seed": 321, "sampling_seed": 321}, 321)]
)
def test_seed_aliases_keep_the_public_nonnegative_integer(params, expected):
    args = types.SimpleNamespace()
    payload = plugin.build_generate_payload(args, sample(), {"max_new_tokens": 12, **params})
    assert payload["sampling_params"]["seed"] == expected
    assert type(payload["sampling_params"]["seed"]) is int
    assert "sampling_seed" not in payload["sampling_params"]


@pytest.mark.parametrize(
    "params",
    [
        {"seed": 2, "sampling_seed": 3},
        {"seed": True},
        {"sampling_seed": True},
        {"seed": -1},
        {"sampling_seed": -1},
        {"seed": 1.5},
        {"sampling_seed": 1.5},
    ],
)
def test_seed_aliases_reject_conflicting_or_invalid_values(params):
    with pytest.raises(ValueError):
        plugin.build_generate_payload(types.SimpleNamespace(), sample(), {"max_new_tokens": 12, **params})


def _native_rollout_for_seed_tests():
    try:
        return importlib.import_module("slime.rollout.sglang_rollout")
    except ModuleNotFoundError as error:
        pytest.skip(f"Original rollout integration needs optional dependency {error.name}")


def test_original_deterministic_group_forwards_sampling_seed_to_omni(monkeypatch):
    rollout = _native_rollout_for_seed_tests()
    args = types.SimpleNamespace(sglang_enable_deterministic_inference=True, group_rm=False)
    state = types.SimpleNamespace(aborted=False, group_sampling_seeds=[321, 322])
    monkeypatch.setattr(rollout, "GenerateState", lambda args: state)
    observed = []

    async def capture_generate(args, item, sampling_params, *, evaluation=False):
        observed.append(plugin.build_generate_payload(args, item, sampling_params)["sampling_params"]["seed"])
        return item

    monkeypatch.setattr(rollout, "generate_and_rm", capture_generate)
    group = [sample(0), sample(1)]
    result = asyncio.run(rollout.generate_and_rm_group(args, group, {"max_new_tokens": 12}))
    assert result == group and observed == [321, 322]


def test_original_deterministic_eval_forwards_sampling_seed_to_omni(monkeypatch):
    from slime.utils.eval_config import EvalDatasetConfig

    rollout = _native_rollout_for_seed_tests()
    args = types.SimpleNamespace(
        sglang_enable_deterministic_inference=True,
        group_rm=False,
        rollout_seed=321,
        hf_checkpoint="fixture",
        multimodal_keys=None,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=True,
        eval_min_new_tokens=None,
        eval_reward_key=None,
        reward_key=None,
    )
    config = EvalDatasetConfig(
        name="omni-seed",
        path="unused",
        n_samples_per_eval_prompt=2,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_response_len=12,
    )
    cache_key = config.cache_key + (args.hf_checkpoint, False, None, None)
    monkeypatch.setattr(rollout, "EVAL_PROMPT_DATASET", {cache_key: types.SimpleNamespace(samples=[sample()])})

    async def capture_generate(args, item, sampling_params, *, evaluation=False):
        assert evaluation
        item.metadata["forwarded_seed"] = plugin.build_generate_payload(args, item, sampling_params)[
            "sampling_params"
        ]["seed"]
        item.reward = 0.0
        item.status = Sample.Status.COMPLETED
        return item

    monkeypatch.setattr(rollout, "generate_and_rm", capture_generate)
    result = asyncio.run(rollout.eval_rollout_single_dataset(args, 0, config))
    assert [item.metadata["forwarded_seed"] for item in result[config.name]["samples"]] == [321, 322]
    items = result[config.name]["samples"]
    assert [item.metadata["eval_sample_index"] for item in items] == [0, 1]
    assert all(item.metadata["eval_prompt_index"] == 0 for item in items)
    assert all(item.metadata["eval_dataset_name"] == config.name for item in items)
    event = items[0].metadata["eval_run_id"]
    assert all(item.metadata["eval_run_id"] == event for item in items)
    repeated = asyncio.run(rollout.eval_rollout_single_dataset(args, 0, config))
    assert repeated[config.name]["samples"][0].metadata["eval_run_id"] != event


def teacher_binding():
    return dict(endpoint="http://teacher")


def teacher_output(request):
    return dict(
        version=2,
        sample_id=request["sample_id"],
        input_sha256=teacher_plugin.teacher_request_hash(request),
        code_logprobs=[[-0.2, -0.3] for _ in request["codes"]],
        selected_stop_logprobs=[math.log(0.75)] * len(request["codes"])
        + ([math.log(0.25)] if request["finish_reason"] == "stop" else []),
        finish_reason=request["finish_reason"],
        temperature=1.0,
        weight_version="frozen_teacher",
        model_identity=geometry().identity,
        logprob_semantics=prism_protocol.LOGPROB_SEMANTICS,
        stop_semantics="pre_frame_threshold_v1",
    )


@pytest.mark.parametrize("reported_identity", [None, {"model": "different-teacher"}])
def test_teacher_scoring_only_needs_an_endpoint_and_accepts_identity_metadata(reported_identity):
    item = plugin.parse_rollout(output(), sample(), geometry())
    routes = []

    async def score(route, request):
        routes.append(route)
        result = teacher_output(request)
        result["model_identity"] = reported_identity
        return types.SimpleNamespace(value=result, batch_size=1, request_seconds=0.1, attempts=1)

    asyncio.run(
        teacher_plugin.score_teacher(
            types.SimpleNamespace(), item, binding=teacher_binding(), client=types.SimpleNamespace(score=score)
        )
    )
    assert routes == [("http://teacher", None)]
    assert item.metadata["omni_teacher_identity"] == reported_identity
    assert item.metadata["omni_teacher_status"] == "success"
    assert "teacher_log_probs" in item.policy_actions["rvq"]
    assert "teacher_log_probs" in item.policy_actions["stop"]


@pytest.mark.parametrize("field,value", [("code_logprobs", [[-0.2], [-0.2]]), ("selected_stop_logprobs", [-0.2])])
def test_teacher_score_shapes_still_match_student_actions(field, value):
    item = plugin.parse_rollout(output(), sample(), geometry())
    request = teacher_plugin.teacher_request(item)
    result = teacher_output(request)
    result[field] = value
    with pytest.raises(ValueError, match="teacher.*logprobs"):
        teacher_plugin.attach_teacher_scores(item, request, result)


@pytest.mark.parametrize("frames,finish", [(2, "stop"), (2, "length"), (0, "stop")])
def test_teacher_scores_exact_student_prefix_and_actual_stop_actions(frames, finish):
    item = plugin.parse_rollout(output(frames, finish), sample(), geometry())
    request = teacher_plugin.teacher_request(item)
    result = teacher_output(request)
    assert request["temperature"] == 1.0 and type(request["temperature"]) is float
    assert request["prompt_rows"] == [[3, 0, 0], [4, 0, 0]]
    assert "stop_decisions" not in request
    teacher_plugin.attach_teacher_scores(item, request, result)
    streams = item.policy_actions
    expected = [math.log(0.75)] * frames + ([math.log(0.25)] if finish == "stop" else [])
    torch.testing.assert_close(
        streams["stop"]["teacher_log_probs"][streams["stop"]["valid_mask"]], torch.tensor(expected)
    )
    assert item.teacher_log_probs is None
    assert streams["rvq"]["teacher_log_probs"][streams["rvq"]["valid_mask"].any(-1)].sum(-1).tolist() == pytest.approx(
        [-0.5] * frames
    )
    assert not streams["rvq"]["teacher_log_probs"].requires_grad


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_sha256", "wrong"),
        ("temperature", 0.7),
        ("sample_id", "wrong"),
        ("version", 1),
    ],
)
def test_teacher_rejects_stale_or_mismatched_scores(field, value):
    item = plugin.parse_rollout(output(), sample(), geometry())
    request = teacher_plugin.teacher_request(item)
    result = teacher_output(request)
    result[field] = value
    with pytest.raises(ValueError, match="differs"):
        teacher_plugin.attach_teacher_scores(item, request, result)


@pytest.mark.parametrize("score", [0.1, float("nan"), float("inf"), -float("inf"), True])
def test_teacher_invalid_selected_scores_are_rejected_without_clipping(score):
    item = plugin.parse_rollout(output(), sample(), geometry())
    request = teacher_plugin.teacher_request(item)
    result = teacher_output(request)
    result["selected_stop_logprobs"][0] = score
    with pytest.raises(ValueError):
        teacher_plugin.attach_teacher_scores(item, request, result)


@pytest.mark.parametrize("logits", [[0.0, 20.0], [200.0, 0.0]])
def test_teacher_selected_stop_scores_survive_probability_saturation(logits):
    item = plugin.parse_rollout(output(frames=1), sample(), geometry())
    request = teacher_plugin.teacher_request(item)
    result = teacher_output(request)
    scores = torch.tensor([logits, logits]).log_softmax(-1)
    expected = scores[torch.arange(2), torch.tensor([0, 1])]
    result["selected_stop_logprobs"] = expected.tolist()
    teacher_plugin.attach_teacher_scores(item, request, result)
    inputs = item.policy_actions
    selected = inputs["stop"]["teacher_log_probs"][inputs["stop"]["valid_mask"]]
    torch.testing.assert_close(selected, expected)
    assert torch.isfinite(selected).all()


@pytest.mark.parametrize("scores", [None, [], [[-0.5, -0.5]] * 2, [-0.5] * 3])
def test_teacher_selected_stop_scores_require_exact_trajectory_shape(scores):
    item = plugin.parse_rollout(output(frames=1), sample(), geometry())
    request = teacher_plugin.teacher_request(item)
    result = teacher_output(request)
    result["selected_stop_logprobs"] = scores
    with pytest.raises(ValueError):
        teacher_plugin.attach_teacher_scores(item, request, result)


def test_teacher_old_score_fields_do_not_replace_selected_scores():
    item = plugin.parse_rollout(output(frames=1), sample(), geometry())
    request = teacher_plugin.teacher_request(item)
    result = teacher_output(request)
    result.pop("selected_stop_logprobs")
    result.update(stop_probabilities=[0.25, 0.25], stop_logprobs=[[math.log(0.75), math.log(0.25)]] * 2)
    with pytest.raises(ValueError, match="selected stop logprobs"):
        teacher_plugin.attach_teacher_scores(item, request, result)


def test_real_hf_artifact_protocol_from_environment():
    """Repeat with PRISM_HF_TEST_ARTIFACT=<HF directory>; no weights are loaded."""
    checkpoint = os.environ.get("PRISM_HF_TEST_ARTIFACT")
    if not checkpoint:
        pytest.skip("Set PRISM_HF_TEST_ARTIFACT to inspect a real exported HF artifact")
    config = prism_protocol.PrismConfig.from_path(checkpoint)
    response = output()
    trace = response["meta_info"]["omni_rollout"]
    trace["model_identity"] = config.identity
    q = config.n_vq
    replay = trace["replay_inputs"]
    replay["prompt_rows"] = [[3] + [0] * q, [config.audio_start_token_id] + [0] * q]
    replay["successor_audio_codes"] = [[0] * q for _ in range(2)]
    replay["target_history_retention_mask"] = [[False] * q for _ in range(2)]
    codes = trace["action_streams"][0]
    codes.update(
        shape=[2, q],
        vocab_size=config.speech_vocab_size,
        actions=[[index % config.speech_vocab_size for index in range(q)] for _ in range(2)],
        logprobs=[[-1.0] * q for _ in range(2)],
        action_mask=[[1] * q for _ in range(2)],
        channel_ids=list(range(q)),
        channel_roles=[f"rvq_{index}" for index in range(q)],
    )
    trace["total_action_count"] = 2 * q + 3
    item = plugin.parse_rollout(response, sample(), config)
    assert item.tokens == [3, config.audio_start_token_id, config.text_pad_idx, config.text_pad_idx]
    assert item.policy_actions["rvq"]["rollout_log_probs"][item.policy_actions["rvq"]["valid_mask"].any(-1)].sum(
        -1
    ).tolist() == [-float(q), -float(q)]
    assert item.multimodal_train_inputs["audio_codes"].shape == (4, q)
    assert item.policy_actions["rvq"]["valid_mask"].sum().item() == 2 * q


def test_public_teacher_stage_reuses_score_endpoint_and_rejects_unmapped_domain(monkeypatch):
    item = plugin.parse_rollout(output(), sample(), geometry())
    binding = teacher_binding()
    args = types.SimpleNamespace(
        use_opd=True,
        opd_teacher_config=dict(
            score_function="slime_plugins.prism.teacher.score_teacher",
            models={"a": binding},
            by_domain={"chinese": "a"},
            default_teacher="a",
        ),
    )
    calls = []

    async def fake_post(url, payload, **kwargs):
        calls.append((url, kwargs))
        return {"version": 1, "results": [teacher_output(payload["samples"][0])]}

    monkeypatch.setattr(prism_http, "post", fake_post)
    assert asyncio.run(opd.score_teacher(args, item)) is item
    assert calls[0][0] == "http://teacher/score_actions" and calls[0][1]["max_retries"] == 1
    assert item.policy_teacher == "a"
    item = plugin.parse_rollout(output(), sample(), geometry())
    item.metadata["domain"] = "unmapped"
    with pytest.raises(ValueError, match="Unmapped"):
        asyncio.run(opd.score_teacher(args, item))
    assert len(calls) == 1


def named_teacher_binding():
    binding = teacher_binding()
    binding.pop("endpoint")
    binding["deployment"] = "teacher_a"
    return binding


def named_teacher_args(binding, configuration=None):
    return types.SimpleNamespace(
        use_opd=True,
        opd_teacher_config={
            "score_function": "slime_plugins.prism.teacher.score_teacher",
            "models": {"a": binding},
            "by_domain": {"chinese": "a"},
        },
        omni_model_name="student",
        sglang_model_routers={"teacher_a": ("127.0.0.1", 30001)},
        sglang_config=configuration,
        rollout_health_check_timeout=0.2,
    )


def worker(url="http://worker-a", *, health="healthy", routable=True, disabled=False):
    return dict(url=url, health_state=health, routable=routable, disabled=disabled, model=None)


def test_named_teacher_discovers_live_workers_then_scores_directly(monkeypatch):
    from slime.backends.sglang_utils.sglang_config import ModelConfig, SglangConfig

    binding = named_teacher_binding()
    configuration = SglangConfig(models=[ModelConfig(name="teacher_a", update_weights=False)])
    args = named_teacher_args(binding, configuration)
    gets, posts = [], []

    async def fake_get(url, **kwargs):
        gets.append(url)
        if url.endswith("/workers"):
            return {
                "workers": [
                    worker(),
                    worker("http://worker-b"),
                    worker("http://disabled", disabled=True),
                    worker("http://dead", health="dead"),
                ]
            }
        assert url in ("http://worker-a/health", "http://worker-b/health")
        return {"running": True}

    async def fake_post(url, payload, **kwargs):
        assert url in ("http://worker-a/score_actions", "http://worker-b/score_actions")
        posts.append(url)
        return {"version": 1, "results": [teacher_output(payload["samples"][0])]}

    monkeypatch.setattr(teacher_plugin, "get", fake_get)
    monkeypatch.setattr(prism_http, "post", fake_post)
    for _ in range(2):
        item = plugin.parse_rollout(output(), sample(), geometry())
        asyncio.run(opd.score_teacher(args, item))
        assert item.metadata["omni_teacher_deployment"] == "teacher_a"
        assert item.metadata["omni_teacher_endpoint"] + "/score_actions" == posts[-1]
        assert "teacher_log_probs" in item.policy_actions["stop"]
    assert posts[0] == posts[1]  # Stable group/sample dispatch, no global mutable round-robin state.
    assert gets.count("http://127.0.0.1:30001/workers") == 2


@pytest.mark.parametrize(
    "binding",
    [
        {"endpoint": "http://teacher", "deployment": "teacher_a"},
        {},
        {"endpoint": ""},
    ],
)
def test_teacher_binding_requires_one_valid_endpoint_or_deployment(monkeypatch, binding):
    monkeypatch.setattr(
        teacher_plugin, "get", lambda *args, **kwargs: pytest.fail("Invalid binding made an HTTP request")
    )
    with pytest.raises(ValueError):
        asyncio.run(teacher_plugin._teacher_endpoint(named_teacher_args(binding), sample(), binding))


@pytest.mark.parametrize("role", ["student-name", "update-true", "inferred-student", "unconfigured", "missing-router"])
def test_named_teacher_rejects_student_or_unconfigured_models_before_http(monkeypatch, role):
    from slime.backends.sglang_utils.sglang_config import ModelConfig, SglangConfig

    binding = named_teacher_binding()
    configuration = SglangConfig(models=[ModelConfig(name="teacher_a", update_weights=False)])
    args = named_teacher_args(binding, configuration)
    if role == "student-name":
        args.omni_model_name = "teacher_a"
    elif role == "update-true":
        configuration.models[0].update_weights = True
    elif role == "inferred-student":
        configuration.models[0].update_weights = None
        args.hf_checkpoint = "student-checkpoint"
        args.rollout_num_gpus_per_engine = 1
    elif role == "unconfigured":
        configuration.models = []
    else:
        args.sglang_model_routers = {}
    monkeypatch.setattr(
        teacher_plugin, "get", lambda *args, **kwargs: pytest.fail("Invalid role made an HTTP request")
    )
    with pytest.raises(ValueError):
        asyncio.run(teacher_plugin._teacher_endpoint(args, sample(), binding))


def test_named_teacher_resolves_explicit_frozen_model_from_yaml(monkeypatch, tmp_path):
    path = tmp_path / "deployments.yaml"
    path.write_text("sglang:\n- name: teacher_a\n  model_path: frozen-checkpoint\n  update_weights: false\n")
    binding = named_teacher_binding()
    args = named_teacher_args(binding, str(path))

    async def fake_get(url, **kwargs):
        return {"workers": [worker()]} if url.endswith("/workers") else {"running": True}

    monkeypatch.setattr(teacher_plugin, "get", fake_get)
    assert asyncio.run(teacher_plugin._teacher_endpoint(args, sample(), binding)) == "http://worker-a"


@pytest.mark.parametrize(
    "workers",
    [[], [worker(health="unhealthy")], [worker(routable=False)], [worker(disabled=True)], [worker(routable=1)]],
)
def test_named_teacher_rejects_router_workers_that_are_not_healthy_and_routable(monkeypatch, workers):
    async def fake_get(url, **kwargs):
        assert url.endswith("/workers")
        return {"workers": workers}

    monkeypatch.setattr(teacher_plugin, "get", fake_get)
    binding = named_teacher_binding()
    with pytest.raises(ValueError, match="no healthy routable"):
        asyncio.run(teacher_plugin._teacher_endpoint(named_teacher_args(binding), sample(), binding))


def test_named_teacher_probes_registration_health_and_uses_another_live_replica(monkeypatch):
    probes = []

    async def fake_get(url, **kwargs):
        if url.endswith("/workers"):
            return {"workers": [worker(), worker("http://worker-b")]}
        probes.append(url)
        # Make the first selected registered worker stale, then accept the other.
        return {"running": len(probes) > 1}

    monkeypatch.setattr(teacher_plugin, "get", fake_get)
    binding = named_teacher_binding()
    selected = asyncio.run(teacher_plugin._teacher_endpoint(named_teacher_args(binding), sample(), binding))
    assert len(probes) == 2 and selected + "/health" == probes[-1]


def test_cancellation_drains_the_owned_request_before_returning(monkeypatch):
    state = []

    async def fake_post(*args, **kwargs):
        await asyncio.sleep(0.01)
        state.append("completed")
        return {}

    monkeypatch.setattr(prism_http, "post", fake_post)

    async def exercise():
        args = types.SimpleNamespace(omni_request_timeout_s=1.0, omni_cancel_drain_timeout_s=0.1)
        task = asyncio.create_task(plugin._request(args, "http://router/generate", {}))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert state == ["completed"]

    asyncio.run(exercise())


def test_cancellation_cleanup_does_not_wait_forever_for_remote_transport(monkeypatch):
    async def slow_post(*args, **kwargs):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            # A remote transport may not stop when its local waiter cancels.
            await asyncio.sleep(10)

    monkeypatch.setattr(prism_http, "post", slow_post)

    async def exercise():
        args = types.SimpleNamespace(omni_request_timeout_s=1.0, omni_cancel_drain_timeout_s=0.01)
        task = asyncio.create_task(plugin._request(args, "http://router/generate", {}))
        await asyncio.sleep(0)
        started = time.monotonic()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert time.monotonic() - started < 0.5

    asyncio.run(exercise())


@pytest.mark.parametrize("show_language", [None, False, True])
def test_generate_uses_router_and_preserves_sample_identity(monkeypatch, tmp_path, show_language):
    item = sample()
    if show_language is not None:
        item.metadata.update(language="Chinese", show_language=show_language)
    response = output()
    blob = io.BytesIO()
    with wave.open(blob, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\0\0" * 32)
    response["audio"] = dict(data=base64.b64encode(blob.getvalue()).decode(), format="wav", sample_rate=24000)
    args = types.SimpleNamespace(hf_checkpoint="fixture", omni_audio_dir=str(tmp_path), omni_model_name="prism")
    calls = []

    async def fake_post(url, payload, **kwargs):
        calls.append((url, payload))
        return copy.deepcopy(response)

    monkeypatch.setattr(prism_http, "post", fake_post)
    monkeypatch.setattr(plugin, "_config", lambda path: geometry())
    monkeypatch.setitem(
        sys.modules,
        "slime.rollout.sglang_rollout",
        types.SimpleNamespace(get_model_url=lambda args, model: f"http://python-router/{model}/generate"),
    )
    assert asyncio.run(plugin.generate(args, item, {"max_new_tokens": 12})) is item
    assert calls[0][0] == "http://python-router/prism/generate"
    assert item.index == 7 and item.group_index == 2
    assert item.metadata["omni_audio_path"].endswith(".wav")
    assert item.metadata["omni_audio_seconds"] == pytest.approx(32 / 24000)
    assert item.metadata["transcript_target"] == "你好"
    assert "transcript_target" not in item.metadata["omni_prompt"]
    assert item.metadata["language"] == ("zh" if show_language is None else "Chinese")
    assert item.metadata.get("show_language") is show_language
    assert item.metadata["omni_prompt"] == calls[0][1]["prompt"]
    assert item.metadata["omni_prompt"]["global_instruction"].get("Language") == ("Chinese" if show_language else None)
    assert plugin.generate.abort_mode == "request"


def test_direct_routing_posts_to_an_engine_of_the_model_router(monkeypatch, tmp_path):
    item = sample()
    args = types.SimpleNamespace(
        hf_checkpoint="fixture", omni_audio_dir=str(tmp_path), omni_model_name="prism", omni_routing="direct"
    )
    response = output()
    blob = io.BytesIO()
    with wave.open(blob, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\0\0" * 32)
    response["audio"] = dict(data=base64.b64encode(blob.getvalue()).decode(), format="wav", sample_rate=24000)
    calls = []

    async def post_to_engine(args, router, model, path, payload, *, headers=None):
        calls.append((router, model, path, payload))
        return copy.deepcopy(response)

    monkeypatch.setattr(plugin, "post_to_engine", post_to_engine)
    monkeypatch.setattr(plugin, "_config", lambda path: geometry())
    monkeypatch.setitem(
        sys.modules,
        "slime.rollout.sglang_rollout",
        types.SimpleNamespace(
            get_model_url=lambda args, model, endpoint="/generate": f"http://router/{model}{endpoint}"
        ),
    )
    assert asyncio.run(plugin.generate(args, item, {"max_new_tokens": 12})) is item
    assert [call[:3] for call in calls] == [("http://router/prism", "prism", "/generate")]
    assert item.metadata["omni_prompt"] == calls[0][3]["prompt"]
    assert set(item.metadata["omni_client_timing"]) == {"prepare_s", "http_s", "parse_audio_s", "finished_unix"}


@pytest.mark.parametrize("override,expected", [(None, False), (True, True), (False, False)])
def test_eval_audio_override_does_not_change_training(override, expected):
    args = types.SimpleNamespace(omni_return_audio=False, omni_eval_return_audio=override)
    assert (
        plugin.build_generate_payload(args, sample(), {"max_new_tokens": 12}, evaluation=True)["return_audio"]
        is expected
    )
    assert plugin.build_generate_payload(args, sample(), {"max_new_tokens": 12})["return_audio"] is False
    assert args.omni_return_audio is False


def test_public_teacher_stage_can_skip_eval_without_changing_training():
    args = types.SimpleNamespace(
        use_opd=True, opd_eval_teacher_scoring=False, opd_teacher_config={"models": {}, "by_domain": {}}
    )
    item = sample()
    assert asyncio.run(opd.score_teacher(args, item, evaluation=True)) is item
    with pytest.raises(ValueError, match="Unmapped"):
        asyncio.run(opd.score_teacher(args, item, evaluation=False))


@pytest.mark.parametrize("frames,finish", [(0, "stop"), (1, "stop"), (2, "length")])
def test_compact_replay_reconstructs_the_same_teacher_request_and_training_tensors(frames, finish):
    item = plugin.parse_rollout(output(frames, finish), sample(), geometry())
    expected = teacher_plugin.teacher_request(item)
    tensors = {name: tensor.clone() for name, tensor in item.multimodal_train_inputs.items()}
    plugin.finalize_sample(types.SimpleNamespace(omni_compact_replay=True), item)
    assert "replay_inputs" not in item.metadata["omni_rollout"]
    assert "action_streams" not in item.metadata["omni_rollout"]
    assert teacher_plugin.teacher_request(item) == expected
    for name, tensor in tensors.items():
        torch.testing.assert_close(item.multimodal_train_inputs[name], tensor, rtol=0, atol=0)


def test_teacher_batches_are_bounded_and_match_results_by_sample_id(tmp_path, monkeypatch):
    binding = teacher_binding()
    path = tmp_path / "teachers.json"
    path.write_text(
        json.dumps(
            {
                "score_function": "slime_plugins.prism.teacher.score_teacher",
                "models": {"a": binding},
                "by_domain": {"chinese": "a"},
                "client": {"batch_size": 4, "max_inflight": 2, "batch_wait_ms": 1},
            }
        )
    )
    args = types.SimpleNamespace(use_opd=True, opd_teacher_config=str(path))
    batches, active, maximum = [], 0, 0
    reads = []
    original = type(path).read_text

    def read(self, *positional, **keyword):
        if self == path:
            reads.append(self)
        return original(self, *positional, **keyword)

    async def request(_args, url, payload, **kwargs):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        batches.append(payload["samples"])
        await asyncio.sleep(0.005)
        active -= 1
        return {"version": 1, "results": [teacher_output(row) for row in reversed(payload["samples"])]}

    monkeypatch.setattr(type(path), "read_text", read)
    monkeypatch.setattr(teacher_plugin, "_request", request)

    async def run():
        runtime = opd.TeacherRuntime(args)
        items = [plugin.parse_rollout(output(), sample(index), geometry()) for index in range(11)]
        await asyncio.gather(*(opd.score_teacher(args, item, runtime=runtime) for item in items))
        path.write_text("invalid config after the binding was frozen")
        await opd.score_teacher(args, plugin.parse_rollout(output(), sample(20), geometry()), runtime=runtime)
        await runtime.close()
        assert all(item.metadata["omni_teacher_status"] == "success" for item in items)

    asyncio.run(run())
    assert len(reads) == 1
    assert sorted(len(batch) for batch in batches) == [1, 3, 4, 4]
    assert maximum == 2


def test_cancelling_one_teacher_sample_keeps_its_batch_peer_alive(monkeypatch):
    binding = teacher_binding()
    args = types.SimpleNamespace(
        use_opd=True,
        opd_teacher_config={
            "score_function": "slime_plugins.prism.teacher.score_teacher",
            "models": {"a": binding},
            "by_domain": {"chinese": "a"},
        },
    )
    args.opd_teacher_config["client"] = {"batch_size": 2, "batch_wait_ms": 100}

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        async def request(_args, url, payload, **kwargs):
            entered.set()
            await release.wait()
            return {"version": 1, "results": [teacher_output(row) for row in payload["samples"]]}

        monkeypatch.setattr(teacher_plugin, "_request", request)
        runtime = opd.TeacherRuntime(args)
        items = [plugin.parse_rollout(output(), sample(index), geometry()) for index in range(2)]
        first, second = [asyncio.create_task(opd.score_teacher(args, item, runtime=runtime)) for item in items]
        await entered.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        release.set()
        assert await second is items[1]
        await runtime.close()
        assert items[0].metadata["omni_teacher_status"] == "cancelled"
        assert items[1].metadata["omni_teacher_status"] == "success"

    asyncio.run(run())


def test_compact_replay_preserves_reference_audio_edges():
    response = output()
    response["meta_info"]["prompt_tokens"] = 4
    replay = response["meta_info"]["omni_rollout"]["replay_inputs"]
    replay.update(
        prompt_rows=[[4, 0, 0], [9, 1, 2], [9, 3, 4], [4, 0, 0]],
        item_kind=[0, 1, 1, 0],
        audio_role=[0, 1, 1, 0],
        successor_audio_mask=[True, True, False, False],
        successor_audio_codes=[[1, 2], [3, 4], [0, 0], [0, 0]],
        target_history_retention_mask=[[False, False]] * 4,
    )
    item = plugin.parse_rollout(response, sample(), geometry())
    expected = teacher_plugin.teacher_request(item)
    plugin.finalize_sample(types.SimpleNamespace(omni_compact_replay=True), item)
    assert teacher_plugin.teacher_request(item) == expected


def test_loss_selection_preserves_teacher_alignment_and_canonical_replay():
    item = plugin.parse_rollout(output(), sample(), geometry())
    expected = teacher_plugin.teacher_request(item)
    item.policy_actions["rvq"]["loss_mask"][1] = False
    item.policy_actions["stop"]["loss_mask"][1] = False
    # Wire traces are diagnostic snapshots, never another source of policy targets.
    item.metadata["omni_rollout"]["action_streams"][0]["actions"][0][0] = 7
    request = teacher_plugin.teacher_request(item)
    assert request == expected
    teacher_plugin.attach_teacher_scores(item, request, teacher_output(request))
    stream = item.policy_actions["rvq"]
    assert stream["teacher_log_probs"][stream["valid_mask"]].numel() == 4
    assert stream["teacher_log_probs"][stream["loss_mask"]].numel() == 2
    assert item.teacher_log_probs is None
