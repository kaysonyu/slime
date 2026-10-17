"""Native text regression and explicit action-stream contracts."""

from argparse import Namespace
from types import SimpleNamespace

import _cp_dist_helpers  # noqa: F401
import pytest
import torch
from test_policy_actions import sample as action_sample

from slime.backends.megatron_utils import loss
from slime.data.batch_builder import BatchBuilder
from slime.utils.types import Sample

NUM_GPUS = 0


def arguments(**overrides):
    from test_policy_actions import arguments as policy_arguments

    return policy_arguments(
        **{
            "custom_megatron_log_prob_function_path": None,
            "rollout_temperature": 1.0,
            "rollout_top_p": 1.0,
            "log_probs_chunk_size": 0,
            "use_rollout_entropy": True,
            **overrides,
        }
    )


@pytest.fixture(autouse=True)
def single_rank(monkeypatch):
    monkeypatch.setattr(loss.mpu, "is_pipeline_last_stage", lambda: True, raising=False)
    monkeypatch.setattr(loss.mpu, "get_tensor_model_parallel_group", lambda: None, raising=False)
    native = loss.compute_policy_loss
    monkeypatch.setattr(loss, "compute_policy_loss", getattr(native, "__wrapped__", native))


@pytest.mark.parametrize("empty_audio", [False, True])
@pytest.mark.parametrize("removed", [False, True])
def test_three_named_streams_use_the_same_frozen_opd_objective(monkeypatch, empty_audio, removed):
    from slime.utils import misc

    args = arguments(
        custom_megatron_log_prob_function_path="fixture.score",
        use_opd=True,
        use_rollout_entropy=False,
        policy_loss_weights={"semantic": 0.4, "acoustic": 1.3, "termination": 0.2},
    )
    output = torch.linspace(-2.5, -0.5, 20).reshape(1, 5, 4).requires_grad_()
    slices = {"semantic": 0, "acoustic": slice(1, 3), "termination": 3}
    masks = {
        "semantic": torch.tensor([True, False, True, False, False]),
        "acoustic": torch.tensor([[False, False], [True, True], [True, True], [False, False], [False, False]]),
        "termination": torch.tensor([False, False, False, True, False]),
    }
    if empty_audio:
        masks["acoustic"].zero_()
    streams = {}
    for name, column in slices.items():
        old = output[0, :, column].detach().clone() - 0.15
        streams[name] = {
            "valid_mask": masks[name],
            "loss_mask": (masks[name]).clone(),
            "actions": torch.zeros_like(masks[name], dtype=torch.long),
            "old_log_probs": old,
            "rollout_log_probs": old - 0.03,
            "teacher_log_probs": old + torch.linspace(-0.4, 0.3, old.numel()).reshape_as(old),
        }

    def score(_args, _batch, value):
        return {f"policy/{name}/log_probs": [value[0, :, column]] for name, column in slices.items()}

    monkeypatch.setattr(misc, "load_function", lambda path: score)
    batch = {
        "policy_actions": streams,
        "policy_sample_masks": torch.tensor([not removed]),
        "total_lengths": [5],
        "response_lengths": [2],
        "loss_masks": [torch.ones(2)],
        "log_probs": [torch.zeros(2)],
    }
    rollout = {"policy_actions": [streams], "rewards": [0.7], "total_lengths": [5], "response_lengths": [2]}
    calls = []
    native_opd = loss.apply_opd_kl_to_advantages

    def record(*args, **kwargs):
        calls.append(True)
        return native_opd(*args, **kwargs)

    monkeypatch.setattr(loss, "apply_opd_kl_to_advantages", record)
    loss.compute_advantages_and_returns(args, rollout)
    frozen = {name: stream["advantages"].clone() for name, stream in streams.items()}
    actual, metrics = loss.policy_loss_function(args, batch, output, lambda values: values.sum())
    expected = output.sum() * 0
    for name, column in slices.items():
        mask = masks[name]
        if not mask.any():
            continue
        old = streams[name]["old_log_probs"][mask]
        teacher = streams[name]["teacher_log_probs"][mask]
        advantage = 0.7 - args.opd_kl_coef * (old - teacher)
        ratio = (output[0, :, column][mask] - old).exp()
        clipped = ratio.clamp(0.8, 1.2)
        expected = expected + args.policy_loss_weights[name] * torch.maximum(
            -advantage * ratio, -advantage * clipped
        ).mean() * (not removed)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, output, retain_graph=True)[0], torch.autograd.grad(expected, output)[0]
    )
    loss.policy_loss_function(args, batch, output + 0.1, lambda values: values.sum())
    assert calls == [True]
    for name in streams:
        torch.testing.assert_close(streams[name]["advantages"], frozen[name])
        assert not streams[name]["advantages"].requires_grad
    assert metrics["acoustic/active_ratio"] == int(not empty_audio and not removed)


@pytest.mark.parametrize("use_opd", [False, True])
@pytest.mark.parametrize("overrides", [{}, {"tensor_model_parallel_size": 2}, {"use_tis": True}])
def test_text_builder_always_keeps_the_native_contract(use_opd, overrides):
    args = arguments(
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        rewards_normalization=False,
        reward_key=None,
        use_opd=use_opd,
        **overrides,
    )
    sample = Sample(
        tokens=[0, 1, 2],
        response_length=1,
        reward=1.0,
        loss_mask=[1],
        rollout_log_probs=[-0.4],
        teacher_log_probs=[-0.2] if use_opd else None,
    )
    data = BatchBuilder(args).convert([sample])
    assert sample.policy_actions is None
    assert "policy_actions" not in data
    assert "policy_sample_masks" not in data
    assert data.get("multimodal_train_inputs") is None
    assert data["tokens"] == [[0, 1, 2]]
    assert data["response_lengths"] == [1]
    assert data["loss_masks"] == [[1]]
    assert data["rollout_log_probs"] == [[-0.4]]
    if use_opd:
        assert data["teacher_log_probs"] == [[-0.2]]


@pytest.mark.parametrize("use_opd", [False, True])
def test_plain_text_batch_uses_native_advantages_scores_and_loss(monkeypatch, use_opd):
    args = arguments(
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        rewards_normalization=False,
        reward_key=None,
        use_opd=use_opd,
    )
    sample = Sample(
        tokens=[0, 1, 2, 3, 1],
        response_length=3,
        reward=0.7,
        loss_mask=[1, 0, 1],
        rollout_log_probs=[-1.4, -1.7, -1.2],
        teacher_log_probs=[-1.2, -2.0, -1.0] if use_opd else None,
    )
    batch = BatchBuilder(args).convert([sample])
    mask = torch.tensor(batch["loss_masks"][0])
    old = torch.tensor(batch["rollout_log_probs"][0])
    batch.update(
        unconcat_tokens=[torch.tensor(sample.tokens)],
        total_lengths=[len(sample.tokens)],
        loss_masks=[mask],
        rollout_mask_sums=torch.tensor([mask.sum()]),
        log_probs=[old],
        rollout_log_probs=[old],
    )
    if use_opd:
        batch["teacher_log_probs"] = [torch.tensor(sample.teacher_log_probs)]

    def unexpected(*args, **kwargs):
        pytest.fail("Ordinary LLM data reached action-stream training")

    monkeypatch.setattr(loss, "prepare_action_advantages", unexpected)
    monkeypatch.setattr(loss, "get_policy_scores", unexpected)
    loss.compute_advantages_and_returns(args, batch)
    expected_advantages = torch.full_like(old, 0.7)
    if use_opd:
        expected_advantages -= args.opd_kl_coef * (old - batch["teacher_log_probs"][0])
    torch.testing.assert_close(batch["advantages"][0], expected_advantages)
    logits = torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(7), requires_grad=True)
    value, metrics = loss.policy_loss_function(args, batch, logits, lambda values: (values * mask).sum() / mask.sum())
    value.backward()
    assert torch.isfinite(value)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert "text/pg_loss" not in metrics


@pytest.mark.parametrize("removed", [False, True])
def test_builder_preserves_explicit_prism_actions(removed):
    args = arguments(
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        rewards_normalization=False,
        reward_key=None,
    )
    streams = action_sample()["streams"]
    sample = Sample(
        tokens=[0, 1, 2, 3, 4],
        response_length=2,
        reward=1.0,
        policy_actions=streams,
        remove_sample=removed,
    )

    data = BatchBuilder(args).convert([sample])

    assert data["policy_actions"][0] is streams
    assert tuple(streams) == ("rvq", "stop")
    assert data["policy_sample_masks"][0].item() is (not removed)
    assert sample.policy_actions is streams


def test_builder_rejects_mixed_implicit_text_and_explicit_actions():
    args = arguments(
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        rewards_normalization=False,
        reward_key=None,
    )
    plain = Sample(index=0, tokens=[0, 1, 2], response_length=1, reward=1.0)
    explicit = Sample(
        index=1,
        tokens=[0, 1, 2],
        response_length=1,
        reward=1.0,
        policy_actions=action_sample(rows=3, frames=1)["streams"],
    )

    with pytest.raises(ValueError, match="cannot mix implicit and explicit"):
        BatchBuilder(args).convert([plain, explicit])
    assert plain.policy_actions is None


def test_policy_streams_survive_sample_and_tensor_buffer_roundtrip():
    from slime.utils.tensor_serialization import SerializedTensorPayload

    sample = Sample(tokens=[0, 1, 2], policy_actions=action_sample(rows=3, frames=1)["streams"])
    restored = Sample.from_dict(sample.to_dict())
    torch.testing.assert_close(
        restored.policy_actions["rvq"]["rollout_log_probs"], sample.policy_actions["rvq"]["rollout_log_probs"]
    )
    payload = SerializedTensorPayload({"policy_actions": [sample.policy_actions]})
    import pickle

    pickled_sample = pickle.loads(pickle.dumps(sample))
    torch.testing.assert_close(
        pickled_sample.policy_actions["rvq"]["actions"], sample.policy_actions["rvq"]["actions"]
    )
    restored = pickle.loads(pickle.dumps(payload))
    # The transport deliberately returns an ordinary mutable dictionary.
    torch.testing.assert_close(
        restored["policy_actions"][0]["rvq"]["valid_mask"], sample.policy_actions["rvq"]["valid_mask"]
    )
    restored["policy_actions"][0]["rvq"]["actions"].fill_(1)
    assert sample.policy_actions["rvq"]["actions"].sum() == 0


def test_action_streams_survive_durable_transport(tmp_path):
    pytest.importorskip("straw")
    from slime.data.transport import pack_rollout_payload, seal_rollout_store, unpack_rollout_payload

    args = Namespace(rollout_data_transport="straw", rollout_data_dir=str(tmp_path), rollout_queue_run_id="streams")
    sample = Sample(tokens=[0, 1, 2], response_length=1, policy_actions=action_sample(rows=3, frames=1)["streams"])
    reference = pack_rollout_payload([[sample]], args, 0)
    restored = unpack_rollout_payload(reference)[0][0]
    torch.testing.assert_close(restored.policy_actions["rvq"]["actions"], sample.policy_actions["rvq"]["actions"])
    torch.testing.assert_close(
        restored.policy_actions["rvq"]["rollout_log_probs"], sample.policy_actions["rvq"]["rollout_log_probs"]
    )
    seal_rollout_store(args)


@pytest.mark.parametrize("mask,postprocess", [([0.5], None), ([1], "custom.postprocess")])
def test_text_extensions_keep_the_original_data_contract(mask, postprocess):
    args = arguments(
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        rewards_normalization=False,
        reward_key=None,
        rollout_data_postprocess_path=postprocess,
    )
    sample = Sample(tokens=[0, 1, 2], response_length=1, reward=1.0, loss_mask=mask, rollout_log_probs=[-0.4])
    data = BatchBuilder(args).convert([sample])
    assert "policy_actions" not in data
    assert data["loss_masks"] == [mask]


def test_custom_extractor_receives_named_actions_and_unmodified_model_data(monkeypatch):
    from slime.backends.megatron_utils.policy_actions import get_policy_scores
    from slime.utils import misc

    streams = {
        "rvq": {
            "valid_mask": torch.ones(3, 2, dtype=torch.bool),
            "loss_mask": (torch.ones(3, 2, dtype=torch.bool)).clone(),
            "actions": torch.zeros_like(torch.ones(3, 2, dtype=torch.bool), dtype=torch.long),
        },
        "stop": {
            "valid_mask": torch.ones(3, dtype=torch.bool),
            "loss_mask": (torch.ones(3, dtype=torch.bool)).clone(),
            "actions": torch.zeros(3, dtype=torch.long),
        },
    }
    inputs = {"audio_codes": torch.zeros(3, 2, dtype=torch.long)}
    batch = {
        "policy_actions": streams,
        "multimodal_train_inputs": inputs,
        "total_lengths": [3],
        "response_lengths": [2],
    }
    output = torch.zeros(1, 3, 3)

    def score(args, received, output):
        assert received is batch
        assert received["policy_actions"] is streams
        assert received["multimodal_train_inputs"] is inputs
        return {
            "log_probs": [torch.zeros(2)],
            "policy/rvq/log_probs": [output[0, :, :2]],
            "policy/stop/log_probs": [output[0, :, 2]],
        }

    monkeypatch.setattr(misc, "load_function", lambda path: score)
    scores = get_policy_scores(
        arguments(custom_megatron_log_prob_function_path="custom.score", use_rollout_entropy=False), batch, output
    )
    assert "policy/rvq/log_probs" in scores
    assert batch["multimodal_train_inputs"] is inputs and set(inputs) == {"audio_codes"}


def test_segmented_text_rollouts_keep_the_existing_rollout_mean_reducer():
    args = arguments(
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        rewards_normalization=False,
        reward_key=None,
    )
    samples = [
        Sample(index=i, rollout_id=7, tokens=[0, 1, 2], response_length=1, reward=1.0, loss_mask=[1]) for i in range(2)
    ]
    data = BatchBuilder(args).convert(samples)
    assert "policy_actions" not in data
    assert data["rollout_ids"] == [7, 7]
    for sample in samples:
        sample.policy_actions = action_sample(rows=3, frames=1)["streams"]
    with pytest.raises(ValueError, match="one Sample per rollout"):
        BatchBuilder(args).convert(samples)


@pytest.mark.parametrize("extractor", [None, "model.score"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_model_forward_only_receives_actions_and_masks(extractor, wrapped):
    from slime.backends.megatron_utils.policy_actions import policy_forward_kwargs

    streams = {
        "rvq": {
            "valid_mask": torch.ones(2, 2, dtype=torch.bool),
            "loss_mask": (torch.ones(2, 2, dtype=torch.bool)).clone(),
            "actions": torch.zeros_like(torch.ones(2, 2, dtype=torch.bool), dtype=torch.long),
            "old_log_probs": torch.zeros(2, 2),
        },
        "stop": {
            "valid_mask": torch.ones(2, dtype=torch.bool),
            "loss_mask": (torch.ones(2, dtype=torch.bool)).clone(),
            "actions": torch.zeros(2, dtype=torch.long),
            "advantages": torch.ones(2),
        },
    }
    batch = {"policy_actions": streams, "multimodal_train_inputs": {"audio_codes": torch.zeros(2, 2)}}
    args = arguments(custom_megatron_log_prob_function_path=extractor)
    model = SimpleNamespace(accepts_policy_actions=True)
    if wrapped:
        model = SimpleNamespace(module=SimpleNamespace(module=model))
    actual = policy_forward_kwargs(args, batch, model)
    assert set(actual["policy_actions"]["rvq"]) == {"valid_mask", "actions"}
    assert set(actual["policy_actions"]["stop"]) == {"valid_mask", "actions"}
    assert actual["policy_return_entropy"] is True


@pytest.mark.parametrize("extractor", [None, "model.score"])
@pytest.mark.parametrize("explicit_false", [False, True])
def test_score_hook_does_not_change_a_standard_model_forward_signature(extractor, explicit_false):
    from slime.backends.megatron_utils.policy_actions import policy_forward_kwargs

    class TextModel(torch.nn.Module):
        def forward(self, input_ids):
            return input_ids

    model = TextModel()
    if explicit_false:
        model.accepts_policy_actions = False
    tokens = torch.tensor([0, 1, 2])
    batch = {"policy_actions": action_sample(rows=3, frames=1)["streams"]}
    kwargs = policy_forward_kwargs(
        arguments(custom_megatron_log_prob_function_path=extractor), batch, SimpleNamespace(module=model)
    )
    assert kwargs == {}
    assert model(input_ids=tokens, **kwargs) is tokens


def test_action_model_requires_targets_even_without_a_score_hook():
    from slime.backends.megatron_utils.policy_actions import policy_forward_kwargs

    with pytest.raises(ValueError, match="requires action streams"):
        policy_forward_kwargs(arguments(), {}, SimpleNamespace(accepts_policy_actions=True))


@pytest.mark.parametrize("stream_name", ["rvq", "text"])
def test_explicit_action_streams_require_their_own_extractor(stream_name):
    from slime.backends.megatron_utils.policy_actions import get_policy_scores

    stream = action_sample(rows=3, frames=1)["streams"]["rvq"]
    batch = {"policy_actions": {stream_name: stream}, "total_lengths": [3]}
    with pytest.raises(ValueError, match="require --custom-megatron-log-prob-function-path"):
        get_policy_scores(arguments(), batch, torch.zeros(1, 3, 2))


def test_training_masks_must_select_only_real_actions():
    from slime.utils.policy import validate_policy_actions

    streams = action_sample(rows=3, frames=1)["streams"]
    streams["rvq"]["loss_mask"][0, 0] = True
    with pytest.raises(ValueError, match="invalid action"):
        validate_policy_actions(streams, 3)


def test_opd_metrics_use_actual_action_advantages_without_frame_placeholders():
    from slime.backends.megatron_utils.policy_actions import policy_rollout_metrics

    streams = action_sample(rows=3, frames=1)["streams"]
    for stream in streams.values():
        stream["old_log_probs"].fill_(-0.7)
        stream["teacher_log_probs"] = torch.full_like(stream["old_log_probs"], -0.2)
    data = {"policy_actions": [streams], "rewards": [0.0], "response_lengths": [1], "total_lengths": [3]}
    loss.compute_advantages_and_returns(arguments(use_opd=True, opd_kl_coef=0.4), data)
    metrics = policy_rollout_metrics(data)
    assert metrics["policy/rvq/advantages"] == pytest.approx((0.2, 1))
    assert metrics["policy/stop/advantages"] == pytest.approx((0.2, 1))
    assert metrics["policy/rvq/train_actions"] == (2, 1)
    assert metrics["policy/stop/train_actions"] == (2, 1)
    assert not {"advantages", "returns", "kl"}.intersection(data)


def test_debug_capture_retains_every_action_stream_without_collapsing_rows():
    streams = {
        "speech": torch.tensor([[-0.2, -0.3], [-0.4, -0.5]]),
        "termination": torch.tensor([-0.6, -0.7]),
    }
    loss.enable_log_prob_capture()
    loss._maybe_capture_log_probs({"partition": [17]}, [streams])
    captured = loss.drain_captured_log_probs()[17]
    for name, values in streams.items():
        torch.testing.assert_close(captured[name], values)
        assert captured[name].data_ptr() != values.data_ptr()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
