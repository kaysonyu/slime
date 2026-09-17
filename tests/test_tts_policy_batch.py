"""Action alignment, CP ownership and Local causal replay contracts."""

from types import SimpleNamespace

import pytest
import torch
from tools.convert_moss_tts_2_0_for_sglang_omni import convert_local_tensor

from slime.backends.megatron_utils.policy_batch import collate_policy_batch, sum_sample_means
from slime_plugins.models.moss_tts_local.config import MossLocalConfig
from slime_plugins.models.moss_tts_local.data import MossLocalTrajectory
from slime_plugins.models.moss_tts_local.local_transformer import LocalTransformer

NUM_GPUS = 0


def config():
    return MossLocalConfig.from_dict(
        dict(
            model_type="moss_tts_local",
            n_vq=3,
            audio_codebook_sizes=[8] * 3,
            audio_pad_token_id=8,
            audio_assistant_slot_token_id=5,
            audio_end_token_id=6,
            pad_token_id=0,
            language_config=dict(
                hidden_size=8,
                num_hidden_layers=2,
                vocab_size=16,
                num_attention_heads=2,
                num_key_value_heads=1,
                intermediate_size=12,
            ),
            gpt2_config=dict(
                n_embd=8,
                n_head=2,
                n_inner=12,
                n_layer=1,
                activation_function="silu",
                layer_norm_epsilon=1e-6,
                position_embedding_type="rope",
                rope_base=10000,
            ),
        )
    )


def trajectory(prompt=3, frames=5, stop=True):
    cfg = config()
    identity = dict(
        n_vq=3,
        audio_vocab_size=8,
        audio_pad_code=8,
        audio_assistant_slot_token_id=5,
        audio_end_token_id=6,
        text_vocab_size=16,
        hidden_size=8,
        global_layers=2,
        global_num_attention_heads=2,
        global_num_query_groups=1,
        global_ffn_hidden_size=12,
        global_rope_base=1000000,
        global_layer_norm_epsilon=1e-6,
        qk_layernorm=True,
        local_layers=1,
        local_num_attention_heads=2,
        local_ffn_hidden_size=12,
        local_rope_base=10000,
        local_layer_norm_epsilon=1e-6,
        local_activation="silu",
        sample_rate=48000,
        tie_audio_embeddings_and_output_weights=False,
    )
    return MossLocalTrajectory(
        prompt_rows=torch.tensor([[1, 8, 8, 8]] * prompt),
        decisions=torch.tensor([0] * frames + ([1] if stop else [])),
        codes=torch.arange(frames * cfg.n_vq).reshape(frames, cfg.n_vq) % 8,
        decision_logprobs=torch.full((frames + int(stop),), -0.7),
        code_logprobs=torch.full((frames, cfg.n_vq), -2.0),
        finish_reason="stop" if stop else "length",
        weight_version="1",
        request_id="sample",
        sampling=dict(
            text_temperature=1,
            audio_temperature=1,
            text_top_p=1,
            audio_top_p=1,
            text_top_k=-1,
            audio_top_k=-1,
            audio_repetition_penalty=1,
        ),
        model_identity=identity,
    )


def batch(traces, cp_size=1, cp_rank=0):
    cfg = config()
    return collate_policy_batch(
        [SimpleNamespace(advantage=1, teacher_scores=None) for _ in traces],
        [trace.training_tensors(cfg) for trace in traces],
        torch.tensor([0, 8, 8, 8]),
        [[1] * 4 for _ in traces],
        cp_size=cp_size,
        cp_rank=cp_rank,
        pad_multiple=8,
    )


def test_terminal_is_a_decision_without_audio_labels():
    trace = trajectory()
    rows, positions, targets, mask, old = trace.training_tensors(config())
    assert rows.shape == (8, 4)
    assert positions.tolist() == [2, 3, 4, 5, 6, 7]
    assert mask[-1].tolist() == [True, False, False, False]
    assert targets[-1, 0] == 1
    assert int(mask.sum()) == trace.num_actions == 21
    assert torch.equal(old[:5, 1:], trace.code_logprobs)


def test_truncation_does_not_synthesize_stop():
    trace = trajectory(stop=False)
    _, positions, _, mask, _ = trace.training_tensors(config())
    assert positions.tolist() == [2, 3, 4, 5, 6]
    assert mask.all()
    trace.decisions = torch.cat((trace.decisions, torch.tensor([1])))
    with pytest.raises(ValueError, match="truncation"):
        trace.validate(config())


def test_cp_action_owner_is_prediction_source_not_target_row():
    trace = trajectory()
    left, right = batch([trace], 2, 0), batch([trace], 2, 1)
    assert left.global_row_indices.tolist() == [0, 1, 6, 7]
    assert right.global_row_indices.tolist() == [2, 3, 4, 5]
    assert left.action_row_indices.tolist() == [4, 5]
    assert right.action_row_indices.tolist() == [0, 1, 2, 3]
    assert right.targets[-1].tolist() == [0, *trace.codes[3].tolist()]
    assert right.prediction_positions[-1] == 3
    assert sorted(torch.cat((left.action_row_indices, right.action_row_indices)).tolist()) == list(range(6))


def test_cp_packing_preserves_loss_and_gradient_with_uneven_samples():
    traces = [trajectory(), trajectory(prompt=1, frames=1, stop=False)]
    weight = torch.tensor(0.5, requires_grad=True)
    whole = batch(traces)
    baseline = sum_sample_means(weight * whole.targets.float().square(), whole)
    (expected_grad,) = torch.autograd.grad(baseline, weight)
    partials = []
    for rank in range(4):
        part = batch(traces, 4, rank)
        partials.append(sum_sample_means(weight * part.targets.float().square(), part))
    reduced = sum(partials)
    (actual_grad,) = torch.autograd.grad(reduced, weight)
    torch.testing.assert_close(reduced, baseline)
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("shape", [(24, 8), (24,)])
def test_offline_local_qkv_conversion_has_expected_row_order(shape):
    original = torch.arange(torch.tensor(shape).prod()).reshape(shape)
    hf_name = "local_transformer.layers.0.attention.query_key_value.weight"
    native_name, native = convert_local_tensor(hf_name, original, 2)
    assert native_name == "local_transformer.h.0.attn.c_attn.weight"
    rows = [0, 2, 1, 3, 12, 14, 13, 15, 4, 6, 5, 7, 16, 18, 17, 19, 8, 9, 10, 11, 20, 21, 22, 23]
    torch.testing.assert_close(native, original[rows])


def test_local_depth_is_causal_and_frames_are_independent():
    torch.manual_seed(3)
    model = LocalTransformer(config())
    inputs = torch.randn(2, 3, 8, requires_grad=True)
    first = model(inputs)
    changed = inputs.detach().clone()
    changed[0, 2] += torch.arange(8) * 10
    changed[1] *= 10
    second = model(changed)
    torch.testing.assert_close(first[0, :2], second[0, :2])
    (first[0, 0].square().sum()).backward()
    assert inputs.grad[0, 1:].eq(0).all()
    assert inputs.grad[1].eq(0).all()


def test_local_decoder_matches_hf_neox_after_coordinate_conversion():
    from transformers import GPTNeoXConfig, GPTNeoXModel

    torch.manual_seed(7)
    cfg = config()
    native = LocalTransformer(cfg).eval()
    hf_config = GPTNeoXConfig(
        vocab_size=1,
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=12,
        num_hidden_layers=1,
        hidden_act="silu",
        use_parallel_residual=False,
        attention_bias=True,
        layer_norm_eps=1e-6,
        rotary_pct=1.0,
        rotary_emb_base=10000,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        rope_parameters={"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 1.0},
    )
    hf_config._attn_implementation = "eager"
    reference = GPTNeoXModel(hf_config).eval()
    state = {}
    for name, tensor in reference.state_dict().items():
        if name == "embed_in.weight":
            continue
        mapped, value = convert_local_tensor("local_transformer." + name, tensor, 2)
        state[mapped.removeprefix("local_transformer.")] = value
    native.load_state_dict(state, strict=True)
    inputs = torch.randn(2, 3, 8)
    torch.testing.assert_close(
        native(inputs), reference(inputs_embeds=inputs, use_cache=False).last_hidden_state, atol=1e-5, rtol=1e-5
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
