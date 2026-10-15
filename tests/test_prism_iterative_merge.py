import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from slime_plugins.prism.iterative_merge import MergeOwner, apply_task_vectors, decide, load_plan, restore_state
from slime_plugins.prism.task_vectors import TaskVectorStore, file_sha256, physical_tensors

NUM_GPUS = 0


def vector_store(tmp_path, delta):
    root = tmp_path / "vector"
    root.mkdir()
    save_file({"delta": delta.flatten()}, root / "tensor.safetensors")
    manifest = {
        "version": 1,
        "layout": "prism_megatron_physical_v1",
        "dtype": "float32",
        "tensors": {
            "weight": {
                "shape": list(delta.shape),
                "file": "tensor.safetensors",
                "sha256": file_sha256(root / "tensor.safetensors"),
            }
        },
    }
    (root / "index.json").write_text(json.dumps(manifest))
    checksum = file_sha256(root / "index.json")
    return TaskVectorStore(root, checksum), {"path": str(root), "sha256": checksum}


def test_fp32_merge_survives_the_next_adam_step_without_resetting_moments(tmp_path, monkeypatch):
    model = torch.nn.Linear(5, 2, bias=False).to(torch.bfloat16)
    master = torch.nn.Parameter(model.weight.detach().float().clone() + 0.0000123)
    adam = torch.optim.Adam([master], lr=0.001)
    master.grad = torch.ones_like(master)
    adam.step()
    adam.zero_grad()
    model.weight.data.copy_(master)
    baseline = master.detach().clone()
    old_state = {key: value.clone() for key, value in adam.state[master].items()}
    delta = torch.arange(10).reshape(2, 5).float() / 31
    store, _ = vector_store(tmp_path, delta)
    owner = type("DistributedOptimizer", (), {})()
    owner.config = SimpleNamespace(use_precision_aware_optimizer=False)
    owner.ddp_config = SimpleNamespace(overlap_param_gather=False, use_megatron_fsdp=False)
    owner.model_float16_groups = [[model.weight]]
    owner.shard_fp32_from_float16_groups = [[master]]
    owner.model_fp32_groups = owner.shard_fp32_groups = []
    owner._get_model_param_range_map = lambda p: {"param": SimpleNamespace(start=0, end=p.numel())}
    owner._copy_main_params_to_model_params = lambda: model.weight.data.copy_(master)
    owner.start_param_sync_for_bucket_group_subset = lambda force_sync: None
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda value: None)
    norm = apply_task_vectors([model], owner, {"sing": store}, {"sing": 0.3})
    torch.testing.assert_close(master, baseline + 0.3 * delta, rtol=0, atol=3e-8)
    assert norm == pytest.approx(float((0.3 * delta).double().norm()))
    assert torch.equal(model.weight, master.to(model.weight.dtype))
    for key, value in old_state.items():
        assert torch.equal(adam.state[master][key], value)
    expected = torch.nn.Parameter(master.detach().clone())
    oracle = torch.optim.Adam([expected], lr=0.001)
    oracle.state[expected] = {key: value.clone() for key, value in old_state.items()}
    master.grad = torch.full_like(master, 0.2)
    expected.grad = master.grad.clone()
    adam.step()
    oracle.step()
    torch.testing.assert_close(master, expected, rtol=0, atol=0)


def test_merge_plan_recovery_and_committed_events_are_replay_safe(tmp_path):
    _, entry = vector_store(tmp_path, torch.ones(2, 3))
    plan = {
        "version": 1,
        "vectors": {"sing": entry},
        "initial": {"sing": 0.1},
        "steps": [50],
        "delta": 0.1,
        "gamma": 0.4,
        "anchors": {"sing": {"base": 0.2, "teacher": 0.7}},
        "datasets": {"sing": "sing_threshold"},
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    plan, digest = load_plan(path)
    event = decide(plan, 50, {"sing_threshold": 0.3})
    assert event["adds"] == {"sing": 0.1}
    assert event["recovery"]["sing"] == pytest.approx(0.2)
    assert decide(plan, 50, {"sing_threshold": 0.6})["adds"] == {}
    owner = MergeOwner.__new__(MergeOwner)
    owner.state = {"version": 1, "plan_sha256": digest, "events": [{**event, "delta_l2": 0.123}]}
    owner.save(tmp_path, 49)
    restored = restore_state(SimpleNamespace(load=str(tmp_path), finetune=False), 49, digest)
    assert restored == owner.state
    # Replaying an already committed event never even visits model/optimizer data.
    assert owner.apply(None, None, event) == restored
    with pytest.raises(ValueError, match="different coefficients"):
        owner.apply(None, None, {**event, "adds": {"sing": 0.3}})
    with pytest.raises(ValueError, match="differs"):
        restore_state(SimpleNamespace(load=str(tmp_path), finetune=False), 49, "wrong-plan")
    plan["anchors"]["sing"]["teacher"] = 0.2
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="recovery gap"):
        load_plan(path)


def test_inverse_prism_mapping_roundtrip_preserves_all_physical_weights():
    from slime.backends.megatron_utils.megatron_to_hf.prism import convert_prism_to_hf

    config = {
        "language_config": {
            "hidden_size": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "num_hidden_layers": 2,
        },
        "n_vq": 2,
        "prism_rvq_heads_per_execution_unit": [1, 1],
        "prism_mlp_only_layers": [1],
    }
    args = SimpleNamespace(
        hidden_size=4,
        num_attention_heads=2,
        num_query_groups=1,
        kv_channels=2,
        num_layers=2,
        n_vq=2,
        padded_vocab_size=8,
        speech_vocab_size=3,
        ffn_hidden_size=6,
        prism_mlp_only_layers=[1],
        prism_rvq_heads_per_execution_unit=[1, 1],
    )
    values = {"transformer.embed_tokens.weight": torch.randn(8, 4), "stop_head.weight": torch.randn(2, 4)}
    for q in range(2):
        values[f"audio_embeddings.{q}.weight"] = torch.randn(3, 4)
        values[f"audio_lm_heads.{q}.weight"] = torch.randn(3, 4)
        values[f"prediction_norms.{q}.weight"] = torch.randn(4)
    for layer in range(2):
        root = f"transformer.layers.{layer}"
        values[root + ".post_attention_layernorm.weight"] = torch.randn(4)
        values[root + ".mlp.gate_proj.weight"] = torch.randn(6, 4)
        values[root + ".mlp.up_proj.weight"] = torch.randn(6, 4)
        values[root + ".mlp.down_proj.weight"] = torch.randn(4, 6)
    values["transformer.layers.0.input_layernorm.weight"] = torch.randn(4)
    for key, shape in {
        "q_proj": (4, 4),
        "k_proj": (2, 4),
        "v_proj": (2, 4),
        "o_proj": (4, 4),
        "q_norm": (2,),
        "k_norm": (2,),
    }.items():
        values[f"transformer.layers.0.self_attn.{key}.weight"] = torch.randn(*shape)
    fake = SimpleNamespace(config=config, index=values, get=values.__getitem__)
    restored = {}
    for name, value in physical_tensors(fake):
        restored.update(convert_prism_to_hf(args, name, value))
    assert restored.keys() == values.keys()
    for key in values:
        torch.testing.assert_close(restored[key], values[key], rtol=0, atol=0)
