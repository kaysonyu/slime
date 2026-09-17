"""The offline artifact boundary preserves head identity and rejects destructive reuse."""

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tools.convert_moss_tts_2_0_for_sglang_omni import convert

NUM_GPUS = 0


def source_artifact(path, *, untied_text=False):
    path.mkdir()
    config = dict(
        model_type="moss_tts_local",
        n_vq=2,
        audio_codebook_sizes=[3, 3],
        audio_pad_token_id=3,
        tie_word_embeddings=True,
        gpt_neox_config=dict(
            hidden_size=4,
            num_attention_heads=1,
            intermediate_size=8,
            num_hidden_layers=1,
            hidden_act="silu",
            use_parallel_residual=False,
            attention_bias=True,
            layer_norm_eps=1e-6,
            rope_parameters={"rope_theta": 10000, "partial_rotary_factor": 1},
        ),
    )
    embedding = torch.arange(28, dtype=torch.float32).reshape(7, 4)
    tensors = {
        "transformer.embed_tokens.weight": embedding,
        "local_transformer.layers.0.attention.query_key_value.weight": torch.arange(48.0).reshape(12, 4),
        "local_transformer.layers.0.attention.query_key_value.bias": torch.arange(12.0),
    }
    for channel in range(2):
        tensors[f"audio_embeddings.{channel}.weight"] = torch.arange(12.0).reshape(3, 4) + channel
        tensors[f"audio_lm_heads.{channel}.weight"] = torch.arange(12.0).reshape(3, 4) + channel + 10
    save_file(tensors, path / "model-1.safetensors")
    save_file({"text_lm_head.weight": embedding + int(untied_text)}, path / "model-2.safetensors")
    weight_map = dict.fromkeys(tensors, "model-1.safetensors")
    weight_map["text_lm_head.weight"] = "model-2.safetensors"
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (path / "config.json").write_text(json.dumps(config))
    (path / "processing_moss_tts.py").write_text("# Original processor\n")
    (path / "tokenizer.json").write_text('{"marker":"original tokenizer"}')
    return tensors


def test_offline_conversion_retains_untied_heads_and_empty_shard(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "converted"
    original = source_artifact(source)
    convert(source, destination)
    index = json.loads((destination / "model.safetensors.index.json").read_text())
    assert "text_lm_head.weight" not in index["weight_map"]
    with safe_open(destination / "model-2.safetensors", framework="pt") as empty:
        assert list(empty.keys()) == []
    with safe_open(destination / "model-1.safetensors", framework="pt") as saved:
        for name, tensor in original.items():
            if name.startswith(("audio_", "transformer.")):
                assert torch.equal(saved.get_tensor(name), tensor)
        assert not torch.equal(
            saved.get_tensor("audio_embeddings.0.weight"), saved.get_tensor("audio_lm_heads.0.weight")
        )
    assert (destination / "tokenizer.json").read_bytes() == (source / "tokenizer.json").read_bytes()
    config = json.loads((destination / "config.json").read_text())
    assert config["gpt2_config"]["n_head"] == 1
    assert config["sglang_omni_compat"]["local_layout"] == "gpt2_qkv_interleaved_v1"
    assert config["tie_audio_embeddings_and_output_weights"] is False
    assert "v1.5-compat shim" in (destination / "processing_moss_tts.py").read_text()
    assert not destination.with_name("converted.incomplete").exists()
    with pytest.raises(FileExistsError):
        convert(source, destination)


def test_offline_conversion_refuses_to_discard_an_untied_text_head(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "converted"
    source_artifact(source, untied_text=True)
    with pytest.raises(ValueError, match="untied text head"):
        convert(source, destination)
    assert not destination.exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
