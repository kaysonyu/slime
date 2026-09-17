from __future__ import annotations

import re
import torch
from .common import SafetensorReader, merge_gate_up, merge_qkv, strip_mcore_wrappers


def _direct_tensor(name: str, reader: SafetensorReader, config) -> torch.Tensor | None:
    mapping = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": (
            "model.embed_tokens.weight"
            if getattr(config, "tie_word_embeddings", False) or "lm_head.weight" not in reader
            else "lm_head.weight"
        ),
    }
    return reader.get_tensor(mapping[name]) if name in mapping else None


def _layer(name: str) -> tuple[int, str]:
    match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
    if not match:
        raise KeyError(f"Unsupported Megatron parameter {name!r}")
    return int(match.group(1)), match.group(2)


def _attention_tensor(
    rest: str,
    prefix: str,
    reader: SafetensorReader,
    config,
) -> torch.Tensor | None:
    mapping = {
        "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
        "self_attention.linear_proj.bias": "self_attn.o_proj.bias",
        "self_attention.linear_qkv.layer_norm_weight": "input_layernorm.weight",
        "self_attention.q_layernorm.weight": "self_attn.q_norm.weight",
        "self_attention.k_layernorm.weight": "self_attn.k_norm.weight",
        "self_attention.core_attention.softmax_offset": "self_attn.sinks",
    }
    if rest in mapping:
        return reader.get_tensor(f"{prefix}.{mapping[rest]}")
    match = re.fullmatch(r"self_attention\.linear_qkv\.(weight|bias)", rest)
    if match:
        suffix = match.group(1)
        return merge_qkv(
            *(reader.get_tensor(f"{prefix}.self_attn.{projection}_proj.{suffix}") for projection in "qkv"),
            config,
        )
    return None


def qwen_hf_tensor(
    name: str,
    reader: SafetensorReader,
    config,
    *,
    layers_prefix="model.layers",
    embedding_key="model.embed_tokens.weight",
    norm_key="model.norm.weight",
) -> torch.Tensor:
    name = strip_mcore_wrappers(name)
    if name == "embedding.word_embeddings.weight":
        return reader.get_tensor(embedding_key)
    if name == "decoder.final_layernorm.weight":
        return reader.get_tensor(norm_key)
    if (tensor := _direct_tensor(name, reader, config)) is not None:
        return tensor

    layer, rest = _layer(name)
    prefix = f"{layers_prefix}.{layer}"
    if (tensor := _attention_tensor(rest, prefix, reader, config)) is not None:
        return tensor

    mapping = {
        "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
        "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
        "mlp.linear_fc2.weight": "mlp.down_proj.weight",
    }
    if rest in mapping:
        return reader.get_tensor(f"{prefix}.{mapping[rest]}")
    if rest == "mlp.linear_fc1.weight":
        return merge_gate_up(
            reader.get_tensor(f"{prefix}.mlp.gate_proj.weight"),
            reader.get_tensor(f"{prefix}.mlp.up_proj.weight"),
        )
    raise KeyError(f"Unsupported Qwen/Llama Megatron parameter {name!r}")
