# Copyright (c) 2025, OpenMOSS. All rights reserved.
"""Prism configuration contracts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from slime_plugins.models.prism_topology import (
    ResolvedPrismTopology,
    _positive_integer,
    normalize_prism_input_rvq_channels,
    resolve_prism_topology,
)


@dataclass(frozen=True)
class PrismModelSpec:
    """Architecture values owned by the inference artifact, independent of runtime TP."""

    topology: ResolvedPrismTopology
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    kv_channels: int
    text_vocab_size: int
    speech_vocab_size: int
    max_position_embeddings: int
    norm_epsilon: float
    rotary_base: float
    text_pad_idx: int
    audio_start_token_id: int
    input_rvq_channels: tuple[int, ...]
    embedding_head_usage: str
    dtype: str

    @classmethod
    def from_hf_config(cls, config: Mapping[str, Any]):
        if config.get("model_type") != "moss_tts_prism":
            raise ValueError("expected a moss_tts_prism Hugging Face configuration")
        language = config.get("language_config")
        if not isinstance(language, Mapping) or language.get("model_type") != "qwen3":
            raise ValueError("Prism requires language_config describing Qwen3")
        for name, expected in (("hidden_act", "silu"), ("attention_bias", False)):
            if language.get(name) != expected:
                raise ValueError(f"unsupported Prism language_config.{name}")
        if language.get("attention_dropout", 0.0) != 0.0 or language.get("use_sliding_window", False):
            raise ValueError("Prism requires zero dropout and full causal attention")
        fields = {
            "n_vq": config["n_vq"],
            "prism_loop_block_ranges": config["prism_loop_block_ranges"],
            "prism_loop_block_repetitions": config["prism_loop_block_repetitions"],
            "prism_rvq_heads_per_execution_unit": config["prism_rvq_heads_per_execution_unit"],
            "prism_mlp_only_layers": config.get("prism_mlp_only_layers", []),
            "reapply_rvq_conditioning_in_loop": config.get("reapply_rvq_conditioning_in_loop", False),
        }
        topology = resolve_prism_topology(SimpleNamespace(**fields))
        if language["num_hidden_layers"] != topology.num_layers:
            raise ValueError("Prism language depth disagrees with physical topology")
        if any(kind != "full_attention" for kind in language.get("layer_types", [])):
            raise ValueError("Prism requires full_attention layer types")
        dimensions = {
            "hidden_size": language["hidden_size"],
            "ffn_hidden_size": language["intermediate_size"],
            "num_attention_heads": language["num_attention_heads"],
            "num_query_groups": language["num_key_value_heads"],
            "kv_channels": language["head_dim"],
            "text_vocab_size": language["vocab_size"],
            "speech_vocab_size": config["speech_vocab_size"],
            "max_position_embeddings": language["max_position_embeddings"],
        }
        for name, value in dimensions.items():
            _positive_integer(value, name)
        if dimensions["num_attention_heads"] % dimensions["num_query_groups"] or dimensions["kv_channels"] % 2:
            raise ValueError("Prism requires valid GQA and even full-head RoPE dimensions")
        rope = language.get("rope_parameters")
        if rope is None:
            if language.get("rope_scaling") is not None:
                raise ValueError("Prism requires unscaled RoPE")
            theta = language["rope_theta"]
        else:
            if rope.get("rope_type") not in ("default", "rope") or set(rope) != {"rope_type", "rope_theta"}:
                raise ValueError("Prism requires the default, unscaled RoPE descriptor")
            theta = rope["rope_theta"]
        eps = language["rms_norm_eps"]
        if any(
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0
            for value in (theta, eps)
        ):
            raise ValueError("Prism norm epsilon and rotary base must be finite and positive")
        usage = config.get("embedding_head_usage")
        if usage not in ("untied", "tied_head_authoritative"):
            raise ValueError("Prism requires an exact embedding_head_usage")
        audio_start = config["audio_start_token_id"]
        if type(audio_start) is not int or audio_start != 151652:
            raise ValueError("Prism's moss_tts_v2 protocol requires audio_start_token_id=151652")
        dtype = config.get("dtype", config.get("torch_dtype"))
        if dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError("Prism artifact must declare its trainable dtype")
        if language.get("dtype", language.get("torch_dtype", dtype)) != dtype:
            raise ValueError("Prism language and root dtype disagree")
        return cls(
            topology=topology,
            **dimensions,
            norm_epsilon=float(eps),
            rotary_base=float(theta),
            text_pad_idx=config["text_pad_idx"],
            audio_start_token_id=audio_start,
            embedding_head_usage=usage,
            dtype=dtype,
            input_rvq_channels=normalize_prism_input_rvq_channels(
                config["prism_input_rvq_channels"], n_vq=topology.n_vq
            ),
        )


def read_prism_spec(hf_checkpoint: str | Path) -> PrismModelSpec:
    path = Path(hf_checkpoint)
    config = path if path.is_file() else path / "config.json"
    return PrismModelSpec.from_hf_config(json.loads(config.read_text()))


def prism_megatron_cli_args(hf_checkpoint: str | Path) -> list[str]:
    """Generate ordinary Megatron architecture flags before its parser runs.

    The provider fills Prism's private topology/RVQ fields after normal CLI
    validation. Attention backend, parallelism and optimizer choices remain
    caller-owned. Standard text/output tying is absent from Prism; audio
    embedding/head use is independently described by embedding_head_usage.
    """
    spec = read_prism_spec(hf_checkpoint)
    if not spec.rotary_base.is_integer():
        raise ValueError("Megatron's --rotary-base requires an integer Prism RoPE base")
    values = (
        ("--num-layers", spec.topology.num_layers),
        ("--hidden-size", spec.hidden_size),
        ("--ffn-hidden-size", spec.ffn_hidden_size),
        ("--num-attention-heads", spec.num_attention_heads),
        ("--num-query-groups", spec.num_query_groups),
        ("--kv-channels", spec.kv_channels),
        ("--vocab-size", spec.text_vocab_size),
        ("--padded-vocab-size", spec.text_vocab_size),
        ("--max-position-embeddings", spec.max_position_embeddings),
        ("--normalization", "RMSNorm"),
        ("--layernorm-epsilon", spec.norm_epsilon),
        ("--position-embedding-type", "rope"),
        ("--rotary-base", int(spec.rotary_base)),
        ("--rotary-percent", 1.0),
        ("--attention-dropout", 0.0),
        ("--hidden-dropout", 0.0),
        ("--transformer-impl", "transformer_engine"),
    )
    result = [token for flag, field_value in values for token in (flag, str(field_value))]
    result.extend(("--swiglu", "--qk-layernorm", "--group-query-attention", "--disable-bias-linear"))
    result.append("--untie-embeddings-and-output-weights")
    if spec.dtype in ("bfloat16", "float16"):
        result.append("--bf16" if spec.dtype == "bfloat16" else "--fp16")
    return result


def configure_prism_args(args) -> PrismModelSpec:
    """Populate provider-owned Prism construction values from its HF artifact.

    Normal entry points pass prism_megatron_cli_args before CLI validation;
    this provider helper fills the topology, RVQ and forward-semantics fields.
    """
    spec = read_prism_spec(args.hf_checkpoint)
    topology = spec.topology
    values = {
        "num_layers": topology.num_layers,
        "hidden_size": spec.hidden_size,
        "ffn_hidden_size": spec.ffn_hidden_size,
        "num_attention_heads": spec.num_attention_heads,
        "num_query_groups": spec.num_query_groups,
        "kv_channels": spec.kv_channels,
        "padded_vocab_size": spec.text_vocab_size,
        "max_position_embeddings": spec.max_position_embeddings,
        "n_vq": topology.n_vq,
        "speech_vocab_size": spec.speech_vocab_size,
        "text_pad_idx": spec.text_pad_idx,
        "audio_start_token_id": spec.audio_start_token_id,
        "prism_loop_block_ranges": topology.prism_loop_block_ranges,
        "prism_loop_block_repetitions": topology.prism_loop_block_repetitions,
        "prism_rvq_heads_per_execution_unit": topology.prism_rvq_heads_per_execution_unit,
        "prism_mlp_only_layers": topology.prism_mlp_only_layers,
        "reapply_rvq_conditioning_in_loop": topology.reapply_rvq_conditioning_in_loop,
        "prism_input_rvq_channels": spec.input_rvq_channels,
        "qk_layernorm": True,
        "normalization": "RMSNorm",
        "layernorm_epsilon": spec.norm_epsilon,
        "position_embedding_type": "rope",
        "rotary_base": spec.rotary_base,
        "rotary_percent": 1.0,
        "rotary_interleaved": False,
        "rotary_seq_len_interpolation_factor": None,
        "use_rope_scaling": False,
        "swiglu": True,
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
        "add_bias_linear": False,
        "add_qkv_bias": False,
        "apply_query_key_layer_scaling": False,
        "group_query_attention": True,
        "untie_embeddings_and_output_weights": True,
    }
    for key, value in values.items():
        setattr(args, key, value)
    return spec


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Print Prism's native Megatron architecture arguments, one per line.", allow_abbrev=False
    )
    parser.add_argument("--hf-checkpoint", required=True)
    # Read the same argv as train.py so the last --hf-checkpoint override wins.
    # All training options belong to Slime's parser.
    arguments, _ = parser.parse_known_args()
    print("\n".join(prism_megatron_cli_args(arguments.hf_checkpoint)))
