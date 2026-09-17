"""Canonical Local HF <-> Megatron mapping; Local tensors need no runtime permutation."""

import re
from types import SimpleNamespace

import torch

from slime.backends.megatron_utils.hf_to_megatron.common import SafetensorReader, merge_qkv, shard_mcore_tensor
from slime.backends.megatron_utils.megatron_to_hf.qwen2 import convert_qwen2_to_hf

checkpoint_aliases = {}


def expected_hf_names(path):
    return set(SafetensorReader(path).weight_map) - {"text_lm_head.weight"}


def _name(name):
    while name.startswith("module."):
        name = name.removeprefix("module.")
    return name


def hf_tensor(name, reader, config):
    name = _name(name)
    if name == "text_embedding.word_embeddings.weight":
        return reader.get_tensor("transformer.embed_tokens.weight")
    if name == "decoder.final_layernorm.weight":
        return reader.get_tensor("transformer.norm.weight")
    if name.startswith(("audio_embeddings.", "audio_lm_heads.", "local_text_lm_head.")):
        return reader.get_tensor(name)
    if name.startswith("local_transformer."):
        return reader.get_tensor(name)
    match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
    if not match:
        raise ValueError(f"Unknown Local training parameter {name}")
    layer, rest = match.groups()
    prefix = f"transformer.layers.{layer}"
    direct = {
        "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
        "self_attention.linear_qkv.layer_norm_weight": "input_layernorm.weight",
        "input_layernorm.weight": "input_layernorm.weight",
        "self_attention.q_layernorm.weight": "self_attn.q_norm.weight",
        "self_attention.k_layernorm.weight": "self_attn.k_norm.weight",
        "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
        "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
        "mlp.linear_fc2.weight": "mlp.down_proj.weight",
    }
    if rest in direct:
        return reader.get_tensor(f"{prefix}.{direct[rest]}")
    if rest == "self_attention.linear_qkv.weight":
        return merge_qkv(
            *(reader.get_tensor(f"{prefix}.self_attn.{p}_proj.weight") for p in "qkv"),
            SimpleNamespace(**config.language),
        )
    if rest == "mlp.linear_fc1.weight":
        return torch.cat([reader.get_tensor(f"{prefix}.mlp.{p}_proj.weight") for p in ("gate", "up")])
    raise ValueError(f"Unknown Local global parameter {name}")


@torch.no_grad()
def load_weights(model, path, config):
    reader = SafetensorReader(path)
    for name, parameter in model.named_parameters():
        tensor = hf_tensor(name, reader, config)
        if _name(name) == "text_embedding.word_embeddings.weight":
            from megatron.core import mpu

            padded = parameter.shape[0] * mpu.get_tensor_model_parallel_world_size()
            if padded > tensor.shape[0]:
                tensor = torch.nn.functional.pad(tensor, (0, 0, 0, padded - tensor.shape[0]))
        tensor = shard_mcore_tensor(name, tensor, parameter)
        if tensor.shape != parameter.shape:
            raise ValueError(f"Checkpoint shape mismatch for {name}: {tensor.shape} != {parameter.shape}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Checkpoint contains non-finite parameter {name}")
        parameter.copy_(tensor)


def export_parameter(args, name, tensor):
    name = _name(name)
    config = args.policy_config
    if name.startswith(("audio_embeddings.", "audio_lm_heads.", "local_text_lm_head.")):
        return [(name, tensor)]
    if name.startswith("local_transformer."):
        return [(name, tensor)]
    if name == "text_embedding.word_embeddings.weight":
        return [("transformer.embed_tokens.weight", tensor[: config.language["vocab_size"]].contiguous())]
    converted = convert_qwen2_to_hf(args, "module.module." + name, tensor)
    return [(key.replace("model.", "transformer.", 1), value) for key, value in converted]
