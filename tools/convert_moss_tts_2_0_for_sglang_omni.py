#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline HF GPTNeoX -> Omni Local artifact conversion.

The runtime uses GPT-2 names, q|k|v blocks and interleaved RoPE. Conversion
preserves all audio heads and verifies the tied text head before omitting it.
An incomplete destination is never published as a ready model directory.
"""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LOCAL_RENAMES = {
    "input_layernorm": "ln_1",
    "post_attention_layernorm": "ln_2",
    "attention.query_key_value": "attn.c_attn",
    "attention.dense": "attn.c_proj",
    "mlp.dense_h_to_4h": "mlp.fc_in",
    "mlp.dense_4h_to_h": "mlp.fc_out",
}
RUNTIME_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "processor_config.json",
    "configuration_moss_tts.py",
    "modeling_moss_tts.py",
    "inference_utils.py",
    "README.md",
    "__init__.py",
]
PROCESSOR_SHIM = """

# --- sglang-omni v1.5-compat shim (offline Local artifact conversion) ---
_moss_tts_v2_build_user_message = MossTTSLocalProcessor.build_user_message


def _moss_tts_v15_build_user_message_compat(
    script=None, reference=None, tokens=None, global_instruction=None,
    sample_id="<unknown>", prompt_protocol=None,
    text=None, instruction=None, language=None, quality=None,
    sound_event=None, ambient_sound=None,
):
    if script is None:
        script = text
    if global_instruction is None:
        global_instruction = instruction
    if language:
        note = f"language: {language}"
        global_instruction = f"{global_instruction}\\n{note}" if global_instruction else note
    return _moss_tts_v2_build_user_message(
        script=script, reference=reference, tokens=tokens,
        global_instruction=global_instruction, sample_id=sample_id,
        prompt_protocol=prompt_protocol,
    )


MossTTSLocalProcessor.build_user_message = staticmethod(_moss_tts_v15_build_user_message_compat)
"""


def convert_local_tensor(name, tensor, num_heads):
    """Convert only at the artifact boundary; serving/training do not call this."""
    if name.startswith("local_transformer.final_layer_norm."):
        return name.replace("final_layer_norm.", "ln_f."), tensor
    match = re.fullmatch(r"local_transformer\.layers\.(\d+)\.(.+)\.(weight|bias)", name)
    if match is None or match[2] not in LOCAL_RENAMES:
        raise ValueError(f"Unknown Local tensor {name}")
    layer, leaf, suffix = match.groups()
    if leaf == "attention.query_key_value":
        dim, tail = tensor.shape[0] // (3 * num_heads), tensor.shape[1:]
        if dim % 2 or tensor.shape[0] != 3 * num_heads * dim:
            raise ValueError("Local QKV requires an even head dimension")
        q, k, v = tensor.reshape(num_heads, 3, dim, *tail).unbind(1)
        q, k = [value.reshape(num_heads, 2, dim // 2, *tail).transpose(1, 2).reshape_as(value) for value in (q, k)]
        tensor = torch.stack((q, k, v)).reshape(3 * num_heads * dim, *tail).contiguous()
    return f"local_transformer.h.{layer}.{LOCAL_RENAMES[leaf]}.{suffix}", tensor


def convert(src_dir, dst_dir):
    source, destination = Path(src_dir).resolve(), Path(dst_dir).resolve()
    staging = destination.with_name(destination.name + ".incomplete")
    if destination.exists() or staging.exists():
        raise FileExistsError(f"Choose a new destination: {destination}")
    config = json.loads((source / "config.json").read_text())
    index = json.loads((source / "model.safetensors.index.json").read_text())
    local = config.get("gpt_neox_config")
    if not isinstance(local, dict) or config.get("gpt2_config"):
        raise ValueError("Input must be the original unconverted GPTNeoX HF artifact")
    hidden, heads = local["hidden_size"], local["num_attention_heads"]
    if hidden % heads or (hidden // heads) % 2:
        raise ValueError("Invalid Local attention shape")
    rope = local.get("rope_parameters") or {}
    if local["hidden_act"] != "silu" or local["use_parallel_residual"] or not local["attention_bias"]:
        raise ValueError("Conversion requires SiLU, sequential residual and biased attention")
    if rope.get("rope_type", "default") != "default" or rope.get("partial_rotary_factor", 1) != 1:
        raise ValueError("Conversion requires default full-head RoPE")
    books = config["audio_codebook_sizes"]
    if len(books) != config["n_vq"] or len(set(books)) != 1:
        raise ValueError("Local requires equal-sized audio codebooks")
    processor = (source / "processing_moss_tts.py").read_text()
    if "v1.5-compat shim" in processor:
        raise ValueError("Source processor already contains a compatibility shim")
    weight_map = index["weight_map"]
    text_head = "text_lm_head.weight"
    embedding = "transformer.embed_tokens.weight"
    if text_head in weight_map:
        with safe_open(source / weight_map[text_head], framework="pt") as a, safe_open(
            source / weight_map[embedding], framework="pt"
        ) as b:
            if not config.get("tie_word_embeddings") or not torch.equal(
                a.get_tensor(text_head), b.get_tensor(embedding)
            ):
                raise ValueError("Refusing to discard an untied text head")
    staging.mkdir(parents=True)
    mapping, size = {}, 0
    stats = dict(copied=0, renamed=0, rope_permuted=0, tie_skipped=0)
    for shard in sorted(set(weight_map.values())):
        tensors = {}
        with safe_open(source / shard, framework="pt") as original:
            for name in sorted(key for key, value in weight_map.items() if value == shard):
                if name == text_head:
                    stats["tie_skipped"] += 1
                    continue
                value = original.get_tensor(name)
                target = name
                if name.startswith("local_transformer."):
                    target, value = convert_local_tensor(name, value, heads)
                    stats["renamed"] += 1
                    stats["rope_permuted"] += "query_key_value" in name
                else:
                    stats["copied"] += 1
                if target in mapping:
                    raise ValueError(f"Duplicate converted tensor {target}")
                tensors[target] = value
                mapping[target] = shard
                size += value.numel() * value.element_size()
        save_file(tensors, staging / shard, metadata={"format": "pt"})
        # Reopen every shard and compare bytes, including every unchanged tensor.
        with safe_open(staging / shard, framework="pt") as saved:
            if set(saved.keys()) != set(tensors):
                raise ValueError(f"Shard index mismatch: {shard}")
            for name, value in tensors.items():
                if not torch.equal(saved.get_tensor(name).view(torch.uint8), value.contiguous().view(torch.uint8)):
                    raise ValueError(f"Written tensor differs: {name}")
        print(f"[shard] {shard}: {len(tensors)} tensors verified", flush=True)
    config["gpt2_config"] = dict(
        model_type="gpt2",
        activation_function="silu",
        n_embd=hidden,
        n_head=heads,
        n_inner=local["intermediate_size"],
        n_layer=local["num_hidden_layers"],
        n_positions=config["n_vq"] + 1,
        n_ctx=config["n_vq"] + 1,
        layer_norm_epsilon=local["layer_norm_eps"],
        position_embedding_type="rope",
        rope_base=rope["rope_theta"],
    )
    config.update(
        audio_vocab_size=books[0],
        audio_pad_code=config["audio_pad_token_id"],
        local_transformer_layers=local["num_hidden_layers"],
        local_text_head_mode="binary",
        use_static_local_kv_cache=True,
        tie_audio_embeddings_and_output_weights=False,
    )
    config["sglang_omni_compat"] = dict(
        converted_from=str(source),
        converter=Path(__file__).name,
        local_layout="gpt2_qkv_interleaved_v1",
        reference_commit="92a53c268a167e153cbe7f36b1d916b8a72c0b4f",
    )
    for name in RUNTIME_FILES:
        if (source / name).exists():
            shutil.copy2(source / name, staging / name)
    (staging / "processing_moss_tts.py").write_text(processor + PROCESSOR_SHIM)
    (staging / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    (staging / "model.safetensors.index.json").write_text(
        json.dumps(dict(metadata={"total_size": size}, weight_map=mapping), indent=2) + "\n"
    )
    report = dict(
        source=str(source),
        destination=str(destination),
        tensors=len(mapping),
        shards=len(set(weight_map.values())),
        total_size=size,
        statistics=stats,
        source_config_sha256=hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
        converter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        verification="all output tensor bytes checked against source or exact permutation",
    )
    (staging / "conversion_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    if destination.exists():
        raise FileExistsError(destination)
    staging.rename(destination)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src_dir")
    parser.add_argument("dst_dir")
    args = parser.parse_args()
    torch.set_num_threads(8)
    convert(args.src_dir, args.dst_dir)
