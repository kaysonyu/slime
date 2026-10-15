"""Prepare immutable FP32 task vectors in Prism's physical Megatron layout."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from slime.data.checkpoint_boundary import write_checkpoint_json


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


class HFWeights:
    def __init__(self, root):
        self.root = Path(root)
        self.config = json.loads((self.root / "config.json").read_text())
        self.index = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        self.handles = {}

    def get(self, name):
        file = self.index[name]
        if file not in self.handles:
            self.handles[file] = safe_open(self.root / file, framework="pt", device="cpu")
        return self.handles[file].get_tensor(name)

    def identity(self):
        files = {"config.json", "tokenizer.json", "tokenizer_config.json", *self.index.values()}
        return {
            "path": str(self.root.resolve()),
            "sha256": {name: file_sha256(self.root / name) for name in sorted(files)},
        }


def physical_tensors(weights):
    """Inverse of megatron_to_hf.prism; repeated execution sites remain shared."""
    config = weights.config
    language = config["language_config"]
    heads, groups = language["num_attention_heads"], language["num_key_value_heads"]
    width = language["hidden_size"]
    dim = language.get("head_dim", width // heads)
    consumed = set()

    def get(name):
        consumed.add(name)
        return weights.get(name)

    yield "text_embedding.word_embeddings.weight", get("transformer.embed_tokens.weight")
    yield "stop_head.weight", get("stop_head.weight")
    for q in range(config["n_vq"]):
        yield f"audio_embeddings.{q}.word_embeddings.weight", get(f"audio_embeddings.{q}.weight")
        yield f"audio_lm_heads.{q}.weight", get(f"audio_lm_heads.{q}.weight")
    norms = sum(count > 0 for count in config["prism_rvq_heads_per_execution_unit"])
    for index in range(norms):
        yield f"prediction_norms.{index}.weight", get(f"prediction_norms.{index}.weight")
    for layer in range(language["num_hidden_layers"]):
        native, hf = f"decoder.layers.{layer}", f"transformer.layers.{layer}"
        yield native + ".mlp.linear_fc1.layer_norm_weight", get(hf + ".post_attention_layernorm.weight")
        yield native + ".mlp.linear_fc1.weight", torch.cat(
            (get(hf + ".mlp.gate_proj.weight"), get(hf + ".mlp.up_proj.weight"))
        )
        yield native + ".mlp.linear_fc2.weight", get(hf + ".mlp.down_proj.weight")
        if layer in config.get("prism_mlp_only_layers", []):
            continue
        query = get(hf + ".self_attn.q_proj.weight").reshape(groups, heads // groups, dim, width)
        key = get(hf + ".self_attn.k_proj.weight").reshape(groups, 1, dim, width)
        value = get(hf + ".self_attn.v_proj.weight").reshape(groups, 1, dim, width)
        yield native + ".self_attention.linear_qkv.weight", torch.cat((query, key, value), dim=1).reshape(-1, width)
        yield native + ".self_attention.linear_proj.weight", get(hf + ".self_attn.o_proj.weight")
        yield native + ".self_attention.linear_qkv.layer_norm_weight", get(hf + ".input_layernorm.weight")
        for head in ("q", "k"):
            yield native + f".self_attention.{head}_layernorm.weight", get(hf + f".self_attn.{head}_norm.weight")
    if consumed != set(weights.index):
        raise ValueError(f"Unmapped HF tensors: {sorted(set(weights.index) - consumed)}")


def build_vectors(base_path, teacher_path, output):
    from slime_plugins.prism.protocol import PrismConfig

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    base, teacher = HFWeights(base_path), HFWeights(teacher_path)
    a, b = PrismConfig.from_path(base_path), PrismConfig.from_path(teacher_path)
    if a.replay_geometry() != b.replay_geometry() or a.tokenizer_sha256 != b.tokenizer_sha256:
        raise ValueError("Task-vector sources have different Prism geometry/tokenizer")
    base_identity, teacher_identity = base.identity(), teacher.identity()
    index, squared_norm = {}, 0.0
    for number, ((name, source), (target_name, target)) in enumerate(
        zip(physical_tensors(base), physical_tensors(teacher), strict=True)
    ):
        if target_name != name or source.shape != target.shape:
            raise ValueError(f"Task-vector source mismatch: {name}")
        delta = target.float() - source.float()
        if not torch.isfinite(delta).all():
            raise ValueError(f"Non-finite task vector: {name}")
        norm = float(torch.linalg.vector_norm(delta.double()))
        squared_norm += norm**2
        file = f"tensor-{number:04d}.safetensors"
        save_file({"delta": delta.flatten().contiguous()}, output / file)
        index[name] = {"file": file, "shape": list(delta.shape), "sha256": file_sha256(output / file), "norm": norm}
    manifest = {
        "version": 1,
        "layout": "prism_megatron_physical_v1",
        "dtype": "float32",
        "base": base_identity,
        "teacher": teacher_identity,
        "tensors": index,
        "norm": math.sqrt(squared_norm),
    }
    write_checkpoint_json(output / "index.json", manifest)
    return manifest


class TaskVectorStore:
    def __init__(self, path, checksum):
        self.root = Path(path)
        index = self.root / "index.json"
        if file_sha256(index) != checksum:
            raise ValueError("Task-vector manifest checksum changed")
        metadata = json.loads(index.read_text())
        if (
            metadata["version"] != 1
            or metadata["layout"] != "prism_megatron_physical_v1"
            or metadata["dtype"] != "float32"
        ):
            raise ValueError("Unsupported task-vector format")
        self.tensors = metadata["tensors"]
        self.handles = {}

    def slice(self, name, shape, start, end):
        entry = self.tensors[name]
        if list(shape) != entry["shape"] or not 0 <= start <= end <= math.prod(shape):
            raise ValueError(f"Task-vector shape or shard mismatch: {name}")
        path = self.root / entry["file"]
        if name not in self.handles:
            # The training owner verifies payloads on rank zero before merging.
            self.handles[name] = safe_open(path, framework="pt", device="cpu")
        value = self.handles[name].get_slice("delta")[start:end]
        if value.dtype != torch.float32:
            raise ValueError("Task vectors must retain FP32 arithmetic")
        return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_vectors(args.base, args.teacher, args.output)
    print(json.dumps({"tensors": len(result["tensors"]), "norm": result["norm"]}))
