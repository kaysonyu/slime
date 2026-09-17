"""Map the discrete Higgs policy while leaving its codec frozen in Omni."""

from types import SimpleNamespace

import torch

from slime.backends.megatron_utils.hf_to_megatron.common import (
    SafetensorReader,
    shard_mcore_tensor,
    strip_mcore_wrappers,
)
from slime.backends.megatron_utils.hf_to_megatron.qwen import qwen_hf_tensor
from slime.backends.megatron_utils.megatron_to_hf.qwen2 import convert_qwen2_to_hf

AUDIO_EMBEDDING = "tied.embedding.modality_embeddings.0.embedding.weight"
AUDIO_HEAD = "tied.head.modality_heads.0.weight"


def expected_hf_names(path):
    return {
        name
        for name in SafetensorReader(path).weight_map
        if name.startswith("body.") or name in {"tied.embedding.text_embedding.weight", AUDIO_EMBEDDING, AUDIO_HEAD}
    }


@torch.no_grad()
def load_weights(model, path, config):
    reader = SafetensorReader(path)
    language = SimpleNamespace(**config["text_config"])
    for name, parameter in model.named_parameters():
        plain = strip_mcore_wrappers(name)
        if plain == "audio_embedding.weight":
            tensor = reader.get_tensor(AUDIO_EMBEDDING)
        elif plain == "audio_head.weight":
            tensor = reader.get_tensor(AUDIO_HEAD)
        else:
            normalized = plain.replace("text_embedding.word_embeddings.", "embedding.word_embeddings.")
            tensor = qwen_hf_tensor(
                normalized,
                reader,
                language,
                layers_prefix="body.layers",
                embedding_key="tied.embedding.text_embedding.weight",
                norm_key="body.norm.weight",
            )
            if plain == "text_embedding.word_embeddings.weight":
                from megatron.core import mpu

                padded = parameter.shape[0] * mpu.get_tensor_model_parallel_world_size()
                if padded > tensor.shape[0]:
                    tensor = torch.nn.functional.pad(tensor, (0, 0, 0, padded - tensor.shape[0]))
        tensor = shard_mcore_tensor(name, tensor, parameter)
        if tensor.shape != parameter.shape:
            raise ValueError(f"Higgs parameter shape mismatch: {name}: {tensor.shape} != {parameter.shape}")
        parameter.copy_(tensor)


def export_parameter(args, name, tensor):
    name = strip_mcore_wrappers(name)
    if name == "audio_embedding.weight":
        return [(AUDIO_EMBEDDING, tensor)]
    if name == "audio_head.weight":
        return [(AUDIO_HEAD, tensor)]
    if name == "text_embedding.word_embeddings.weight":
        return [("tied.embedding.text_embedding.weight", tensor[: args.vocab_size].contiguous())]
    result = convert_qwen2_to_hf(args, "module.module." + name, tensor)
    return [(key.replace("model.", "body.", 1), value) for key, value in result]
