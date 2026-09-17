"""Offline real-weight Local parity and converted processor contract check."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, GPTNeoXModel


def main():
    from sglang_omni.models.moss_tts.hf_loading import load_moss_processor_class, moss_transformers_processor_compat
    from sglang_omni.models.moss_tts_local.local_transformer import MossTTSLocalTransformer

    from slime.backends.megatron_utils.hf_to_megatron.common import SafetensorReader
    from slime_plugins.models.moss_tts_local.config import MossLocalConfig
    from slime_plugins.models.moss_tts_local.local_transformer import LocalTransformer

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--converted", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(17)
    source = SafetensorReader(args.source)
    converted = SafetensorReader(args.converted)
    config = MossLocalConfig.from_pretrained(args.converted)
    hf_config = AutoConfig.from_pretrained(args.source, trust_remote_code=True).gpt_neox_config
    hf_config._attn_implementation = "eager"
    reference = GPTNeoXModel(hf_config).float().eval()
    original_state = {
        key.removeprefix("local_transformer."): source.get_tensor(key).float()
        for key in source.weight_map
        if key.startswith("local_transformer.")
    }
    result = reference.load_state_dict(original_state, strict=False)
    if set(result.missing_keys) != {"embed_in.weight"} or result.unexpected_keys:
        raise ValueError(f"Original Local state mismatch: {result}")
    converted_state = {
        key.removeprefix("local_transformer."): converted.get_tensor(key).float()
        for key in converted.weight_map
        if key.startswith("local_transformer.")
    }
    local = config.local
    serving = (
        MossTTSLocalTransformer(
            hidden_size=config.hidden_size,
            num_heads=local["num_attention_heads"],
            inner_size=local["intermediate_size"],
            num_layers=local["num_hidden_layers"],
            max_positions=config.channels,
            rope_base=config.local_rope_base,
            layer_norm_eps=local["layer_norm_eps"],
        )
        .float()
        .eval()
    )
    serving.load_state_dict(converted_state, strict=True)
    training = LocalTransformer(config).float().eval()
    training.load_state_dict(converted_state, strict=True)
    inputs = torch.randn(2, config.channels, config.hidden_size)
    with torch.no_grad():
        expected = reference(inputs_embeds=inputs, use_cache=False).last_hidden_state
        actual = torch.stack([serving.step(inputs[:, position], position) for position in range(config.channels)], 1)
        train_output = training(inputs)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-5)
    torch.testing.assert_close(train_output, expected, atol=2e-4, rtol=2e-5)
    runtime_config = AutoConfig.from_pretrained(args.converted, trust_remote_code=True)
    processor_class = load_moss_processor_class(args.converted)
    tokenizer = AutoTokenizer.from_pretrained(args.converted, trust_remote_code=True)
    with moss_transformers_processor_compat():
        processor = processor_class(tokenizer=tokenizer, audio_tokenizer=None)
    message = processor.build_user_message(
        text="Hello, this is a test.", instruction="Speak clearly.", language="English"
    )
    encoded = processor([[message]], mode="generation")
    ids = encoded["input_ids"]
    if ids.ndim != 3 or ids.shape[-1] != config.channels:
        raise ValueError(f"Processor lost multi-channel layout: {ids.shape}")
    if (
        int(ids[0, -1, 0]) != int(runtime_config.audio_start_token_id)
        or not ids[0, -1, 1:].eq(config.audio_pad_id).all()
    ):
        raise ValueError("Processor did not retain the converted checkpoint's special IDs")
    report = dict(
        omni_fp32_max_abs=float((actual - expected).abs().max()),
        megatron_local_fp32_max_abs=float((train_output - expected).abs().max()),
        tensor_count=len(converted.weight_map),
        input_shape=list(ids.shape),
        audio_start=int(ids[0, -1, 0]),
        audio_pad=config.audio_pad_id,
        context_length=runtime_config.language_config.max_position_embeddings,
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
