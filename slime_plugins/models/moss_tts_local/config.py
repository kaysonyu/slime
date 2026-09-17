"""Read the actual HF artifact; model dimensions are never launch-profile constants."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MossLocalConfig:
    language: dict
    local: dict
    n_vq: int
    audio_vocab_size: int
    audio_pad_id: int
    audio_slot_id: int
    audio_end_id: int
    text_pad_id: int
    sample_rate: int
    config_sha256: str

    @classmethod
    def from_dict(cls, config: dict):
        if config.get("model_type") != "moss_tts_local":
            raise ValueError("Expected a complete MOSS TTS Local checkpoint")
        language = config["language_config"]
        runtime = config.get("gpt2_config")
        if not isinstance(runtime, dict):
            raise ValueError(
                "MOSS requires the converted Local HF directory; run tools/convert_moss_tts_2_0_for_sglang_omni.py"
            )
        local = dict(
            hidden_size=runtime["n_embd"],
            num_attention_heads=runtime["n_head"],
            intermediate_size=runtime["n_inner"],
            num_hidden_layers=runtime["n_layer"],
            hidden_act=runtime["activation_function"],
            layer_norm_eps=runtime["layer_norm_epsilon"],
            rope_parameters={"rope_theta": runtime["rope_base"]},
        )
        if runtime.get("position_embedding_type", "rope") != "rope":
            raise ValueError("Local requires interleaved RoPE")
        books = config["audio_codebook_sizes"]
        n_vq = config["n_vq"]
        if not isinstance(n_vq, int) or n_vq < 1 or len(books) != n_vq or len(set(books)) != 1:
            raise ValueError("MOSS Local requires n_vq positive, equal-size codebooks")
        if books[0] < 2 or local["hidden_size"] != language["hidden_size"]:
            raise ValueError("Invalid codebook vocabulary or global/local hidden-size mismatch")
        if local["hidden_act"] != "silu":
            raise ValueError("Unsupported Local decoder: expected sequential residual, SiLU and QKV bias")
        rope = local.get("rope_parameters", {})
        if rope.get("rope_type", "default") != "default" or rope.get("partial_rotary_factor", 1) != 1:
            raise ValueError("Local decoder requires default full-head RoPE")
        head_dim = local["hidden_size"] // local["num_attention_heads"]
        if head_dim % 2 or local["hidden_size"] % local["num_attention_heads"]:
            raise ValueError("Local attention requires an even, integral head dimension")
        digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return cls(
            language=dict(language),
            local=dict(local),
            n_vq=n_vq,
            audio_vocab_size=books[0],
            audio_pad_id=config["audio_pad_token_id"],
            audio_slot_id=config["audio_assistant_slot_token_id"],
            audio_end_id=config["audio_end_token_id"],
            text_pad_id=config["pad_token_id"],
            sample_rate=config.get("sampling_rate", 48000),
            config_sha256=digest,
        )

    @classmethod
    def from_pretrained(cls, path):
        return cls.from_dict(json.loads((Path(path) / "config.json").read_text()))

    @property
    def channels(self):
        return self.n_vq + 1

    @property
    def hidden_size(self):
        return self.language["hidden_size"]

    @property
    def local_rope_base(self):
        return self.local.get("rope_parameters", {}).get("rope_theta", 1000000.0)

    def validate_rollout_identity(self, identity):
        expected = {
            "n_vq": self.n_vq,
            "audio_vocab_size": self.audio_vocab_size,
            "audio_pad_code": self.audio_pad_id,
            "audio_assistant_slot_token_id": self.audio_slot_id,
            "audio_end_token_id": self.audio_end_id,
            "text_vocab_size": self.language["vocab_size"],
            "hidden_size": self.hidden_size,
            "global_layers": self.language["num_hidden_layers"],
            "global_num_attention_heads": self.language["num_attention_heads"],
            "global_num_query_groups": self.language["num_key_value_heads"],
            "global_ffn_hidden_size": self.language["intermediate_size"],
            "global_rope_base": self.language.get("rope_parameters", {}).get(
                "rope_theta", self.language.get("rope_theta", 1000000)
            ),
            "global_layer_norm_epsilon": self.language.get("rms_norm_eps", 1e-6),
            "qk_layernorm": True,
            "local_layers": self.local["num_hidden_layers"],
            "local_num_attention_heads": self.local["num_attention_heads"],
            "local_ffn_hidden_size": self.local["intermediate_size"],
            "local_rope_base": self.local_rope_base,
            "local_layer_norm_epsilon": self.local["layer_norm_eps"],
            "local_activation": self.local["hidden_act"],
            "sample_rate": self.sample_rate,
            "tie_audio_embeddings_and_output_weights": False,
        }
        for key, value in expected.items():
            if identity.get(key) != value:
                raise ValueError(f"Rollout model identity mismatch for {key}: {identity.get(key)!r} != {value!r}")
