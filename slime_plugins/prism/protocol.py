"""Prism replay identities, geometry, and typed tensor validation."""

from __future__ import annotations
import hashlib
import json
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import torch

# mossLite.models.MossTTSPrism.run_policy: PrismItemKind / PrismAudioRole.
TEXT, AUDIO = 0, 1
NONE, REFERENCE, TARGET = 0, 1, 2
LAYOUT = "moss_prism_typed_v1"
LOGPROB_SEMANTICS = "temperature_scaled_full_vocab_v1"
STOP_SEMANTICS = "pre_frame_bernoulli_v1"
REPLAY_FIELDS = (
    "layout",
    "prompt_rows",
    "item_kind",
    "audio_role",
    "successor_audio_mask",
    "successor_audio_codes",
    "target_history_retention_mask",
)


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _tensor(value, shape, dtype, name):
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    flat = value
    if len(shape) == 2:
        if any(not isinstance(row, list) or len(row) != shape[1] for row in value):
            raise ValueError(f"{name} has invalid row widths")
        flat = [item for row in value for item in row]
    expected = (bool,) if dtype == torch.bool else (int,) if dtype == torch.long else (int, float)
    if any(type(item) not in expected for item in flat):
        raise ValueError(f"{name} has invalid scalar types")
    result = torch.tensor(value, dtype=dtype)
    if shape[0] == 0 and value == []:
        result = result.reshape(shape)
    if tuple(result.shape) != tuple(shape):
        raise ValueError(f"{name} shape {tuple(result.shape)} != {shape}")
    if dtype.is_floating_point and not torch.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite scores")
    return result


def _copy_wire_data(value):
    """Snapshot decoded protocol containers without Python's per-element deepcopy dispatch."""
    # Only deserialize bytes produced locally from the already decoded payload.
    return pickle.loads(pickle.dumps(value, protocol=5))


@dataclass(frozen=True)
class PrismConfig:
    n_vq: int
    speech_vocab_size: int
    text_vocab_size: int
    text_pad_idx: int
    pad_token_id: int
    audio_start_token_id: int
    sample_rate: int
    context_length: int
    input_rvq_channels: tuple[int, ...]
    identity: dict[str, Any]
    tokenizer_sha256: dict[str, str] | None = None

    def replay_geometry(self):
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in vars(self).items()
            if name not in ("identity", "tokenizer_sha256")
        }

    @classmethod
    def from_path(cls, checkpoint):
        root = Path(checkpoint)
        raw = (root / "config.json").read_bytes()
        config = json.loads(raw)
        if config.get("model_type") != "moss_tts_prism":
            raise ValueError("Omni rollout requires a moss_tts_prism inference artifact")
        language = config["language_config"]
        depth = _integer(config["n_vq"], "n_vq", minimum=1)
        channels = config["prism_input_rvq_channels"]
        if (
            not isinstance(channels, list)
            or not channels
            or channels != sorted(set(channels))
            or any(type(channel) is not int or not 1 <= channel <= depth for channel in channels)
        ):
            raise ValueError("prism_input_rvq_channels must be a sorted one-based channel subset")
        identity = {
            "policy_family": "moss_tts_prism",
            "architecture": "MOSS-TTS-Prism",
            "config_sha256": hashlib.sha256(raw).hexdigest(),
            "n_vq": depth,
            "audio_vocab_size": config["speech_vocab_size"],
            "sample_rate": config["sampling_rate"],
            "stop_semantics": "pre_frame_threshold_v1",
        }
        for key, filename in (
            ("modeling_sha256", "modeling_moss_tts.py"),
            ("processor_sha256", "processing_moss_tts.py"),
            ("prompt_protocol_sha256", "prompt_protocol.py"),
        ):
            identity[key] = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        return cls(
            depth,
            _integer(config["speech_vocab_size"], "speech_vocab_size", minimum=1),
            _integer(language["vocab_size"], "vocab_size", minimum=1),
            _integer(config["text_pad_idx"], "text_pad_idx"),
            _integer(config["pad_token_id"], "pad_token_id"),
            _integer(config["audio_start_token_id"], "audio_start_token_id"),
            _integer(config["sampling_rate"], "sampling_rate", minimum=1),
            _integer(language["max_position_embeddings"], "max_position_embeddings", minimum=1),
            tuple(channels),
            identity,
            {
                filename: hashlib.sha256((root / filename).read_bytes()).hexdigest()
                for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
            },
        )


@lru_cache(maxsize=8)
def _config(checkpoint):
    return PrismConfig.from_path(checkpoint)
