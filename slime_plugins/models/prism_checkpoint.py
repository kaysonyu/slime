# Copyright (c) 2025, OpenMOSS. All rights reserved.
"""Prism checkpoint contracts."""

from __future__ import annotations
import json
import re
from dataclasses import replace
from pathlib import Path
import torch


def select_native_checkpoint(path: str | Path, step: int | None = None):
    """Select a native model component, accepting a RUN, checkpoint root or iteration."""
    root = Path(path)
    if step is not None and (type(step) is not int or step < 0):
        raise ValueError("checkpoint step must be a nonnegative integer")
    if (root / "checkpoints").is_dir():
        root = root / "checkpoints"
    if (root / "checkpoint.json").is_file():
        selected = root
    else:
        if step is None:
            marker = (root / "latest_checkpointed_iteration.txt").read_text().strip()
            if marker == "release":
                selected = root / "release"
            else:
                if not marker.isdecimal() or int(marker) <= 0:
                    raise ValueError("invalid native checkpoint tracker")
                selected = root / f"iter_{int(marker):07d}"
        else:
            selected = root / "release" if step == 0 else root / f"iter_{step:07d}"
    manifest = json.loads((selected / "checkpoint.json").read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("Prism requires the native logical checkpoint manifest schema 2")
    expected_iteration = 0 if selected.name == "release" else int(selected.name.removeprefix("iter_"))
    if manifest.get("iteration") != expected_iteration:
        raise ValueError("native selected directory and manifest iteration disagree")
    if step is not None and step != expected_iteration:
        raise ValueError("--ckpt-step disagrees with the explicit native checkpoint directory")
    component = selected / "model"
    if not (component / ".metadata").is_file():
        raise ValueError("native checkpoint model DCP is missing")
    return component, manifest


def _align_native_shards(value, tensor_metadata, num_layers):
    """Reconcile MCore's per-layer template with older layer-stacked DCP keys.

    Keep factory merge functions intact so SwiGLU gate/up TP placement remains
    correct. Only logical metadata gains an axis; local parameter shapes do not.
    """
    from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory

    if isinstance(value, dict):
        return {key: _align_native_shards(item, tensor_metadata, num_layers) for key, item in value.items()}
    if isinstance(value, list):
        return [_align_native_shards(item, tensor_metadata, num_layers) for item in value]
    if isinstance(value, ShardedTensorFactory):
        original_build = value.build_fn

        def build(key, tensor, replica_id, flattened_range):
            return _align_native_shards(
                original_build(key, tensor, replica_id, flattened_range), tensor_metadata, num_layers
            )

        return replace(value, build_fn=build)
    if not isinstance(value, ShardedTensor):
        return value
    if value.key in tensor_metadata:
        expected = tuple(tensor_metadata[value.key].global_shape)
        if tuple(value.global_shape) != expected:
            raise ValueError(f"native shape mismatch for {value.key}: {value.global_shape} versus {expected}")
        return value
    match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", value.key)
    if match:
        logical = f"decoder.layers.{match[2]}"
        if logical in tensor_metadata:
            expected = (num_layers, *value.global_shape)
            if tuple(tensor_metadata[logical].global_shape) != expected:
                raise ValueError(f"native stacked shape mismatch for {logical}")
            return replace(
                value,
                key=logical,
                global_shape=expected,
                global_offset=(int(match[1]), *value.global_offset),
                axis_fragmentations=(num_layers, *value.axis_fragmentations),
                prepend_axis_num=value.prepend_axis_num + 1,
            )
    raise ValueError(f"native Prism weight is missing: {value.key}")


def load_prism_pretrained(models, args) -> None:
    """Import only the SFT model component; slime owns RL optimizer/resume state.

    Use --custom-megatron-model-initializer-path with --pretrained-checkpoint
    and optional --ckpt-step. Slime dispatches this only without --load, owns
    master-parameter refresh, and keeps all RL progress/optimizer state fresh.
    """
    from slime_plugins.models.prism import _require_megatron

    _require_megatron()
    from megatron.core import dist_checkpointing

    if len(models) != 1:
        raise ValueError("Prism imports exactly one PP=1 model")
    model = models[0]
    while hasattr(model, "module"):
        model = model.module
    spec = model.prism_spec
    source = getattr(args, "pretrained_checkpoint", None)
    if source is None:
        raise ValueError("Prism's model initializer requires --pretrained-checkpoint")
    component, manifest = select_native_checkpoint(source, getattr(args, "ckpt_step", None))
    common = dist_checkpointing.load_common_state_dict(str(component))
    if common.get("iteration") != manifest["iteration"] or common.get("checkpoint_version") != 3.0:
        raise ValueError("native model common state disagrees with selected manifest")
    saved_metadata = dist_checkpointing.load_tensors_metadata(str(component))
    expected_dtype = getattr(torch, spec.dtype)
    if any(metadata.dtype != expected_dtype for key, metadata in saved_metadata.items() if key.endswith(".weight")):
        raise ValueError("native trainable dtype disagrees with the inference artifact")
    template = model.sharded_state_dict(metadata={"non_homogeneous_layers": True})
    # TE/FA auxiliary states are parameter-free and excluded from the model
    # importer. The target engine initializes its own auxiliary kernel state.
    template = {key: value for key, value in template.items() if "_extra_state" not in key}
    template = _align_native_shards(template, saved_metadata, spec.topology.num_layers)
    loaded = dist_checkpointing.load({"model": template}, str(component))
    incompatible = model.load_state_dict(loaded["model"], strict=False)
    missing = [key for key in incompatible.missing_keys if "_extra_state" not in key]
    if missing or incompatible.unexpected_keys:
        raise ValueError(
            f"native Prism tensor load mismatch: missing={missing}, unexpected={incompatible.unexpected_keys}"
        )
    return None
