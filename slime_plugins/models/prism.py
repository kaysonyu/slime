# Copyright (c) 2025, OpenMOSS. All rights reserved.
"""Prism's native Megatron model and selected-action scoring adapter.

A chronological row is a text token or a complete RVQ frame. The Tensor
returned by ``forward`` is [1, T, Q + 1]: Q selected code log probabilities,
then the selected binary stop log probability. There is no extra token shift.
An optional second Q + 1 columns contain per-head entropy. All distributions
use temperature one; policy masks describe real sampled actions separately.

The pure topology, configuration and carrier helpers can be used without
Megatron installed. Constructing the training model requires Megatron Core.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from copy import copy
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import ClassVar, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from slime_plugins.models.prism_configuration import configure_prism_args
from slime_plugins.models.prism_policy import prepare_policy_inputs, selected_logprob_and_entropy
from slime_plugins.models.prism_topology import (
    PrismAudioRole,
    PrismExecutionUnit,
    PrismItemKind,
    ResolvedPrismTopology,
    normalize_prism_input_rvq_channels,
    normalize_prism_rvq_label_smoothing,
    resolve_prism_topology,
    validate_prism_recompute_policy,
    validate_prism_stop_gradient_layer,
)

__all__ = [
    "MossTTSPrismConfig",
    "language_model_embedding_with_weight",
    "LanguageModelEmbeddingWithPad",
    "FP32StopHead",
    "EmbeddingHead",
    "PrismMLPOnlyLayer",
    "prism_block_spec",
    "MossTTSPrismModel",
    "vocab_parallel_entropy",
    "model_provider",
]

_MEGATRON_IMPORT_ERROR = None
try:
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    from megatron.core.models.common.language_module.language_module import LanguageModule
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.enums import ModelType
    from megatron.core.transformer.module import MegatronModule
    from megatron.core.transformer.spec_utils import ModuleSpec, build_module
    from megatron.core.transformer.transformer_block import LayerNormImpl, TransformerBlock, TransformerBlockSubmodules
    from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
    from megatron.core.utils import init_method_normal
except (ImportError, OSError) as error:
    _MEGATRON_IMPORT_ERROR = error
    LanguageModule = LanguageModelEmbedding = TransformerLayer = nn.Module
    TransformerConfig = object
    tensor_parallel = SimpleNamespace(ColumnParallelLinear=nn.Module)


def _require_megatron():
    if _MEGATRON_IMPORT_ERROR is not None:
        raise RuntimeError("Prism training requires a working Megatron Core installation") from _MEGATRON_IMPORT_ERROR


@dataclass
class MossTTSPrismConfig(TransformerConfig):
    """Transformer and immutable execution-topology config for Prism."""

    _cli_hidden_fields: ClassVar[frozenset[str]] = frozenset(
        {
            "prism_loop_block_ranges",
            "prism_loop_block_repetitions",
            "prism_rvq_heads_per_execution_unit",
            "prism_mlp_only_layers",
            # These options have custom vector/scalar parsers in train.py.
            "rvq_label_smoothing",
            "prism_stop_gradient_layer",
            "prism_input_rvq_channels",
            "prism_randomize_loop_prefix",
        }
    )

    transformer_impl: str = "transformer_engine"
    max_position_embeddings: int = 128000
    pad_token_id: int = 151667
    text_pad_idx: int = 151643
    audio_start_token_id: int = 151652

    n_vq: int = 8
    speech_vocab_size: int = 1024
    speech_embedding_init_method_std: float = 0.0
    speech_head_init_method_std: float = 0.02
    speech_pad_index: int = field(init=False)

    prism_loop_block_ranges: tuple[tuple[int, int], ...] = ()
    prism_mlp_only_layers: tuple[int, ...] = ()
    prism_loop_block_repetitions: tuple[int, ...] = ()
    prism_rvq_heads_per_execution_unit: tuple[int, ...] = ()
    reapply_rvq_conditioning_in_loop: bool = False
    rvq_label_smoothing: tuple[float, ...] | None = None
    prism_stop_gradient_layer: int | None = None
    prism_input_rvq_channels: tuple[int, ...] | None = None
    prism_topology: ResolvedPrismTopology = field(init=False, repr=False)

    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    kv_channels: int = 128

    def __post_init__(self) -> None:
        _require_megatron()
        if self.normalization != "RMSNorm":
            raise ValueError("MOSS-TTS Prism requires RMSNorm")
        topology = resolve_prism_topology(self)
        if self.num_layers not in (0, topology.num_layers):
            raise ValueError(
                "MOSS-TTS Prism physical layer count is derived from its concrete "
                f"topology ({topology.num_layers}), not num_layers={self.num_layers}"
            )
        self.num_layers = topology.num_layers
        # Shared layers run backward in reverse call order. TE's first-call
        # flag would overwrite fused main_grad after later calls accumulated it.
        # Passing None through the existing cache setting keeps accumulation on.
        self.disable_parameter_transpose_cache = True
        super().__post_init__()
        if isinstance(self.text_pad_idx, bool) or not isinstance(self.text_pad_idx, int) or self.text_pad_idx < 0:
            raise ValueError("text_pad_idx must be a nonnegative non-Boolean integer")
        if (
            isinstance(self.speech_vocab_size, bool)
            or not isinstance(self.speech_vocab_size, int)
            or self.speech_vocab_size <= 0
        ):
            raise ValueError("speech_vocab_size must be a positive integer")
        for name in ("speech_embedding_init_method_std", "speech_head_init_method_std"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        self.speech_pad_index = self.speech_vocab_size

        self.prism_loop_block_ranges = topology.prism_loop_block_ranges
        self.prism_mlp_only_layers = topology.prism_mlp_only_layers
        self.prism_loop_block_repetitions = topology.prism_loop_block_repetitions
        self.prism_rvq_heads_per_execution_unit = topology.prism_rvq_heads_per_execution_unit
        self.reapply_rvq_conditioning_in_loop = topology.reapply_rvq_conditioning_in_loop
        self.rvq_label_smoothing = normalize_prism_rvq_label_smoothing(self.rvq_label_smoothing, n_vq=self.n_vq)
        self.prism_stop_gradient_layer = validate_prism_stop_gradient_layer(self.prism_stop_gradient_layer, topology)
        self.prism_input_rvq_channels = normalize_prism_input_rvq_channels(
            self.prism_input_rvq_channels, n_vq=self.n_vq
        )
        self.prism_topology = topology


def language_model_embedding_with_weight(
    embedding: LanguageModelEmbedding,
    input_ids: Tensor,
    position_ids: Tensor | None,
    weight: Tensor,
    *,
    pad_token_id: int | None = None,
) -> Tensor:
    """Run a MOSS-TTS embedding with an externally supplied TP-sharded weight."""

    if embedding.config.sequence_parallel:
        raise ValueError("external-weight MOSS-TTS embeddings reject sequence parallelism")
    if embedding.tokentype_embeddings is not None:
        raise ValueError("external-weight MOSS-TTS embeddings reject token types")

    parallel_embedding = embedding.word_embeddings
    expected_shape = (
        parallel_embedding.num_embeddings_per_partition,
        parallel_embedding.embedding_dim,
    )
    if tuple(weight.shape) != expected_shape:
        raise ValueError(f"external embedding weight has shape {tuple(weight.shape)}; " f"expected {expected_shape}")

    pad_mask = None
    lookup_ids = input_ids
    if pad_token_id is not None:
        pad_mask = input_ids.eq(pad_token_id)
        lookup_ids = input_ids.masked_fill(pad_mask, parallel_embedding.vocab_start_index)

    if parallel_embedding.tp_group.size() > 1:
        input_mask = (lookup_ids < parallel_embedding.vocab_start_index) | (
            lookup_ids >= parallel_embedding.vocab_end_index
        )
        local_ids = lookup_ids.clone() - parallel_embedding.vocab_start_index
        local_ids.masked_fill_(input_mask, 0)
    else:
        input_mask = None
        local_ids = lookup_ids

    if parallel_embedding.deterministic_mode:
        output_parallel = weight[local_ids]
    else:
        output_parallel = F.embedding(local_ids, weight)
    if input_mask is not None:
        output_parallel = output_parallel.masked_fill(input_mask.unsqueeze(-1), 0)
    word_embeddings = tensor_parallel.reduce_from_tensor_model_parallel_region(
        output_parallel, group=parallel_embedding.tp_group
    )

    if embedding.add_position_embedding:
        if position_ids is None:
            raise ValueError("position_ids are required for learned position embeddings")
        word_embeddings = word_embeddings + embedding.position_embeddings(position_ids)
    if pad_mask is not None:
        word_embeddings = word_embeddings.masked_fill(pad_mask.unsqueeze(-1), 0)

    output = word_embeddings.transpose(0, 1).contiguous()
    if embedding.config.fp32_residual_connection:
        output = output.float()
    return embedding.embedding_dropout(output)


class LanguageModelEmbeddingWithPad(LanguageModelEmbedding):
    """Language model embedding that masks out pad tokens."""

    def __init__(
        self,
        pad_token_id: int,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        position_embedding_type: Literal["learned_absolute", "rope", "none"] = ("learned_absolute"),
        num_tokentypes: int = 0,
        scatter_to_sequence_parallel: bool = True,
        tp_group: torch.distributed.ProcessGroup | None = None,
        *,
        word_embedding_init_method: Callable[[Tensor], object] | None = None,
    ) -> None:
        if word_embedding_init_method is None:
            super().__init__(
                config=config,
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
                position_embedding_type=position_embedding_type,
                num_tokentypes=num_tokentypes,
                scatter_to_sequence_parallel=scatter_to_sequence_parallel,
                tp_group=tp_group,
            )
        else:
            allocation_config = copy(config)
            allocation_config.perform_initialization = False
            super().__init__(
                config=allocation_config,
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
                position_embedding_type=position_embedding_type,
                num_tokentypes=num_tokentypes,
                scatter_to_sequence_parallel=scatter_to_sequence_parallel,
                tp_group=tp_group,
            )
            self.config = config
            self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
                num_embeddings=self.vocab_size,
                embedding_dim=config.hidden_size,
                init_method=word_embedding_init_method,
                reduce_scatter_embeddings=self.reduce_scatter_embeddings,
                config=config,
                tp_group=self.tp_group,
            )
            if config.perform_initialization:
                if self.add_position_embedding:
                    config.embedding_init_method(self.position_embeddings.weight)
                if self.num_tokentypes > 0:
                    config.embedding_init_method(self.tokentype_embeddings.weight)
        self.pad_token_id = pad_token_id

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None,
        tokentype_ids: int | None = None,
    ) -> Tensor:
        """
        Inputs:
            input_ids: [b, s]
            position_ids: [b, s] or None
            tokentype_ids: int or None
        Outputs:
            output: [s, b, h]
        """
        b, s = input_ids.shape
        h = self.config.hidden_size
        input_mask = input_ids != self.pad_token_id  # Binary mask of shape [b, s]
        input_ids_masked = input_ids[input_mask].unsqueeze(0)  # [1, C]
        position_ids_masked = position_ids[input_mask].unsqueeze(0) if position_ids is not None else None  # [1, C]
        masked_output = super().forward(input_ids_masked, position_ids_masked, tokentype_ids)  # [C, 1, H]
        output = torch.zeros(s, b, h, device=input_ids.device, dtype=self.word_embeddings.weight.dtype)
        output[input_mask.transpose(0, 1)] = masked_output.squeeze(1)
        return output  # [s, b, h]


class FP32StopHead(nn.Linear):
    """Binary projection in FP32; initialization belongs to the model."""

    def reset_parameters(self) -> None:
        pass

    def forward(self, hidden: Tensor) -> Tensor:
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return F.linear(hidden.float(), self.weight.float())


class EmbeddingHead(tensor_parallel.ColumnParallelLinear):
    """Enter the weight owner's forward hooks for both projection and lookup."""

    def forward(self, input_, *, embedding=None, position_ids=None, pad_token_id=None, **kwargs):
        if embedding is not None:
            return language_model_embedding_with_weight(
                embedding, input_, position_ids, self.weight, pad_token_id=pad_token_id
            )
        return super().forward(input_, **kwargs)


class PrismMLPOnlyLayer(TransformerLayer):
    """Remove attention while retaining the native pre-norm MLP and residual."""

    def _forward_attention(self, hidden_states, *, context=None, **_kwargs):
        return hidden_states, context


def prism_block_spec(
    layer_spec: ModuleSpec, *, num_layers: int, mlp_only_layers: tuple[int, ...]
) -> TransformerBlockSubmodules:
    submodules = layer_spec.submodules
    mlp_spec = replace(
        layer_spec,
        module=PrismMLPOnlyLayer,
        submodules=TransformerLayerSubmodules(
            pre_mlp_layernorm=submodules.pre_mlp_layernorm,
            mlp=submodules.mlp,
            mlp_bda=submodules.mlp_bda,
            sharded_state_dict_keys_map=submodules.sharded_state_dict_keys_map,
        ),
    )
    return TransformerBlockSubmodules(
        layer_specs=[mlp_spec if index in mlp_only_layers else layer_spec for index in range(num_layers)],
        layer_norm=LayerNormImpl,
    )


_ApplyRange = Callable[[int, int, Tensor], Tensor]
_ReadSite = Callable[[int, PrismExecutionUnit, tuple[int, ...], Tensor, bool], Tensor | None]


def _normalize_loop_block_prefix_counts(
    value: Tensor | Sequence[int] | None, topology: ResolvedPrismTopology
) -> tuple[int, ...] | None:
    """Normalize a per-forward prefix vector before schedule construction."""

    if value is None:
        return None
    if torch.is_tensor(value):
        if value.dtype != torch.int64 or value.ndim != 1:
            raise ValueError("loop_block_prefix_counts must be a one-dimensional int64 tensor")
        value = value.detach().to(device="cpu").tolist()
    topology.execution_unit_indices_for_prefix_counts(value)
    return tuple(value)


def _add_delta(base: Tensor, delta: Tensor | None) -> Tensor:
    return base if delta is None else base + delta


def _accumulate_delta(accumulator: Tensor | None, delta: Tensor | None) -> Tensor | None:
    if accumulator is None:
        return delta
    return accumulator if delta is None else accumulator + delta


def _detach_optional(value: Tensor | None) -> Tensor | None:
    """Detach an optional conditioning path without changing its forward value."""

    return None if value is None else value.detach()


def _validate_delta(raw: Tensor, delta: Tensor | None) -> Tensor | None:
    if delta is not None and (not torch.is_tensor(delta) or tuple(delta.shape) != tuple(raw.shape)):
        raise RuntimeError("Prism conditioning delta must match the raw site state")
    return delta


def _execute_prism_schedule(
    topology: ResolvedPrismTopology,
    pre_layer_residual: Tensor,
    *,
    apply_range: _ApplyRange,
    read_site: _ReadSite,
    stop_gradient_layer: int | None = None,
    loop_block_prefix_counts: Sequence[int] | None = None,
) -> Tensor:
    """Execute the recurrence, optionally cutting one physical-layer boundary.

    ``stop_gradient_layer`` is a physical boundary rather than an expanded
    execution-unit index.  The backbone residual and incoming conditioning
    delta are detached together when they enter the first unit after that
    boundary.  This keeps the boundary site's prediction graph intact and
    naturally gives a repeated loop block one detach event, even when the
    block has several applications. Conditioning deltas produced after the
    boundary remain differentiable.
    """

    selected_indices = topology.execution_unit_indices_for_prefix_counts(loop_block_prefix_counts)
    units = tuple(topology.execution_units[index] for index in selected_indices)
    groups = tuple(topology.co_site_groups[index] for index in selected_indices)
    if not units or len(groups) != len(units):
        raise RuntimeError("resolved Prism topology has inconsistent execution units")

    previous_raw: Tensor | None = None
    incoming_delta: Tensor | None = None
    unit_index = 0
    while unit_index < len(units):
        unit = units[unit_index]
        if unit.site.kind == "ordinary":
            unit_base = pre_layer_residual if previous_raw is None else previous_raw
            if stop_gradient_layer == unit.layer_start:
                # Cut every path crossing the boundary. The values are still
                # added below, so detach changes only the backward graph.
                unit_base = unit_base.detach()
                incoming_delta = _detach_optional(incoming_delta)
            unit_input = unit_base if incoming_delta is None else unit_base + incoming_delta
            raw = apply_range(unit.layer_start, unit.layer_stop, unit_input)
            if not torch.is_tensor(raw) or tuple(raw.shape) != tuple(unit_input.shape):
                raise RuntimeError("a registered Prism layer changed the residual shape")
            incoming_delta = _validate_delta(
                raw,
                read_site(
                    selected_indices[unit_index],
                    unit,
                    groups[unit_index],
                    raw,
                    unit_index == len(units) - 1,
                ),
            )
            previous_raw = raw
            unit_index += 1
            continue

        block = unit.site.index
        block_start = unit_index
        block_stop = block_start
        while (
            block_stop < len(units) and units[block_stop].site.kind == "loop" and units[block_stop].site.index == block
        ):
            block_stop += 1

        pre_block_residual = pre_layer_residual if previous_raw is None else previous_raw
        if stop_gradient_layer == unit.layer_start:
            # The whole loop range is one schedule unit at this boundary.
            # Detach the fixed anchor and incoming accumulator once before
            # application 1; post-boundary recurrence and deltas stay live.
            pre_block_residual = pre_block_residual.detach()
            incoming_delta = _detach_optional(incoming_delta)
        accumulator = incoming_delta
        for application_index in range(block_start, block_stop):
            application = units[application_index]
            coupled_base = (
                pre_block_residual if application_index == block_start else previous_raw + pre_block_residual
            )
            unit_input = _add_delta(coupled_base, accumulator)
            raw = apply_range(application.layer_start, application.layer_stop, unit_input)
            if not torch.is_tensor(raw) or tuple(raw.shape) != tuple(unit_input.shape):
                raise RuntimeError("a registered Prism loop range changed the residual shape")
            delta = _validate_delta(
                raw,
                read_site(
                    selected_indices[application_index],
                    application,
                    groups[application_index],
                    raw,
                    application_index == len(units) - 1,
                ),
            )
            previous_raw = raw
            if application_index + 1 < block_stop:
                accumulator = (
                    _accumulate_delta(accumulator, delta) if topology.reapply_rvq_conditioning_in_loop else delta
                )
            else:
                # Block-local history never escapes. Only the final site's own
                # delta can reach the immediate successor.
                incoming_delta = delta
        unit_index = block_stop

    if previous_raw is None:  # pragma: no cover - topology rejects this state
        raise RuntimeError("Prism execution produced no raw state")
    return previous_raw


class MossTTSPrismModel(LanguageModule):
    """Dense stop/RVQ training model with a unified recurrent layer schedule."""

    config: MossTTSPrismConfig

    def __init__(
        self,
        config: MossTTSPrismConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal["learned_absolute", "rope", "none"] = "rope",
        rotary_percent: float = 1.0,
        rotary_base: int = 1000000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        seq_len_interpolation_factor: float | None = None,
    ) -> None:
        _require_megatron()
        super().__init__(config=config)
        self._validate_constructor_envelope(
            config,
            pre_process=pre_process,
            post_process=post_process,
            parallel_output=parallel_output,
            position_embedding_type=position_embedding_type,
        )

        self.model_type = ModelType.encoder_or_decoder
        self.vocab_size = int(vocab_size)
        self.max_sequence_length = int(max_sequence_length)
        self.pre_process = pre_process
        self.post_process = post_process
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type
        self.max_position_embeddings = max_sequence_length
        self.rotary_percent = rotary_percent
        self.rotary_base = rotary_base
        self.rotary_scaling = rope_scaling
        self.prism_topology = config.prism_topology
        self._full_activation_recompute = validate_prism_recompute_policy(config, effective_config=True)

        speech_embedding_init_method = init_method_normal(config.speech_embedding_init_method_std)
        speech_head_init_method = init_method_normal(config.speech_head_init_method_std)

        self.text_embedding = LanguageModelEmbeddingWithPad(
            pad_token_id=config.text_pad_idx,
            config=config,
            vocab_size=self.vocab_size,
            max_sequence_length=max_sequence_length,
            position_embedding_type=position_embedding_type,
        )
        self.audio_embeddings = nn.ModuleList(
            [
                LanguageModelEmbeddingWithPad(
                    pad_token_id=config.speech_pad_index,
                    config=config,
                    vocab_size=config.speech_vocab_size,
                    max_sequence_length=max_sequence_length,
                    position_embedding_type=position_embedding_type,
                    word_embedding_init_method=speech_embedding_init_method,
                )
                for _ in range(config.n_vq)
            ]
        )

        self.rotary_pos_emb = RotaryEmbedding(
            kv_channels=config.kv_channels,
            rotary_percent=rotary_percent,
            rotary_interleaved=config.rotary_interleaved,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
            rope_scaling_factor=rope_scaling_factor,
            use_cpu_initialization=config.use_cpu_initialization,
        )
        self.decoder = TransformerBlock(
            config=config,
            spec=prism_block_spec(
                transformer_layer_spec,
                num_layers=config.num_layers,
                mlp_only_layers=self.prism_topology.prism_mlp_only_layers,
            ),
            post_layer_norm=False,
            pre_process=True,
            post_process=True,
        )
        if len(self.decoder.layers) != config.num_layers:
            raise RuntimeError("MOSS-TTS Prism must register exactly num_layers physical layers")
        if self.decoder.final_layernorm is not None:
            raise RuntimeError("MOSS-TTS Prism must not register a global final norm")
        self.prediction_norms = nn.ModuleList(
            [
                build_module(
                    self.decoder.submodules.layer_norm,
                    config=config,
                    hidden_size=config.hidden_size,
                    eps=config.layernorm_epsilon,
                )
                for _ in range(self.prism_topology.num_prediction_norms)
            ]
        )

        self.stop_head = FP32StopHead(config.hidden_size, 2, bias=False)
        if config.perform_initialization:
            nn.init.normal_(self.stop_head.weight, mean=0.0, std=config.init_method_std)
        self.stop_head.weight.average_gradients_across_tp_domain = True
        self.audio_lm_heads = nn.ModuleList(
            [
                EmbeddingHead(
                    config.hidden_size,
                    config.speech_vocab_size,
                    config=config,
                    init_method=speech_head_init_method,
                    bias=False,
                    gather_output=False,
                    skip_weight_param_allocation=False,
                )
                for _ in range(config.n_vq)
            ]
        )
        self.setup_embeddings_and_output_layer()

    @staticmethod
    def _validate_constructor_envelope(
        config: MossTTSPrismConfig,
        *,
        pre_process: bool,
        post_process: bool,
        parallel_output: bool,
        position_embedding_type: str,
    ) -> None:
        for attribute, label in (
            ("pipeline_model_parallel_size", "pipeline parallel"),
            ("context_parallel_size", "context parallel"),
        ):
            if getattr(config, attribute, 1) != 1:
                raise ValueError(f"MOSS-TTS Prism requires {label} size one")
        if not pre_process or not post_process:
            raise ValueError("MOSS-TTS Prism requires pre_process and post_process")
        if not parallel_output:
            raise ValueError("MOSS-TTS Prism requires tensor-parallel output shards")
        if position_embedding_type != "rope":
            raise ValueError("MOSS-TTS Prism requires RoPE")
        if getattr(config, "multi_latent_attention", False):
            raise ValueError("MOSS-TTS Prism does not support multi-latent attention")
        if getattr(config, "sequence_parallel", False):
            raise ValueError("MOSS-TTS Prism does not support sequence parallelism")
        validate_prism_recompute_policy(config, effective_config=True)
        unsupported = []
        if getattr(config, "virtual_pipeline_model_parallel_size", None) is not None:
            unsupported.append("virtual pipeline parallelism")
        if getattr(config, "fp8", None) is not None:
            unsupported.append("FP8")
        for attribute, label in (
            ("cpu_offloading", "CPU offload"),
            ("enable_cuda_graph", "CUDA graphs"),
            ("flash_decode", "flash decoding"),
        ):
            if getattr(config, attribute, False):
                unsupported.append(label)
        if unsupported:
            raise ValueError("MOSS-TTS Prism does not support " + ", ".join(unsupported))
        if config.attention_dropout != 0.0 or config.hidden_dropout != 0.0:
            raise ValueError("MOSS-TTS Prism requires zero attention and hidden dropout")

    def setup_embeddings_and_output_layer(self) -> None:
        """Mark canonical parameters and freeze redundant tied embeddings."""

        text_weight = self.text_embedding.word_embeddings.weight
        text_weight.is_embedding_or_output_parameter = True
        self.stop_head.weight.is_embedding_or_output_parameter = True
        for embedding in self.audio_embeddings:
            embedding.word_embeddings.weight.is_embedding_or_output_parameter = True
        for head in self.audio_lm_heads:
            head.weight.is_embedding_or_output_parameter = True
        if self.share_embeddings_and_output_weights:
            for embedding in self.audio_embeddings:
                embedding.word_embeddings.weight.requires_grad_(False)
            for head in self.audio_lm_heads:
                head.weight.zero_out_wgrad = True

    def _audio_embedding(self, channel_index: int, input_ids: Tensor) -> Tensor:
        embedding = self.audio_embeddings[channel_index]
        if not self.share_embeddings_and_output_weights:
            return embedding(input_ids, None)
        return self.audio_lm_heads[channel_index](
            input_ids, embedding=embedding, pad_token_id=self.config.speech_pad_index
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Preserve TP metadata for every RVQ embedding and head."""

        if self.prism_topology.prism_mlp_only_layers:
            # Mixed layers cannot stack attention tensors across physical depth.
            metadata = {**(metadata or {}), "non_homogeneous_layers": True}
        sharded_state = MegatronModule.sharded_state_dict(
            self, prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata
        )
        module_list_prefixes = (
            f"{prefix}audio_embeddings.",
            f"{prefix}audio_lm_heads.",
        )
        for key in list(sharded_state):
            if key.startswith(module_list_prefixes):
                del sharded_state[key]
        for index, embedding in enumerate(self.audio_embeddings):
            sharded_state.update(
                embedding.sharded_state_dict(
                    prefix=f"{prefix}audio_embeddings.{index}.",
                    sharded_offsets=sharded_offsets,
                    metadata=metadata,
                )
            )
        for index, head in enumerate(self.audio_lm_heads):
            sharded_state.update(
                head.sharded_state_dict(
                    prefix=f"{prefix}audio_lm_heads.{index}.",
                    sharded_offsets=sharded_offsets,
                    metadata=metadata,
                )
            )
        return sharded_state

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        if isinstance(input_tensor, list):
            if len(input_tensor) != 1:
                raise ValueError("expected one pipeline input tensor")
            input_tensor = input_tensor[0]
        self.decoder.set_input_tensor(input_tensor)

    @staticmethod
    def _tensor(
        value: object,
        *,
        name: str,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        device: torch.device,
    ) -> Tensor:
        if not torch.is_tensor(value) or value.dtype != dtype or tuple(value.shape) != shape or value.device != device:
            raise ValueError(f"{name} must be a {dtype} tensor on {device} with shape {shape}")
        return value

    def _prepare_inputs(
        self,
        input_ids: Tensor,
        item_kind: Tensor,
        audio_role: Tensor,
        target_history_retention_mask: Tensor,
    ) -> Tensor:
        text = item_kind == int(PrismItemKind.TEXT)
        target = audio_role == int(PrismAudioRole.TARGET)
        selected_channels = getattr(
            self.config,
            "prism_input_rvq_channels",
            tuple(range(1, self.config.n_vq + 1)),
        )
        text_ids = input_ids[..., 0].masked_fill(~text, self.config.text_pad_idx)
        hidden_states = self.text_embedding(text_ids, None)
        for channel_index in range(self.config.n_vq):
            if channel_index + 1 in selected_channels:
                retained = ~target | target_history_retention_mask[..., channel_index]
                active = ~text & retained
            else:
                # The channel selection belongs to the target/output side.
                # Reference audio is user input and always exposes its full
                # RVQ frame to the backbone.
                active = ~text & ~target
            audio_ids = input_ids[..., channel_index + 1].masked_fill(~active, self.config.speech_pad_index)
            hidden_states = hidden_states + self._audio_embedding(channel_index, audio_ids)
        return hidden_states

    def _successor_conditioning_delta(
        self,
        successor_audio_codes: Tensor,
        successor_audio_mask: Tensor,
        group: tuple[int, ...],
    ) -> Tensor | None:
        delta = None
        for q in group:
            channel_index = q - 1
            successor_ids = successor_audio_codes[..., channel_index].masked_fill(
                ~successor_audio_mask, self.config.speech_pad_index
            )
            contribution = self._audio_embedding(channel_index, successor_ids)
            delta = contribution if delta is None else delta + contribution
        return delta

    def _prediction_read(self, unit_index: int, raw: Tensor) -> Tensor:
        """Apply the normalization owned by one occupied execution unit."""

        norm_index = self.prism_topology.prediction_norm_index_by_execution_unit[unit_index]
        if norm_index is None:
            raise RuntimeError("an unoccupied Prism execution unit has no prediction norm")
        return self.prediction_norms[norm_index](raw)

    def _apply_registered_range(
        self,
        start: int,
        stop: int,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None,
        context: Tensor | None,
        context_mask: Tensor | None,
        rotary_pos_emb: Tensor | None,
        packed_seq_params: PackedSeqParams,
    ) -> tuple[Tensor, Tensor | None]:
        def bind_layer(layer):
            def layer_forward(
                value: Tensor,
                layer_attention_mask: Tensor | None,
                layer_context: Tensor | None,
                layer_context_mask: Tensor | None,
                layer_rotary_pos_emb: Tensor | None,
            ) -> tuple[Tensor, Tensor | None]:
                return layer(
                    hidden_states=value,
                    attention_mask=layer_attention_mask,
                    context=layer_context,
                    context_mask=layer_context_mask,
                    rotary_pos_emb=layer_rotary_pos_emb,
                    rotary_pos_cos=None,
                    rotary_pos_sin=None,
                    attention_bias=None,
                    inference_context=None,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=None,
                )

            return layer_forward

        checkpoint_layers = self.training and getattr(self, "_full_activation_recompute", False)
        for layer_index in range(start, stop):
            layer = self.decoder.layers[layer_index]
            layer_forward = bind_layer(layer)
            if checkpoint_layers:
                if not hidden_states.requires_grad:
                    # Reentrant checkpoint needs a differentiable input to build
                    # parameter gradients after a stop-gradient boundary.
                    hidden_states = hidden_states.detach().requires_grad_(True)
                hidden_states, context = tensor_parallel.checkpoint(
                    layer_forward,
                    False,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )
            else:
                hidden_states, context = layer_forward(
                    hidden_states, attention_mask, context, context_mask, rotary_pos_emb
                )
        return hidden_states, context

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None,
        attention_mask: Tensor | None,
        *,
        packed_seq_params=None,
        labels=None,
        loss_mask=None,
        audio_codes: Tensor,
        item_kind: Tensor,
        audio_role: Tensor,
        successor_audio_codes: Tensor,
        successor_audio_mask: Tensor,
        target_history_retention_mask: Tensor | None = None,
        loop_block_prefix_counts: Tensor | Sequence[int] | None = None,
        policy_code_mask: Tensor,
        policy_stop_mask: Tensor,
        policy_stop_actions: Tensor,
        policy_return_entropy: bool = False,
        **policy_fields,
    ) -> Tensor:
        """Score the supplied trajectory, preserving the native recurrence.

        Auxiliary ``policy_*`` tensors are consumed by slime's objective. They
        never change the backbone input or the successor-conditioning edge.
        """
        unexpected = [key for key in policy_fields if not key.startswith("policy_")]
        if unexpected:
            raise TypeError(f"unexpected Prism inputs: {unexpected}")
        if position_ids is not None or attention_mask is not None:
            raise ValueError("Prism uses THD packed RoPE with position_ids=attention_mask=None")
        if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
            raise ValueError("Prism requires THD PackedSeqParams")
        batch = prepare_policy_inputs(
            input_ids,
            audio_codes=audio_codes,
            item_kind=item_kind,
            audio_role=audio_role,
            successor_audio_codes=successor_audio_codes,
            successor_audio_mask=successor_audio_mask,
            target_history_retention_mask=target_history_retention_mask,
            policy_code_mask=policy_code_mask,
            policy_stop_mask=policy_stop_mask,
            policy_stop_actions=policy_stop_actions,
            n_vq=self.config.n_vq,
            speech_vocab_size=self.config.speech_vocab_size,
            text_vocab_size=self.vocab_size,
            text_pad_idx=self.config.text_pad_idx,
            audio_start_token_id=self.config.audio_start_token_id,
            cu_seqlens=packed_seq_params.cu_seqlens_q,
        )
        prefix_counts = _normalize_loop_block_prefix_counts(loop_block_prefix_counts, self.prism_topology)
        hidden = self._prepare_inputs(batch["tokens"], batch["item_kind"], batch["audio_role"], batch["retention"])
        rotary_length = self.rotary_pos_emb.get_rotary_seq_len(
            None, self.decoder, hidden, self.config, packed_seq_params
        )
        rotary = self.rotary_pos_emb(rotary_length, packed_seq=True)
        code_scores = [None] * self.config.n_vq
        code_entropies = [None] * self.config.n_vq
        stop_score = stop_entropy = None
        context = None

        def apply_range(start, stop, value):
            nonlocal context
            value, context = self._apply_registered_range(
                start,
                stop,
                value,
                attention_mask=None,
                context=context,
                context_mask=None,
                rotary_pos_emb=rotary,
                packed_seq_params=packed_seq_params,
            )
            return value

        def read_site(unit_index, _unit, group, raw, is_global_final):
            nonlocal stop_score, stop_entropy
            if not group:
                return None
            normalized = self._prediction_read(unit_index, raw)
            if 1 in group:
                stop_logits = self.stop_head(normalized).transpose(0, 1)
                stop_score, stop_entropy = selected_logprob_and_entropy(
                    stop_logits,
                    batch["stop_actions"],
                    return_entropy=policy_return_entropy,
                )
            # Each co-site head reads this same state. No selected-code
            # feedback occurs until every head in the group has been scored.
            for q in group:
                logits, _ = self.audio_lm_heads[q - 1](normalized)
                logits = logits.transpose(0, 1)
                targets = batch["successor_codes"][..., q - 1]
                if policy_return_entropy:
                    code_entropies[q - 1] = vocab_parallel_entropy(logits)
                # MCore CE exponentiates its FP32 argument in place. Keep the
                # head output intact for other differentiable statistics.
                code_scores[q - 1] = -tensor_parallel.vocab_parallel_cross_entropy(logits.float().clone(), targets)
            if is_global_final:
                return None
            return self._successor_conditioning_delta(batch["successor_codes"], batch["successor_mask"], group)

        _execute_prism_schedule(
            self.prism_topology,
            hidden,
            apply_range=apply_range,
            read_site=read_site,
            loop_block_prefix_counts=prefix_counts,
        )
        if stop_score is None or any(value is None for value in code_scores):
            raise RuntimeError("Prism schedule did not score every action head")
        scores = torch.cat((torch.stack(code_scores, dim=-1), stop_score.unsqueeze(-1)), dim=-1)
        if policy_return_entropy:
            entropies = torch.cat((torch.stack(code_entropies, dim=-1), stop_entropy.unsqueeze(-1)), dim=-1)
            scores = torch.cat((scores, entropies), dim=-1)
        return scores


def vocab_parallel_entropy(logits: Tensor) -> Tensor:
    """Exact entropy over the complete TP vocabulary with differentiable reductions."""
    group = parallel_state.get_tensor_model_parallel_group()
    if torch.distributed.get_world_size(group) == 1:
        logprobs = logits.float().log_softmax(-1)
        return -(logprobs.exp() * logprobs).sum(-1)
    # The maximum is a numerical offset and cancels from the entropy.
    maximum = logits.detach().float().amax(-1)
    torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX, group=group)
    shifted = logits.float() - maximum.unsqueeze(-1)
    exps = shifted.exp()
    # TP replicas consume this same scalar. The region reduction performs a
    # SUM in forward and passes the local gradient through in backward.
    denominator = tensor_parallel.reduce_from_tensor_model_parallel_region(exps.sum(-1), group=group)
    weighted = tensor_parallel.reduce_from_tensor_model_parallel_region((exps * shifted).sum(-1), group=group)
    return denominator.log() - weighted / denominator


def model_provider(pre_process=True, post_process=True, vp_stage=None):
    """Build Prism through slime's existing custom-model-provider entry point."""
    _require_megatron()
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.training import get_args
    from megatron.training.arguments import core_transformer_config_from_args

    if vp_stage is not None:
        raise ValueError("Prism does not support virtual pipeline stages")
    args = get_args()
    spec = configure_prism_args(args)
    if args.transformer_impl != "transformer_engine":
        raise ValueError("Prism uses Transformer Engine parameter names to preserve its native checkpoint contract")
    if getattr(args, "fp8_param_gather", False):
        raise ValueError("Prism does not support FP8 parameter gather")
    config = core_transformer_config_from_args(args, config_class=MossTTSPrismConfig)
    if config.params_dtype != getattr(torch, spec.dtype):
        raise ValueError("Prism Megatron precision must match the inference artifact dtype")
    # MCore will cast/wrap parameters after construction. Native import runs
    # at slime's checkpoint-load boundary, after DDP/optimizer allocation.
    layer_spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True)
    model = MossTTSPrismModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=spec.text_vocab_size,
        max_sequence_length=spec.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        share_embeddings_and_output_weights=spec.embedding_head_usage == "tied_head_authoritative",
        rotary_base=spec.rotary_base,
    )
    model.prism_spec = spec
    return model
