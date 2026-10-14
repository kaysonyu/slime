# Copyright (c) 2025, OpenMOSS. All rights reserved.
"""Prism topology contracts."""

from __future__ import annotations
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from numbers import Real
from typing import Any, Literal, TypeAlias


class PrismItemKind(IntEnum):
    TEXT = 0
    AUDIO = 1


class PrismAudioRole(IntEnum):
    NONE = 0
    REFERENCE = 1
    TARGET = 2


PrismSiteRecord: TypeAlias = tuple[Literal["ordinary"], int] | tuple[Literal["loop"], int, int]


@dataclass(frozen=True, slots=True)
class PrismExecutionSite:
    """Identity of one prediction site in physical execution order.

    Ordinary physical-layer indices are zero-based. Loop block and application
    indices are one-based, matching the architecture notation ``C(b, r)``.
    """

    kind: Literal["ordinary", "loop"]
    index: int
    application: int | None = None

    def canonical_record(self) -> PrismSiteRecord:
        if self.kind == "ordinary":
            return ("ordinary", self.index)
        if self.application is None:  # pragma: no cover - construction invariant
            raise RuntimeError("loop execution site is missing its application")
        return ("loop", self.index, self.application)


@dataclass(frozen=True, slots=True)
class PrismExecutionUnit:
    """One ordinary layer or one complete loop-range application."""

    site: PrismExecutionSite
    layer_start: int
    layer_stop: int


@dataclass(frozen=True, slots=True)
class ResolvedPrismTopology:
    """Immutable, canonical Prism construction and execution topology."""

    n_vq: int
    num_layers: int
    prism_mlp_only_layers: tuple[int, ...]
    prism_loop_block_ranges: tuple[tuple[int, int], ...]
    prism_loop_block_repetitions: tuple[int, ...]
    prism_rvq_heads_per_execution_unit: tuple[int, ...]
    reapply_rvq_conditioning_in_loop: bool
    execution_units: tuple[PrismExecutionUnit, ...]
    sigma: tuple[PrismExecutionSite, ...]
    co_site_groups: tuple[tuple[int, ...], ...]
    occupied_execution_unit_indices: tuple[int, ...]
    prediction_norm_index_by_execution_unit: tuple[int | None, ...]

    @property
    def num_loop_blocks(self) -> int:
        return len(self.prism_loop_block_ranges)

    @property
    def num_execution_units(self) -> int:
        return len(self.execution_units)

    @property
    def num_prediction_norms(self) -> int:
        return len(self.occupied_execution_unit_indices)

    @property
    def num_effective_layer_calls(self) -> int:
        return sum(unit.layer_stop - unit.layer_start for unit in self.execution_units)

    @property
    def execution_sites(self) -> tuple[PrismExecutionSite, ...]:
        return tuple(unit.site for unit in self.execution_units)

    @property
    def final_site(self) -> PrismExecutionSite:
        return self.execution_units[-1].site

    @property
    def site_records(self) -> tuple[PrismSiteRecord, ...]:
        return tuple(site.canonical_record() for site in self.execution_sites)

    @property
    def sigma_records(self) -> tuple[PrismSiteRecord, ...]:
        return tuple(site.canonical_record() for site in self.sigma)

    @property
    def loop_block_prefix_counts(self) -> tuple[int, ...]:
        """Return leading unoccupied applications before each block's first head.

        The compact head-site vector already contains this information.  A
        prefix count is therefore a runtime scheduling property rather than a
        second topology field.  Every valid block has an occupied final
        application, so a first occupied application always exists.
        """

        spans: list[tuple[int, int, int]] = []
        index = 0
        while index < len(self.execution_units):
            unit = self.execution_units[index]
            if unit.site.kind != "loop":
                index += 1
                continue
            block = unit.site.index
            start = index
            while index < len(self.execution_units):
                candidate = self.execution_units[index]
                if candidate.site.kind != "loop" or candidate.site.index != block:
                    break
                index += 1
            spans.append((block, start, index))

        if len(spans) != self.num_loop_blocks:  # pragma: no cover - resolver invariant
            raise RuntimeError("Prism topology did not contain every loop block")
        result: list[int] = []
        for block, block_start, block_stop in spans:
            block_groups = self.co_site_groups[block_start:block_stop]
            first_head = next((offset for offset, group in enumerate(block_groups) if group), None)
            if first_head is None:  # pragma: no cover - resolver invariant
                raise RuntimeError(f"Prism loop block {block} has no RVQ head")
            result.append(first_head)
        return tuple(result)

    def execution_unit_indices_for_prefix_counts(self, prefix_counts: Sequence[int] | None = None) -> tuple[int, ...]:
        """Select execution units for one forward's loop prefix schedule.

        ``prefix_counts[b]`` is the number of unoccupied applications retained
        immediately before block ``b``'s first occupied site.  The omitted
        applications are only the leading zero-head applications; all
        occupied sites and all units after the first occupied site remain in
        their canonical order.  ``None`` selects the complete static schedule.
        """

        maximums = self.loop_block_prefix_counts
        if prefix_counts is None:
            return tuple(range(len(self.execution_units)))
        if isinstance(prefix_counts, (str, bytes, bytearray)) or not isinstance(prefix_counts, Sequence):
            raise TypeError("prefix_counts must be an ordered integer vector")
        if len(prefix_counts) != self.num_loop_blocks:
            raise ValueError("prefix_counts must contain exactly " f"{self.num_loop_blocks} entries")
        normalized: list[int] = []
        for block, (value, maximum) in enumerate(zip(prefix_counts, maximums, strict=True), start=1):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"prefix_counts[{block - 1}] must be a non-Boolean integer")
            if not 0 <= value <= maximum:
                raise ValueError(f"prefix_counts[{block - 1}] must be in [0, {maximum}]")
            normalized.append(value)

        selected: list[int] = []
        block = 0
        index = 0
        while index < len(self.execution_units):
            unit = self.execution_units[index]
            if unit.site.kind != "loop":
                selected.append(index)
                index += 1
                continue
            block_index = unit.site.index - 1
            expected_block = block_index + 1
            block_start = index
            while index < len(self.execution_units):
                candidate = self.execution_units[index]
                if candidate.site.kind != "loop" or candidate.site.index != expected_block:
                    break
                index += 1
            leading = maximums[block_index]
            retained_prefix = normalized[block_index]
            selected.extend(range(block_start, block_start + retained_prefix))
            selected.extend(range(block_start + leading, index))
            block += 1
        if block != self.num_loop_blocks:  # pragma: no cover - resolver invariant
            raise RuntimeError("Prism schedule did not visit every loop block")
        return tuple(selected)

    def canonical_dict(self) -> dict[str, Any]:
        """Return the alias-free, JSON-compatible topology representation."""

        return {
            "n_vq": self.n_vq,
            "num_layers": self.num_layers,
            "prism_mlp_only_layers": list(self.prism_mlp_only_layers),
            "prism_loop_block_ranges": [[start, stop] for start, stop in self.prism_loop_block_ranges],
            "prism_loop_block_repetitions": list(self.prism_loop_block_repetitions),
            "prism_rvq_heads_per_execution_unit": list(self.prism_rvq_heads_per_execution_unit),
            "sigma": [list(record) for record in self.sigma_records],
            "reapply_rvq_conditioning_in_loop": (self.reapply_rvq_conditioning_in_loop),
        }


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive non-Boolean integer")
    return value


def normalize_prism_rvq_label_smoothing(value: object, *, n_vq: int) -> tuple[float, ...]:
    """Return one finite label-smoothing value for every RVQ channel.

    ``None`` is the intentional omitted-option spelling and means that
    smoothing is disabled for every audio head.  Once supplied, the vector is
    deliberately exact-length: scalar broadcasting and an empty explicit
    vector are both rejected.
    """

    n_vq = _positive_integer(n_vq, "n_vq")
    if value is None:
        return (0.0,) * n_vq
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("rvq_label_smoothing must be an ordered vector")
    if len(value) != n_vq:
        raise ValueError("rvq_label_smoothing must contain exactly " f"n_vq={n_vq} entries")
    result: list[float] = []
    for index, entry in enumerate(value):
        if isinstance(entry, bool) or not isinstance(entry, Real):
            raise ValueError(f"rvq_label_smoothing[{index}] must be a real non-Boolean number")
        number = float(entry)
        if not math.isfinite(number) or not 0.0 <= number < 1.0:
            raise ValueError(f"rvq_label_smoothing[{index}] must be finite and in [0, 1)")
        result.append(number)
    return tuple(result)


def normalize_prism_input_rvq_channels(value: object, *, n_vq: int) -> tuple[int, ...]:
    """Return the canonical 1-based RVQ channels used at backbone input."""

    n_vq = _positive_integer(n_vq, "n_vq")
    if value is None:
        return tuple(range(1, n_vq + 1))
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("prism_input_rvq_channels must be an ordered vector")
    if not value:
        raise ValueError("prism_input_rvq_channels must not be empty")
    result: list[int] = []
    previous = 0
    for index, entry in enumerate(value):
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise ValueError(f"prism_input_rvq_channels[{index}] must be a positive " "non-Boolean integer")
        if not 1 <= entry <= n_vq:
            raise ValueError(f"prism_input_rvq_channels[{index}] must be in [1, {n_vq}]")
        if entry <= previous:
            raise ValueError("prism_input_rvq_channels must be strictly increasing without " "duplicates")
        result.append(entry)
        previous = entry
    return tuple(result)


def _ordered_collection(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an ordered collection")
    return value


def _canonical_ranges(value: object) -> tuple[tuple[int, int], ...]:
    ranges = _ordered_collection(value, "prism_loop_block_ranges")
    result: list[tuple[int, int]] = []
    previous_stop = 0
    for block_offset, raw_range in enumerate(ranges):
        name = f"prism_loop_block_ranges[{block_offset}]"
        pair = _ordered_collection(raw_range, name)
        if len(pair) != 2:
            raise ValueError(f"{name} must contain exactly [start, stop]")
        start, stop = pair
        if any(isinstance(endpoint, bool) or not isinstance(endpoint, int) for endpoint in pair):
            raise ValueError(f"{name} endpoints must be non-Boolean integers")
        if not 0 <= start < stop:
            raise ValueError(f"{name} must satisfy 0 <= start < stop")
        if result and previous_stop > start:
            raise ValueError("Prism loop block ranges must be ordered and nonoverlapping")
        result.append((start, stop))
        previous_stop = stop
    return tuple(result)


def _canonical_integer_vector(value: object, *, name: str, positive: bool) -> tuple[int, ...]:
    sequence = _ordered_collection(value, name)
    result: list[int] = []
    for index, entry in enumerate(sequence):
        if isinstance(entry, bool) or not isinstance(entry, int):
            qualifier = "positive " if positive else "nonnegative "
            raise ValueError(f"{name}[{index}] must be a {qualifier}non-Boolean integer")
        if entry < (1 if positive else 0):
            qualifier = "positive" if positive else "nonnegative"
            raise ValueError(f"{name}[{index}] must be {qualifier}")
        result.append(entry)
    return tuple(result)


def _execution_units(
    *,
    num_layers: int,
    ranges: tuple[tuple[int, int], ...],
    repetitions: tuple[int, ...],
) -> tuple[PrismExecutionUnit, ...]:
    result: list[PrismExecutionUnit] = []
    layer = 0
    for block_offset, ((start, stop), repeat) in enumerate(zip(ranges, repetitions, strict=True)):
        while layer < start:
            site = PrismExecutionSite("ordinary", layer)
            result.append(PrismExecutionUnit(site, layer, layer + 1))
            layer += 1
        block = block_offset + 1
        for application in range(1, repeat + 1):
            site = PrismExecutionSite("loop", block, application)
            result.append(PrismExecutionUnit(site, start, stop))
        layer = stop
    while layer < num_layers:
        site = PrismExecutionSite("ordinary", layer)
        result.append(PrismExecutionUnit(site, layer, layer + 1))
        layer += 1
    return tuple(result)


def resolve_prism_topology(values: object) -> ResolvedPrismTopology:
    """Resolve exact Prism config/argument attributes without mutating ``values``."""

    n_vq = _positive_integer(values.n_vq, "n_vq")
    ranges = _canonical_ranges(values.prism_loop_block_ranges)
    repetitions = _canonical_integer_vector(
        values.prism_loop_block_repetitions,
        name="prism_loop_block_repetitions",
        positive=True,
    )
    if len(repetitions) != len(ranges):
        raise ValueError("prism_loop_block_repetitions must have one entry per loop block range")
    if len(ranges) > n_vq:
        raise ValueError("the number of Prism loop blocks cannot exceed n_vq")

    counts = _canonical_integer_vector(
        values.prism_rvq_heads_per_execution_unit,
        name="prism_rvq_heads_per_execution_unit",
        positive=False,
    )
    if not counts:
        raise ValueError("prism_rvq_heads_per_execution_unit must contain at least one " "execution unit")
    num_layers = len(counts) + sum(
        (stop - start) - repeat for (start, stop), repeat in zip(ranges, repetitions, strict=True)
    )
    if num_layers <= 0:
        raise ValueError("the concrete Prism topology must infer a positive physical layer count")
    mlp_only_layers = _canonical_integer_vector(
        getattr(values, "prism_mlp_only_layers", ()),
        name="prism_mlp_only_layers",
        positive=False,
    )
    if mlp_only_layers != tuple(sorted(set(mlp_only_layers))):
        raise ValueError("prism_mlp_only_layers must be strictly increasing without duplicates")
    if any(index >= num_layers for index in mlp_only_layers):
        raise ValueError(f"prism_mlp_only_layers entries must be in [0, {num_layers})")
    for block_offset, (_start, stop) in enumerate(ranges):
        if stop > num_layers:
            raise ValueError(
                f"prism_loop_block_ranges[{block_offset}] must stop at or before "
                f"the inferred physical layer count ({num_layers})"
            )

    reapply = getattr(values, "reapply_rvq_conditioning_in_loop", False)
    if not isinstance(reapply, bool):
        raise ValueError("reapply_rvq_conditioning_in_loop must be exactly Boolean")

    units = _execution_units(num_layers=num_layers, ranges=ranges, repetitions=repetitions)
    if len(counts) != len(units):
        raise RuntimeError("inferred Prism physical depth did not reproduce the declared " "execution-unit count")
    if sum(counts) != n_vq:
        raise ValueError("prism_rvq_heads_per_execution_unit must sum exactly to n_vq")
    if counts[-1] == 0:
        raise ValueError("the final Prism execution site must own at least one RVQ head")

    co_site_groups: list[tuple[int, ...]] = []
    sigma: list[PrismExecutionSite] = []
    next_q = 1
    for unit, count in zip(units, counts, strict=True):
        group = tuple(range(next_q, next_q + count))
        co_site_groups.append(group)
        sigma.extend((unit.site,) * count)
        next_q += count

    occupied_execution_unit_indices = tuple(index for index, group in enumerate(co_site_groups) if group)
    prediction_norm_index_by_execution_unit: list[int | None] = []
    next_prediction_norm = 0
    for group in co_site_groups:
        prediction_norm_index_by_execution_unit.append(next_prediction_norm if group else None)
        if group:
            next_prediction_norm += 1

    count_by_site = {unit.site: count for unit, count in zip(units, counts, strict=True)}
    for block, repeat in enumerate(repetitions, start=1):
        final_block_site = PrismExecutionSite("loop", block, repeat)
        if count_by_site[final_block_site] == 0:
            raise ValueError(f"Prism loop block {block} final application must own an RVQ head")

    if sigma[-1] != units[-1].site:  # defensive expression of sigma(n_vq)=v_M
        raise ValueError("RVQ n_vq must be assigned to the final execution site")

    return ResolvedPrismTopology(
        n_vq=n_vq,
        num_layers=num_layers,
        prism_mlp_only_layers=mlp_only_layers,
        prism_loop_block_ranges=ranges,
        prism_loop_block_repetitions=repetitions,
        prism_rvq_heads_per_execution_unit=counts,
        reapply_rvq_conditioning_in_loop=reapply,
        execution_units=units,
        sigma=tuple(sigma),
        co_site_groups=tuple(co_site_groups),
        occupied_execution_unit_indices=occupied_execution_unit_indices,
        prediction_norm_index_by_execution_unit=tuple(prediction_norm_index_by_execution_unit),
    )


def validate_prism_stop_gradient_layer(value: object, topology: ResolvedPrismTopology) -> int | None:
    """Validate one physical-layer gradient boundary without changing topology.

    The boundary is expressed in physical coordinates: ``K`` means that the
    first ``K`` layers have run and the first post-boundary unit receives
    detached copies of both the crossing residual and incoming
    successor-conditioning delta.  A boundary strictly inside a loop range has
    no single place in the expanded recurrence, so it is rejected.  Ordinary
    boundaries and range endpoints are valid; an endpoint is handled by the
    schedule as one block transition.  Validation does not change topology.
    """

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("prism_stop_gradient_layer must be None or a non-Boolean integer")
    if not 1 <= value < topology.num_layers:
        raise ValueError(
            "prism_stop_gradient_layer must satisfy " f"1 <= K < physical_num_layers={topology.num_layers}"
        )
    for start, stop in topology.prism_loop_block_ranges:
        if start < value < stop:
            raise ValueError(
                "prism_stop_gradient_layer cannot be inside loop block " f"[{start}, {stop}); got K={value}"
            )
    return value


def validate_prism_recompute_policy(values: object, *, effective_config: bool = False) -> bool:
    """Validate Prism's one supported Megatron recomputation tuple.

    ``effective_config`` accounts for ``TransformerConfig`` replacing an
    absent ``recompute_modules`` value with its inert ``["core_attn"]``
    default. Argument validation still rejects every explicit module choice.
    The return value says whether full-layer recomputation is enabled.
    """

    modules = getattr(values, "recompute_modules", None)
    if effective_config:
        if modules not in (None, ["core_attn"], ("core_attn",)):
            raise ValueError(
                "MOSS-TTS Prism does not support an effective --recompute-modules " f"selection; got {modules!r}"
            )
    elif modules is not None:
        raise ValueError(
            "MOSS-TTS Prism does not support explicit --recompute-modules, " "including an empty selection"
        )

    granularity = getattr(values, "recompute_granularity", None)
    method = getattr(values, "recompute_method", None)
    count = getattr(values, "recompute_num_layers", None)
    distributed = getattr(values, "distribute_saved_activations", False)
    tensor_parallel_size = getattr(values, "tensor_model_parallel_size", 1)
    distributed_disabled = distributed is False or (effective_config and distributed is None)

    if granularity is None and method is None and count is None and not distributed:
        return False

    supported = (
        granularity == "full"
        and method == "uniform"
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count == 1
        and distributed_disabled
        and isinstance(tensor_parallel_size, int)
        and not isinstance(tensor_parallel_size, bool)
        and tensor_parallel_size == 1
    )
    if not supported:
        raise ValueError(
            "MOSS-TTS Prism activation recomputation supports only "
            "--recompute-granularity full --recompute-method uniform "
            "--recompute-num-layers 1 at --tensor-model-parallel-size 1, "
            "without --distribute-saved-activations; got "
            f"--recompute-granularity={granularity!r}, "
            f"--recompute-method={method!r}, "
            f"--recompute-num-layers={count!r}, "
            f"--tensor-model-parallel-size={tensor_parallel_size!r}, "
            f"--distribute-saved-activations={distributed!r}"
        )
    return True
