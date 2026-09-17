"""Purpose-aware reference similarity aggregation around the timbre service."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import soundfile as sf
from typing_extensions import assert_never

from slime.rollout.tts_schema import (
    REFERENCE_AUDIO_USE_ORDER,
    ReferenceAudioUse,
    TTSReferenceAudio,
    parse_reference_audios_metadata,
)
from slime.utils.types import Sample

from . import sim
from .config import ComponentReward
from .runtime import RewardHttpRuntime


class ReferenceDimensionMetadata(TypedDict):
    reference_id: str
    reward: float
    raw: float
    implemented: bool


class ReferenceSimilarityInputError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GeneratedAudioFailure:
    category: str = "reference_similarity.invalid_generated_audio"


@dataclass(frozen=True, slots=True)
class PreparedReferenceSample:
    sample: Sample
    references: tuple[TTSReferenceAudio, ...]
    required_uses: tuple[ReferenceAudioUse, ...]


@dataclass(frozen=True, slots=True)
class ReferenceDimensionResult:
    use: ReferenceAudioUse
    reference_id: str
    reward: float
    raw_value: float
    implemented: bool

    def to_metadata(self) -> ReferenceDimensionMetadata:
        return {
            "reference_id": self.reference_id,
            "reward": self.reward,
            "raw": self.raw_value,
            "implemented": self.implemented,
        }


@dataclass(frozen=True, slots=True)
class ReferenceSimilarityResult:
    component: ComponentReward
    dimensions: tuple[ReferenceDimensionResult, ...]
    required_uses: tuple[ReferenceAudioUse, ...]


def reset_diagnostics(samples: Sequence[Sample]) -> None:
    for sample in samples:
        if not isinstance(sample.metadata, dict):
            sample.metadata = {}
        sample.metadata["reference_similarity_dimensions"] = {}
        sample.metadata["reference_similarity_required_uses"] = []


def _require_readable_wav(path: Path, field: str) -> Path:
    if not path.is_absolute():
        raise ReferenceSimilarityInputError(f"reference_similarity requires an absolute {field} WAV path")
    try:
        stat = path.stat()
    except OSError as error:
        raise ReferenceSimilarityInputError(f"reference_similarity requires a readable {field} WAV file") from error
    if not path.is_file():
        raise ReferenceSimilarityInputError(f"reference_similarity requires a regular {field} WAV file")
    if stat.st_size <= 0:
        raise ReferenceSimilarityInputError(f"reference_similarity {field} WAV must be non-empty")
    try:
        info = sf.info(path)
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ReferenceSimilarityInputError(f"reference_similarity requires a decodable {field} WAV file") from error
    if info.format != "WAV" or info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise ReferenceSimilarityInputError(f"reference_similarity requires a non-empty {field} WAV stream")
    return resolved


def _reference_for_use(
    references: tuple[TTSReferenceAudio, ...],
    use: ReferenceAudioUse,
) -> TTSReferenceAudio:
    matches = [reference for reference in references if use in reference.uses]
    if len(matches) != 1:
        raise ReferenceSimilarityInputError(f"reference_similarity requires exactly one reference for use {use!r}")
    return matches[0]


def prepare_batch(
    samples: Sequence[Sample],
) -> tuple[PreparedReferenceSample | GeneratedAudioFailure | ReferenceSimilarityResult, ...]:
    """Validate reference/candidate audio before optional SIM batching."""
    prepared: list[PreparedReferenceSample | GeneratedAudioFailure] = []
    for sample in samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        references = parse_reference_audios_metadata(metadata.get("reference_audios"))
        if not references:
            raise ReferenceSimilarityInputError("reference_similarity requires at least one reference use")
        for reference in references:
            _require_readable_wav(reference.path, "reference audio")
        required_uses = tuple(
            use for use in REFERENCE_AUDIO_USE_ORDER if any(use in reference.uses for reference in references)
        )
        generated_audio = sample.audio_path
        if generated_audio is None and sample.trajectory is not None and sample.trajectory.num_frames == 0:
            prepared.append(
                ReferenceSimilarityResult(ComponentReward("reference_similarity", 0.0, 0.0), (), required_uses)
            )
            continue
        if not isinstance(generated_audio, str) or not generated_audio:
            prepared.append(GeneratedAudioFailure())
            continue
        try:
            _require_readable_wav(Path(generated_audio), "generated audio")
        except ReferenceSimilarityInputError:
            prepared.append(GeneratedAudioFailure())
            continue
        prepared.append(
            PreparedReferenceSample(
                sample=sample,
                references=references,
                required_uses=required_uses,
            )
        )
    return tuple(prepared)


def requires_timbre_service(
    prepared: Sequence[PreparedReferenceSample | GeneratedAudioFailure],
) -> bool:
    return any("timbre" in item.required_uses for item in prepared if isinstance(item, PreparedReferenceSample))


def _dimension_result(
    prepared: PreparedReferenceSample,
    use: ReferenceAudioUse,
    timbre_result: ComponentReward | None,
) -> ReferenceDimensionResult:
    reference = _reference_for_use(prepared.references, use)
    match use:
        case "timbre":
            if timbre_result is None or timbre_result.name != "timbre":
                raise AssertionError("Timbre reference similarity result is missing or malformed.")
            return ReferenceDimensionResult(
                use=use,
                reference_id=reference.id,
                reward=timbre_result.reward,
                raw_value=timbre_result.raw_value,
                implemented=True,
            )
        case "accent" | "prosody" | "emotion":
            # These dimensions remain explicit zero-score placeholders so the
            # aggregate still reflects every required reference use.
            return ReferenceDimensionResult(
                use=use,
                reference_id=reference.id,
                reward=0.0,
                raw_value=0.0,
                implemented=False,
            )
        case unreachable:
            assert_never(unreachable)


async def score_prepared_batch(
    prepared: Sequence[PreparedReferenceSample | GeneratedAudioFailure],
    runtime: RewardHttpRuntime | None,
) -> list[ReferenceSimilarityResult | sim.SimBatchFailure | GeneratedAudioFailure]:
    timbre_positions = [
        index
        for index, item in enumerate(prepared)
        if isinstance(item, PreparedReferenceSample) and "timbre" in item.required_uses
    ]
    timbre_by_position: dict[int, ComponentReward | sim.SimBatchFailure] = {}
    if timbre_positions:
        if runtime is None:
            raise AssertionError("Timbre reference similarity requires an HTTP runtime.")
        timbre_results = await sim.score_batch(
            [prepared[index].sample for index in timbre_positions],
            runtime,
        )
        timbre_by_position.update(zip(timbre_positions, timbre_results, strict=True))

    results: list[ReferenceSimilarityResult | sim.SimBatchFailure | GeneratedAudioFailure] = []
    for index, item in enumerate(prepared):
        if isinstance(item, (GeneratedAudioFailure, ReferenceSimilarityResult)):
            results.append(item)
            continue
        timbre_result = timbre_by_position.get(index)
        if isinstance(timbre_result, sim.SimBatchFailure):
            results.append(timbre_result)
            continue
        dimensions = tuple(_dimension_result(item, use, timbre_result) for use in item.required_uses)
        aggregate = sum(dimension.reward for dimension in dimensions) / len(dimensions)
        results.append(
            ReferenceSimilarityResult(
                component=ComponentReward("reference_similarity", aggregate, aggregate),
                dimensions=dimensions,
                required_uses=item.required_uses,
            )
        )
    return results


def apply_diagnostics(sample: Sample, result: ReferenceSimilarityResult) -> None:
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    sample.metadata["reference_similarity_dimensions"] = {
        dimension.use: dimension.to_metadata() for dimension in result.dimensions
    }
    sample.metadata["reference_similarity_required_uses"] = list(result.required_uses)
