"""Batched shared-path timbre similarity client."""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from slime.utils.types import Sample

from .config import ComponentReward
from slime.rollout.tts_schema import parse_reference_audios_metadata
from .runtime import RewardHttpRuntime, RewardServiceError

MAX_SIM_ITEMS_PER_REQUEST = 16
_AFFINITY_PREFIX = b"mosstts-sim-ref-v1\0"
_ZERO_REWARD_ITEM_ERROR_CODES = frozenset({"audio_too_short"})


@dataclass(frozen=True, slots=True)
class _SimilarityItem:
    result_index: int
    item_id: str
    reference_path: Path
    candidate_path: Path


class SimBatchFailure(Enum):
    RETRY_EXHAUSTED = "retry_exhausted"


def _paths(sample: Sample) -> tuple[Path, Path]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    references = parse_reference_audios_metadata(metadata.get("reference_audios"))
    candidate = sample.audio_path
    timbre_references = [reference for reference in references if "timbre" in reference.uses]
    if len(timbre_references) != 1:
        raise ValueError("Timbre similarity requires exactly one reference audio with the timbre use.")
    if not isinstance(candidate, str):
        raise ValueError("SIM reward requires a gen_audio metadata path.")
    reference_path = timbre_references[0].path
    candidate_path = Path(candidate)
    if not reference_path.is_file() or not candidate_path.is_file():
        raise ValueError("SIM audio inputs must be readable files.")
    return reference_path.resolve(strict=True), candidate_path.resolve(strict=True)


def reference_affinity_key(reference_path: Path) -> str:
    resolved_path = reference_path.resolve(strict=True)
    return hashlib.sha256(_AFFINITY_PREFIX + os.fsencode(resolved_path)).hexdigest()


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RewardServiceError("timbre_sim", "protocol", f"SIM result {field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise RewardServiceError("timbre_sim", "protocol", f"SIM result {field} must be a finite number")
    return result


def _parse_results(response: object, items: Sequence[_SimilarityItem]) -> list[tuple[int, ComponentReward]]:
    if not isinstance(response, Mapping):
        raise RewardServiceError("timbre_sim", "protocol", "SIM response must be an object")
    rows = response.get("results")
    if not isinstance(rows, list):
        raise RewardServiceError("timbre_sim", "protocol", "SIM response requires a results list")
    expected_ids = {item.item_id for item in items}
    rows_by_id: dict[str, Mapping[object, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise RewardServiceError("timbre_sim", "protocol", "SIM result rows must be objects")
        item_id = row.get("id")
        if not isinstance(item_id, str) or item_id in rows_by_id:
            raise RewardServiceError("timbre_sim", "protocol", "SIM result ids must be unique strings")
        rows_by_id[item_id] = row
    if set(rows_by_id) != expected_ids:
        raise RewardServiceError("timbre_sim", "protocol", "SIM response ids do not match the request")

    parsed: list[tuple[int, ComponentReward]] = []
    for item in items:
        row = rows_by_id[item.item_id]
        item_error = row.get("error")
        if item_error is not None:
            code = item_error.get("code") if isinstance(item_error, Mapping) else None
            safe_code = code if isinstance(code, str) and code else "unknown"
            if safe_code in _ZERO_REWARD_ITEM_ERROR_CODES:
                parsed.append(
                    (
                        item.result_index,
                        ComponentReward("timbre", 0.0, 0.0),
                    )
                )
                continue
            raise RewardServiceError("timbre_sim", "request", f"SIM item failed with code {safe_code}")
        raw_similarity = _finite_number(row.get("similarity"), "similarity")
        reward_similarity = _finite_number(row.get("reward_similarity"), "reward_similarity")
        if not 0.0 <= reward_similarity <= 1.0:
            raise RewardServiceError("timbre_sim", "protocol", "SIM reward_similarity must be within [0, 1]")
        parsed.append(
            (
                item.result_index,
                ComponentReward("timbre", reward_similarity, raw_similarity),
            )
        )
    return parsed


async def _score_chunk(runtime: RewardHttpRuntime, items: Sequence[_SimilarityItem]) -> list[tuple[int, ComponentReward]]:
    reference_path = items[0].reference_path
    response = await runtime.post_json(
        "timbre_sim",
        {
            "items": [
                {
                    "id": item.item_id,
                    "reference_path": os.fspath(item.reference_path),
                    "candidate_path": os.fspath(item.candidate_path),
                }
                for item in items
            ]
        },
        request_headers={"x-inspire-inference-key": reference_affinity_key(reference_path)},
    )
    return _parse_results(response, items)


async def score_batch(samples: Sequence[Sample], runtime: RewardHttpRuntime) -> list[ComponentReward | SimBatchFailure]:
    """Batch timbre requests by reference while preserving input order."""
    grouped_items: dict[Path, list[_SimilarityItem]] = {}
    for result_index, sample in enumerate(samples):
        if sample.index is None:
            raise RewardServiceError("timbre_sim", "configuration", "SIM reward requires sample.index")
        reference_path, candidate_path = _paths(sample)
        grouped_items.setdefault(reference_path, []).append(
            _SimilarityItem(
                result_index=result_index,
                item_id=f"slime-{result_index}-{sample.index}",
                reference_path=reference_path,
                candidate_path=candidate_path,
            )
        )

    # A failed chunk is recoverable for exactly the samples in that chunk; do
    # not turn an unrelated reference group into a failed sample.
    chunks = [items[start : start + MAX_SIM_ITEMS_PER_REQUEST] for items in grouped_items.values() for start in range(0, len(items), MAX_SIM_ITEMS_PER_REQUEST)]
    chunk_results = await asyncio.gather(*(_score_chunk(runtime, chunk) for chunk in chunks), return_exceptions=True)
    results: list[ComponentReward | SimBatchFailure | None] = [None] * len(samples)
    for chunk, chunk_result in zip(chunks, chunk_results, strict=True):
        if isinstance(chunk_result, asyncio.CancelledError):
            raise chunk_result
        if isinstance(chunk_result, RewardServiceError):
            if not chunk_result.failure.retryable:
                raise chunk_result
            failure = SimBatchFailure.RETRY_EXHAUSTED
            for item in chunk:
                results[item.result_index] = failure
            continue
        if isinstance(chunk_result, Exception):
            raise chunk_result
        for result_index, reward in chunk_result:
            results[result_index] = reward
    if any(result is None for result in results):
        raise RuntimeError("SIM batch result is incomplete")
    return [result for result in results if result is not None]
