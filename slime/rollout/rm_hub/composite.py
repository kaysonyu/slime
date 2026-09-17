"""Composite group reward and valid-only GRPO normalization."""

from __future__ import annotations

import asyncio
import logging
from argparse import Namespace
from collections import Counter
from typing_extensions import assert_never

import torch

from slime.rollout.failures import RecoverableRolloutError, mark_sample_failed
from slime.utils.types import Sample

from .config import get_reward_config
from .config import ComponentReward
from . import judge, reference_similarity, sim, wer
from .runtime import RewardHttpRuntime, RewardServiceError, get_reward_http_runtime

logger = logging.getLogger(__name__)


async def _score_sample(
    sample: Sample,
    runtime: RewardHttpRuntime | None,
    reference_result: (reference_similarity.ReferenceSimilarityResult | reference_similarity.GeneratedAudioFailure | sim.SimBatchFailure | None),
    args: Namespace,
) -> list[ComponentReward]:
    """Score one sample while preserving component-local failure semantics."""
    config = get_reward_config(args)
    if isinstance(reference_result, reference_similarity.GeneratedAudioFailure):
        error = RecoverableRolloutError(reference_result.category)
        mark_sample_failed(sample, error.failure)
        sample.metadata["reward_failure_categories"] = [reference_result.category]
        raise error
    tasks = []
    names = []
    for component in config.components:
        if component.weight == 0:
            continue
        names.append(component.name)
        match component.name:
            case "noop":
                tasks.append(_noop_score())
            case "wer":
                if runtime is None:
                    raise AssertionError("Active WER reward requires an HTTP runtime.")
                tasks.append(wer.score(sample, runtime))
            case "reference_similarity":
                if reference_result is None:
                    raise AssertionError("Active reference_similarity reward requires a batch result.")
                tasks.append(_precomputed_reference_score(sample, reference_result))
            case "judge":
                if runtime is None:
                    raise AssertionError("Active judge reward requires an HTTP runtime.")
                tasks.append(judge.score(sample, runtime))
            case unreachable:
                assert_never(unreachable)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    component_rewards: list[ComponentReward] = []
    failures: list[str] = []
    recoverable_categories: list[str] = []
    for name, result in zip(names, results, strict=True):
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, RewardServiceError):
            if not result.failure.retryable:
                raise result
            failures.append(f"{name}:{result.category}")
            recoverable_categories.append(result.failure.category)
        elif isinstance(result, Exception):
            raise result
        else:
            component_rewards.append(result)
    if failures:
        error = RecoverableRolloutError(recoverable_categories[0])
        mark_sample_failed(sample, error.failure)
        sample.metadata["reward_failure_categories"] = failures
        raise error
    return component_rewards


async def _noop_score() -> ComponentReward:
    return ComponentReward("noop", 0.0, 0.0)


async def _precomputed_reference_score(
    sample: Sample,
    result: reference_similarity.ReferenceSimilarityResult | sim.SimBatchFailure,
) -> ComponentReward:
    if isinstance(result, sim.SimBatchFailure):
        raise RewardServiceError("timbre_sim", result.value, "SIM request retry exhausted")
    reference_similarity.apply_diagnostics(sample, result)
    return result.component


async def reward_batch(args: Namespace, samples: list[Sample], **_kwargs) -> list[float]:
    """Run active reward components and return one raw scalar per sample."""
    config = get_reward_config(args)
    reference_similarity.reset_diagnostics(samples)
    if config.weight("judge") > 0:
        for sample in samples:
            if isinstance(sample.metadata, dict):
                sample.metadata.pop("judge_rubric_scores", None)
        for sample in samples:
            judge.parse_sample_rubrics(sample)
    reference_active = config.weight("reference_similarity") > 0
    prepared_references = reference_similarity.prepare_batch(samples) if reference_active else ()
    requires_http_runtime = config.weight("wer") > 0 or config.weight("judge") > 0 or reference_similarity.requires_timbre_service(prepared_references)
    if requires_http_runtime:
        runtime = get_reward_http_runtime()
        runtime.add_services(config.services)
        if reference_active:
            reference_results: list[reference_similarity.ReferenceSimilarityResult | reference_similarity.GeneratedAudioFailure | sim.SimBatchFailure | None] = list(await reference_similarity.score_prepared_batch(prepared_references, runtime))
        else:
            reference_results = [None] * len(samples)
        results = await asyncio.gather(
            *(_score_sample(sample, runtime, reference_result, args) for sample, reference_result in zip(samples, reference_results, strict=True)),
            return_exceptions=True,
        )
    else:
        if reference_active:
            reference_results = list(await reference_similarity.score_prepared_batch(prepared_references, None))
        else:
            reference_results = [None] * len(samples)
        results = await asyncio.gather(
            *(_score_sample(sample, None, reference_result, args) for sample, reference_result in zip(samples, reference_results, strict=True)),
            return_exceptions=True,
        )

    rewards: list[float] = []
    failure_count = 0
    failure_categories: Counter[str] = Counter()
    for sample, result in zip(samples, results, strict=True):
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, RecoverableRolloutError):
            failure_count += 1
            mark_sample_failed(sample, result.failure)
            rewards.append(0.0)
            metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
            categories = metadata.get("reward_failure_categories")
            if isinstance(categories, list) and all(isinstance(category, str) for category in categories):
                failure_categories.update(categories)
            else:
                failure_categories[result.failure.category] += 1
            continue
        if isinstance(result, Exception):
            raise result
        by_name = {component.name: component for component in result}
        # Component failures are removed before this point; an active component
        # therefore cannot silently disappear from the configured weighted sum.
        value = sum(component.weight * by_name[component.name].reward for component in config.components if component.weight > 0)
        if not torch.isfinite(torch.tensor(value)):
            raise FloatingPointError("Composite reward is NaN or Inf.")
        if sample.metadata is None:
            sample.metadata = {}
        sample.metadata["reward_components"] = {name: {"reward": component.reward, "raw": component.raw_value} for name, component in by_name.items()}
        rewards.append(float(value))
    if failure_count:
        rendered_categories = ", ".join(f"{category}={count}" for category, count in sorted(failure_categories.items()))
        logger.warning(
            "MOSS-TTS reward components invalidated %d/%d samples; categories=%s.",
            failure_count,
            len(samples),
            rendered_categories or "unknown",
        )
    return rewards
