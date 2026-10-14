"""Shared component clients and scalar/batch contracts for category rewards.

Category files own their formulas. This module owns frozen service/cache
lifetime, complete component records, failure draining, and final reward logs.
"""

import asyncio
import inspect
import math
from collections.abc import Mapping
from numbers import Real

from slime_plugins import tts_reward


class _RewardState:
    def __init__(self, config):
        if not isinstance(config, Mapping) or set(config) not in ({"services"}, {"services", "weights"}):
            raise ValueError("omni_reward_config requires services and optional weights mappings")
        self.scorer = tts_reward.RewardScorer(config["services"])
        self.weights = None
        if "weights" in config:
            weights = config["weights"]
            if not isinstance(weights, Mapping) or set(weights) - {"wer", "sim", "judge_clap"}:
                raise ValueError("reward weights must map wer, sim, and judge_clap to coefficients")
            self.weights = {name: tts_reward._finite(value, f"{name} weight") for name, value in weights.items()}
            if any(value < 0 for value in self.weights.values()):
                raise ValueError("reward weights must be nonnegative")


def _state(args):
    # Preserve the existing category hook's cache. Clients live on its scorer.
    state = getattr(args, "_prism_category_rewards", None)
    if state is None:
        state = _RewardState(tts_reward._config(getattr(args, "omni_reward_config", None)))
        args._prism_category_rewards = state
    return state


def component_weights(args, components):
    """Validate coefficients for categories that explicitly use a weighted sum."""
    weights = _state(args).weights
    if weights is None:
        return {name: 1.0 for name in components}
    missing = set(components) - weights.keys()
    if missing:
        raise ValueError(f"reward category requires component weights: {sorted(missing)}")
    selected = {name: weights[name] for name in components}
    if any(weight <= 0 for weight in selected.values()):
        raise ValueError(f"reward category requires positive weights for {components}")
    return selected


async def gather_scores(coroutines):
    """Wait for all component readers before propagating a scoring failure."""
    outcomes = await asyncio.gather(*coroutines, return_exceptions=True)
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome
    return outcomes


async def score(args, sample, *, category, components, aggregate, weights=None):
    """Score one category with its synchronous aggregate(sample, records) formula."""
    if isinstance(sample, list):
        return await gather_scores(
            score(args, item, category=category, components=components, aggregate=aggregate, weights=weights)
            for item in sample
        )
    if not isinstance(getattr(sample, "metadata", None), dict):
        raise ValueError("Prism category rewards require sample.metadata")
    sample.metadata.pop("omni_reward", None)
    try:
        records = await _state(args).scorer.score_components(sample, components)
        outcome = {"status": "success", "components": records}
        if weights is not None:
            outcome["weights"] = dict(weights)
        sample.metadata["omni_reward"] = outcome
        try:
            # A legitimate immediate stop stays a zero reward and a real
            # GRPO candidate, including for formulas with an additive bonus.
            total = 0.0 if sample.metadata["omni_frames"] == 0 else aggregate(sample, records)
            if inspect.isawaitable(total):
                if inspect.iscoroutine(total):
                    total.close()
                raise TypeError("category aggregate must return a scalar synchronously")
            if isinstance(total, bool) or not isinstance(total, Real) or not math.isfinite(total):
                raise ValueError("category reward must return a finite scalar")
            total = float(total)
        except Exception as error:
            outcome.update(status="error", errors={"aggregation": type(error).__name__})
            raise
        outcome["total"] = total
        return total
    finally:
        outcome = sample.metadata.get("omni_reward")
        if isinstance(outcome, dict):
            outcome["category"] = category
