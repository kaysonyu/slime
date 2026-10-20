"""Opt-in component math examples; select a dotted path in rewards.yaml.

These functions receive service outputs, not Prism samples. Category reward
files choose components and aggregate their results; filters run afterwards.
"""

import math

from slime_plugins.audio_rewards.protocol import finite
from slime_plugins.audio_rewards.scoring import default_judge_clap, default_wer


def exponential_wer(*, reference, hypothesis, language, identity, alpha=2.0):
    """Keep raw WER/CER and use exp(-alpha * error_rate) as component reward."""
    alpha = finite(alpha, "exponential_wer alpha")
    if alpha <= 0:
        raise ValueError("exponential_wer alpha must be positive")
    result = default_wer(reference=reference, hypothesis=hypothesis, language=language, identity=identity)
    result["transformed"] = math.exp(-alpha * result["raw"])
    result["details"].update(reward_transform="exponential", alpha=alpha)
    return result


def minimum_judge_clap(*, audio, texts, queries, identity):
    """Use the lowest calibrated attribute probability for raw and reward."""
    result = default_judge_clap(audio=audio, texts=texts, queries=queries, identity=identity)
    minimum = min(result["details"]["attribute_scores"])
    result.update(raw=minimum, transformed=minimum)
    result["details"]["aggregation"] = "minimum"
    return result
