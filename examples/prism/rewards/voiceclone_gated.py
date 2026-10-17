"""Nonlinear example: optimize SIM only when raw WER/CER passes a gate.

The threshold is an illustrative experiment setting, owned by this category.
Use this file as a starting point for another per-category formula.
"""

from .common import reward_states, score

MAX_ERROR_RATE = 0.15


def aggregate(sample, components):
    wer, sim = components["wer"], components["sim"]
    passed = wer["status"] == sim["status"] == "success" and wer["raw"] <= MAX_ERROR_RATE
    sample.metadata["omni_reward"]["gate"] = {"max_error_rate": MAX_ERROR_RATE, "passed": passed}
    return sim["transformed"] if passed else 0.0


async def reward(args, sample, **kwargs):
    return await score(args, sample, category="voiceclone_gated", components=("wer", "sim"), aggregate=aggregate)


reward.close = reward_states.close
