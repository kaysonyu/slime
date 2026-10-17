"""Base speech: content accuracy measured by WER/CER."""

from .common import component_weights, reward_states, score


async def reward(args, sample, **kwargs):
    weights = component_weights(args, ("wer",))

    def aggregate(sample, components):
        return weights["wer"] * components["wer"]["transformed"]

    return await score(args, sample, category="base", components=("wer",), aggregate=aggregate, weights=weights)


reward.close = reward_states.close
