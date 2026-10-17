"""Voice clone: this file owns the content/speaker reward combination."""

from .common import component_weights, reward_states, score


async def reward(args, sample, **kwargs):
    weights = component_weights(args, ("wer", "sim"))

    def aggregate(sample, components):
        return weights["wer"] * components["wer"]["transformed"] + weights["sim"] * components["sim"]["transformed"]

    return await score(
        args, sample, category="voiceclone", components=("wer", "sim"), aggregate=aggregate, weights=weights
    )


reward.close = reward_states.close
