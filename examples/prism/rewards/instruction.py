"""Instruction speech: this file owns the content/instruction reward combination."""

from .common import component_weights, score


async def reward(args, sample, **kwargs):
    weights = component_weights(args, ("wer", "judge_clap"))

    def aggregate(sample, components):
        return (
            weights["wer"] * components["wer"]["transformed"]
            + weights["judge_clap"] * components["judge_clap"]["transformed"]
        )

    return await score(
        args, sample, category="instruction", components=("wer", "judge_clap"), aggregate=aggregate, weights=weights
    )
