"""Voice clone: combine content and explicitly assigned reference attributes."""

from collections.abc import Mapping

from .common import component_weights, gather_scores, reward_states, score


async def reward(args, sample, **kwargs):
    if isinstance(sample, list):
        return await gather_scores(reward(args, item, **kwargs) for item in sample)
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict) or not isinstance(metadata.get("omni_prompt"), Mapping):
        raise ValueError("Prism voiceclone rewards require sample.metadata.omni_prompt")
    # SIM retains its existing name and timbre contract. Optional assignments
    # never become zero-valued observations, and emotion is not implemented.
    attributes = metadata["omni_prompt"].get("reference_attributes", {})
    if not isinstance(attributes, Mapping):
        raise ValueError("Prism reference_attributes must be a mapping")
    selected = ("wer", "sim", *(name for name in ("accent", "prosody") if attributes.get(name, "none") != "none"))
    weights = component_weights(args, selected)

    def aggregate(sample, components):
        return sum(weights[name] * components[name]["transformed"] for name in selected)

    return await score(args, sample, category="voiceclone", components=selected, aggregate=aggregate, weights=weights)


reward.close = reward_states.close
