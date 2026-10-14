"""Select an example-owned reward file by metadata.reward_category."""

from .base import reward as reward_base
from .common import gather_scores
from .instruction import reward as reward_instruction
from .voiceclone import reward as reward_voiceclone
from .voiceclone_gated import reward as reward_voiceclone_gated

CATEGORY_REWARDS = {
    "base": reward_base,
    "voiceclone": reward_voiceclone,
    "instruction": reward_instruction,
    "voiceclone_gated": reward_voiceclone_gated,
}


def _category(sample):
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        raise ValueError("Prism category rewards require sample.metadata")
    category = metadata.get("reward_category")
    if not isinstance(category, str) or category not in CATEGORY_REWARDS:
        raise ValueError(f"metadata.reward_category must be one of {tuple(CATEGORY_REWARDS)}; got {category!r}")
    return category


async def reward(args, sample, **kwargs):
    """Dispatch a sample or ordered mixed-category batch to its category file."""
    if isinstance(sample, list):
        categories = [_category(item) for item in sample]
        group_categories = {}
        for item, category in zip(sample, categories, strict=True):
            group = getattr(item, "group_index", None)
            if group is not None and group_categories.setdefault(group, category) != category:
                raise ValueError("All candidates in a GRPO prompt group must use the same reward_category")
        return await gather_scores(
            CATEGORY_REWARDS[category](args, item, **kwargs) for item, category in zip(sample, categories, strict=True)
        )
    return await CATEGORY_REWARDS[_category(sample)](args, sample, **kwargs)
