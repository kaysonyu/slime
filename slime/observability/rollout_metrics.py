"""Common speech rollout metrics plus model-owned diagnostics."""

import logging
import numpy as np
from slime.observability import logging_utils
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def log_tts_rollout_data(rollout_id, args, samples, extra, elapsed, prefix="rollout"):
    from importlib import import_module

    values = {
        f"{prefix}/step": rollout_id,
        f"{prefix}/samples": len(samples),
        f"{prefix}/frames": sum(s.trajectory.num_frames for s in samples),
        f"{prefix}/actions": sum(s.trajectory.num_actions for s in samples),
        f"{prefix}/seconds": elapsed,
        f"{prefix}/reward": float(np.mean([s.get_reward_value(args) for s in samples])),
        f"{prefix}/truncated_fraction": float(np.mean([s.status == Sample.Status.TRUNCATED for s in samples])),
    }
    wers = [s.metadata["wer"] for s in samples if "wer" in s.metadata]
    if wers:
        values[f"{prefix}/wer"] = float(np.mean(wers))
        measured = [s for s in samples if "errors" in s.metadata]
        errors = sum(s.metadata["errors"] for s in measured)
        reference_tokens = sum(s.metadata["reference_tokens"] for s in measured)
        if reference_tokens:
            values[f"{prefix}/corpus_wer"] = errors / reference_tokens
    components = sorted({key for sample in samples for key in sample.metadata.get("reward_components", {})})
    for component in components:
        scores = [
            sample.metadata["reward_components"][component]
            for sample in samples
            if component in sample.metadata.get("reward_components", {})
        ]
        values[f"{prefix}/reward/{component}"] = float(np.mean([score["reward"] for score in scores]))
        values[f"{prefix}/reward/{component}/raw"] = float(np.mean([score["raw"] for score in scores]))
    groups = {}
    for sample in samples:
        groups.setdefault(sample.group_index, []).append(sample)
    deviations = [float(np.std([s.get_reward_value(args) for s in group])) for group in groups.values()]
    values[f"{prefix}/group_reward_std"] = float(np.mean(deviations))
    values[f"{prefix}/zero_variance_group_fraction"] = float(np.mean([value == 0 for value in deviations]))
    for domain in sorted({s.metadata["teacher_domain"] for s in samples if "teacher_domain" in s.metadata}):
        values[f"{prefix}/teacher/{domain}/samples"] = sum(s.metadata.get("teacher_domain") == domain for s in samples)
    diagnostics = import_module(f"slime_plugins.models.{args.model_family}.diagnostics")
    values.update({f"{prefix}/{key}": value for key, value in diagnostics.rollout_metrics(samples).items()})
    values.update(extra)
    logger.info("TTS %s %d: %s", prefix, rollout_id, values)
    logging_utils.log(args, values, step_key=f"{prefix}/step")
