"""Opt-in post-reward filtering through Slime's existing rollout hooks.

Both examples require raw WER/CER to remain an error rate. Copy this module to
choose a different threshold or policy; no filter runs unless selected by CLI.
"""

from collections.abc import Mapping

from slime.rollout.base_types import iter_samples
from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime_plugins.audio_rewards.protocol import finite

MAX_ERROR_RATE = 0.15


def _wer_passes(sample):
    metadata = sample.metadata
    outcome = metadata.get("omni_reward") if isinstance(metadata, Mapping) else None
    if not isinstance(outcome, Mapping) or outcome.get("status") != "success":
        raise ValueError("WER filters require a successfully scored Prism sample")
    components = outcome.get("components")
    wer = components.get("wer") if isinstance(components, Mapping) else None
    if not isinstance(wer, Mapping):
        raise ValueError("WER filters require the wer component")
    if wer.get("status") == "invalid_audio":
        return False
    if wer.get("status") != "success":
        raise ValueError("WER filters cannot turn scoring errors into rejected samples")
    raw = finite(wer.get("raw"), "WER filter raw error rate")
    if raw < 0:
        raise ValueError("WER filters require a nonnegative raw error rate")
    return raw <= MAX_ERROR_RATE


def keep_group_with_wer_pass(args, samples, **kwargs):
    """Keep the whole prompt group if at least one candidate passes raw WER."""
    passed = [_wer_passes(sample) for sample in iter_samples(samples)]
    if not passed:
        raise ValueError("WER group filter requires a nonempty prompt group")
    keep = any(passed)
    return DynamicFilterOutput(keep=keep, reason=None if keep else "wer_no_passing_candidate")


def mask_high_wer(args, groups):
    """Mask failing samples from loss, preserving rewards and group statistics."""
    # Validate the whole batch before mutating any sample.
    decisions = [(sample, _wer_passes(sample)) for sample in iter_samples(groups)]
    for sample, passed in decisions:
        if not passed:
            sample.remove_sample = True
        # Preserve any removal already requested by another hook.
        sample.metadata["omni_reward"].setdefault("filters", {})["wer_loss_mask"] = {
            "max_error_rate": MAX_ERROR_RATE,
            "passed": passed,
        }
