"""Configure cyclic collection in dedicated rollout processes."""

import gc
import logging

logger = logging.getLogger(__name__)


def configure_rollout_gc(threshold: int | None) -> None:
    if threshold is None:
        return
    if threshold <= 0:
        raise ValueError("--rollout-gc-threshold must be positive")
    previous = gc.get_threshold()
    gc.set_threshold(threshold, *previous[1:])
    logger.info("Rollout GC thresholds: %s -> %s", previous, gc.get_threshold())
