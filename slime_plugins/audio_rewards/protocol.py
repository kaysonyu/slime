"""Lightweight contracts shared by audio reward clients and services."""

import math

PROTOCOL = "voxrl.embedding.v1"


class ScoringError(RuntimeError):
    """A technical failure that must not become a numeric reward."""


def finite(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ScoringError(f"{field} must be a finite number")
    return float(value)
