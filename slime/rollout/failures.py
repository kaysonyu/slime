from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from slime.utils.types import Sample


_CATEGORY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.:-]*$")
_METADATA_KEY = "rollout_failure"


@dataclass(frozen=True, slots=True)
class SampleFailure:
    """Safe failure metadata shared by rollout plugins and retry budgeting."""

    category: str
    retryable: bool

    def __post_init__(self) -> None:
        if not _CATEGORY_PATTERN.fullmatch(self.category):
            raise ValueError("failure category must contain only lowercase letters, digits, '.', '_', ':', or '-'.")
        if not isinstance(self.retryable, bool):
            raise TypeError("failure retryable must be a bool.")


class RolloutFailure(RuntimeError):
    def __init__(self, failure: SampleFailure, detail: str | None = None) -> None:
        message = failure.category if detail is None else f"{failure.category}: {detail}"
        super().__init__(message)
        self.failure = failure
        self.detail = detail

    def __reduce__(
        self,
    ) -> tuple[
        Callable[[type[RolloutFailure], SampleFailure, str | None], RolloutFailure],
        tuple[type[RolloutFailure], SampleFailure, str | None],
        dict[str, object],
    ]:
        """Preserve typed failure metadata across Ray/cloudpickle process boundaries."""
        return _restore_rollout_failure, (type(self), self.failure, self.detail), self.__dict__.copy()


def _restore_rollout_failure(
    exception_type: type[RolloutFailure],
    failure: SampleFailure,
    detail: str | None,
) -> RolloutFailure:
    error = exception_type.__new__(exception_type)
    RolloutFailure.__init__(error, failure, detail)
    return error


class RecoverableRolloutError(RolloutFailure):
    def __init__(self, category: str, detail: str | None = None) -> None:
        super().__init__(SampleFailure(category=category, retryable=True), detail)


class FatalRolloutError(RolloutFailure):
    def __init__(self, category: str, detail: str | None = None) -> None:
        super().__init__(SampleFailure(category=category, retryable=False), detail)


def mark_sample_failed(sample: Sample, failure: SampleFailure) -> None:
    sample.status = Sample.Status.FAILED
    sample.remove_sample = True
    sample.metadata[_METADATA_KEY] = {
        "category": failure.category,
        "retryable": failure.retryable,
    }


def get_sample_failure(sample: Sample) -> SampleFailure | None:
    raw = sample.metadata.get(_METADATA_KEY)
    if not isinstance(raw, dict) or set(raw) != {"category", "retryable"}:
        return None
    category = raw["category"]
    retryable = raw["retryable"]
    if not isinstance(category, str) or not isinstance(retryable, bool):
        return None
    try:
        return SampleFailure(category=category, retryable=retryable)
    except ValueError:
        return None


def iter_group_samples(group: Sequence[Sample] | Sequence[Sequence[Sample]]) -> Iterable[Sample]:
    for item in group:
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            yield from item
        else:
            yield item


class RecoverableFailureBudget:
    def __init__(self, maximum: int | None) -> None:
        if maximum is not None and maximum < 0:
            raise ValueError("maximum recoverable rollout failures must be non-negative.")
        self.maximum = maximum
        self.total = 0
        self.categories: Counter[str] = Counter()

    def record(self, group: Sequence[Sample] | Sequence[Sequence[Sample]]) -> None:
        samples = list(iter_group_samples(group))
        failures = [failure for sample in samples if (failure := get_sample_failure(sample))]
        if any(sample.status == Sample.Status.FAILED and get_sample_failure(sample) is None for sample in samples):
            raise FatalRolloutError("sample.unclassified_failure")
        fatal = [failure.category for failure in failures if not failure.retryable]
        if fatal:
            rendered = ",".join(sorted(fatal))
            raise FatalRolloutError("sample.fatal", rendered)
        recoverable = [failure.category for failure in failures if failure.retryable]
        self.total += len(recoverable)
        self.categories.update(recoverable)
        if self.maximum is not None and self.total > self.maximum:
            rendered = ",".join(f"{category}={count}" for category, count in sorted(self.categories.items()))
            raise FatalRolloutError(
                "failure_budget.exhausted",
                f"recoverable={self.total}, maximum={self.maximum}, categories={rendered or 'none'}",
            )

    def collect_metrics(self) -> dict[str, int]:
        metrics = {"rollout/recoverable_failures/total": self.total}
        metrics.update(
            {f"rollout/recoverable_failures/{category}": count for category, count in sorted(self.categories.items())}
        )
        return metrics
