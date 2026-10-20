"""Divide one training run's service budget across its execution workers."""


def worker_budget(total, worker_index=0, worker_count=1):
    for name, value, minimum in (
        ("concurrency", total, 1),
        ("worker_count", worker_count, 1),
        ("worker_index", worker_index, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if worker_index >= worker_count:
        raise ValueError("worker_index must be smaller than worker_count")
    if total < worker_count:
        raise ValueError(
            f"service concurrency={total} is smaller than {worker_count} rollout workers; "
            "increase the total budget or use fewer workers"
        )
    quotient, remainder = divmod(total, worker_count)
    return quotient + (worker_index < remainder)
