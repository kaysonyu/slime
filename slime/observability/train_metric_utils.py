"""Distributed metric reduction shared by all speech policies."""

import torch.distributed as dist


def reduce_train_step_metrics(
    losses_reduced: list[dict],
    *,
    calculate_per_token_loss: bool,
    step_global_batch_size: int,
    cp_size: int,
    dp_with_cp_group,
) -> dict[str, float]:
    """Aggregate per-mb log dicts into the dict ``train_one_step`` reports.

    Pipeline (1:1 with what the train loop used to do inline):
      1. Sum each metric's per-mb ``values`` tensor locally on this rank.
      2. All-reduce across the DP*CP group (``dp_with_cp_group``).
      3. Apply the per-mode divisor / cp_factor:
         - per-token-loss: divisor = ``values[0]`` = all-reduced ``num_tokens``,
           CP-inflated by ``cp_size`` because every CP rank computes the same
           num_tokens off the FULL (not chunked) masks; the
           ``cp_factor = cp_size`` multiplier cancels that inflation, leaving
           the genuine per-token average.
         - per-rollout-mean: divisor = constant ``step_global_batch_size`` from
           the rollout side, never all-reduced, so no CP inflation to cancel
           and ``cp_factor = 1``.

    Tests pass a mock ``dp_with_cp_group`` and monkeypatch ``dist.all_reduce``
    to a no-op, then pre-aggregate virtual ranks themselves — this exercises
    the same call shape as production while staying single-process.
    """
    keys = losses_reduced[0]["keys"]
    values = None
    for item in losses_reduced:
        values = item["values"] if values is None else values + item["values"]
    assert len(keys) + 1 == values.numel()
    dist.all_reduce(values, group=dp_with_cp_group)
    values = values.tolist()

    if calculate_per_token_loss:
        num_samples_or_tokens = values[0]
        cp_factor = cp_size
    else:
        num_samples_or_tokens = step_global_batch_size
        cp_factor = 1
    return {key: value * cp_factor / num_samples_or_tokens for key, value in zip(keys, values[1:], strict=False)}


def rollout_log_metric_contribution(
    per_rank_reducer_sum: float,
    *,
    cp_size: int,
    num_rollouts_in_rollout: int,
    dp_size: int,
) -> tuple[float, float]:
    """``(sum, count)`` tuple for a per-rollout-mean metric.

    Sum across DP*CP ranks of ``count`` lands on ``num_rollouts_in_rollout``
    (``dp_size`` here is the no-CP DP width; the gather covers ``dp_size *
    cp_size`` ranks, and each rank emits the same ``count``, so the totals
    cancel out the ``cp_size`` in the sum). Result: ``Σsum / Σcount =
    sum_DP_full / num_rollouts`` — the same number ``train_one_step`` reports
    for the same samples (when ``num_steps_per_rollout == 1``).

    Pair with :func:`gather_and_reduce_log_dict` to do the full end-to-end
    in tests.
    """
    sum_value = cp_size * per_rank_reducer_sum
    count = num_rollouts_in_rollout / dp_size
    return sum_value, count


def gather_and_reduce_log_dict(
    log_dict: dict,
    *,
    dp_size: int,
    dp_src_rank: int,
    dp_group,
) -> dict | None:
    """Gather per-rank log dicts and reduce each metric on ``dp_src_rank``.

    ``(sum, count)`` tuples reduce to ``Σsum / Σcount``; plain values reduce
    to a mean across ranks. The helper stays free of reporting side effects so
    CPU multi-process tests can exercise it with real ``torch.distributed``.
    """
    if dist.get_rank() == dp_src_rank:
        gathered = [None] * dp_size
        dist.gather_object(log_dict, gathered, dst=dp_src_rank, group=dp_group)
        reduced: dict = {}
        for key in log_dict:
            values = [item[key] for item in gathered]
            first = values[0]
            if isinstance(first, tuple) and len(first) == 2:
                total_sum = sum(value[0] for value in values)
                total_count = sum(value[1] for value in values)
                reduced[key] = total_sum / total_count if total_count else 0.0
            else:
                reduced[key] = sum(values) / dp_size
        return reduced
    dist.gather_object(log_dict, None, dst=dp_src_rank, group=dp_group)
    return None
