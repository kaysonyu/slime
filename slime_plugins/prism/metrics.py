"""Audio, scoring coverage, and grouped reward statistics."""

import math
import statistics
from collections import defaultdict

from slime_plugins.prism.experiment import path_component


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _statistics(values):
    return {"mean": statistics.fmean(values), "std": statistics.pstdev(values), "min": min(values), "max": max(values)}


def _percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def compute_metrics(records, group_size):
    """Use scored/active populations; missing components are never zero scores."""
    if not records:
        return {}
    metrics = {"sample_count": len(records)}
    for key, predicate in {
        "zero_frame_ratio": lambda r: r["frames"] == 0,
        "truncated_ratio": lambda r: r["generation_status"] == "truncated",
        "scoring_error_ratio": lambda r: r["reward_status"] == "error",
        "pending_ratio": lambda r: r["reward_status"] == "pending",
        "cancelled_ratio": lambda r: r["reward_status"] == "cancelled",
    }.items():
        metrics[key] = sum(predicate(record) for record in records) / len(records)
    scored = [record for record in records if record["reward_status"] == "success"]
    metrics["scored_count"] = len(scored)
    metrics["scored_ratio"] = len(scored) / len(records)
    for field in ("frames", "audio_seconds"):
        values = [record[field] for record in records if _number(record.get(field))]
        metrics[f"{field}/coverage_ratio"] = len(values) / len(records)
        if values:
            metrics.update({f"{field}/{key}": value for key, value in _statistics(values).items()})
            metrics[f"{field}/p50"] = _percentile(values, 0.5)
            metrics[f"{field}/p95"] = _percentile(values, 0.95)
    duration_ratios = [
        r["frames"] / r["source_target_frames"]
        for r in records
        if _number(r.get("frames")) and _number(r.get("source_target_frames")) and r["source_target_frames"] > 0
    ]
    if duration_ratios:
        metrics["source_duration_ratio/p50"] = _percentile(duration_ratios, 0.5)
        metrics["source_duration_ratio/p95"] = _percentile(duration_ratios, 0.95)
        metrics["source_duration_ratio/over_2x_ratio"] = sum(value > 2 for value in duration_ratios) / len(
            duration_ratios
        )
    values = [r["reward"] for r in scored if _number(r["reward"])]
    if values:
        metrics.update({f"reward/{key}": value for key, value in _statistics(values).items()})
    components = defaultdict(list)
    groups = defaultdict(list)
    for record in scored:
        groups[(record["dataset"], record["prompt_index"])].append(record)
    # Retain completed components even when a different service/aggregation failed.
    for record in records:
        outcome = record.get("reward_details") or {}
        returned = outcome.get("components", {})
        for name, component in returned.items():
            components[name].append(component)
        for name in set(outcome.get("errors", {})) - returned.keys():
            if name in {"wer", "sim", "judge_clap"}:
                components[name].append({"status": "error"})
    for name, items in components.items():
        prefix = f"component/{path_component(name)}"
        metrics[f"{prefix}/active_count"] = len(items)
        metrics[f"{prefix}/active_ratio"] = len(items) / len(records)
        valid = [item for item in items if item.get("status") == "success" and _number(item.get("raw"))]
        metrics[f"{prefix}/valid_count"] = len(valid)
        metrics[f"{prefix}/coverage_ratio"] = len(valid) / len(records)
        metrics[f"{prefix}/valid_ratio"] = len(valid) / len(items)
        metrics[f"{prefix}/error_ratio"] = sum(item.get("status") == "error" for item in items) / len(items)
        metrics[f"{prefix}/invalid_audio_ratio"] = sum(item.get("status") == "invalid_audio" for item in items) / len(
            items
        )
        for field in ("raw", "transformed"):
            values = [item[field] for item in items if _number(item.get(field))]
            if values:
                metrics.update({f"{prefix}/{field}/{key}": value for key, value in _statistics(values).items()})
        if name == "wer":
            # Never pool character and word denominators, or different languages.
            languages = defaultdict(list)
            for item in valid:
                detail = item.get("details") or {}
                errors, tokens = detail.get("errors"), detail.get("reference_tokens")
                language = detail.get("language")
                if language and _number(errors) and errors >= 0 and _number(tokens) and tokens > 0:
                    languages[str(language)].append((errors, tokens))
            for language, counts in languages.items():
                key = f"{prefix}/language/{path_component(language)}"
                errors = sum(value[0] for value in counts)
                tokens = sum(value[1] for value in counts)
                metrics.update(
                    {
                        f"{key}/errors": errors,
                        f"{key}/reference_tokens": tokens,
                        f"{key}/corpus_error_rate": errors / tokens,
                        f"{key}/scored_count": len(counts),
                    }
                )
        elif name == "judge_clap":
            attributes = defaultdict(list)
            for item in valid:
                detail = item.get("details") or {}
                for query, score in zip(detail.get("queries", []), detail.get("attribute_scores", []), strict=True):
                    if isinstance(query, str) and _number(score):
                        attributes[query.partition(":")[0]].append(score)
            for attribute, scores in attributes.items():
                key = f"{prefix}/attribute/{path_component(attribute)}"
                metrics[f"{key}/mean"] = statistics.fmean(scores)
                metrics[f"{key}/coverage_ratio"] = len(scores) / len(records)
    complete = [group for group in groups.values() if len(group) == group_size]
    metrics["complete_group_count"] = len(complete)
    metrics["incomplete_group_count"] = len(groups) - len(complete)
    if group_size == 1:
        return metrics
    for name in ("total", *sorted(components)):
        values_by_group = []
        for group in complete:
            values = [
                (
                    record["reward"]
                    if name == "total"
                    else (record.get("reward_details") or {}).get("components", {}).get(name, {}).get("transformed")
                )
                for record in group
            ]
            if all(_number(value) for value in values):
                values_by_group.append(values)
        if values_by_group:
            deviations = [statistics.pstdev(values) for values in values_by_group]
            prefix = f"group_reward/{path_component(name)}"
            metrics.update(
                {
                    f"{prefix}/mean": statistics.fmean(statistics.fmean(values) for values in values_by_group),
                    f"{prefix}/std_mean": statistics.fmean(deviations),
                    f"{prefix}/std_p50": _percentile(deviations, 0.5),
                    f"{prefix}/std_p90": _percentile(deviations, 0.9),
                    f"{prefix}/zero_std_ratio": sum(value <= 1e-12 for value in deviations) / len(deviations),
                }
            )
    return metrics
