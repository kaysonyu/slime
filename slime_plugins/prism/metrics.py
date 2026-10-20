"""Audio, scoring coverage, and grouped reward statistics."""

import hashlib
import math
import re
import statistics
from collections import defaultdict

from slime_plugins.audio_rewards.timing import STAGES


def path_component(value):
    """Keep metric labels and archive names readable without path collisions."""
    raw = str(value)
    safe = re.sub(r"[^\w.-]", "_", raw, flags=re.UNICODE).strip(".")[:80]
    if not safe or safe != raw:
        safe = f"{safe or 'unnamed'}_{hashlib.sha256(raw.encode()).hexdigest()[:12]}"
    return safe


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


def _timing_statistics(metrics, prefix, profiles):
    metrics[f"{prefix}/count"] = len(profiles)
    for field in ("total_s", *STAGES):
        values = [p[field] for p in profiles if _number(p.get(field)) and p[field] >= 0]
        if values:
            metrics[f"{prefix}/{field}/mean"] = statistics.fmean(values)
            metrics[f"{prefix}/{field}/max"] = max(values)
            for q in (50, 95, 99) if field == "total_s" else (95,):
                metrics[f"{prefix}/{field}/p{q}"] = _percentile(values, q / 100)
    for field in ("http_requests", "reference_cache_hit", "reference_cache_miss", "text_cache_hit", "text_cache_miss"):
        values = [p[field] for p in profiles if _number(p.get(field)) and p[field] >= 0]
        if values:
            metrics[f"{prefix}/{field}"] = sum(values)


def _component_timings(record):
    outcome = record.get("reward_details") or {}
    return {
        name: item["timing"]
        for name, item in outcome.get("components", {}).items()
        if isinstance(item.get("timing"), dict)
        and _number(item["timing"].get("total_s"))
        and item["timing"]["total_s"] >= 0
        and _number(item["timing"].get("finished_at"))
    }


def last_completed_score(records):
    """Return the last observed completion, not a causal speedup estimate."""
    completions = [
        (profile["finished_at"], name, record, profile)
        for record in records
        for name, profile in _component_timings(record).items()
    ]
    return max(completions, key=lambda item: item[0]) if completions else None


def _reward_timing_metrics(records):
    metrics = {}
    joint, components, tails = [], defaultdict(list), defaultdict(list)
    last_counts, last_finished = defaultdict(int), {}
    for record in records:
        profile = (record.get("reward_details") or {}).get("timing")
        if isinstance(profile, dict) and _number(profile.get("total_s")):
            joint.append(profile)
        profiles = _component_timings(record)
        for name, profile in profiles.items():
            components[name].append(profile)
            last_finished[name] = max(last_finished.get(name, -math.inf), profile["finished_at"])
        # Single-component recipes do not compete with another score. Keep
        # them out of the joint last-finish/tail population.
        if len(profiles) > 1:
            ordered = sorted(profiles, key=lambda name: profiles[name]["finished_at"])
            last = ordered[-1]
            latest = profiles[last]["finished_at"]
            for name in profiles:
                last_counts[name] += int(profiles[name]["finished_at"] == latest)
                tails[name].append(max(0, latest - profiles[ordered[-2]]["finished_at"]) if name == last else 0)
    if joint:
        _timing_statistics(metrics, "reward_timing", joint)
    for name, profiles in components.items():
        prefix = f"component/{path_component(name)}/timing"
        _timing_statistics(metrics, prefix, profiles)
        metrics[f"{prefix}/error_ratio"] = sum(p.get("status") == "error" for p in profiles) / len(profiles)
        metrics[f"{prefix}/cancelled_ratio"] = sum(p.get("status") == "cancelled" for p in profiles) / len(profiles)
        if tails[name]:
            metrics[f"{prefix}/joint_count"] = len(tails[name])
            metrics[f"{prefix}/last_ratio"] = last_counts[name] / len(tails[name])
            metrics[f"{prefix}/sample_tail_s/mean"] = statistics.fmean(tails[name])
            metrics[f"{prefix}/sample_tail_s/max"] = max(tails[name])
        # Cross-worker wall timestamps require synchronized host clocks.
        gap = max(last_finished.values()) - last_finished[name]
        metrics[f"{prefix}/finish_gap_s"] = gap
        metrics[f"{prefix}/last_completed"] = int(gap == 0)
    return metrics


def reward_timing_summary(records, metrics):
    """Compact per-step diagnostics for the existing Prism archive logger."""
    last = last_completed_score(records)
    if last is None:
        return None
    parts = []
    names = sorted({name for record in records for name in _component_timings(record)})
    for name in names:
        prefix = f"component/{path_component(name)}/timing"
        parts.append(
            f"{name}: n={metrics[prefix + '/count']} "
            f"p50/p95/max={metrics[prefix + '/total_s/p50']:.2f}/"
            f"{metrics[prefix + '/total_s/p95']:.2f}/{metrics[prefix + '/total_s/max']:.2f}s "
            f"queue_p95={metrics.get(prefix + '/client_queue_s/p95', 0):.2f}s "
            f"http_p95={metrics.get(prefix + '/http_s/p95', 0):.2f}s"
        )
    _, name, record, profile = last
    parts.append(
        f"last_completed={name} sample={record.get('sample_index')} "
        f"request={profile.get('request_id')} audio_s={record.get('audio_seconds')} "
        f"component_s={profile['total_s']:.2f}"
    )
    return "; ".join(parts)


def compute_metrics(records, group_size):
    """Use scored/active populations; missing components are never zero scores."""
    if not records:
        return {}
    metrics = {"sample_count": len(records), **_reward_timing_metrics(records)}
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
            if name in {"wer", "sim", "accent", "prosody", "judge_clap"}:
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
        elif name == "prosody":
            for group in ("f0", "energy", "rate", "pause"):
                values = [(item.get("details") or {}).get("groups", {}).get(group) for item in valid]
                values = [value for value in values if _number(value)]
                if values:
                    metrics[f"{prefix}/group/{group}/mean"] = statistics.fmean(values)
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
