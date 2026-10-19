"""Read-only bottleneck analysis of completed scaling cases and ASR snapshots."""

import argparse
import ast
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from analyze import percentile, rows
from prepare import write_json


def distribution(values):
    return (
        {
            "count": len(values),
            "mean": statistics.fmean(values),
            "min": min(values),
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
            "max": max(values),
        }
        if values
        else {}
    )


def gpu_profile(paths, start, end):
    devices = defaultdict(list)
    for path in paths:
        with path.open() as stream:
            for fields in csv.reader(stream):
                if len(fields) not in (6, 7):
                    continue
                instant = (
                    datetime.strptime(fields[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                )
                if start <= instant < end:
                    devices[fields[2].strip()].append(float(fields[3]))
    values = list(devices.values())
    return {
        "devices": len(devices),
        "device_mean_busy_percent": distribution([statistics.fmean(v) for v in values]),
        "busy_percent_all_samples": distribution([x for v in values for x in v]),
        "per_device": {key: distribution(value) for key, value in sorted(devices.items())},
    }


def router_profile(path):
    data = json.loads(path.read_text())
    frames = data["shared"]["frames"]
    leaves, inclusive = Counter(), Counter()
    total = count = 0
    for profile in data["profiles"]:
        for stack, weight in zip(profile["samples"], profile["weights"], strict=True):
            total += weight
            count += 1
            if stack:
                leaves[stack[-1]] += weight
            for frame in set(stack):
                inclusive[frame] += weight

    def top(counter):
        return [dict(frames[index], fraction=weight / total) for index, weight in counter.most_common(30)]

    return {
        "samples": count,
        "sampled_weight_seconds": total,
        "top_leaf": top(leaves),
        "top_inclusive": top(inclusive),
        "caveat": "Fractions of successfully captured GIL-held samples, not wall-clock fractions; "
        "nonblocking errors/partial stacks are possible. See py-spy.log and raw profile.",
    }


def diagnose(run):
    summary = json.loads((run / "summary.json").read_text())
    steady = [window for window in summary["windows"] if window["steady"]]
    if not steady:
        return {"case": run.name, "complete_success": summary["valid"], "diagnosable": False}
    root = run.parent.parent
    service_dir = root / "services/history" / run.name
    result = {
        "case": run.name,
        "complete_success": summary["valid"],
        "issues": summary["issues"],
        "diagnosable": True,
        "steady_rollouts": [w["step"] for w in steady],
        "scope": "Complete timing windows only; not a replacement for lifecycle/checkpoint validation.",
        "learner_rollout_gpu": gpu_profile(run.glob("node-*/gpu.csv"), steady[0]["started"], steady[-1]["ended"]),
        "phase_fraction": {k: v / summary["step_seconds"] for k, v in summary["phase_seconds"].items()},
    }
    profile = run / "router-profile/router.speedscope.json"
    if profile.exists():
        result["router_profile"] = router_profile(profile)
    processes = defaultdict(list)
    for sample in rows(root / "platform/head-processes.jsonl"):
        if steady[0]["started"] <= sample["time"] < steady[-1]["ended"]:
            for process in sample["processes"]:
                if process["role"].startswith("omni_router") and "cpu_cores_used" in process:
                    processes[(process["pid"], process["role"])].append(process["cpu_cores_used"])
    result["router_cpu_cores"] = [
        dict(pid=pid, role=role, **distribution(samples)) for (pid, role), samples in sorted(processes.items())
    ]
    if (service_dir / "snapshot.json").exists():
        result["asr_gpu"] = gpu_profile(service_dir.glob("*/gpu.csv"), steady[0]["started"], steady[-1]["ended"])
        result["asr_snapshot"] = str(service_dir)
    intervals = []
    for window in steady:
        selected = [
            r
            for r in rows(run / "experiment/records/train" / f"rollout_{window['step']:07d}.jsonl")
            if r.get("selected_for_training")
        ]
        buckets = defaultdict(list)
        starts, finishes = [], []
        for row in selected:
            timing = row.get("generation_client_timing") or {}
            if "http_s" not in timing or "finished_unix" not in timing:
                continue
            duration = timing["http_s"]
            label = "0-5s" if duration < 5 else "5-15s" if duration < 15 else "15-25s" if duration < 25 else "25s+"
            buckets[label].append(row["frames"])
            finishes.append(timing["finished_unix"] - window["started"])
            starts.append(
                timing["finished_unix"]
                - sum(timing.get(key, 0) for key in ("prepare_s", "http_s", "parse_audio_s"))
                - window["started"]
            )
        intervals.append(
            {
                "rollout": window["step"],
                "http_latency_buckets": {
                    key: {"requests": len(value), "mean_frames": statistics.fmean(value)}
                    for key, value in sorted(buckets.items())
                },
                "request_start_offset_seconds": distribution(starts),
                "request_finish_offset_seconds": distribution(finishes),
                "after_last_generation_response_seconds": (
                    window["phases"]["generate"] - max(finishes) if finishes else None
                ),
            }
        )
    result["generation_timeline"] = intervals
    worker_cpu = {}
    pattern = re.compile(r"perf (\d+): (\{.*\})")
    with (run / "node-0/launch.log").open(errors="replace") as stream:
        for line in stream:
            if "rollout/execution_worker_cpu_seconds" not in line:
                continue
            match = pattern.search(re.sub(r"\x1b\[[0-9;]*m", "", line))
            if match:
                worker_cpu[int(match[1])] = ast.literal_eval(match[2])["rollout/execution_worker_cpu_seconds"]
    result["execution_worker_cpu_seconds"] = distribution(
        [worker_cpu[w["step"]] for w in steady if w["step"] in worker_cpu]
    )
    result["generation_http_request_counts"] = dict(
        sum(
            (
                Counter({key: value["requests"] for key, value in item["http_latency_buckets"].items()})
                for item in intervals
            ),
            Counter(),
        )
    )
    write_json(run / "bottleneck.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    for run in parser.parse_args().runs:
        report = diagnose(run)
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in ["generation_timeline", "learner_rollout_gpu", "asr_gpu"]}
            )
        )
        for group in ("learner_rollout_gpu", "asr_gpu"):
            if group in report:
                print(json.dumps({group: {k: v for k, v in report[group].items() if k != "per_device"}}))
