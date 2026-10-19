"""Validate complete cases and report strong/weak scaling without dropping tails."""

import argparse
import ast
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from prepare import write_json


def rows(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def summarize_case(run):
    case = json.loads((run / "case.json").read_text())
    issues = []

    def check(condition, message):
        if not condition:
            issues.append(message)

    receipts = [json.loads(path.read_text()) for path in run.glob("done-*.json")]
    check(len(receipts) == 8 and all(item["returncode"] == 0 for item in receipts), "node failure/incomplete cleanup")
    phases = rows(run / "phases.jsonl")
    check(bool(phases) and all(item["error"] is None for item in phases), "missing/failed driver phases")
    generated = {item["rollout"]: item for item in phases if item["phase"] == "generate"}
    first_step = 100 if case.get("resume") else 0
    expected = list(range(first_step, case["steps"]))
    check(sorted(generated) == expected, "incomplete generate steps")
    trained = [item for item in phases if item["phase"] == "async_train"]
    check(sorted(item["rollout"] for item in trained) == expected, "incomplete training steps")
    log_path = run / "node-0/launch.log"
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    metrics = []
    perf = []
    for line in log.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        if "train/grad_norm" not in line and "perf/actor_train_time" not in line:
            continue
        try:
            value = ast.literal_eval(line[line.find("{") : line.rfind("}") + 1])
        except (SyntaxError, ValueError):
            continue
        if "train/grad_norm" in value:
            metrics.append(value)
        if "perf/actor_train_time" in value:
            perf.append(value)
    check(len(metrics) == len(expected), f"training metric count {len(metrics)} != {len(expected)}")
    check(
        all(
            math.isfinite(item["train/grad_norm"]) and item["train/grad_norm"] > 0 and math.isfinite(item["train/loss"])
            for item in metrics
        ),
        "invalid loss/gradients",
    )
    publications = [
        (int(v), int(n)) for v, n in re.findall(r"Published policy version=(\d+) to (\d+) rollout replicas", log)
    ]
    check(bool(publications) and all(n == case["gpus"] for _, n in publications), "incorrect publication replica count")
    check(
        {v for v, _ in publications} == set(range(first_step + 1, case["steps"] + 2)), "incomplete policy publications"
    )
    schedules = [json.loads(path.read_text()) for path in (run / "validation").glob("schedule-*.json")]
    check(len(schedules) == len(expected) * case["gpus"], "incomplete rank packing audit")
    for step in expected:
        current = [item for item in schedules if item["rollout"] == step]
        check({item["rank"] for item in current} == set(range(case["gpus"])), f"missing ranks in step {step}")
        check(sum(item["samples"] for item in current) == case["batch"], f"incorrect sample count in step {step}")
    sample_stats = {}
    timing = defaultdict(list)
    identities = set()
    for step in sorted(generated):
        selected = [
            item
            for item in rows(run / "experiment/records/train" / f"rollout_{step:07d}.jsonl")
            if item.get("selected_for_training")
        ]
        check(len(selected) == case["batch"], f"incomplete selected archive for step {step}")
        check(all(item["reward_status"] == "success" for item in selected), f"failed rewards in step {step}")
        check(all(str(item["policy_version"]) == str(step + 1) for item in selected), f"stale policy in step {step}")
        groups = Counter((item["dataset"], item["prompt_index"]) for item in selected)
        check(all(size == 8 for size in groups.values()), f"broken GRPO groups in step {step}")
        for item in selected:
            component = (item.get("reward_details") or {}).get("components", {}).get("wer", {})
            if component.get("status") == "success":
                identities.add(json.dumps(component.get("identity"), sort_keys=True))
            if step >= first_step + 3 and step < case["steps"] - 1:
                for key, value in (item.get("generation_client_timing") or {}).items():
                    if key.endswith("_s"):
                        timing[f"generation/{key}"].append(value)
                for key, value in (component.get("client_timing") or {}).items():
                    timing[f"asr/{key}"].append(value)
        sample_stats[step] = {
            "samples": len(selected),
            "frames": sum(item.get("frames") or 0 for item in selected),
            "zero_frames": sum(item.get("frames") == 0 for item in selected),
            "truncated": sum(item.get("finish_reason") == "length" for item in selected),
        }
    check(len(identities) == 1, f"ASR identity count {len(identities)}")
    if case.get("choose_best") or case.get("resume"):
        checkpoint_path = run / "checkpoint-verification.json"
        check(
            checkpoint_path.exists() and json.loads(checkpoint_path.read_text()).get("valid"),
            "checkpoint/optimizer/scheduler validation failed",
        )
    windows = []
    for step in sorted(generated):
        if step + 1 not in generated:
            continue
        start, end = generated[step], generated[step + 1]
        seconds = (end["start_monotonic_ns"] - start["start_monotonic_ns"]) / 1e9
        windows.append(
            {
                "step": step,
                "seconds": seconds,
                "started": start["started"],
                "ended": end["started"],
                "steady": step >= first_step + 3,
                **sample_stats.get(step, {}),
                "phases": {item["phase"]: item["seconds"] for item in phases if item["rollout"] == step},
            }
        )
    steady = [item for item in windows if item["steady"]]
    result = {
        "case": case["name"],
        "valid": not issues,
        "issues": issues,
        "updates": len(trained),
        "steady_steps": len(steady),
        "windows": windows,
        "microbatch_schedules": dict(Counter(str(item["microbatches"]) for item in schedules)),
        "asr_identity_count": len(identities),
        "asr_fingerprints": [json.loads(item).get("fingerprint") for item in identities if json.loads(item)],
        "publications": publications,
        "timing": {
            key: {
                "mean": statistics.fmean(values),
                "p50": percentile(values, 0.5),
                "p95": percentile(values, 0.95),
                "p99": percentile(values, 0.99),
            }
            for key, values in timing.items()
            if values
        },
    }
    if metrics:
        result["gradient_norm_range"] = [
            min(item["train/grad_norm"] for item in metrics),
            max(item["train/grad_norm"] for item in metrics),
        ]
    if steady:
        seconds = sum(item["seconds"] for item in steady)
        result.update(
            step_seconds=seconds / len(steady),
            p50_seconds=percentile([item["seconds"] for item in steady], 0.5),
            p95_seconds=percentile([item["seconds"] for item in steady], 0.95),
            samples_per_second=sum(item["samples"] for item in steady) / seconds,
            frames_per_second=sum(item["frames"] for item in steady) / seconds,
            mean_frames=sum(item["frames"] for item in steady) / max(1, sum(item["samples"] for item in steady)),
            phase_seconds={
                key: statistics.fmean(item["phases"].get(key, 0) for item in steady)
                for key in {key for item in steady for key in item["phases"]}
            },
        )
        busy, memory, uuids = [], [], set()
        for path in run.glob("node-*/gpu.csv"):
            with path.open() as stream:
                for fields in csv.reader(stream):
                    if len(fields) != 7:
                        continue
                    instant = (
                        datetime.strptime(fields[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
                        .replace(tzinfo=timezone.utc)
                        .timestamp()
                    )
                    if steady[0]["started"] <= instant <= steady[-1]["ended"]:
                        uuids.add(fields[2].strip())
                        busy.append(float(fields[3]))
                        memory.append(float(fields[5]) / 1024)
        result.update(
            gpu_busy_mean=statistics.fmean(busy) if busy else None,
            gpu_peak_memory_gib=max(memory) if memory else None,
            gpu_uuids=sorted(uuids),
        )
        if len(perf) == len(expected):
            result["actor_train_seconds"] = statistics.fmean(item["perf/actor_train_time"] for item in perf[3:-1])
    return result


def analyze(root):
    reports = {path.parent.name: json.loads(path.read_text()) for path in (root / "runs").glob("*/summary.json")}
    manifest = json.loads((root / "manifest.json").read_text())
    order = [case["name"] for case in manifest["cases"]]
    aggregate = {
        "complete": set(reports) == set(order),
        "all_valid": bool(reports) and all(r["valid"] for r in reports.values()),
        "method": "Adjacent generate starts; exclude first three updates and final window; retain all other steps. HTTP latency includes remote queue and transport, not pure GPU time.",
        "runs": reports,
    }
    anchors = [
        reports[name]
        for name in ("32-b2048-a4-w16-start", "32-b2048-a4-w16-end")
        if name in reports and reports[name]["valid"] and reports[name].get("steady_steps")
    ]
    if anchors:
        baseline = sum(r["step_seconds"] * r["steady_steps"] for r in anchors) / sum(r["steady_steps"] for r in anchors)
        aggregate["baseline_32_b2048_seconds"] = baseline
        aggregate["baseline_32_b2048_cases"] = [r["case"] for r in anchors]
        aggregate["baseline_drift_percent"] = (
            (anchors[1]["step_seconds"] / anchors[0]["step_seconds"] - 1) * 100 if len(anchors) == 2 else None
        )
        aggregate["baseline_drift_measured"] = len(anchors) == 2
        aggregate["weak_scaling"] = {
            name: {"throughput_speedup": 2 * baseline / r["step_seconds"], "efficiency": baseline / r["step_seconds"]}
            for name, r in reports.items()
            if (name.startswith("64-b4096") or name == "64-confirm") and r.get("valid") and r.get("step_seconds")
        }
    a, b = reports.get("32-b4096-a8-w32", {}), reports.get("64-b4096-a8-w32", {})
    if a.get("valid") and b.get("valid") and a.get("step_seconds") and b.get("step_seconds"):
        aggregate["strong_scaling_b4096"] = {
            "speedup": a["step_seconds"] / b["step_seconds"],
            "efficiency": a["step_seconds"] / b["step_seconds"] / 2,
        }
    write_json(root / "analysis/report.json", aggregate)
    table = ["case,valid,updates,steady_steps,step_seconds,p95_seconds,samples_per_second,frames_per_second"]
    keys = table[0].split(",")
    for name in order:
        if name in reports:
            table.append(",".join(str(reports[name].get(key, "")) for key in keys))
    (root / "analysis/results.csv").write_text("\n".join(table) + "\n")
    return aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    print(json.dumps({key: value for key, value in result.items() if key != "runs"}, indent=2))
