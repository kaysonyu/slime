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
    allocated_nodes = case.get("allocated_nodes", 8)
    check(
        len(receipts) == allocated_nodes
        and {item.get("rank") for item in receipts} == set(range(allocated_nodes))
        and all(item["returncode"] == 0 for item in receipts),
        "node failure/incomplete cleanup",
    )
    phases = rows(run / "phases.jsonl")
    check(bool(phases) and all(item["error"] is None for item in phases), "missing/failed driver phases")
    generated = {item["rollout"]: item for item in phases if item["phase"] == "generate"}
    first_step = case.get("first_step", 100 if case.get("resume") else 0)
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
            math.isfinite(item["train/grad_norm"])
            and item["train/grad_norm"] > 0
            and math.isfinite(item["train/loss"])
            for item in metrics
        ),
        "invalid loss/gradients",
    )
    publications = [
        (int(v), int(n)) for v, n in re.findall(r"Published policy version=(\d+) to (\d+) rollout replicas", log)
    ]
    check(
        bool(publications) and all(n == case["gpus"] for _, n in publications), "incorrect publication replica count"
    )
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


def acceptance_status(root, manifest, reports):
    """Business acceptance is separate from finishing a matrix of attempts.

    Failed exploratory attempts remain evidence, but cannot veto a later valid
    confirmation. Conversely, successful short runs cannot replace confirmation
    and a verified checkpoint restart.
    """
    required = [case for case in manifest["cases"] if case.get("choose_best") or case.get("resume")]
    results = {}
    for case in required:
        name = case["name"]
        run = root / "runs" / name
        summary = reports.get(name)
        if summary is None:
            results[name] = {"status": "failed" if (run / "skip.json").exists() else "pending"}
            continue
        verification = run / "checkpoint-verification.json"
        checkpoint_valid = verification.exists() and json.loads(verification.read_text()).get("valid") is True
        commit_exists = (run / "checkpoint/rollout" / f"training_commit_{case['steps'] - 1}.json").is_file()
        passed = summary.get("valid") is True and checkpoint_valid and commit_exists
        results[name] = {
            "status": "passed" if passed else "failed",
            "summary_valid": summary.get("valid") is True,
            "updates": summary.get("updates", 0),
            "checkpoint_verified": checkpoint_valid,
            "checkpoint_commit_exists": commit_exists,
        }
    configured = any(case.get("choose_best") for case in required) and any(case.get("resume") for case in required)
    statuses = {item["status"] for item in results.values()}
    accepted = configured and statuses == {"passed"}
    status = "passed" if accepted else ("failed" if "failed" in statuses else "pending")
    if not configured:
        status = "not_configured"
    return {
        "accepted": accepted,
        "status": status,
        "required_cases": results,
        "scope": "Requires independent confirmation plus checkpoint/optimizer/scheduler restart validation. Platform exit and complete/all_valid matrix fields are not acceptance.",
    }


def analyze(root):
    reports = {path.parent.name: json.loads(path.read_text()) for path in (root / "runs").glob("*/summary.json")}
    manifest = json.loads((root / "manifest.json").read_text())
    order = [case["name"] for case in manifest["cases"]]
    aggregate = {
        "complete": set(reports) == set(order),
        "all_valid": bool(reports) and all(r["valid"] for r in reports.values()),
        "method": "Adjacent generate starts; exclude first three updates and final window; retain all other steps. HTTP latency includes remote queue and transport, not pure GPU time.",
        "runs": reports,
        "acceptance": acceptance_status(root, manifest, reports),
    }
    aggregate["planned_comparisons"] = {}
    for comparison in manifest.get("comparisons", []):
        controls = [reports.get(name, {}) for name in comparison["controls"]]
        candidate = reports.get(comparison["candidate"], {})
        if not all(row.get("valid") and row.get("steady_steps", 0) >= 21 for row in [*controls, candidate]):
            continue
        control_seconds = sum(row["step_seconds"] * row["steady_steps"] for row in controls) / sum(
            row["steady_steps"] for row in controls
        )
        speedup = comparison["batch_ratio"] * control_seconds / candidate["step_seconds"]
        aggregate["planned_comparisons"][comparison["name"]] = {
            **comparison,
            "control_seconds": control_seconds,
            "candidate_seconds": candidate["step_seconds"],
            "throughput_speedup": speedup,
            "scaling_efficiency": speedup / comparison["gpu_ratio"],
            "control_drift_percent": (
                (controls[-1]["step_seconds"] / controls[0]["step_seconds"] - 1) * 100 if len(controls) > 1 else None
            ),
        }
    anchors = [
        reports[name]
        for name in ("32-b2048-a4-w16-start", "32-b2048-a4-w16-end")
        if name in reports and reports[name]["valid"] and reports[name].get("steady_steps")
    ]
    if anchors:
        baseline = sum(r["step_seconds"] * r["steady_steps"] for r in anchors) / sum(
            r["steady_steps"] for r in anchors
        )
        aggregate["baseline_32_b2048_seconds"] = baseline
        aggregate["baseline_32_b2048_cases"] = [r["case"] for r in anchors]
        aggregate["baseline_drift_percent"] = (
            (anchors[1]["step_seconds"] / anchors[0]["step_seconds"] - 1) * 100 if len(anchors) == 2 else None
        )
        aggregate["baseline_drift_measured"] = len(anchors) == 2
        aggregate["weak_scaling_scope"] = (
            "Observed B2048-to-B4096 throughput relative to the valid original-router 32-GPU anchor(s). "
            "Candidates may also change router implementation, ASR replicas, execution workers or concurrency. "
            "The legacy efficiency field normalizes by learner/rollout GPU count only; values above one "
            "are not evidence of superlinear hardware scaling and do not account for additional ASR GPUs. "
            "Separate-run means do not establish statistical significance; baseline drift is unknown "
            "unless both anchors are valid."
        )
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

    def latest_valid_attempt(name):
        family = {name}
        for case in manifest["cases"]:
            if case.get("retry_of") in family:
                family.add(case["name"])
        # Choose the latest complete valid attempt, never the fastest attempt.
        for case in reversed(manifest["cases"]):
            result = reports.get(case["name"], {})
            if case["name"] in family and result.get("valid") and result.get("steady_steps", 0) >= 21:
                return case["name"]
        return name

    aggregate["router_comparisons"] = {}
    for label, control, candidate in (
        ("single_process_policy", "64-b4096-a8-w32", "64-b4096-a8-w32-r1rr-retry"),
        ("native_four_process", "64-b4096-a8-w32-r1rr-retry", "64-b4096-a8-w32-r4rr-retry"),
        ("native_eight_process", "64-b4096-a8-w32-r1rr-retry", "64-b4096-a8-w32-r8rr"),
        ("native_eight_vs_four_process", "64-b4096-a8-w32-r4rr-retry", "64-b4096-a8-w32-r8rr"),
        ("asr16_vs_asr8_native_four_process", "64-b4096-a8-w32-r4rr-retry", "64-b4096-a16-w32-r4rr"),
        ("asr16_vs_asr8_native_eight_process", "64-b4096-a8-w32-r8rr", "64-b4096-a16-w32-r8rr"),
        ("matched_strong_scaling_native_eight_process", "32-b4096-a8-w32-r8rr", "64-b4096-a8-w32-r8rr"),
    ):
        control, candidate = latest_valid_attempt(control), latest_valid_attempt(candidate)
        a, b = reports.get(control, {}), reports.get(candidate, {})
        if not all(
            item.get("valid") and item.get("steady_steps", 0) >= 21 and item.get("step_seconds") for item in (a, b)
        ):
            continue
        aggregate["router_comparisons"][label] = {
            "control": control,
            "candidate": candidate,
            "control_seconds": a["step_seconds"],
            "candidate_seconds": b["step_seconds"],
            "speedup": a["step_seconds"] / b["step_seconds"],
            "time_reduction_percent": (1 - b["step_seconds"] / a["step_seconds"]) * 100,
            "phase_seconds_saved": {
                key: value - b.get("phase_seconds", {}).get(key, 0)
                for key, value in a.get("phase_seconds", {}).items()
            },
            "caveat": "Descriptive means from separate runs, not statistical significance. Native multiprocess also changes upstream HTTP pool implementation; the first native comparison is not process-count-only.",
        }
    if "target_weak_scaling_efficiency" in manifest:
        confirmation = root / "runs/128-confirm/case.json"
        selected = json.loads(confirmation.read_text()).get("selected_from") if confirmation.exists() else None
        comparison = aggregate["planned_comparisons"].get(selected, {})
        efficiency = comparison.get("scaling_efficiency")
        aggregate["scaling_target"] = {
            "selected_from": selected,
            "efficiency": efficiency,
            "required_efficiency": manifest["target_weak_scaling_efficiency"],
            "met": efficiency is not None and efficiency >= manifest["target_weak_scaling_efficiency"],
            "scope": "Same-allocation 64/B4096 versus selected 128/B8192 short-run throughput; requires valid pre/post controls, and is separate from long-run/checkpoint acceptance.",
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
