"""Step attribution must respect overlapping work and late-arriving samples."""

import pytest

from slime_plugins.prism.metrics import compute_metrics, last_completed_score, reward_timing_summary

NUM_GPUS = 0


def record(index, start, durations, *, error=None):
    components = {}
    for name, seconds in durations.items():
        status = "error" if name == error else "success"
        components[name] = {
            "status": status,
            "raw": None if status == "error" else 0.5,
            "transformed": None if status == "error" else 0.5,
            "timing": {
                "status": status,
                "started_at": start,
                "finished_at": start + seconds,
                "total_s": seconds,
                "client_queue_s": 0,
                "http_s": seconds,
                "request_id": f"sample-{index}-{name}",
            },
        }
    return {
        "dataset": "train",
        "prompt_index": index,
        "sample_index": index,
        "frames": 100,
        "audio_seconds": 8.0,
        "generation_status": "completed",
        "reward_status": "error" if error else "success",
        "reward": None if error else 0.5,
        "reward_details": {
            "components": components,
            "errors": {error: "TimeoutError"} if error else {},
            "timing": {"total_s": max(durations.values()), "started_at": start},
        },
    }


def test_last_completed_component_is_not_the_longest_single_request():
    rows = [
        record(0, 0, {"wer": 20, "sim": 5, "judge_clap": 15}),
        record(1, 30, {"wer": 1, "sim": 2, "judge_clap": 0.5}, error="judge_clap"),
        record(2, 27, {"wer": 1}),
    ]
    metrics = compute_metrics(rows, group_size=1)
    assert metrics["reward_timing/total_s/max"] == 20
    assert metrics["reward_timing/total_s/mean"] == pytest.approx(23 / 3)
    assert metrics["component/wer/timing/total_s/max"] == 20
    assert metrics["component/wer/timing/count"] == 3
    assert metrics["component/wer/timing/joint_count"] == 2
    assert metrics["component/wer/timing/last_ratio"] == 0.5
    assert metrics["component/sim/timing/last_ratio"] == 0.5
    assert metrics["component/judge_clap/timing/last_ratio"] == 0
    assert metrics["component/wer/timing/sample_tail_s/mean"] == 2.5
    assert metrics["component/sim/timing/sample_tail_s/max"] == 1
    assert metrics["component/wer/timing/finish_gap_s"] == 1
    assert metrics["component/sim/timing/last_completed"] == 1
    assert metrics["component/wer/timing/last_completed"] == 0
    assert metrics["component/judge_clap/timing/error_ratio"] == 0.5
    # Error profiles participate in latency percentiles, never numeric rewards.
    assert metrics["component/judge_clap/timing/count"] == 2
    assert metrics["component/judge_clap/valid_count"] == 1
    assert last_completed_score(rows)[1:3] == ("sim", rows[1])
    summary = reward_timing_summary(rows, metrics)
    assert "last_completed=sim sample=1" in summary
    assert "request=sample-1-sim" in summary
    assert "wer: n=3" in summary


def test_unmeasured_zero_frame_or_old_records_are_not_zero_latency_samples():
    row = record(0, 0, {"wer": 2})
    old = record(1, 5, {"wer": 10})
    del old["reward_details"]["timing"]
    del old["reward_details"]["components"]["wer"]["timing"]
    zero = record(2, 0, {"wer": 0})
    zero["frames"] = 0
    zero["reward_details"] = {"components": {"wer": {"raw": None, "transformed": 0, "status": "invalid_audio"}}}
    metrics = compute_metrics([row, old, zero], group_size=1)
    assert metrics["component/wer/timing/count"] == 1
    assert metrics["component/wer/timing/total_s/mean"] == 2
    assert "component/wer/timing/last_ratio" not in metrics
    assert last_completed_score([old, zero]) is None
    assert reward_timing_summary([old, zero], {}) is None


def test_parallel_request_work_is_not_reported_as_component_wall_time():
    row = record(0, 0, {"sim": 5})
    row["reward_details"]["components"]["sim"]["timing"].update(http_s=9, client_queue_s=1, http_requests=2)
    metrics = compute_metrics([row], group_size=1)
    assert metrics["component/sim/timing/http_s/mean"] == 9
    assert metrics["component/sim/timing/total_s/mean"] == 5
    assert metrics["component/sim/timing/http_requests"] == 2
