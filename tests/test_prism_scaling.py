"""Measurement and selection contracts for a single-allocation scaling run."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/prism/benchmarks/scaling"))

from analyze import analyze, percentile, summarize_case
from campaign import resolve_case
from prepare import convert_prompt, write_json
from slime_plugins import tts_reward
from slime_plugins.omni_rollout import _prompt


def test_converted_prompts_pass_production_parser_with_and_without_language():
    old = {
        "task_type": "fullclone",
        "transcript_target": "测试语音。",
        "language": "Chinese",
        "references": [{"id": "ref0", "uri": "/reference.wav"}],
        "reference_targets": {"timbre": "ref0"},
    }
    for visible in (True, False):
        prompt = convert_prompt(dict(old, show_language=visible))
        expected = {"Language": "Chinese"} if visible else {}
        assert prompt["global_instruction"] == expected
        parsed = _prompt(SimpleNamespace(prompt=prompt, metadata={}))
        assert parsed["global_instruction"] == expected


def test_reward_timing_measures_client_queue_without_mutating_response(monkeypatch):
    response = {"protocol": tts_reward.PROTOCOL, "request_id": "one", "result": {"text": "hello"}}

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def post(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                await release.wait()
            return response

        monkeypatch.setattr(tts_reward, "post", post)
        client = tts_reward._Service({"endpoint": "http://asr", "max_inflight": 1})
        first = asyncio.create_task(client.evaluate("score", {"request_id": "one"}))
        await entered.wait()
        second = asyncio.create_task(client.evaluate("score", {"request_id": "one"}))
        await asyncio.sleep(0.01)
        assert calls == 1
        release.set()
        a, b = await asyncio.gather(first, second)
        assert b["_client_timing"]["queue_s"] >= 0.005
        assert a["_client_timing"]["http_s"] >= 0.005
        assert "_client_timing" not in response

    asyncio.run(run())


def test_confirmation_selects_only_complete_valid_cases(tmp_path):
    for name, speed, valid, steps in (("bad", 0.1, False, 21), ("short", 0.2, True, 2), ("good", 16, True, 21)):
        run = tmp_path / "runs" / f"64-b4096-{name}"
        write_json(run / "summary.json", {"valid": valid, "steady_steps": steps, "step_seconds": speed})
        write_json(
            run / "case.json",
            {"name": run.name, "asr_replicas": 8, "workers": 32, "concurrency": 16, "asr_inflight": 4},
        )
    chosen = resolve_case(tmp_path, {"name": "confirmation", "steps": 100, "choose_best": True})
    assert chosen["selected_from"] == "64-b4096-good"
    assert chosen["steps"] == 100


def test_empty_case_is_invalid_and_not_zero_time(tmp_path):
    write_json(tmp_path / "case.json", {"name": "incomplete", "steps": 5, "gpus": 32, "batch": 2048})
    result = summarize_case(tmp_path)
    assert not result["valid"]
    assert result["issues"]
    assert "step_seconds" not in result


def test_percentile_keeps_slow_steps():
    assert percentile([10, 10, 10, 100], 0.95) > 80


def test_single_valid_anchor_does_not_claim_drift_or_erase_failure(tmp_path):
    names = ["32-b2048-a4-w16-start", "32-b2048-a4-w16-end", "64-b4096-a8-w32"]
    write_json(tmp_path / "manifest.json", {"cases": [{"name": name} for name in names]})
    for name, valid, seconds in zip(names, [False, True, True], [0.1, 18, 18], strict=True):
        write_json(
            tmp_path / "runs" / name / "summary.json",
            {
                "case": name,
                "valid": valid,
                "step_seconds": seconds,
                "steady_steps": 21,
            },
        )
    report = analyze(tmp_path)
    assert report["complete"] and not report["all_valid"]
    assert report["baseline_32_b2048_seconds"] == 18
    assert report["baseline_32_b2048_cases"] == [names[1]]
    assert report["baseline_drift_percent"] is None
    assert not report["baseline_drift_measured"]
    assert report["weak_scaling"][names[2]]["throughput_speedup"] == 2
