"""Measurement and selection contracts for a single-allocation scaling run."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/prism/benchmarks/scaling"))

from analyze import analyze, percentile, summarize_case
from campaign import resolve_case
from diagnose import gpu_profile
from observe_processes import role
from profile_router import select_router
from prepare import convert_prompt, write_json
import router_python
import router_extension
from watch import admission_held, prepare_case_recipe, snapshot_service_telemetry
from slime_plugins import tts_reward
from slime_plugins.omni_rollout import _prompt
from slime.utils.misc import should_run_periodic_action


def test_extension_replacement_requires_complete_cleanup_and_held_parent(tmp_path):
    case = {"name": "supplemental"}
    run = tmp_path / "runs" / case["name"]
    router_extension.assert_idle(tmp_path, [case])
    write_json(run / "services-ready.json", {})
    with pytest.raises(RuntimeError, match="still running"):
        router_extension.assert_idle(tmp_path, [case])
    write_json(run / "summary.json", {"valid": False})
    with pytest.raises(RuntimeError, match="Cleanup incomplete"):
        router_extension.assert_idle(tmp_path, [case])
    for rank in range(8):
        write_json(run / f"done-{rank}.json", {})
    router_extension.assert_idle(tmp_path, [case])
    write_json(tmp_path / "runs" / router_extension.HOLD_CASE / "services-ready.json", {})
    with pytest.raises(RuntimeError, match="Original parent was admitted"):
        router_extension.assert_idle(tmp_path, [case])


def test_extension_inherits_exact_original_parent_environment_without_persisting_secrets(tmp_path, monkeypatch):
    import psutil
    import socket

    source = tmp_path / "frozen"
    write_json(tmp_path / "manifest.json", {"source": str(source)})
    write_json(tmp_path / "platform/node-2.json", {"address": "10.1.2.3"})
    environment = {
        "PET_NODE_RANK": "2",
        "PET_NNODES": "8",
        "GLOO_SOCKET_IFNAME": "bond0",
        "NCCL_SOCKET_IFNAME": "bond0",
        "NCCL_IB_HCA": "=mlx5_0",
        "PRIVATE_TOKEN": "secret",
    }
    parent = SimpleNamespace(
        pid=32,
        info={"cmdline": ["python", str(source / "examples/prism/benchmarks/scaling/campaign.py"), str(tmp_path)]},
        environ=lambda: dict(environment),
    )
    monkeypatch.setattr(psutil, "process_iter", lambda *args: [parent])
    monkeypatch.setattr(
        psutil, "net_if_addrs", lambda: {"bond0": [SimpleNamespace(address="10.1.2.3", family=socket.AF_INET)]}
    )
    inherited, receipt = router_extension.inherited_job_environment(tmp_path, 2)
    assert inherited == environment
    assert "secret" not in json.dumps(receipt)
    assert receipt["network"]["GLOO_SOCKET_IFNAME"] == "bond0"
    environment.pop("GLOO_SOCKET_IFNAME")
    with pytest.raises(RuntimeError, match="missing GLOO"):
        router_extension.inherited_job_environment(tmp_path, 2)


def test_extension_environment_rejects_unrelated_or_ambiguous_parent(tmp_path, monkeypatch):
    import psutil

    write_json(tmp_path / "manifest.json", {"source": "/frozen"})
    monkeypatch.setattr(
        psutil,
        "process_iter",
        lambda *args: [SimpleNamespace(info={"cmdline": ["python", "campaign.py", str(tmp_path)]})],
    )
    with pytest.raises(RuntimeError, match="found 0"):
        router_extension.inherited_job_environment(tmp_path, 0)


def test_retry_plan_preserves_evidence_and_retries_only_failed_leaves(tmp_path):
    cases = [
        {"name": "r1"},
        {"name": "r4"},
        {"name": "r1-retry", "retry_of": "r1"},
        {"name": "r4-retry", "retry_of": "r4", "router_source_commit": "unchanged"},
        {"name": "pending"},
    ]
    for name, valid in (("r1", False), ("r4", False), ("r1-retry", True), ("r4-retry", False)):
        write_json(tmp_path / "runs" / name / "summary.json", {"valid": valid})
    plan, retries = router_extension.retry_plan(tmp_path, cases)
    assert [case["name"] for case in plan] == ["r1", "r4", "r1-retry", "r4-retry", "r4-retry-retry", "pending"]
    assert retries == [{"name": "r4-retry-retry", "retry_of": "r4-retry", "router_source_commit": "unchanged"}]
    assert router_extension.retry_plan(tmp_path, plan)[1] == []
    write_json(tmp_path / "runs/r4-retry-retry/evidence.json", {})
    with pytest.raises(FileExistsError):
        router_extension.retry_plan(tmp_path, cases)


def test_manual_admission_hold_is_scoped_and_expires(tmp_path, monkeypatch):
    import watch

    write_json(tmp_path / "platform/admission-hold.json", {"cases": ["pending"], "expires": 200})
    monkeypatch.setattr(watch.time, "time", lambda: 100)
    assert admission_held(tmp_path, "pending")
    assert not admission_held(tmp_path, "running")
    monkeypatch.setattr(watch.time, "time", lambda: 200)
    assert not admission_held(tmp_path, "pending")


def test_matched_controls_only_change_asr_or_gpu_count_and_preserve_unrun_plan(tmp_path):
    name = "64-b4096-a8-w32-r8rr"
    base = dict(name=name, gpus=64, asr_replicas=8, router_source_commit="pinned", batch=4096)
    obsolete = dict(base, name="64-b4096-a16-w32-r4rr", asr_replicas=16)
    cases = [base, obsolete]
    write_json(tmp_path / "runs" / name / "summary.json", {"valid": True, "steady_steps": 21})
    updated, old = router_extension.matched_controls(tmp_path, cases)
    assert old == obsolete
    assert cases == [base, obsolete]
    assert updated[0] == base
    assert updated[1] == dict(base, name="64-b4096-a16-w32-r8rr", asr_replicas=16, matched_control_of=name)
    assert updated[2] == dict(base, name="32-b4096-a8-w32-r8rr", gpus=32, matched_control_of=name)
    write_json(tmp_path / "runs" / obsolete["name"] / "services-ready.json", {})
    with pytest.raises(RuntimeError, match="admitted"):
        router_extension.matched_controls(tmp_path, cases)


def test_no_save_recipe_disables_final_checkpoint_before_admission(tmp_path):
    write_json(tmp_path / "recipe.json", {"save": None, "other": 42})
    prepare_case_recipe(tmp_path)
    recipe = json.loads((tmp_path / "recipe.json").read_text())
    assert recipe == {"save": None, "save_interval": None, "other": 42}
    assert not should_run_periodic_action(24, recipe["save_interval"], num_rollout=25)
    assert (tmp_path / "recipe-before-save-fix.json").exists()


def test_checkpoint_confirmation_and_admitted_recipes_are_not_rewritten(tmp_path):
    recipe = {"save": str(tmp_path / "checkpoint"), "save_interval": 10000}
    write_json(tmp_path / "recipe.json", recipe)
    prepare_case_recipe(tmp_path)
    assert json.loads((tmp_path / "recipe.json").read_text()) == recipe
    assert should_run_periodic_action(99, recipe["save_interval"], num_rollout=100)
    write_json(tmp_path / "services-ready.json", {})
    write_json(tmp_path / "recipe.json", {"save": None})
    prepare_case_recipe(tmp_path)
    assert json.loads((tmp_path / "recipe.json").read_text()) == {"save": None}


def test_extension_hold_releases_only_after_all_helpers_finished(tmp_path):
    assert not admission_held(tmp_path, "baseline")
    write_json(tmp_path / "platform/router-extension.json", {"hold_case": "baseline"})
    assert admission_held(tmp_path, "baseline")
    assert not admission_held(tmp_path, "supplemental")
    write_json(tmp_path / "platform/router-extension/finished.json", {})
    assert not admission_held(tmp_path, "baseline")


@pytest.mark.parametrize("router", [True, False])
def test_router_adapter_preserves_engine_arguments_and_pins_only_router(tmp_path, monkeypatch, router):
    run = tmp_path / "runs/case"
    source = tmp_path / "original"
    write_json(tmp_path / "manifest.json", {"source": str(source)})
    write_json(run / "case.json", {"router_processes": 4, "router_policy": "round_robin"})
    monkeypatch.setenv("PRISM_SCALE_RUN", str(run))
    arguments = (
        [str(source / "slime/backends/sglang_utils/omni_router.py"), json.dumps([{"policy": "least_request"}, False])]
        if router
        else ["-m", "sglang_omni.cli", "serve", "--config", "same-config"]
    )
    monkeypatch.setattr(router_python.sys, "argv", ["adapter", *arguments])
    calls = []
    monkeypatch.setattr(router_python.os, "execv", lambda *args: calls.append(args))
    router_python.main()
    executable, command = calls[0]
    assert executable == command[0] == "/opt/venvs/core/bin/python"
    if router:
        config, disabled = json.loads(command[2])
        assert config["router_processes"] == 4
        assert config["policy"] == "round_robin"
        assert disabled is False
    else:
        assert command[1:] == arguments


@pytest.mark.parametrize("diagnostic", [False, True])
def test_confirmation_and_resume_inherit_selected_pinned_router(tmp_path, diagnostic):
    import hashlib

    source = tmp_path / "source"
    adapter = source / "examples/prism/benchmarks/scaling/router_python.py"
    entry = source / "slime/backends/sglang_utils/omni_router.py"
    write_json(entry, {"router": "fixed"})
    variant = dict(
        router_python=str(adapter),
        router_processes=4,
        router_policy="round_robin",
        router_entry_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),
        router_source_commit="commit",
    )
    write_json(tmp_path / "runs/candidate/case.json", variant)
    expected_adapter, expected_commit = adapter, "commit"
    if diagnostic:
        entry = tmp_path / "diagnostic/slime/backends/sglang_utils/omni_router.py"
        write_json(entry, {"router": "error-only diagnostics"})
        expected_adapter = tmp_path / "diagnostic/examples/prism/benchmarks/scaling/router_python.py"
        expected_commit = "diagnostic-commit"
        write_json(tmp_path / "manifest.json", {"router_diagnostic_override": {
            "parent_commits": ["commit"],
            "router_python": str(expected_adapter),
            "router_source_commit": expected_commit,
            "router_entry_sha256": hashlib.sha256(entry.read_bytes()).hexdigest(),
        }})
    for name, link in (("confirm", {"selected_from": "candidate"}), ("resume", {"resume": "confirm"})):
        run = tmp_path / "runs" / name
        write_json(run / "case.json", link)
        write_json(run / "recipe.json", {"save": "checkpoint", "save_interval": 10000})
        prepare_case_recipe(run)
        case = json.loads((run / "case.json").read_text())
        assert case["router_processes"] == 4
        recipe = json.loads((run / "recipe.json").read_text())
        assert recipe["omni_python"] == str(expected_adapter)
        assert case["router_source_commit"] == expected_commit
        assert recipe["save"] == "checkpoint"
    assert json.loads((tmp_path / "runs/candidate/case.json").read_text()) == variant


def test_unadmitted_supplemental_case_uses_repaired_manifest_router(tmp_path):
    import hashlib

    run = tmp_path / "runs" / "supplemental"
    entry = tmp_path / "new/slime/backends/sglang_utils/omni_router.py"
    write_json(entry, {"revision": "repaired"})
    case = dict(
        name="supplemental",
        supplemental=True,
        gpus=64,
        batch=4096,
        asr_replicas=8,
        workers=32,
        steps=25,
        router_processes=4,
        router_policy="round_robin",
        router_python="old",
    )
    planned = dict(
        case,
        router_python=str(tmp_path / "new/examples/prism/benchmarks/scaling/router_python.py"),
        router_entry_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),
        router_source_commit="new",
    )
    write_json(run / "case.json", case)
    write_json(run / "recipe.json", {"save": None, "save_interval": None})
    write_json(tmp_path / "manifest.json", {"cases": [planned]})
    prepare_case_recipe(run)
    assert json.loads((run / "case.json").read_text())["router_source_commit"] == "new"
    assert json.loads((run / "recipe.json").read_text())["omni_python"] == planned["router_python"]
    planned["batch"] = 2048
    write_json(tmp_path / "manifest.json", {"cases": [planned]})
    with pytest.raises(RuntimeError, match="non-router"):
        prepare_case_recipe(run)
    write_json(run / "services-ready.json", {})
    prepare_case_recipe(run)  # An admitted case is immutable, even after a manifest change.


def test_asr_snapshot_survives_replica_log_reuse(tmp_path):
    manifest = {"cases": [{"name": "first"}, {"name": "second"}]}
    write_json(tmp_path / "runs/first/summary.json", {"valid": True})
    pod = tmp_path / "services/asr/pod-0"
    write_json(pod / "launch.json", {"started": 1})
    snapshot_service_telemetry(tmp_path, manifest, "second")
    write_json(pod / "launch.json", {"started": 2})
    snapshot_service_telemetry(tmp_path, manifest, "second")
    saved = tmp_path / "services/history/first/pod-0/launch.json"
    assert json.loads(saved.read_text())["started"] == 1


def test_gpu_profiles_use_unique_devices_and_half_open_steady_interval(tmp_path):
    path = tmp_path / "gpu.csv"
    path.write_text(
        "1970/01/01 00:16:39.000,0,GPU-1,99,0,50,200\n"
        "1970/01/01 00:16:40.000,0,GPU-1,0,0,50,200\n"
        "1970/01/01 00:16:41.000,0,GPU-1,100,0,50,200\n"
        "1970/01/01 00:16:42.000,0,GPU-1,99,0,50,200\n"
        "1970/01/01 00:16:40.000,0,GPU-2,20,50,200\n"
    )
    result = gpu_profile([path], 1000, 1002)
    assert result["devices"] == 2
    assert result["device_mean_busy_percent"]["mean"] == 35
    assert result["busy_percent_all_samples"]["mean"] == 40
    assert gpu_profile([], 1000, 1002)["devices"] == 0


def test_process_observer_only_classifies_scoped_roles(tmp_path):
    assert role(["python", str(tmp_path / "slime/backends/sglang_utils/omni_router.py")], tmp_path) == "omni_router"
    assert role(["ray::GroupWorker"], tmp_path) == "GroupWorker"
    assert role(["python", "/another/job/omni_router.py"], tmp_path) is None
    assert role([], tmp_path) is None
    assert (
        role(["python", str(tmp_path / "slime/backends/sglang_utils/omni_router.py"), "--data-plane"], tmp_path)
        == "omni_router_data"
    )


def test_profiler_selects_busy_current_data_plane_not_supervisor_or_stale_router(tmp_path):
    source = tmp_path / "pinned"
    run = tmp_path / "runs/current"
    entry = str(source / "slime/backends/sglang_utils/omni_router.py")

    def process(pid, arguments, cpu, scoped_run):
        return SimpleNamespace(
            pid=pid,
            info={"cmdline": ["python", entry, *arguments]},
            cpu_times=lambda: SimpleNamespace(user=cpu, system=0),
            environ=lambda: {"PRISM_SCALE_RUN": str(scoped_run)},
        )

    processes = [
        process(1, [], 1000, run),
        process(2, ["--control-plane"], 999, run),
        process(3, ["--data-plane"], 998, tmp_path / "old"),
        process(4, ["--data-plane"], 20, run),
        process(5, ["--data-plane"], 40, run),
    ]
    selected, eligible = select_router(processes, run, source, multiprocess=True)
    assert selected.pid == 5 and eligible == [4, 5]
    assert select_router(processes, run, source)[0].pid == 1
    with pytest.raises(RuntimeError, match="No omni_router_data"):
        select_router(processes, tmp_path / "unrelated", source, multiprocess=True)


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


def test_weak_scaling_uses_valid_anchor_and_discloses_configuration_confounding(tmp_path):
    names = ["32-b2048-a4-w16-start", "32-b2048-a4-w16-end", "64-b4096-a16-w32-r8rr"]
    write_json(tmp_path / "manifest.json", {"cases": [{"name": name} for name in names]})
    for name, valid, seconds in zip(names, [False, True, True], [1, 20, 16], strict=True):
        write_json(
            tmp_path / "runs" / name / "summary.json",
            {"case": name, "valid": valid, "steady_steps": 21, "step_seconds": seconds},
        )
    report = analyze(tmp_path)
    assert report["baseline_32_b2048_seconds"] == 20
    assert report["baseline_32_b2048_cases"] == [names[1]]
    assert report["baseline_drift_measured"] is False
    assert report["baseline_drift_percent"] is None
    assert report["weak_scaling"][names[2]] == {"throughput_speedup": 2.5, "efficiency": 1.25}
    assert "not evidence of superlinear hardware scaling" in report["weak_scaling_scope"]
    assert "additional ASR GPUs" in report["weak_scaling_scope"]


def test_router_comparisons_require_complete_valid_controls(tmp_path):
    names = ["64-b4096-a8-w32", "64-b4096-a8-w32-r1rr-retry", "64-b4096-a8-w32-r4rr-retry"]
    write_json(tmp_path / "manifest.json", {"cases": [{"name": name} for name in names]})
    for name, seconds, valid in zip(names, [40, 38, 10], [True, True, False], strict=True):
        write_json(
            tmp_path / "runs" / name / "summary.json",
            {
                "case": name,
                "valid": valid,
                "steady_steps": 21,
                "step_seconds": seconds,
                "phase_seconds": {"generate": seconds - 2},
            },
        )
    comparison = analyze(tmp_path)["router_comparisons"]
    assert set(comparison) == {"single_process_policy"}
    assert comparison["single_process_policy"]["speedup"] == 40 / 38
    assert comparison["single_process_policy"]["phase_seconds_saved"]["generate"] == 2
    retry = names[-1] + "-retry"
    cases = [{"name": name} for name in names] + [{"name": retry, "retry_of": names[-1]}]
    write_json(tmp_path / "manifest.json", {"cases": cases})
    write_json(
        tmp_path / "runs" / retry / "summary.json",
        {"case": retry, "valid": True, "steady_steps": 21, "step_seconds": 25, "phase_seconds": {"generate": 23}},
    )
    comparison = analyze(tmp_path)["router_comparisons"]
    assert comparison["native_four_process"]["candidate"] == retry
    assert comparison["native_four_process"]["speedup"] == 38 / 25


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
