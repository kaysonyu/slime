"""Replacement lifecycle attempts preserve scope and keep old failures visible."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/prism/benchmarks/scaling"))
from analyze import acceptance_status
from lifecycle_extension import recipe_for, require_held, takeover_at_barrier
from prepare import write_json
from resume_watch import collect_events

NUM_GPUS = 0
pytestmark = pytest.mark.unit


def test_replacement_is_pending_not_accepted_from_prior_short(tmp_path):
    old = dict(name="old", choose_best=True, steps=100, gpus=64, batch=4096,
               acceptance_superseded_by="new")
    new = dict(name="new", choose_best=True, steps=100, gpus=64, batch=4096)
    resume = dict(name="resume", resume="new", steps=102, gpus=64, batch=4096)
    status = acceptance_status(tmp_path, {"cases": [old, new, resume]}, {"old": {"valid": False}})
    assert status["status"] == "pending" and not status["accepted"]
    assert status["superseded_attempts"] == {"old": "new"}
    assert set(status["required_cases"]) == {"new", "resume"}
    new["steps"] = 25
    with pytest.raises(ValueError, match="short run"):
        acceptance_status(tmp_path, {"cases": [old, new, resume]}, {})


def test_lifecycle_recipe_preserves_periodic_and_resume_saving(tmp_path):
    base = dict(name="long", router_python="/pinned/router", router_policy="round_robin")
    assert recipe_for(tmp_path, base)["save"] is None
    long = recipe_for(tmp_path, dict(base, choose_best=True, steps=100, save_interval=25))
    assert long["save"] == str(tmp_path / "runs/long/checkpoint") and long["save_interval"] == 25
    resume = recipe_for(tmp_path, dict(base, resume="long", steps=102))
    assert resume["save"] and resume["save_interval"] == 10000


def test_controller_requires_unadmitted_original_parent(tmp_path):
    write_json(tmp_path / "services/request.json", {"case": "held"})
    write_json(tmp_path / "runs/held/prepared.json", {})
    require_held(tmp_path, {"held_case": "held"})
    write_json(tmp_path / "runs/held/services-ready.json", {})
    with pytest.raises(RuntimeError, match="overlap"):
        require_held(tmp_path, {"held_case": "held"})


def test_failed_recovery_controller_wakes_existing_thread(tmp_path):
    write_json(tmp_path / "manifest.json", {"cases": [], "nodes": 16})
    write_json(tmp_path / "platform/lifecycle-extension/failed-3.json", {"error": "preflight"})
    assert any(e.startswith("recovery-controller-failed:lifecycle-extension:failed-3:")
               for e in collect_events(tmp_path, 10))


@pytest.mark.parametrize("obstacle", [None, "child", "identity", "admitted", "invalid"])
def test_helper_takeover_never_interrupts_active_or_unverified_work(tmp_path, monkeypatch, obstacle):
    import psutil
    import lifecycle_extension

    plan = {
        "record": "platform/new",
        "takeover": {"record": "platform/old", "completed_case": "short", "held_case": "next"},
    }
    for rank in range(16):
        write_json(tmp_path / f"platform/old/short-clean-{rank}.json", {})
        if rank:
            write_json(tmp_path / f"platform/new/takeover-{rank}.json", {})
    write_json(tmp_path / "platform/old/controller-0.json", {"pid": 123, "command": ["old-helper"]})
    write_json(tmp_path / "runs/short/summary.json", {"valid": obstacle != "invalid"})
    write_json(tmp_path / "services/request.json", {"case": "next"})
    write_json(tmp_path / "runs/next/prepared.json", {})
    if obstacle == "admitted":
        write_json(tmp_path / "runs/next/services-ready.json", {})
    signals = []

    class Process:
        pid = 123

        def cmdline(self):
            return ["other" if obstacle == "identity" else "old-helper"]

        def children(self, recursive):
            return [1] if obstacle == "child" else []

        def create_time(self):
            return 1

        def terminate(self):
            signals.append("terminate")

        def wait(self, timeout):
            signals.append("wait")

    monkeypatch.setattr(psutil, "Process", lambda pid: Process())
    monkeypatch.setattr(lifecycle_extension, "wait_for", lambda predicate, *args: predicate() or pytest.fail("not ready"))
    if obstacle:
        with pytest.raises(RuntimeError):
            takeover_at_barrier(tmp_path, 0, plan)
        assert not signals
    else:
        takeover_at_barrier(tmp_path, 0, plan)
        assert signals == ["terminate", "wait"]
        assert (tmp_path / "platform/new/takeover-0.json").is_file()
