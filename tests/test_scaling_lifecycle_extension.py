"""Replacement lifecycle attempts preserve scope and keep old failures visible."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/prism/benchmarks/scaling"))
from analyze import acceptance_status
from lifecycle_extension import recipe_for, require_held
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
    assert any(e.startswith("recovery-controller-failed:failed-3:") for e in collect_events(tmp_path, 10))
