"""Run repaired lifecycle tests inside the existing, explicitly held allocation.

The original per-node campaign parents are suspended only at a verified idle
admission barrier. They are not killed, and no original result is overwritten.
The bounded replacement controllers use the same GPU Job/environment. After
business verification the operator stops the Job, rather than resuming obsolete
parents whose admission deadlines have elapsed.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from campaign import atomic_json, resolve_case, run_node, wait_for
from router_extension import inherited_job_environment


def require_held(root, plan):
    held = root / "runs" / plan["held_case"]
    request = json.loads((root / "services/request.json").read_text())
    if request["case"] != plan["held_case"] or not (held / "prepared.json").exists():
        raise RuntimeError("Original parents are not at the expected prepared barrier")
    if (held / "services-ready.json").exists():
        raise RuntimeError("Never overlap an admitted original GPU case")


def recipe_for(root, case):
    run = root / "runs" / case["name"]
    checkpoint = bool(case.get("choose_best") or case.get("resume"))
    return {
        "save": str(run / "checkpoint") if checkpoint else None,
        "save_interval": case.get("save_interval", 10000) if checkpoint else None,
        "prism_artifact_dir": str(run / "experiment"),
        "omni_audio_dir": str(run / "audio"),
        "rollout_sample_observer_path": ["slime_plugins.prism.artifacts.observe"],
        "omni_audio_path_function_path": "slime_plugins.prism.artifacts.audio_path",
        "custom_rollout_log_function_path": "slime_plugins.prism.artifacts.log_rollout_data",
        "omni_python": case["router_python"],
        "omni_router_policy": case["router_policy"],
    }


def audit_cleanup(root, case, script, output):
    with output.open("w") as log:
        subprocess.run(
            [sys.executable, str(script), str(root), case, "--apply"],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=60,
            check=True,
        )


def run(root, rank, plan):
    from analyze import summarize_case

    record = root / plan.get("record", "platform/lifecycle-extension")
    manifest = json.loads((root / "manifest.json").read_text())
    nodes = manifest["nodes"]
    source = Path(plan["source"])
    runtime = dict(manifest, source=str(source))
    scripts = source / "examples/prism/benchmarks/scaling"
    control_scripts = Path(__file__).resolve().parent
    atomic_json(record / f"ready-{rank}.json", {"time": time.time(), "rank": rank, "commit": plan["commit"]})
    wait_for(lambda: all((record / f"ready-{i}.json").exists() for i in range(nodes)), "All recovery ranks", 600)
    # The failed case is complete on all nodes; exact process provenance is
    # checked by the existing reaper before it sends any signals.
    audit_cleanup(root, plan["cleanup_case"], scripts / "reap_orphans.py", record / f"initial-cleanup-{rank}.log")
    atomic_json(record / f"clean-{rank}.json", {"time": time.time(), "rank": rank})
    wait_for(lambda: all((record / f"clean-{i}.json").exists() for i in range(nodes)), "Initial cleanup", 180)
    address = json.loads((root / "platform" / f"node-{rank}.json").read_text())["address"]
    master = json.loads((root / "platform/head.json").read_text())["address"]
    failed = None
    for index, candidate in enumerate(plan["cases"], start=60):
        case = dict(candidate, allocated_nodes=nodes, runtime_source=str(source), runtime_commit=plan["commit"])
        output = root / "runs" / case["name"]
        if (output / "summary.json").exists():
            raise FileExistsError(f"Recovery case already exists: {case['name']}")
        if rank == 0:
            if case.get("resume"):
                case = resolve_case(root, case)
            elif case.get("selected_from"):
                control = root / "runs" / case["selected_from"]
                stats = json.loads((control / "summary.json").read_text())
                config = json.loads((control / "case.json").read_text())
                if not stats["valid"] or stats.get("steady_steps", 0) < 21:
                    raise RuntimeError("Confirmation requires a valid repaired short control")
                for key in ("gpus", "batch", "workers", "concurrency", "asr_replicas", "asr_inflight"):
                    if case[key] != config[key]:
                        raise RuntimeError(f"Confirmation workload differs: {key}")
            atomic_json(output / "case.json", case)
            atomic_json(output / "recipe.json", recipe_for(root, case))
            atomic_json(
                output / "rewards.json",
                {
                    "services": {"wer": {
                        "endpoint": manifest["asr_endpoint"], "auth_token_env": "INF_API_KEY",
                        "audio_transport": "path", "max_inflight": case["asr_inflight"],
                        "timeout_s": 300, "max_retries": 2,
                    }},
                    "weights": {"wer": 1.0},
                },
            )
            atomic_json(output / "prepared.json", {"time": time.time()})
            atomic_json(root / "services/request.json", {"case": case["name"], "replicas": case["asr_replicas"]})
            if index == 60 and plan.get("takeover"):
                if case["name"] != plan["takeover"].get("admit_case", plan["takeover"]["held_case"]):
                    raise RuntimeError("Takeover must start at its explicitly planned admission case")
                hold_path = root / "platform/admission-hold.json"
                hold = json.loads(hold_path.read_text())
                if case["name"] not in hold["cases"] or time.time() >= hold["expires"]:
                    raise RuntimeError("Takeover admission hold was not maintained")
                hold["cases"].remove(case["name"])
                atomic_json(record / "admission-release.json", {
                    "time": time.time(), "case": case["name"], "all_takeovers": nodes,
                    "all_cleanup": nodes, "preserved_holds": hold["cases"],
                })
                atomic_json(hold_path, hold)
        wait_for(lambda: (output / "prepared.json").exists(), "Recovery case prepared", 180)
        case = json.loads((output / "case.json").read_text())
        wait_for(lambda: (output / "services-ready.json").exists(), "Recovery ASR ready", 3600)
        try:
            receipt = run_node(root, runtime, case, rank, master, address, index)
        except Exception as error:
            receipt = {"rank": rank, "returncode": 1, "error": repr(error)}
        atomic_json(output / f"done-{rank}.json", receipt)
        wait_for(lambda: all((output / f"done-{i}.json").exists() for i in range(nodes)), "Recovery done", 6000)
        if rank == 0:
            if case.get("choose_best") or case.get("resume"):
                with (output / "checkpoint-verification.log").open("w") as log:
                    subprocess.run(
                        [sys.executable, str(control_scripts / "checkpoint.py"), str(output)],
                        stdout=log, stderr=subprocess.STDOUT, timeout=300,
                    )
            atomic_json(output / "summary.json", summarize_case(output))
        wait_for(lambda: (output / "summary.json").exists(), "Recovery summary", 600)
        audit_cleanup(root, case["name"], scripts / "reap_orphans.py", record / f"{case['name']}-cleanup-{rank}.log")
        atomic_json(record / f"{case['name']}-clean-{rank}.json", {"time": time.time(), "rank": rank})
        wait_for(
            lambda: all((record / f"{case['name']}-clean-{i}.json").exists() for i in range(nodes)),
            "Recovery cleanup barrier", 180,
        )
        if not json.loads((output / "summary.json").read_text())["valid"]:
            failed = case["name"]
            break  # Preserve allocation, await same-thread diagnostic trigger.
    atomic_json(record / f"finished-{rank}.json", {"time": time.time(), "failed": failed})
    if rank == 0:
        wait_for(lambda: all((record / f"finished-{i}.json").exists() for i in range(nodes)), "Recovery finished", 180)
        atomic_json(record / "finished.json", {"time": time.time(), "failed": failed, "business_acceptance": False})


def takeover_at_barrier(root, rank, plan):
    """Replace only the previous idle helper, after its full cleanup barrier."""
    import psutil

    takeover = plan["takeover"]
    old_record = root / takeover["record"]
    completed = takeover["completed_case"]
    held = takeover["held_case"]
    record = root / plan["record"]

    def at_barrier():
        if not all((old_record / f"{completed}-clean-{i}.json").exists() for i in range(16)):
            return False
        summary = json.loads((root / "runs" / completed / "summary.json").read_text())
        if not summary["valid"]:
            raise RuntimeError("Cannot advance from failed short run")
        request = json.loads((root / "services/request.json").read_text())
        return request["case"] == held and (root / "runs" / held / "prepared.json").exists()

    wait_for(at_barrier, "Previous helper reached clean admission barrier", 3600)
    require_held(root, {"held_case": held})
    receipt = json.loads((old_record / f"controller-{rank}.json").read_text())
    process = psutil.Process(receipt["pid"])
    if process.cmdline() != receipt["command"] or process.children(recursive=True):
        raise RuntimeError("Previous helper identity/idle check failed; not signalling")
    identity = {"pid": process.pid, "created": process.create_time(), "command": process.cmdline()}
    process.terminate()
    process.wait(timeout=15)
    atomic_json(record / f"takeover-{rank}.json", {
        "time": time.time(), "previous": identity, "held_case": held,
        "completed_case": completed, "no_children": True,
    })
    wait_for(
        lambda: all((record / f"takeover-{i}.json").exists() for i in range(16)),
        "All helper handoffs", 180,
    )


def main():
    import psutil

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--plan", default="platform/lifecycle-extension.json")
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / args.plan).read_text())
    record = root / plan.get("record", "platform/lifecycle-extension")
    record.mkdir(exist_ok=True)
    if args.launch:
        if not plan.get("takeover"):
            require_held(root, plan)
        ranks = [
            json.loads(path.read_text())["rank"] for path in (root / "platform").glob("node-*.json")
            if json.loads(path.read_text())["host"] == socket.gethostname()
        ]
        if len(ranks) != 1:
            raise RuntimeError("Not one of the existing allocation's nodes")
        rank = ranks[0]
        if (record / f"controller-{rank}.json").exists():
            raise FileExistsError("Recovery controller already launched")
        environment, provenance = inherited_job_environment(root, rank)
        parent = psutil.Process(provenance["parent_pid"])
        if parent.children(recursive=True):
            raise RuntimeError("Original parent still has subprocesses")
        provenance.update(parent_created=parent.create_time(), before_status=parent.status())
        atomic_json(record / f"parent-{rank}.json", provenance)
        parent.suspend()
        wait_for(lambda: parent.status() == psutil.STATUS_STOPPED, "Original parent suspension", 10)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), str(root), "--plan", args.plan]
        with (record / f"rank-{rank}.log").open("a") as log:
            process = subprocess.Popen(
                command, env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                start_new_session=True,
            )
        atomic_json(record / f"controller-{rank}.json", {
            "pid": process.pid, "command": command, "time": time.time(), "parent_suspended": parent.pid,
        })
        print(json.dumps({"rank": rank, "pid": process.pid, "parent_suspended": parent.pid}))
    else:
        rank = int(os.environ["PET_NODE_RANK"])
        try:
            if plan.get("takeover"):
                takeover_at_barrier(root, rank, plan)
            run(root, rank, plan)
        except Exception as error:
            atomic_json(record / f"failed-{rank}.json", {"time": time.time(), "error": repr(error)})
            raise


if __name__ == "__main__":
    main()
