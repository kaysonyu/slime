"""CPU-only lifecycle controller: poll the queue and satisfy ASR scale requests.

No language-model polling is involved. Only the named campaign ASR is changed.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from campaign import atomic_json


def cli(*arguments):
    result = subprocess.run(["inspire", "--json", *arguments], text=True, capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError(f"inspire {arguments[:2]} failed: {result.stderr[-500:]}")
    value = json.loads(result.stdout)
    if not value.get("success"):
        raise RuntimeError(f"inspire {arguments[:2]} returned failure")
    return value.get("data", {})


def smoke(manifest):
    token = Path(manifest["auth_file"]).read_text().strip()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(manifest["asr_endpoint"] + "/info", headers=headers)
    with opener.open(request, timeout=20) as response:
        info = json.load(response)
    if info.get("protocol") != "voxrl.scoring.v1":
        raise ValueError("Incorrect ASR protocol")
    with (Path(manifest["root"]) / "data/train.jsonl").open() as stream:
        row = json.loads(next(stream))
    audio = next(iter(row["metadata"]["omni_prompt"]["references"].values()))
    payload = {"request_id": f"scale-smoke-{time.time_ns()}", "audio": {"path": audio}}
    request = urllib.request.Request(
        manifest["asr_endpoint"] + "/score", headers=headers, data=json.dumps(payload).encode()
    )
    with opener.open(request, timeout=90) as response:
        result = json.load(response)
    if result.get("request_id") != payload["request_id"] or not isinstance(result.get("result", {}).get("text"), str):
        raise ValueError("ASR business smoke failed")
    return {"info": info, "score": result, "time": time.time()}


def prepare_case_recipe(run):
    """Validate save settings before releasing the immutable launch barrier.

    Also repairs recipes emitted by the original, already-running campaign
    driver: save=None alone does not disable its final periodic save action.
    Never rewrite a recipe after the training processes have been admitted.
    """
    if (run / "services-ready.json").exists():
        return
    path = run / "recipe.json"
    recipe = json.loads(path.read_text())
    if recipe.get("save") is None and recipe.get("save_interval", 10000) is not None:
        previous = dict(recipe)
        recipe["save_interval"] = None
        atomic_json(run / "recipe-before-save-fix.json", previous)
        atomic_json(path, recipe)
        atomic_json(
            run / "recipe-save-fix.json",
            {
                "time": time.time(),
                "change": "Disable periodic and final checkpoint when save is null",
                "before_training_admission": True,
            },
        )
    case_path = run / "case.json"
    if case_path.exists():
        case = json.loads(case_path.read_text())
        selected = case.get("selected_from") or case.get("resume")
        source_case = json.loads((run.parent / selected / "case.json").read_text()) if selected else case
        manifest_path = run.parent.parent / "manifest.json"
        if not selected and case.get("supplemental") and manifest_path.exists():
            # A live helper may hold a pre-repair case list in memory. Apply
            # only the explicitly pinned router revision, before admission.
            planned = next(
                item for item in json.loads(manifest_path.read_text())["cases"] if item["name"] == case["name"]
            )
            for key in ("gpus", "batch", "asr_replicas", "workers", "steps", "router_processes", "router_policy"):
                if planned[key] != case[key]:
                    raise RuntimeError(f"Unexpected non-router launch-plan change: {key}")
            source_case = planned
        if selected and manifest_path.exists():
            diagnostic = json.loads(manifest_path.read_text()).get("router_diagnostic_override", {})
            if source_case.get("router_processes", 1) > 1 and source_case.get(
                "router_source_commit"
            ) in diagnostic.get("parent_commits", []):
                source_case = dict(source_case)
                for key in ("router_python", "router_entry_sha256", "router_source_commit"):
                    source_case[key] = diagnostic[key]
                case["router_diagnostic_parent_commit"] = json.loads(
                    (run.parent / selected / "case.json").read_text()
                )["router_source_commit"]
        if "router_python" in source_case:
            adapter = Path(source_case["router_python"])
            entry = adapter.parents[4] / "slime/backends/sglang_utils/omni_router.py"
            if hashlib.sha256(entry.read_bytes()).hexdigest() != source_case["router_entry_sha256"]:
                raise RuntimeError("Pinned router entry changed")
            for key in (
                "router_python",
                "router_processes",
                "router_policy",
                "router_entry_sha256",
                "router_source_commit",
            ):
                case[key] = source_case[key]
            recipe.update(omni_python=case["router_python"], omni_router_policy=case["router_policy"])
            atomic_json(path, recipe)
            atomic_json(case_path, case)
            atomic_json(
                run / "router-variant-admission.json",
                {
                    "time": time.time(),
                    "selected_from": selected,
                    "router_processes": case["router_processes"],
                    "router_source_commit": case["router_source_commit"],
                    "before_training_admission": True,
                },
            )


def admission_held(root, name):
    # A scoped, expiring diagnostic hold pauses admission, not observation.
    # Already admitted runs continue on their immutable recipes.
    manual = root / "platform/admission-hold.json"
    if manual.exists():
        hold = json.loads(manual.read_text())
        if name in hold["cases"] and time.time() < hold["expires"]:
            return True
    extension = root / "platform/router-extension.json"
    if not extension.exists():
        return False
    configuration = json.loads(extension.read_text())
    return name == configuration["hold_case"] and not (root / "platform/router-extension/finished.json").exists()


def snapshot_service_telemetry(root, manifest, next_case):
    """Keep completed-case ASR evidence before scale-up reuses pod log paths."""
    names = [case["name"] for case in manifest["cases"]]
    index = names.index(next_case)
    if index == 0:
        return
    previous = names[index - 1]
    output = root / "services/history" / previous
    if (output / "snapshot.json").exists():
        return
    if not (root / "runs" / previous / "summary.json").exists():
        raise RuntimeError(f"Cannot snapshot unfinished case: {previous}")
    copied = []
    for pod in (root / "services/asr").iterdir():
        if not pod.is_dir():
            continue
        for name in ("gpu.csv", "service.log", "launch.json"):
            source = pod / name
            if source.is_file():
                target = output / pod.name / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied.append(str(target.relative_to(output)))
    atomic_json(
        output / "snapshot.json",
        {
            "time": time.time(),
            "completed_case": previous,
            "before_case": next_case,
            "files": copied,
            "method": "Completed case prefix; filter samples by its steady-state wall-clock interval.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    root = args.root
    manifest = json.loads((root / "manifest.json").read_text())
    workspace, service = manifest["workspace"], manifest["asr_name"]
    started = time.monotonic()
    acknowledged = None
    initial_smoke = (root / "services/initial-smoke.json").exists()
    while True:
        try:
            updated = json.loads((root / "manifest.json").read_text())
            if any(updated[key] != manifest[key] for key in ("workspace", "job_name", "asr_name", "root")):
                raise RuntimeError("Campaign identity changed during observation")
            manifest = updated
            job = cli("job", "status", manifest["job_name"], "--workspace", workspace)
            serving = cli("serving", "status", service, "--workspace", workspace)
            status = job.get("status", "UNKNOWN")
            atomic_json(root / "platform/monitor-state.json", {"time": time.time(), "job": job, "serving": serving})
            if (root / "campaign-finished.json").exists() or status in {
                "SUCCEEDED",
                "FAILED",
                "CANCELLED",
                "SUCCEEDED_RETAINING",
                "FAILED_RETAINING",
            }:
                if serving.get("status") != "STOPPED":
                    cli("serving", "stop", service, "--workspace", workspace)
                from analyze import analyze

                analyze(root)
                atomic_json(root / "platform/monitor-finished.json", {"time": time.time(), "job_status": status})
                return
            if not initial_smoke and serving.get("status") == "RUNNING":
                atomic_json(root / "services/initial-smoke.json", smoke(manifest))
                initial_smoke = True
            if status == "RUNNING":
                if serving.get("status") == "STOPPED":
                    cli("serving", "start", service, "--workspace", workspace)
                    acknowledged = None
                    # Starting is asynchronous; scale is accepted only after RUNNING.
                    time.sleep(args.interval)
                    continue
                request_file = root / "services/request.json"
                if request_file.exists():
                    request = json.loads(request_file.read_text())
                    if request["case"] != acknowledged:
                        if admission_held(root, request["case"]):
                            print(json.dumps({"time": time.time(), "admission_held": request["case"]}), flush=True)
                            time.sleep(args.interval)
                            continue
                        prepare_case_recipe(root / "runs" / request["case"])
                        snapshot_service_telemetry(root, manifest, request["case"])
                        replicas = request["replicas"]
                        if replicas not in manifest.get("approved_asr_replicas", [4, 8, 16]):
                            raise ValueError("ASR request exceeds approved matrix")
                        if serving.get("status") != "RUNNING":
                            time.sleep(args.interval)
                            continue
                        if serving.get("replicas") != replicas:
                            cli("serving", "scale", service, "--workspace", workspace, "--replicas", str(replicas))
                        instances = cli("serving", "instances", service, "--workspace", workspace, "--all").get(
                            "items", []
                        )
                        if len(instances) == replicas and all(
                            item["status"] == "INSTANCE_RUNNING" for item in instances
                        ):
                            receipt = smoke(manifest)
                            receipt.update(case=request["case"], replicas=replicas, instances=instances)
                            run = root / "runs" / request["case"]
                            # A controller restart may recheck an already admitted
                            # case. Preserve its original admission evidence.
                            target = (
                                "services-monitor-smoke-latest.json"
                                if (run / "services-ready.json").exists()
                                else "services-ready.json"
                            )
                            atomic_json(run / target, receipt)
                            acknowledged = request["case"]
            elif initial_smoke and time.monotonic() - started > 900 and serving.get("status") == "RUNNING":
                # Retain the tested deployment configuration while waiting for all nodes.
                cli("serving", "stop", service, "--workspace", workspace)
            print(
                json.dumps(
                    {
                        "time": time.time(),
                        "job": status,
                        "serving": serving.get("status"),
                        "acknowledged": acknowledged,
                    }
                ),
                flush=True,
            )
        except Exception as error:
            atomic_json(root / "platform/monitor-error.json", {"time": time.time(), "error": str(error)})
            print(json.dumps({"time": time.time(), "error": str(error)}), flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
