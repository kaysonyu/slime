"""Insert router controls before the final baseline in an existing allocation.

The original eight campaign parents wait at their ASR admission barrier while
these eight helpers use the same nodes. No new Job, Ray service, or GPU quota.
Frozen original learner and Omni sources remain untouched. Only the standalone
Slime router entry is replaced through a pinned interpreter adapter.
"""

import argparse
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from datetime import timedelta

from campaign import atomic_json, run_node, wait_for

HOLD_CASE = "32-b2048-a4-w16-end"
PREVIOUS_CASE = "64-b2048-a4-w16"


def assert_idle(root, cases):
    """Never replace helpers while any admitted GPU case is unfinished."""
    for case in cases:
        output = root / "runs" / case["name"]
        if (output / "services-ready.json").exists() and not (output / "summary.json").exists():
            raise RuntimeError(f"Admitted case still running: {case['name']}")
        if (output / "summary.json").exists() and not all((output / f"done-{i}.json").exists() for i in range(8)):
            raise RuntimeError(f"Cleanup incomplete: {case['name']}")
    if (root / "runs" / HOLD_CASE / "services-ready.json").exists():
        raise RuntimeError("Original parent was admitted; do not overlap GPU work")


def inherited_job_environment(root, rank):
    """An interactive Job shell does NOT inherit the submitted Job environment."""
    import psutil

    manifest = json.loads((root / "manifest.json").read_text())
    original = str(Path(manifest["source"]) / "examples/prism/benchmarks/scaling/campaign.py")
    parents = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        command = process.info["cmdline"] or []
        if original in command and str(root) in command:
            parents.append(process)
    if len(parents) != 1:
        raise RuntimeError(f"Expected one original campaign parent, found {len(parents)}")
    parent = parents[0]
    environment = parent.environ()
    if environment.get("PET_NODE_RANK") != str(rank) or environment.get("PET_NNODES") != "8":
        raise RuntimeError("Original Job environment belongs to a different node/allocation")
    for key in ("GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME", "NCCL_IB_HCA"):
        if not environment.get(key):
            raise RuntimeError(f"Original Job is missing {key}")
    interface = environment["GLOO_SOCKET_IFNAME"]
    addresses = [item.address for item in psutil.net_if_addrs().get(interface, []) if item.family == socket.AF_INET]
    address = json.loads((root / "platform" / f"node-{rank}.json").read_text())["address"]
    if address not in addresses or address.startswith("127."):
        raise RuntimeError("Gloo interface does not own this allocation's recorded routable address")
    # Keep the complete inherited environment only in memory; persist no credentials.
    receipt = {
        "parent_pid": parent.pid,
        "rank": rank,
        "address": address,
        "time": time.time(),
        "network": {key: value for key, value in environment.items() if key.startswith(("NCCL_", "GLOO_"))},
    }
    return environment, receipt


def preflight(root, rank, generation):
    """Eight CPU-only ranks validate actual Gloo collectives before admission."""
    import torch
    import torch.distributed as dist

    master = json.loads((root / "platform/head.json").read_text())["address"]
    dist.init_process_group(
        "gloo", init_method=f"tcp://{master}:6490", rank=rank, world_size=8, timeout=timedelta(seconds=120)
    )
    try:
        value = torch.tensor([rank], dtype=torch.int64)
        dist.all_reduce(value)
        if value.item() != 28:
            raise RuntimeError("Cross-node Gloo sum differs from 0+...+7")
        dist.barrier()
    finally:
        dist.destroy_process_group()
    atomic_json(
        root / "platform/router-extension" / f"gloo-{rank}.json",
        {"rank": rank, "generation": generation, "sum": 28, "time": time.time()},
    )


def retry_plan(root, cases):
    """Retry only failed leaves; keep every previous attempt and pinned router."""
    retried = {case.get("retry_of") for case in cases}
    completed, pending, retries = [], [], []
    for case in cases:
        summary = root / "runs" / case["name"] / "summary.json"
        if summary.exists():
            completed.append(case)
            result = json.loads(summary.read_text())
            if not result["valid"] and case["name"] not in retried:
                name = case["name"] + "-retry"
                if (root / "runs" / name).exists() or any(item["name"] == name for item in cases):
                    raise FileExistsError(f"Refuse overwriting retry {name}")
                retries.append(dict(case, name=name, retry_of=case["name"]))
        else:
            pending.append(case)
    return completed + retries + pending, retries


def matched_controls(root, cases):
    """Spend the remaining allocation on controls for the completed fast router."""
    name = "64-b4096-a8-w32-r8rr"
    base = next(case for case in cases if case["name"] == name)
    result = json.loads((root / "runs" / name / "summary.json").read_text())
    if not result["valid"] or result.get("steady_steps", 0) < 21:
        raise RuntimeError("Matched controls require a complete valid eight-process result")
    obsolete = next(case for case in cases if case["name"] == "64-b4096-a16-w32-r4rr")
    if (root / "runs" / obsolete["name"] / "services-ready.json").exists():
        raise RuntimeError("Cannot supersede an admitted experiment")
    additions = [
        dict(base, name="64-b4096-a16-w32-r8rr", asr_replicas=16, matched_control_of=name),
        dict(base, name="32-b4096-a8-w32-r8rr", gpus=32, matched_control_of=name),
    ]
    for case in additions:
        if (root / "runs" / case["name"]).exists() or any(item["name"] == case["name"] for item in cases):
            raise FileExistsError(case["name"])
    return [case for case in cases if case["name"] != obsolete["name"]] + additions, obsolete


def repair(root, checkout, *, add_matched_controls=False):
    config_path = root / "platform/router-extension.json"
    config = json.loads(config_path.read_text())
    assert_idle(root, config["cases"])
    cases, retries = retry_plan(root, config["cases"])
    if add_matched_controls:
        cases, superseded = matched_controls(root, cases)
        config.setdefault("superseded_cases", []).append(
            {"case": superseded, "reason": "Unadmitted ASR16 control moved to the completed faster eight-process router"}
        )
    if subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Commit the repair before pinning it")
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    destination = root / "source" / f"router-extension-{revision[:12]}"
    destination.mkdir()
    archive = destination / "source.tar"
    subprocess.run(["git", "-C", str(checkout), "archive", "--format=tar", "-o", str(archive), revision], check=True)
    with tarfile.open(archive) as bundle:
        bundle.extractall(destination, filter="data")
    manifest = json.loads((root / "manifest.json").read_text())
    old_names = {case["name"] for case in config["cases"]}
    index = next(i for i, case in enumerate(manifest["cases"]) if case["name"] in old_names)
    manifest["cases"] = [case for case in manifest["cases"] if case["name"] not in old_names]
    manifest["cases"][index:index] = cases
    stamp = str(int(time.time()))
    atomic_json(root / "platform" / f"router-extension-before-repair-{stamp}.json", config)
    atomic_json(
        root / "platform" / f"manifest-before-repair-{stamp}.json", json.loads((root / "manifest.json").read_text())
    )
    config.update(source=str(destination), commit=revision, generation=revision, cases=cases, repaired=time.time())
    manifest["router_extension"].update(helper_source=str(destination), helper_commit=revision)
    atomic_json(root / "manifest.json", manifest)
    atomic_json(config_path, config)
    print(json.dumps({"helper_commit": revision, "retries": [case["name"] for case in retries]}))


def pin_router(root, checkout, names, *, diagnostics_only=False):
    """Pin a tested router-only repair for cases not yet admitted."""
    manifest = json.loads((root / "manifest.json").read_text())
    config = json.loads((root / "platform/router-extension.json").read_text())
    cases = [case for case in manifest["cases"] if case["name"] in names]
    if {case["name"] for case in cases} != set(names):
        raise ValueError("Unknown supplemental case")
    for case in cases:
        if not case.get("supplemental") or case["router_processes"] < 2:
            raise ValueError("Only supplemental multiprocess cases may be repinned")
        if (root / "runs" / case["name"] / "services-ready.json").exists():
            raise RuntimeError(f"Case already admitted: {case['name']}")
    if subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Commit the router repair before pinning it")
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    destination = root / "source" / f"router-repair-{revision[:12]}"
    destination.mkdir()
    archive = destination / "source.tar"
    subprocess.run(["git", "-C", str(checkout), "archive", "--format=tar", "-o", str(archive), revision], check=True)
    with tarfile.open(archive) as bundle:
        bundle.extractall(destination, filter="data")
    adapter = destination / "examples/prism/benchmarks/scaling/router_python.py"
    adapter.chmod(0o755)
    entry = destination / "slime/backends/sglang_utils/omni_router.py"
    changes = {
        "router_python": str(adapter),
        "router_entry_sha256": hashlib.sha256(entry.read_bytes()).hexdigest(),
        "router_source_commit": revision,
    }
    if diagnostics_only:
        manifest["router_diagnostic_override"] = {
            "parent_commits": sorted({case["router_source_commit"] for case in cases}),
            "scope": "Error-only transport diagnostics; no retries, pool, timeout, routing or publication changes",
            **changes,
        }
    atomic_json(root / "platform" / f"manifest-before-router-repair-{revision[:12]}.json", manifest)
    for collection in (manifest["cases"], config["cases"]):
        for case in collection:
            if case["name"] in names:
                case.update(changes)
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "platform/router-extension.json", config)
    atomic_json(root / "platform/router-repair.json", {"time": time.time(), "cases": names, **changes})
    print(json.dumps({"cases": names, **changes}))


def prepare(root, checkout):
    if (root / "platform/router-extension.json").exists():
        raise FileExistsError("Router extension already prepared")
    if (root / "runs" / HOLD_CASE / "case.json").exists():
        raise RuntimeError("Prepare the admission hold before the original parent reaches it")
    if subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Commit the extension before pinning it")
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    destination = root / "source" / f"router-extension-{revision[:12]}"
    destination.mkdir()
    archive = destination / "source.tar"
    subprocess.run(["git", "-C", str(checkout), "archive", "--format=tar", "-o", str(archive), revision], check=True)
    with tarfile.open(archive) as bundle:
        bundle.extractall(destination, filter="data")
    adapter = destination / "examples/prism/benchmarks/scaling/router_python.py"
    adapter.chmod(0o755)
    entry = destination / "slime/backends/sglang_utils/omni_router.py"
    cases = []
    for replicas, processes in ((8, 1), (8, 4), (8, 8), (16, 4)):
        cases.append(
            dict(
                name=f"64-b4096-a{replicas}-w32-r{processes}rr",
                gpus=64,
                batch=4096,
                asr_replicas=replicas,
                asr_inflight=4,
                workers=32,
                concurrency=16,
                steps=25,
                router_processes=processes,
                router_policy="round_robin",
                router_python=str(adapter),
                router_entry_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),
                router_source_commit=revision,
                supplemental=True,
            )
        )
    manifest = json.loads((root / "manifest.json").read_text())
    atomic_json(root / "platform/manifest-before-router-extension.json", manifest)
    index = next(i for i, case in enumerate(manifest["cases"]) if case["name"] == HOLD_CASE)
    manifest["cases"][index:index] = cases
    manifest["router_extension"] = {
        "source": str(destination),
        "commit": revision,
        "hold_case": HOLD_CASE,
        "original_parent_keeps_initial_case_list": True,
        "scope": "Only standalone router entry changed; original learner and Omni source stay pinned.",
    }
    atomic_json(root / "manifest.json", manifest)
    atomic_json(
        root / "platform/router-extension.json",
        dict(
            prepared=time.time(),
            hold_case=HOLD_CASE,
            previous_case=PREVIOUS_CASE,
            source=str(destination),
            commit=revision,
            cases=cases,
        ),
    )
    print(json.dumps(manifest["router_extension"]))


def run(root, rank):
    extension = json.loads((root / "platform/router-extension.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    record = root / "platform/router-extension"
    assert_idle(root, extension["cases"])
    with (record / f"rank-{rank}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        generation = extension.get("generation", extension["commit"])
        atomic_json(
            record / f"ready-{rank}.json",
            {"rank": rank, "pid": os.getpid(), "time": time.time(), "generation": generation},
        )

        def all_current(prefix):
            return all(
                (record / f"{prefix}-{i}.json").exists()
                and json.loads((record / f"{prefix}-{i}.json").read_text()).get("generation") == generation
                for i in range(8)
            )

        wait_for(lambda: all_current("ready"), "Current generation of eight helpers", 600)
        preflight(root, rank, generation)
        wait_for(lambda: all_current("gloo"), "All eight Gloo preflight results", 180)
        if rank == 0:

            def original_waiting():
                request = json.loads((root / "services/request.json").read_text())
                return request["case"] == HOLD_CASE and (root / "runs" / PREVIOUS_CASE / "summary.json").exists()

            if not (record / "claimed.json").exists():
                wait_for(original_waiting, "Original parents reach the admission hold", 6 * 3600)
            if (root / "runs" / HOLD_CASE / "services-ready.json").exists():
                raise RuntimeError("Original baseline was admitted; do not overlap GPU work")
            if not (record / "claimed.json").exists():
                atomic_json(record / "claimed.json", {"time": time.time()})
        wait_for(lambda: (record / "claimed.json").exists(), "Extension takes ownership", 6 * 3600)
        address = json.loads((root / "platform" / f"node-{rank}.json").read_text())["address"]
        master = json.loads((root / "platform/head.json").read_text())["address"]
        for index, case in enumerate(extension["cases"], start=30):
            output = root / "runs" / case["name"]
            if (output / "summary.json").exists():
                continue
            if rank == 0:
                atomic_json(output / "case.json", case)
                atomic_json(
                    output / "recipe.json",
                    {
                        "save": None,
                        "save_interval": None,
                        "prism_artifact_dir": str(output / "experiment"),
                        "omni_audio_dir": str(output / "audio"),
                        "rollout_sample_observer_path": ["slime_plugins.prism.artifacts.observe"],
                        "omni_audio_path_function_path": "slime_plugins.prism.artifacts.audio_path",
                        "custom_rollout_log_function_path": "slime_plugins.prism.artifacts.log_rollout_data",
                        "omni_python": case["router_python"],
                        "omni_router_policy": case["router_policy"],
                    },
                )
                atomic_json(
                    output / "rewards.json",
                    {
                        "services": {
                            "wer": {
                                "endpoint": manifest["asr_endpoint"],
                                "auth_token_env": "INF_API_KEY",
                                "audio_transport": "path",
                                "max_inflight": case["asr_inflight"],
                                "timeout_s": 300,
                                "max_retries": 2,
                            }
                        },
                        "weights": {"wer": 1.0},
                    },
                )
                atomic_json(root / "services/request.json", {"case": case["name"], "replicas": case["asr_replicas"]})
            wait_for(lambda: (output / "services-ready.json").exists(), "Supplemental ASR readiness", 1800)
            try:
                receipt = run_node(root, manifest, case, rank, master, address, index)
            except Exception as error:
                receipt = {"rank": rank, "returncode": 1, "participating": True, "error": repr(error)}
            atomic_json(output / f"done-{rank}.json", receipt)
            wait_for(lambda: all((output / f"done-{i}.json").exists() for i in range(8)), "Supplemental cleanup", 6000)
            if rank == 0:
                from analyze import summarize_case

                atomic_json(output / "summary.json", summarize_case(output))
            wait_for(lambda: (output / "summary.json").exists(), "Supplemental summary", 600)
        atomic_json(record / f"finished-{rank}.json", {"time": time.time(), "generation": generation})
        if rank == 0:
            wait_for(
                lambda: all_current("finished"),
                "All extension helpers finish",
                120,
            )
            atomic_json(record / "finished.json", {"time": time.time()})
            held = json.loads((root / "runs" / HOLD_CASE / "case.json").read_text())
            atomic_json(root / "services/request.json", {"case": HOLD_CASE, "replicas": held["asr_replicas"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--prepare", type=Path, metavar="CHECKOUT")
    parser.add_argument("--repair", type=Path, metavar="CHECKOUT")
    parser.add_argument("--matched-controls", action="store_true")
    parser.add_argument("--pin-router", type=Path, metavar="CHECKOUT")
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--replace-idle", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.root, args.prepare)
        return
    if args.repair:
        repair(args.root, args.repair, add_matched_controls=args.matched_controls)
        return
    if args.pin_router:
        if not args.cases:
            parser.error("--pin-router requires --cases")
        pin_router(args.root, args.pin_router, args.cases, diagnostics_only=args.diagnostics_only)
        return
    rank = int(os.environ["PET_NODE_RANK"])
    if int(os.environ["PET_NNODES"]) != 8 or rank not in range(8):
        raise ValueError("Use only the authorized existing eight-node Job")
    record = args.root / "platform/router-extension"
    record.mkdir(exist_ok=True)
    if args.launch:
        import psutil

        config = json.loads((args.root / "platform/router-extension.json").read_text())
        environment, network = inherited_job_environment(args.root, rank)
        if args.replace_idle:
            assert_idle(args.root, config["cases"])
            receipt_path = record / f"controller-{rank}.json"
            previous = json.loads(receipt_path.read_text())
            process = psutil.Process(previous["pid"])
            if process.cmdline() != previous["command"] or process.children(recursive=True):
                raise RuntimeError("Old helper identity mismatch or children still active")
            if (
                str(Path(config["source"]) / "examples/prism/benchmarks/scaling/router_extension.py")
                in previous["command"]
            ):
                raise RuntimeError("This helper generation is already running")
            process.terminate()
            process.wait(timeout=15)
            atomic_json(record / f"controller-{rank}-before-{config['commit'][:12]}.json", previous)
        script = Path(config["source"]) / "examples/prism/benchmarks/scaling/router_extension.py"
        command = [sys.executable, "-u", str(script), str(args.root)]
        with (record / f"rank-{rank}.log").open("a") as log:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=environment,
            )
        atomic_json(record / f"network-{rank}.json", network)
        atomic_json(record / f"controller-{rank}.json", {"pid": process.pid, "command": command})
        print(json.dumps({"rank": rank, "pid": process.pid}))
    else:
        try:
            run(args.root, rank)
        except Exception as error:
            atomic_json(record / f"failed-{rank}.json", {"error": repr(error), "time": time.time()})
            raise


if __name__ == "__main__":
    main()
