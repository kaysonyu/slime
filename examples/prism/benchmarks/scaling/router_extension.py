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
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from campaign import atomic_json, run_node, wait_for

HOLD_CASE = "32-b2048-a4-w16-end"
PREVIOUS_CASE = "64-b2048-a4-w16"


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
    with (record / f"rank-{rank}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(record / f"ready-{rank}.json", {"rank": rank, "pid": os.getpid(), "time": time.time()})
        if rank == 0:
            wait_for(lambda: all((record / f"ready-{i}.json").exists() for i in range(8)), "Extension helpers", 600)

            def original_waiting():
                request = json.loads((root / "services/request.json").read_text())
                return request["case"] == HOLD_CASE and (root / "runs" / PREVIOUS_CASE / "summary.json").exists()

            wait_for(original_waiting, "Original parents reach the admission hold", 6 * 3600)
            if (root / "runs" / HOLD_CASE / "services-ready.json").exists():
                raise RuntimeError("Original baseline was admitted; do not overlap GPU work")
            atomic_json(record / "claimed.json", {"time": time.time()})
        wait_for(lambda: (record / "claimed.json").exists(), "Extension takes ownership", 6 * 3600)
        address = json.loads((root / "platform" / f"node-{rank}.json").read_text())["address"]
        master = json.loads((root / "platform/head.json").read_text())["address"]
        for index, case in enumerate(extension["cases"], start=30):
            output = root / "runs" / case["name"]
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
        atomic_json(record / f"finished-{rank}.json", {"time": time.time()})
        if rank == 0:
            wait_for(
                lambda: all((record / f"finished-{i}.json").exists() for i in range(8)),
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
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.root, args.prepare)
        return
    rank = int(os.environ["PET_NODE_RANK"])
    if int(os.environ["PET_NNODES"]) != 8 or rank not in range(8):
        raise ValueError("Use only the authorized existing eight-node Job")
    record = args.root / "platform/router-extension"
    record.mkdir(exist_ok=True)
    if args.launch:
        config = json.loads((args.root / "platform/router-extension.json").read_text())
        script = Path(config["source"]) / "examples/prism/benchmarks/scaling/router_extension.py"
        command = [sys.executable, "-u", str(script), str(args.root)]
        with (record / f"rank-{rank}.log").open("a") as log:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
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
