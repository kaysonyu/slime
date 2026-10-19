"""Run a complete scaling matrix in one eight-node allocation.

All allocated nodes join barriers, including idle nodes in the four-node cases.
A separate CPU watcher applies ASR replica requests and acknowledges readiness.
"""

import argparse
import ipaddress
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from prepare import write_json


def atomic_json(path, value):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    write_json(temporary, value)
    temporary.replace(path)


def wait_for(predicate, description, seconds=3600):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError(description)
        time.sleep(2)


def resolve_case(root, candidate):
    case = dict(candidate)
    if case.get("choose_best"):
        eligible = []
        for summary in (root / "runs").glob("64-b4096-*/summary.json"):
            stats = json.loads(summary.read_text())
            if stats.get("valid") and stats.get("steady_steps", 0) >= 16:
                config = json.loads((summary.parent / "case.json").read_text())
                eligible.append((stats["step_seconds"], config))
        if not eligible:
            raise RuntimeError("No valid B4096 candidate for independent confirmation")
        _, best = min(eligible, key=lambda pair: pair[0])
        for key in ("asr_replicas", "workers", "concurrency", "asr_inflight"):
            case[key] = best[key]
        case["selected_from"] = best["name"]
    if case.get("resume"):
        parent = root / "runs" / case["resume"]
        previous = json.loads((parent / "case.json").read_text())
        summary = parent / "summary.json"
        verification = parent / "checkpoint-verification.json"
        commit = parent / "checkpoint/rollout" / f"training_commit_{previous['steps'] - 1}.json"
        if not (
            summary.exists() and json.loads(summary.read_text()).get("valid") is True
            and verification.exists() and json.loads(verification.read_text()).get("valid") is True
            and commit.is_file()
        ):
            raise RuntimeError(f"Resume prerequisite failed: {case['resume']} needs valid confirmation and checkpoint")
        for key in ("asr_replicas", "workers", "concurrency", "asr_inflight"):
            case[key] = previous[key]
    return case


def finish_campaign(root):
    from analyze import analyze

    result = analyze(root)  # Reload the manifest, including supplemental attempts.
    receipt = {
        "finished": time.time(),
        "acceptance": result["acceptance"],
        "returncode": 0 if result["acceptance"]["accepted"] else 1,
    }
    atomic_json(root / "campaign-finished.json", receipt)
    return receipt


def run_node(root, manifest, case, rank, master, address, index):
    run = root / "runs" / case["name"]
    node = run / f"node-{rank}"
    node.mkdir(parents=True, exist_ok=True)
    if rank >= case["gpus"] // 8:
        return {"rank": rank, "returncode": 0, "participating": False}
    source = Path(manifest["source"])
    scripts = source / "examples/prism/benchmarks/scaling"
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join([str(source), manifest["omni"], str(scripts), manifest["megatron"]]),
        TRAIN_PYTHON=sys.executable,
        OMNI_PYTHON=sys.executable,
        PYTHONUNBUFFERED="1",
        PYTHONNOUSERSITE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        RAY_USAGE_STATS_ENABLED="0",
        RAY_DEDUP_LOGS="0",
        RAY_USE_UVLOOP="0",
        SLIME_HOST_IP=address,
        PET_NODE_RANK=str(rank),
        PET_NNODES=str(case["gpus"] // 8),
        PET_NPROC_PER_NODE="8",
        MASTER_ADDR=master,
        MASTER_PORT=str(6385 + index),
        OUTPUT_ROOT=str(run / "lifecycle"),
        RUN_NAME="scaling",
        JOB_NAME=case["name"],
        PRISM_SCALE_RUN=str(run),
        SLIME_ROOT=str(source),
        CUDA_DEVICE_MAX_CONNECTIONS="1",
        NCCL_CUMEM_ENABLE="0",
        NCCL_DEBUG="WARN",
    )
    env["INF_API_KEY"] = Path(manifest["auth_file"]).read_text().strip()
    env["PATH"] = str(Path(sys.executable).parent) + ":" + env["PATH"]
    for name in ("NO_PROXY", "no_proxy"):
        env[name] = env.get(name, "") + ",127.0.0.1,localhost,.sii.edu.cn,.shaipower.online," + master
    command = [
        "bash",
        str(source / "examples/prism/run-grpo.sh"),
        "--hf-checkpoint",
        manifest["hf_checkpoint"],
        "--pretrained-checkpoint",
        manifest["native_checkpoint"],
        "--prompt-data",
        str(root / "data/train.jsonl"),
        "--omni-config",
        str(root / "configs/student.json"),
        "--omni-python",
        sys.executable,
        "--omni-reward-config",
        str(run / "rewards.json"),
        "--num-rollout",
        str(case["steps"]),
        "--rollout-batch-size",
        str(case["batch"] // 8),
        "--n-samples-per-prompt",
        "8",
        "--global-batch-size",
        str(case["batch"]),
        "--rollout-max-response-len",
        str(manifest["max_frames"]),
        "--rollout-execution-workers",
        str(case["workers"]),
        "--sglang-server-concurrency",
        str(case["concurrency"]),
        "--seq-length",
        "32768",
        "--max-tokens-per-gpu",
        str(manifest["max_tokens_per_gpu"]),
        "--balance-by-flops",
        "--use-distributed-optimizer",
        "--rollout-gc-threshold",
        "70000",
        "--rollout-tensor-buffer",
        "--no-offload-train",
        "--no-offload-rollout",
        "--resident-cleanup-interval",
        "16",
        "--distributed-timeout-minutes",
        "15",
        "--update-weight-buffer-size",
        "268435456",
        "--save-interval",
        "10000",
        "--custom-config-path",
        str(run / "recipe.json"),
        "--phase-trace-path",
        str(run / "phases.jsonl"),
        "--resolved-config-path",
        str(run / "resolved-args.json"),
        "--tensorboard-dir",
        str(run / "tensorboard"),
        "--rollout-data-postprocess-path",
        "hooks.audit_train",
    ]
    if case.get("resume"):
        command += [
            "--load",
            str(root / "runs" / case["resume"] / "checkpoint"),
            "--use-checkpoint-opt-param-scheduler",
        ]
    write_json(node / "command.json", command)
    started = time.time()
    with (node / "gpu.csv").open("w") as gpu, (node / "launch.log").open("w") as log:
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw",
                "--format=csv,noheader,nounits",
                "-l",
                "1",
            ],
            stdout=gpu,
        )
        try:
            result = subprocess.run(command, env=env, cwd=source, stdout=log, stderr=subprocess.STDOUT, timeout=5400)
            status = result.returncode
        except subprocess.TimeoutExpired:
            subprocess.run([sys.executable, "-m", "ray.scripts.scripts", "stop", "--force"], timeout=90)
            status = 124
        finally:
            monitor.terminate()
            monitor.wait(timeout=15)
    return {"rank": rank, "returncode": status, "participating": True, "started": started, "finished": time.time()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    manifest = json.loads((root / "manifest.json").read_text())
    rank = int(os.environ.get("PET_NODE_RANK", os.environ.get("RANK", 0)))
    nodes = int(os.environ.get("PET_NNODES", os.environ.get("WORLD_SIZE", 1)))
    if nodes != 8:
        raise ValueError(f"Expected eight allocated nodes, got {nodes}")
    from ray._private.services import get_node_ip_address

    address = get_node_ip_address()
    if ipaddress.ip_address(address).is_loopback:
        raise ValueError(f"Unroutable Ray node address: {address}")
    atomic_json(
        root / "platform" / f"node-{rank}.json",
        {"rank": rank, "host": socket.gethostname(), "address": address, "started": time.time()},
    )
    if rank == 0:
        atomic_json(root / "platform/head.json", {"address": address})
    wait_for(lambda: (root / "platform/head.json").exists(), "No head address")
    master = json.loads((root / "platform/head.json").read_text())["address"]
    for index, candidate in enumerate(manifest["cases"]):
        run = root / "runs" / candidate["name"]
        run.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            try:
                case = resolve_case(root, candidate)
                atomic_json(run / "case.json", case)
                write_json(
                    run / "rewards.json",
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
                write_json(
                    run / "recipe.json",
                    {
                        "save": str(run / "checkpoint") if case.get("choose_best") or case.get("resume") else None,
                        "save_interval": 10000 if case.get("choose_best") or case.get("resume") else None,
                        "prism_artifact_dir": str(run / "experiment"),
                        "omni_audio_dir": str(run / "audio"),
                        "rollout_sample_observer_path": ["slime_plugins.prism.artifacts.observe"],
                        "omni_audio_path_function_path": "slime_plugins.prism.artifacts.audio_path",
                        "custom_rollout_log_function_path": "slime_plugins.prism.artifacts.log_rollout_data",
                    },
                )
                atomic_json(root / "services/request.json", {"case": case["name"], "replicas": case["asr_replicas"]})
            except Exception as error:
                # A skipped prerequisite is a terminal invalid attempt, not a
                # missing result or a successful zero-update benchmark.
                atomic_json(run / "summary.json", {
                    "case": candidate["name"], "valid": False, "updates": 0,
                    "steady_steps": 0, "issues": [f"Case preparation failed: {error!r}"],
                })
                atomic_json(run / "skip.json", {"error": repr(error)})
        wait_for(lambda: (run / "case.json").exists() or (run / "skip.json").exists(), "Case not prepared")
        if (run / "skip.json").exists():
            continue
        case = json.loads((run / "case.json").read_text())
        wait_for(lambda: (run / "services-ready.json").exists(), f"ASR readiness for {case['name']}", seconds=7200)
        print(json.dumps({"event": "case_start", "case": case["name"], "rank": rank, "time": time.time()}), flush=True)
        receipt = run_node(root, manifest, case, rank, master, address, index)
        atomic_json(run / f"done-{rank}.json", receipt)
        wait_for(
            lambda: all((run / f"done-{peer}.json").exists() for peer in range(nodes)), "Node cleanup barrier", 6000
        )
        if rank == 0:
            from analyze import summarize_case

            if case.get("choose_best") or case.get("resume"):
                with (run / "checkpoint-verification.log").open("w") as log:
                    subprocess.run(
                        [sys.executable, str(Path(__file__).with_name("checkpoint.py")), str(run)],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=300,
                    )

            atomic_json(run / "summary.json", summarize_case(run))
        wait_for(lambda: (run / "summary.json").exists(), "Case analysis", 600)
    if rank == 0:
        finish_campaign(root)
    wait_for(lambda: (root / "campaign-finished.json").exists(), "Campaign acceptance", 600)
    return json.loads((root / "campaign-finished.json").read_text())["returncode"]


if __name__ == "__main__":
    sys.exit(main())
