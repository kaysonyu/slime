"""Launch one pinned multi-node Prism run from a JSON manifest."""

import argparse
import ipaddress
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    required = {
        "source",
        "source_commit",
        "omni_source",
        "omni_source_commit",
        "auth_token_file",
        "hf_checkpoint",
        "pretrained_checkpoint",
        "prompt_data",
        "student_config",
        "output",
        "ready_file",
        "nodes",
        "mode",
    }
    if not isinstance(manifest, dict) or required - manifest.keys():
        raise ValueError(f"Run manifest requires {sorted(required)}")
    if manifest["mode"] not in {"grpo", "mopd"}:
        raise ValueError("Run mode must be grpo or mopd")
    for name, default in (
        ("nodes", 1),
        ("gpus_per_node", 8),
        ("batch", 2048),
        ("rollouts", 18),
        ("save_interval", 10000),
        ("concurrency", 16),
    ):
        value = manifest.setdefault(name, default)
        if type(value) is not int or value < 1:
            raise ValueError(f"Run {name} must be a positive integer")
    if manifest["nodes"] > 1 and manifest["gpus_per_node"] != 8:
        raise ValueError("Multi-node GPU jobs require full eight-GPU nodes")
    if manifest["batch"] % (manifest["nodes"] * manifest["gpus_per_node"]):
        raise ValueError("Batch must be divisible by the data parallel size")
    for name in ("source", "omni_source", "hf_checkpoint", "pretrained_checkpoint", "output", "ready_file"):
        if not Path(manifest[name]).is_absolute():
            raise ValueError(f"Run {name} must be an absolute shared path")
    for name in (
        "source",
        "omni_source",
        "hf_checkpoint",
        "pretrained_checkpoint",
        "auth_token_file",
        "prompt_data",
        "student_config",
    ):
        if not Path(manifest[name]).exists():
            raise FileNotFoundError(f"Run asset is missing: {name}")
    binding = "reward_config" if manifest["mode"] == "grpo" else "teacher_config"
    if binding not in manifest or not Path(manifest[binding]).is_file():
        raise ValueError(f"Run requires {binding}")
    return manifest


def publish_result(output: Path, value: dict) -> None:
    temporary = output / "result.partial"
    temporary.write_text(json.dumps(value) + "\n")
    temporary.replace(output / "result.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    manifest_path = arguments.manifest.resolve()
    manifest = load_manifest(manifest_path)
    if arguments.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    root = manifest_path.parent.parent
    rank = int(os.environ.get("RANK", "0"))
    nodes = int(os.environ.get("WORLD_SIZE", "1"))
    gpus_per_node = manifest["gpus_per_node"]
    if nodes != manifest["nodes"]:
        raise RuntimeError(f"Allocated {nodes} nodes, expected {manifest['nodes']}")
    output = Path(manifest["output"])
    output.mkdir(parents=True, exist_ok=True)
    node_output = output / f"node-{rank}"
    node_output.mkdir(exist_ok=True)
    source = Path(manifest["source"])
    if manifest.get("source_commit"):
        if (
            subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            != manifest["source_commit"]
        ):
            raise RuntimeError("Slime source changed after manifest preparation")
    omni_source = {"path": manifest["omni_source"]}
    if (
        subprocess.check_output(["git", "-C", omni_source["path"], "rev-parse", "HEAD"], text=True).strip()
        != manifest["omni_source_commit"]
    ):
        raise RuntimeError("Omni source changed after manifest preparation")
    for repository in (str(source), omni_source["path"]):
        if subprocess.check_output(["git", "-C", repository, "status", "--porcelain"], text=True).strip():
            raise RuntimeError(f"Run requires a clean pinned source: {repository}")
    env = dict(
        os.environ,
        PYTHONNOUSERSITE="1",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=str(root / "scripts") + ":" + str(source),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="4",
        CUDA_DEVICE_MAX_CONNECTIONS="1",
        NCCL_CUMEM_ENABLE="0",
        RAY_USAGE_STATS_ENABLED="0",
        RAY_DEDUP_LOGS="0",
        RAY_USE_UVLOOP="0",
        PRISM_PROFILE_OUTPUT=str(output),
        PRISM_REQUIRE_ONE_PACK="1",
        PRISM_SLIME_SOURCE=str(source),
        TENSORBOARD_DIR=str(output / "experiment/tensorboard"),
        PRISM_OMNI_SOURCE=manifest["omni_source"],
    )
    key_file = Path(manifest["auth_token_file"])
    env["INF_API_KEY"] = key_file.read_text().strip()
    if not env["INF_API_KEY"] or "\n" in env["INF_API_KEY"]:
        raise ValueError("The authentication file must contain one nonempty token")
    python = manifest.get("python", sys.executable)
    for package, entry in manifest.get("dependencies", {}).items():
        if (
            subprocess.check_output(["git", "-C", entry["path"], "rev-parse", "HEAD"], text=True).strip()
            != entry["commit"]
        ):
            raise RuntimeError(f"Source changed for {package}")
    gpu_info = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name,memory.total,driver_version", "--format=csv"], text=True
    )
    (node_output / "hardware.csv").write_text(gpu_info)
    from ray._private.services import get_node_ip_address

    ip = get_node_ip_address()
    assert not ipaddress.ip_address(ip).is_loopback, ip
    head_address = output / "head-address.json"
    if rank == 0:
        temporary = output / "head-address.json.partial"
        temporary.write_text(json.dumps({"ip": ip, "host": socket.gethostname()}) + "\n")
        temporary.replace(head_address)
    else:
        deadline = time.monotonic() + 900
        while not head_address.exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Head pod did not publish its address")
            time.sleep(2)
    master = json.loads(head_address.read_text())["ip"]
    address = f"{master}:6385"
    env["RAY_ADDRESS"] = address
    env["SLIME_HOST_IP"] = ip
    if manifest.get("nccl_socket_ifname"):
        env["NCCL_SOCKET_IFNAME"] = manifest["nccl_socket_ifname"]
    else:
        env.pop("NCCL_SOCKET_IFNAME", None)
        os.environ.pop("NCCL_SOCKET_IFNAME", None)
    env["NCCL_DEBUG"] = "WARN"
    os.environ.update(env)
    (node_output / "launch.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "nodes": nodes,
                "host": socket.gethostname(),
                "ip": ip,
                "master": master,
                "ray": address,
                "manifest": manifest,
            },
            indent=2,
        )
        + "\n"
    )
    monitor_log = (node_output / "gpu.csv").open("w")
    monitor = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
            "-l",
            "1",
        ],
        stdout=monitor_log,
        stderr=subprocess.STDOUT,
    )
    ray_log = (node_output / "ray-start.log").open("w")
    ray_tmp = tempfile.mkdtemp(prefix="prism-ray-", dir="/tmp")
    ray_command = [
        python,
        "-m",
        "ray.scripts.scripts",
        "start",
        "--block",
        "--node-ip-address",
        ip,
        "--num-gpus",
        str(gpus_per_node),
        "--num-cpus",
        str(manifest.get("cpus_per_node", 160)),
        "--object-store-memory",
        str(8 * 1024**3),
        "--disable-usage-stats",
    ]
    if rank == 0:
        ray_command += ["--head", "--port", "6385", "--include-dashboard=false", "--temp-dir", ray_tmp]
    else:
        deadline = time.monotonic() + 600
        while True:
            try:
                with socket.create_connection((master, 6385), timeout=2):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError("Ray head did not open its port") from None
                time.sleep(2)
        ray_command += ["--address", address]
    ray_process = subprocess.Popen(
        ray_command, env=env, stdout=ray_log, stderr=subprocess.STDOUT, start_new_session=True
    )
    returncode = 1
    try:
        if rank == 0:
            import ray

            deadline = time.monotonic() + 900
            while True:
                if ray_process.poll() is not None:
                    raise RuntimeError("Ray head exited")
                try:
                    ray.init(address=address, logging_level="ERROR")
                    if int(ray.cluster_resources().get("GPU", 0)) == nodes * gpus_per_node:
                        break
                except (ConnectionError, RuntimeError):
                    pass
                ray.shutdown()
                if time.monotonic() > deadline:
                    raise TimeoutError("Ray cluster did not acquire every allocated GPU")
                time.sleep(3)
            (output / "ray-cluster.json").write_text(
                json.dumps({"resources": ray.cluster_resources(), "nodes": ray.nodes()}, indent=2)
            )
            ray.shutdown()
            subprocess.run(
                [
                    python,
                    str(Path(__file__).with_name("cluster_probe.py")),
                    str(nodes * gpus_per_node),
                    str(output / "nccl-probe.json"),
                ],
                env=env,
                check=True,
            )
            bindings = Path(manifest["ready_file"])
            deadline = time.monotonic() + 900
            while not bindings.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("Frozen service bindings not ready")
                time.sleep(3)
            mode = manifest["mode"]
            batch = manifest.get("batch", nodes * gpus_per_node * 32)
            samples = 8 if mode == "grpo" else 1
            env.update(
                {
                    "TRAIN_PYTHON": python,
                    "OMNI_PYTHON": manifest.get("omni_python", python),
                    "HF_CHECKPOINT": manifest["hf_checkpoint"],
                    "OMNI_CONFIG": manifest.get("student_config", str(root / "configs/student.yaml")),
                    "PROMPT_DATA": manifest["prompt_data"],
                    "TEACHER_CONFIG": manifest.get("teacher_config", str(root / "configs/teachers.yaml")),
                    "PRISM_PRETRAINED_CHECKPOINT": manifest["pretrained_checkpoint"],
                    "OUTPUT_DIR": str(output / "checkpoint"),
                    "AUDIO_DIR": str(output / "experiment/rollout_audio"),
                    "RUN_DIR": str(output / "experiment"),
                    "RUN_NAME": "prism-" + mode,
                    "JOB_TAG": output.name,
                    "RUNNING_ROUND": "0",
                    "USE_TENSORBOARD": "1",
                    "COLOCATE": "1",
                    "ACTOR_NODES": str(nodes),
                    "ACTOR_GPUS_PER_NODE": str(gpus_per_node),
                    "ROLLOUT_GPUS": str(nodes * gpus_per_node),
                    "GPUS_PER_NODE": str(gpus_per_node),
                    "PROMPTS_PER_BATCH": str(batch // samples),
                    "SAMPLES_PER_PROMPT": str(samples),
                    "GLOBAL_BATCH_SIZE": str(batch),
                    "MAX_FRAMES": str(manifest.get("max_frames", 1024)),
                    "TRAIN_SEQ_LENGTH": "20480",
                    "SAVE_INTERVAL": str(manifest["save_interval"]),
                    "NUM_ROLLOUT": str(manifest.get("rollouts", 5)),
                    "ROLLOUT_CONCURRENCY_PER_REPLICA": str(manifest.get("concurrency", 4)),
                    "PRISM_DRY_RUN": "1",
                    "PHASE_TRACE_PATH": str(output / "driver-phases.jsonl"),
                }
            )
            if manifest.get("reward_config"):
                env["REWARD_CONFIG"] = manifest["reward_config"]
            if manifest.get("load_checkpoint"):
                env["RL_LOAD"] = manifest["load_checkpoint"]
            extra = [
                "--rollout-data-transport",
                "object-store",
                "--use-dynamic-batch-size",
                "--max-tokens-per-gpu",
                "20480",
                "--balance-data",
                "--use-distributed-optimizer",
                "--distributed-timeout-minutes",
                "15",
                "--update-weight-buffer-size",
                str(manifest.get("weight_buffer_bytes", 64 * 1024**2)),
                "--rollout-seed",
                str(manifest.get("seed", 6211)),
                "--seed",
                str(manifest.get("seed", 6211)),
            ] + manifest.get("extra_args", [])
            if not manifest.get("features", True):
                extra += [
                    "--use-tensorboard",
                    "--tensorboard-dir",
                    str(output / "experiment/tensorboard"),
                    "--tb-project-name",
                    mode + "-metrics",
                    "--tb-experiment-name",
                    output.name,
                ]
            if manifest.get("save_rollout", False):
                extra += ["--save-debug-rollout-data", str(output / "replay/rollout-{rollout_id}.pt")]
            dry = subprocess.run(
                ["bash", str(source / f"examples/prism/run-{mode}.sh"), *extra],
                cwd=source,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            (output / "command-preparation.log").write_text(dry.stderr)
            command = shlex.split(dry.stdout.strip())
            assert command[:2] == [python, str(source / "train.py")], command[:2]
            if manifest.get("features", True):
                assert command[-2:] == ["--custom-config-path", str(output / "experiment/config/experiment.yaml")]
                command = command[:-2]
            train_env_index = command.index("--train-env-vars") + 1
            train_env = json.loads(command[train_env_index])
            for name in (
                "PRISM_PROFILE_OUTPUT",
                "PRISM_REQUIRE_ONE_PACK",
                "PRISM_SLIME_SOURCE",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NCCL_DEBUG",
                "PRISM_OMNI_SOURCE",
                "TENSORBOARD_DIR",
            ):
                train_env[name] = env[name]
            command[train_env_index] = json.dumps(train_env)
            if manifest.get("features", True):
                command_file = output / "resolved-command.nul"
                subprocess.run(
                    [
                        python,
                        "-m",
                        "slime_plugins.prism.experiment",
                        "--run-dir",
                        str(output / "experiment"),
                        "--run-name",
                        env["RUN_NAME"],
                        "--job-tag",
                        env["JOB_TAG"],
                        "--running-round",
                        "0",
                        "--command-file",
                        str(command_file),
                        "--",
                        *command,
                    ],
                    cwd=source,
                    env=env,
                    check=True,
                )
                command = command_file.read_bytes().decode().rstrip("\0").split("\0")
                command_file.unlink()
                (output / "experiment/logs/driver.log").symlink_to(output / "run.log")
            if manifest.get("driver_wrapper"):
                command = [python, manifest["driver_wrapper"], *command[1:]]
            (output / "training-command.json").write_text(json.dumps(command, indent=2) + "\n")
            env.pop("PRISM_DRY_RUN")
            started = time.time()
            with (output / "run.log").open("w") as log:
                process = subprocess.Popen(
                    command,
                    cwd=source,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                returncode = process.wait()
            publish_result(
                output,
                {
                    "returncode": returncode,
                    "started": started,
                    "elapsed_seconds": time.time() - started,
                    "mode": mode,
                    "gpus": nodes * gpus_per_node,
                    "batch": batch,
                },
            )
        else:
            deadline = (
                time.monotonic() + manifest["timeout_seconds"] if manifest.get("timeout_seconds") else float("inf")
            )
            while not (output / "result.json").exists():
                if ray_process.poll() is not None:
                    raise RuntimeError("Ray worker exited")
                if time.monotonic() > deadline:
                    raise TimeoutError("Training head did not finish")
                time.sleep(3)
            returncode = json.loads((output / "result.json").read_text())["returncode"]
    finally:
        if rank == 0 and not (output / "result.json").exists():
            publish_result(output, {"returncode": 1, "failure": "launcher exception"})
        try:
            os.killpg(ray_process.pid, signal.SIGTERM)
            ray_process.wait(timeout=15)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(ray_process.pid, signal.SIGKILL)
            ray_process.wait()
        subprocess.run(
            [python, "-m", "ray.scripts.scripts", "stop", "--force"],
            stdout=ray_log,
            stderr=subprocess.STDOUT,
            timeout=40,
        )
        for directory in (Path(ray_tmp) / "session_latest/logs", Path("/tmp/ray/session_latest/logs")):
            if directory.exists():
                shutil.copytree(directory, node_output / "ray-logs", dirs_exist_ok=True)
        monitor.terminate()
        monitor.wait()
        monitor_log.close()
        ray_log.close()
        (node_output / "exit-code.txt").write_text(str(returncode) + "\n")
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
