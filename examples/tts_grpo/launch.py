"""Inspire Job submission and fixed-size Omni/Megatron process supervision."""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path("/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models")
SOURCES = Path("/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable")
FORWARD = (
    "PROMPT_DATA",
    "EVAL_DATA",
    "EVAL_CONFIG",
    "WER_LANGUAGE",
    "REWARD_CONFIG",
    "REWARD_SECRET_ENV",
    "ENDPOINT_BUNDLE",
    "ASR_ENDPOINT",
    "ASR_PROTOCOL",
    "ASR_MODEL",
    "ASR_AUTH_TOKEN_ENV",
    "OMNI_ENDPOINTS",
    "MODEL_DIR",
    "TRAIN_CHECKPOINT",
    "CODEC_DIR",
    "ASR_MODEL_DIR",
    "OMNI_DIR",
    "MEGATRON_DIR",
    "OUTPUT_DIR",
    "JOB_NAME",
    "TRAIN_GPUS",
    "TP_SIZE",
    "CP_SIZE",
    "NUM_ROLLOUTS",
    "ROLLOUT_BATCH_SIZE",
    "N_SAMPLES_PER_PROMPT",
    "MAX_RESPONSE_LEN",
    "SAVE_INTERVAL",
    "EVAL_INTERVAL",
    "EVAL_SAMPLES",
    "EVAL_MAX_RESPONSE_LEN",
    "EVAL_TEMPERATURE",
    "LR",
    "MICRO_BATCH_SIZE",
    "ROLLOUT_TEMPERATURE",
    "REWARD_CONCURRENCY",
    "REWARD_TIMEOUT",
    "REWARD_MAX_RETRIES",
    "GROUP_MAX_RETRIES",
    "FAILURE_BUDGET",
    "START_LOCAL_ASR",
    "REF_AUDIO_ROOT",
    "TTS_PORT",
    "ASR_PORT",
    "RAY_PORT",
    "STARTUP_TIMEOUT",
    "INSPIRE_NODES",
    "INSPIRE_QUOTA",
    "INSPIRE_WORKSPACE",
    "INSPIRE_PROJECT",
    "INSPIRE_GROUP",
    "INSPIRE_IMAGE",
    "INSPIRE_SHM_SIZE",
    "INSPIRE_PRIORITY",
    "TTS_ASR_URL",
    "TTS_SIM_URL",
    "TTS_JUDGE_URL",
    "EXTRA_PYTHONPATH",
)


def positive(env, name, default):
    value = int(env.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def environment():
    env = dict(os.environ)
    defaults = {
        "MODEL_DIR": str(ASSETS / "MOSS-TTS-Local-0020000-omni"),
        "TRAIN_CHECKPOINT": str(ASSETS / "MOSS-TTS-Local-0020000-torch_dist"),
        "CODEC_DIR": str(ASSETS / "MOSS-Audio-Tokenizer-v2"),
        "ASR_MODEL_DIR": str(ASSETS / "Qwen3-ASR-1.7B"),
        "OMNI_DIR": str(SOURCES / "sglang-omni"),
        "MEGATRON_DIR": str(SOURCES / "Megatron-LM"),
        "JOB_NAME": "moss-local-grpo-" + time.strftime("%Y%m%d-%H%M%S"),
        "INSPIRE_WORKSPACE": "CQ-科研驾驶舱",
        "INSPIRE_PROJECT": "CQ项目",
        "INSPIRE_GROUP": "科研驾驶舱",
        "INSPIRE_QUOTA": "4,60,800",
        "INSPIRE_NODES": "1",
        "INSPIRE_SHM_SIZE": "128",
        "INSPIRE_PRIORITY": "10",
        "INSPIRE_IMAGE": "docker.sii.shaipower.online/inspire-studio/miles-moss-tts-local-env:20260908-cu130-v1",
        "WER_LANGUAGE": "en",
        "TTS_PORT": "18410",
        "ASR_PORT": "18411",
        "RAY_PORT": "6379",
    }
    for key, value in defaults.items():
        env.setdefault(key, value)
    env.setdefault("OUTPUT_DIR", str(ASSETS.parent / "outputs" / env["JOB_NAME"]))
    if env.get("ENDPOINT_BUNDLE"):
        # Same strict, non-executable format as the serving manager's bundle.
        allowed = {"TTS_ASR_URL", "TTS_SIM_URL", "TTS_JUDGE_URL"}
        seen = set()
        for line in Path(env["ENDPOINT_BUNDLE"]).read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            key, separator, value = line.partition("=")
            if (
                not separator
                or key not in allowed
                or key in seen
                or not value.startswith(("http://", "https://"))
                or any(c.isspace() for c in value)
            ):
                raise ValueError("Malformed endpoint bundle")
            seen.add(key)
            env.setdefault(key, value)
    if "TTS_ASR_URL" in env:
        env.setdefault("ASR_ENDPOINT", env["TTS_ASR_URL"])
        env.setdefault("ASR_PROTOCOL", "qwen3_asr_chat_path")
    env.setdefault("ASR_MODEL", "qwen3-asr-1.7b" if env.get("ASR_PROTOCOL") == "qwen3_asr_chat_path" else "qwen3-asr")
    local_asr = env.get("START_LOCAL_ASR", "0" if env.get("ASR_ENDPOINT") or env.get("REWARD_CONFIG") else "1")
    if local_asr not in ("0", "1"):
        raise ValueError("START_LOCAL_ASR must be 0 or 1")
    env["START_LOCAL_ASR"] = local_asr
    if local_asr == "1":
        env.setdefault("ASR_ENDPOINT", f"http://127.0.0.1:{env['ASR_PORT']}/v1/audio/transcriptions")
    gpu_count, _, memory = map(int, env["INSPIRE_QUOTA"].split(","))
    nodes = positive(env, "INSPIRE_NODES", 1)
    if nodes > 1 and (gpu_count != 8 or local_asr == "1"):
        raise ValueError("Multi-node Jobs require 8 GPUs/node and external reward services")
    if positive(env, "INSPIRE_SHM_SIZE", 128) > memory:
        raise ValueError("Shared memory cannot exceed quota memory")
    reserved = (0 if env.get("OMNI_ENDPOINTS") else 1) + int(local_asr)
    env.setdefault("TRAIN_GPUS", str(gpu_count - reserved))
    train_gpus = positive(env, "TRAIN_GPUS", 2)
    if train_gpus + reserved > gpu_count:
        raise ValueError("Training and local services exceed the GPU allocation")
    parallel = positive(env, "TP_SIZE", 1) * positive(env, "CP_SIZE", 1)
    if nodes * train_gpus % parallel:
        raise ValueError("Training GPU count must divide by TP * CP")
    dp_size = nodes * train_gpus // parallel
    fanout = positive(env, "N_SAMPLES_PER_PROMPT", 4)
    prompt_multiple = dp_size // math.gcd(dp_size, fanout)
    env.setdefault("ROLLOUT_BATCH_SIZE", str(math.ceil(2 / prompt_multiple) * prompt_multiple))
    batch = positive(env, "ROLLOUT_BATCH_SIZE", 2) * fanout
    if positive(env, "N_SAMPLES_PER_PROMPT", 4) < 2 or batch % (nodes * train_gpus // parallel):
        raise ValueError("GRPO fanout must be >=2 and the global batch must be divisible by DP")
    if not env.get("PROMPT_DATA"):
        raise ValueError("PROMPT_DATA is required")
    if not (env.get("EVAL_CONFIG") or env.get("EVAL_DATA")):
        raise ValueError("Set EVAL_DATA or EVAL_CONFIG for independent evaluation")
    if env.get("EVAL_CONFIG") and env.get("EVAL_DATA"):
        raise ValueError("Choose EVAL_DATA or EVAL_CONFIG")
    output = Path(env["OUTPUT_DIR"])
    if not output.is_absolute() or any(
        output.resolve().is_relative_to(Path(env[key]).resolve()) for key in ("MODEL_DIR", "TRAIN_CHECKPOINT")
    ):
        raise ValueError("OUTPUT_DIR must be absolute and outside model/checkpoint input directories")
    return env


def checkpoint_for_round(root, round_number, initial):
    for number in range(round_number - 1, -1, -1):
        checkpoint = root / f"running_round_{number}" / "checkpoints"
        tracker = checkpoint / "latest_checkpointed_iteration.txt"
        if not tracker.is_file():
            continue
        value = tracker.read_text().strip()
        if not value.isdecimal():
            continue
        iteration = int(value)
        directory = checkpoint / f"iter_{iteration:07d}"
        sampler = checkpoint / "rollout" / f"global_dataset_state_dict_{iteration}.pt"
        if directory.is_dir() and any(directory.rglob(".metadata")) and sampler.is_file() and sampler.stat().st_size:
            return str(checkpoint)
    if round_number:
        raise ValueError(
            "Restart found no complete checkpoint plus matching sampler; refusing to restart from initialization"
        )
    return initial


def training_command(env, output, checkpoint, endpoints, ray_address=None):
    batch = positive(env, "ROLLOUT_BATCH_SIZE", 2) * positive(env, "N_SAMPLES_PER_PROMPT", 4)
    command = [
        sys.executable,
        str(ROOT / "train.py"),
        "--model-family",
        "moss_tts_local",
        "--objective",
        "grpo",
        "--hf-checkpoint",
        env["MODEL_DIR"],
        "--load",
        checkpoint,
        "--actor-num-nodes",
        env["INSPIRE_NODES"],
        "--actor-num-gpus-per-node",
        env["TRAIN_GPUS"],
        "--omni-endpoints",
        *endpoints,
        "--prompt-data",
        env["PROMPT_DATA"],
        "--wer-language",
        env["WER_LANGUAGE"],
        "--global-batch-size",
        str(batch),
        "--rollout-shuffle",
        "--rollout-seed",
        "42",
        "--save",
        str(output / "checkpoints"),
        "--metrics-jsonl",
        str(output / "metrics.jsonl"),
        "--audio-output-dir",
        str(output / "audio"),
        "--use-tensorboard",
        "--tb-log-dir",
        str(output / "tensorboard"),
        "--lr-decay-style",
        "constant",
        "--weight-decay",
        "0",
        "--train-scope",
        "full",
        "--no-gradient-accumulation-fusion",
        "--no-masked-softmax-fusion",
        "--no-rope-fusion",
        "--no-persist-layer-norm",
        "--attention-backend",
        "flash",
    ]
    settings = {
        "TP_SIZE": ("tensor-model-parallel-size", "1"),
        "CP_SIZE": ("context-parallel-size", "1"),
        "NUM_ROLLOUTS": ("num-rollout", "100"),
        "ROLLOUT_BATCH_SIZE": ("rollout-batch-size", "2"),
        "N_SAMPLES_PER_PROMPT": ("n-samples-per-prompt", "4"),
        "MAX_RESPONSE_LEN": ("rollout-max-response-len", "512"),
        "SAVE_INTERVAL": ("save-interval", "10"),
        "EVAL_INTERVAL": ("eval-interval", "10"),
        "EVAL_SAMPLES": ("n-samples-per-eval-prompt", "1"),
        "LR": ("lr", "0.000003"),
        "MICRO_BATCH_SIZE": ("micro-batch-size", "1"),
        "ROLLOUT_TEMPERATURE": ("rollout-temperature", "1.0"),
        "REWARD_CONCURRENCY": ("reward-concurrency", "8"),
        "REWARD_TIMEOUT": ("reward-timeout", "120"),
        "REWARD_MAX_RETRIES": ("reward-max-retries", "2"),
        "GROUP_MAX_RETRIES": ("rollout-group-max-retries", "2"),
        "FAILURE_BUDGET": ("max-recoverable-rollout-failures", "32"),
    }
    for variable, (flag, default) in settings.items():
        command += ["--" + flag, env.get(variable, default)]
    for variable, flag in (
        ("EVAL_CONFIG", "eval-config"),
        ("EVAL_DATA", "eval-data"),
        ("REWARD_CONFIG", "reward-config"),
        ("ASR_ENDPOINT", "asr-endpoint"),
        ("ASR_PROTOCOL", "asr-protocol"),
        ("ASR_MODEL", "asr-model"),
        ("ASR_AUTH_TOKEN_ENV", "asr-auth-token-env"),
        ("EVAL_TEMPERATURE", "eval-temperature"),
        ("EVAL_MAX_RESPONSE_LEN", "eval-max-response-len"),
    ):
        if env.get(variable):
            command += ["--" + flag, env[variable]]
    if ray_address:
        command += ["--ray-address", ray_address]
    return command


def preflight(env):
    # Import only the CPU data/reward boundary, never construct a model or Ray cluster.
    from types import SimpleNamespace
    from slime.rollout.data_source import RolloutDataSource
    from slime.rollout.evaluation import evaluation_source
    from slime.rollout.rm_hub.config import get_reward_config
    from slime.utils.eval_config import resolve_eval_datasets

    for name in ("MODEL_DIR", "TRAIN_CHECKPOINT", "OMNI_DIR", "MEGATRON_DIR", "CODEC_DIR"):
        if not Path(env[name]).is_absolute() or not Path(env[name]).is_dir():
            raise ValueError(f"{name} must be an existing absolute directory")
    if not (Path(env["TRAIN_CHECKPOINT"]) / "latest_checkpointed_iteration.txt").is_file():
        raise ValueError("TRAIN_CHECKPOINT must be a torch_dist checkpoint root")
    config = json.loads((Path(env["MODEL_DIR"]) / "config.json").read_text())
    if config.get("sglang_omni_compat", {}).get("local_layout") != "gpt2_qkv_interleaved_v1":
        raise ValueError("MODEL_DIR must use the canonical Local layout")
    args = SimpleNamespace(
        prompt_data=env["PROMPT_DATA"],
        rollout_seed=42,
        rollout_shuffle=True,
        objective="grpo",
        n_samples_per_prompt=int(env.get("N_SAMPLES_PER_PROMPT", 4)),
        hf_checkpoint=env["MODEL_DIR"],
        wer_language=env["WER_LANGUAGE"],
        reward_config=env.get("REWARD_CONFIG"),
        asr_endpoint=env.get("ASR_ENDPOINT"),
        asr_model=env["ASR_MODEL"],
        asr_protocol=env.get("ASR_PROTOCOL", "openai_audio_transcriptions"),
        asr_auth_token_env=env.get("ASR_AUTH_TOKEN_ENV"),
        custom_rm_path=None,
        eval_data=env.get("EVAL_DATA"),
        eval_config=env.get("EVAL_CONFIG"),
        rollout_temperature=float(env.get("ROLLOUT_TEMPERATURE", 1)),
        rollout_max_response_len=int(env.get("MAX_RESPONSE_LEN", 512)),
        n_samples_per_eval_prompt=int(env.get("EVAL_SAMPLES", 1)),
        eval_temperature=float(env["EVAL_TEMPERATURE"]) if env.get("EVAL_TEMPERATURE") else None,
        eval_max_response_len=int(env["EVAL_MAX_RESPONSE_LEN"]) if env.get("EVAL_MAX_RESPONSE_LEN") else None,
    )
    # YAML interpolation sees only the caller's explicit configuration and endpoint bundle.
    os.environ.update({key: value for key, value in env.items() if key.startswith("TTS_")})
    get_reward_config(args)
    source = RolloutDataSource(args)
    checkpoint = Path(env["TRAIN_CHECKPOINT"])
    manifest = checkpoint / "slime_checkpoint.json"
    initialization = manifest.is_file() and json.loads(manifest.read_text()).get("kind") == "initialization"
    if not initialization:
        iteration = int((checkpoint / "latest_checkpointed_iteration.txt").read_text().strip())
        args.load = str(checkpoint)
        args.rollout_global_dataset = True
        source.load(iteration)
    datasets = resolve_eval_datasets(args)
    sizes = {dataset.name: len(evaluation_source(args, dataset)[1]) for dataset in datasets}
    print(
        json.dumps(
            {
                "train_records": len(source),
                "eval_records": sizes,
                "data_identity": source.dataset.identity,
                "nodes": int(env["INSPIRE_NODES"]),
                "train_gpus_per_node": int(env["TRAIN_GPUS"]),
                "rollout_batch_size": int(env["ROLLOUT_BATCH_SIZE"]),
                "samples_per_prompt": args.n_samples_per_prompt,
            }
        ),
        flush=True,
    )


def atomic_json(path, data):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data))
    os.replace(temporary, path)


def check_ports(ports):
    if len(set(ports)) != len(ports) or any(not 0 < port < 65536 for port in ports):
        raise ValueError("Local service ports must be distinct valid TCP ports")
    for port in ports:
        with socket.socket() as probe:
            probe.bind(("0.0.0.0", port))


def run(env):
    rank = int(env.get("PET_NODE_RANK", 0))
    nodes = int(env["INSPIRE_NODES"])
    if not 0 <= rank < nodes or (nodes > 1 and int(env.get("PET_NNODES", 0)) != nodes):
        raise ValueError("Inspire PET rank/node environment disagrees with the requested layout")
    round_number = int(env.get("RUNNING_ROUND", 0))
    root = Path(env["OUTPUT_DIR"])
    output = root / f"running_round_{round_number}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_for_round(root, round_number, env["TRAIN_CHECKPOINT"])
    env["TRAIN_CHECKPOINT"] = checkpoint
    env.update(
        PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS=env.get("OMP_NUM_THREADS", "8"),
        CUDA_DEVICE_MAX_CONNECTIONS="1",
        NCCL_CUMEM_ENABLE="0",
        NCCL_NVLS_ENABLE="0",
    )
    if env.get("REWARD_SECRET_ENV"):
        secret_file = Path(env["REWARD_SECRET_ENV"])
        if secret_file.stat().st_mode & 0o077:
            raise ValueError("REWARD_SECRET_ENV must be readable only by its owner")
        for line in secret_file.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            name, separator, value = line.partition("=")
            if not separator or not name.isidentifier():
                raise ValueError("Malformed non-executable secret environment file")
            env[name] = value
    os.environ.update(env)
    preflight(env)
    visible = env.get("CUDA_VISIBLE_DEVICES")
    gpus = (
        visible.split(",")
        if visible is not None
        else subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True)
        .strip()
        .splitlines()
    )
    if len(gpus) != int(env["INSPIRE_QUOTA"].split(",")[0]):
        raise ValueError("Visible GPUs do not match the Job quota")
    timeout = int(env.get("STARTUP_TIMEOUT", 1200))
    ip = env.get("SLIME_HOST_IP") or socket.gethostbyname(socket.gethostname())
    master = env.get("MASTER_ADDR", ip)
    if nodes > 1 and ip.startswith("127."):
        raise ValueError("Multi-node services require a routable SLIME_HOST_IP")
    processes, streams = [], []
    owns_ray = False
    success, failure = output / "succeeded.json", output / "failed.json"
    if success.exists() or failure.exists():
        raise FileExistsError("This running_round already finished; use a new output directory or a new running round")
    ports = []
    if not env.get("OMNI_ENDPOINTS"):
        ports.append(int(env["TTS_PORT"]))
    if env["START_LOCAL_ASR"] == "1":
        ports.append(int(env["ASR_PORT"]))
    if nodes > 1:
        ports.append(int(env["RAY_PORT"]))
    check_ports(ports)

    def start(name, command, child_env):
        stream = (output / f"{name}-rank{rank}.log").open("w")
        streams.append(stream)
        child = subprocess.Popen(
            command, cwd=ROOT, env=child_env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        processes.append(child)
        return child

    def healthy_children():
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("A local service exited; inspect the per-rank service logs")

    def wait_until(predicate, label):
        deadline = time.monotonic() + timeout
        while not predicate():
            healthy_children()
            if failure.exists():
                raise RuntimeError("A Job rank failed; inspect its logs")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for {label}")
            time.sleep(2)

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def health(url):
        try:
            with opener.open(url + "/health", timeout=3) as response:
                return response.status == 200
        except (urllib.error.URLError, TimeoutError):
            return False

    def stop(signum, _frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        consumed = 0
        endpoint = None
        if not env.get("OMNI_ENDPOINTS"):
            endpoint = f"http://{ip}:{env['TTS_PORT']}"
            command = [
                sys.executable,
                "-m",
                "sglang_omni.cli",
                "serve",
                "--config",
                str(Path(env["MODEL_DIR"]) / "sglang_omni.yaml"),
                "--model-path",
                env["MODEL_DIR"],
                "--host",
                "0.0.0.0",
                "--port",
                env["TTS_PORT"],
                "--preprocessing.factory.codec_model_path",
                env["CODEC_DIR"],
                "--vocoder.factory.codec_model_path",
                env["CODEC_DIR"],
                "--tts_engine.engine.max_running_requests",
                "4",
                "--tts_engine.engine.cuda_graph_max_bs",
                "4",
            ]
            if env.get("REF_AUDIO_ROOT"):
                command += ["--allowed-local-media-path", env["REF_AUDIO_ROOT"]]
            start("tts", command, env | {"CUDA_VISIBLE_DEVICES": gpus[consumed], "PYTHONPATH": env["OMNI_DIR"]})
            consumed += 1
            wait_until(lambda: health(endpoint), "Omni student")
        if env["START_LOCAL_ASR"] == "1":
            command = [
                sys.executable,
                "-m",
                "sglang_omni.cli",
                "serve",
                "--config",
                str(Path(env["OMNI_DIR"]) / "examples/configs/qwen3_asr_rtx4090.yaml"),
                "--model-path",
                env["ASR_MODEL_DIR"],
                "--host",
                "127.0.0.1",
                "--port",
                env["ASR_PORT"],
                "--asr.engine.max_running_requests",
                "4",
                "--asr.engine.cuda_graph_max_bs",
                "4",
            ]
            start("asr", command, env | {"CUDA_VISIBLE_DEVICES": gpus[consumed], "PYTHONPATH": env["OMNI_DIR"]})
            consumed += 1
            wait_until(lambda: health(f"http://127.0.0.1:{env['ASR_PORT']}"), "ASR")
        train_env = env | {
            "CUDA_VISIBLE_DEVICES": ",".join(gpus[consumed : consumed + int(env["TRAIN_GPUS"])]),
            "PYTHONPATH": env["MEGATRON_DIR"]
            + ":"
            + str(ROOT)
            + (":" + env["EXTRA_PYTHONPATH"] if env.get("EXTRA_PYTHONPATH") else ""),
        }
        ray_address = None
        if nodes > 1:
            ray_address = f"{master}:{env['RAY_PORT']}"
            ray_command = [
                "ray",
                "start",
                "--node-ip-address",
                ip,
                "--num-gpus",
                env["TRAIN_GPUS"],
                "--disable-usage-stats",
            ]
            if rank == 0:
                ray_command += ["--head", "--port", env["RAY_PORT"], "--include-dashboard=false"]
            else:
                wait_until(lambda: (output / "ray-head.json").exists(), "Ray head")
                ray_command += ["--address", ray_address]
            owns_ray = True
            subprocess.run(ray_command, env=train_env, check=True, timeout=120)
            if rank == 0:
                atomic_json(output / "ray-head.json", {"address": ray_address})
        atomic_json(output / f"rank-{rank}.json", {"endpoint": endpoint, "rank": rank})
        if rank == 0:
            wait_until(
                lambda: all((output / f"rank-{index}.json").exists() for index in range(nodes)), "all Job ranks"
            )
            endpoints = (
                shlex.split(env["OMNI_ENDPOINTS"])
                if env.get("OMNI_ENDPOINTS")
                else [json.loads((output / f"rank-{index}.json").read_text())["endpoint"] for index in range(nodes)]
            )
            command = training_command(env, output, checkpoint, endpoints, ray_address)
            atomic_json(output / "train-command.json", command)
            child = subprocess.Popen(command, cwd=ROOT, env=train_env, start_new_session=True)
            try:
                while child.poll() is None:
                    healthy_children()
                    if failure.exists():
                        raise RuntimeError("A worker rank failed")
                    time.sleep(2)
                if child.returncode:
                    raise subprocess.CalledProcessError(child.returncode, command)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
            atomic_json(success, {"completed": True})
        else:
            while not success.exists():
                healthy_children()
                if failure.exists():
                    raise RuntimeError("Training rank failed")
                time.sleep(2)
    except BaseException:
        atomic_json(failure, {"rank": rank, "failed": True})
        raise
    finally:
        for child in processes:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in processes:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        if owns_ray:
            subprocess.run(["ray", "stop", "--force"], env=env, timeout=60, check=False)
        for stream in streams:
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "plan", "submit", "run"))
    action = parser.parse_args().action
    env = environment()
    if env.get("EXTRA_PYTHONPATH"):
        sys.path.extend(env["EXTRA_PYTHONPATH"].split(os.pathsep))
    if action == "run":
        run(env)
        return
    preflight(env)
    if action == "check":
        return
    command = [
        "inspire",
        "job",
        "create",
        "--name",
        env["JOB_NAME"],
        "--workspace",
        env["INSPIRE_WORKSPACE"],
        "--project",
        env["INSPIRE_PROJECT"],
        "--group",
        env["INSPIRE_GROUP"],
        "--quota",
        env["INSPIRE_QUOTA"],
        "--image",
        env["INSPIRE_IMAGE"],
        "--nodes",
        env["INSPIRE_NODES"],
        "--shm-size",
        env["INSPIRE_SHM_SIZE"],
        "--priority",
        env["INSPIRE_PRIORITY"],
        "--no-enable-notification",
        "--command",
        shlex.join(["python3", str(Path(__file__).resolve()), "run"]),
    ]
    if env.get("AUTO_FAULT_TOLERANCE", "0") == "1":
        command += ["--auto-fault-tolerance", "--fault-tolerance-max-retry", env.get("MAX_JOB_RETRIES", "3")]
    else:
        command += ["--no-auto-fault-tolerance"]
    for name in FORWARD:
        if name in env:
            command += ["--env", f"{name}={env[name]}"]
    if env.get("MAX_HOURS"):
        command += ["--max-time", env["MAX_HOURS"]]
    if action == "plan":
        command += ["--dry-run"]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
