"""Bounded speech rollout/train/refit validation, with optional native resume.

MOPD uses two independently hosted copies of the supplied checkpoint to test
domain routing. They are not advertised as trained domain teachers.
"""

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    for field in ("model", "train-checkpoint", "omni", "megatron", "output-dir"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--codec")
    parser.add_argument("--asr-model")
    parser.add_argument("--model-family", choices=["moss_tts_local", "higgs_tts"], default="moss_tts_local")
    parser.add_argument("--objective", choices=["grpo", "mopd"], default="grpo")
    parser.add_argument("--rollout-max-response-len", type=int, default=128)
    parser.add_argument("--logprob-parity-tolerance", type=float)
    parser.add_argument("--logprob-parity-mean-tolerance", type=float)
    parser.add_argument("--skip-resume", action="store_true", help="Skip large checkpoint writes and process restart")
    args = parser.parse_args()
    if args.model_family == "moss_tts_local" and not args.codec:
        parser.error("MOSS Local generation requires --codec")
    if args.objective == "grpo" and not args.asr_model:
        parser.error("WER GRPO requires --asr-model")
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "source"
    snapshot.mkdir()
    for package in ("slime", "slime_plugins"):
        shutil.copytree(root / package, snapshot / package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(root / "train.py", snapshot / "train.py")
    manifest = {
        str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in snapshot.rglob("*.py")
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    prompts = [
        "The rural juror rarely worried about the unusual weather.",
        "An enthusiastic archaeologist photographed the extraordinary ruins at dawn.",
        "Several researchers carefully compared the measurements before publishing their report.",
        "I thought the meeting would begin on Thursday, but it was moved to Wednesday.",
        "Please place the blue notebook beside the small wooden box.",
        "Although the weather changed suddenly, the children finished their game.",
    ]
    dataset = output / "train.jsonl"
    dataset.write_text(
        "".join(
            json.dumps({"text": text, "domain": ("domain_a", "domain_b")[i % 2]}) + "\n"
            for i, text in enumerate(prompts)
        )
    )
    (output / "eval.jsonl").write_text(
        json.dumps({"text": "The sun is shining today. Let us take a walk in the park."}) + "\n"
    )
    base_env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "8",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_CUMEM_ENABLE": "0",
        "NCCL_NVLS_ENABLE": "0",
    }
    servers = []
    child = None

    def stop(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    try:
        tts_extra = ["--tts_engine.engine.max_running_requests", "4", "--tts_engine.engine.cuda_graph_max_bs", "4"]
        tts_config = None
        if args.model_family == "moss_tts_local":
            tts_config = "examples/configs/moss_tts_local.yaml"
            tts_extra += [
                "--preprocessing.factory.codec_model_path",
                args.codec,
                "--vocoder.factory.codec_model_path",
                args.codec,
            ]
        configurations = [("tts", "0", 18410, tts_config, args.model, tts_extra)]
        if args.objective == "grpo":
            configurations.append(
                (
                    "asr",
                    "1",
                    18411,
                    "examples/configs/qwen3_asr_rtx4090.yaml",
                    args.asr_model,
                    ["--asr.engine.max_running_requests", "4", "--asr.engine.cuda_graph_max_bs", "4"],
                )
            )
        else:
            for index, domain in enumerate(("domain_a", "domain_b"), 1):
                configurations.append(
                    (
                        domain,
                        str(index),
                        18411 + index,
                        f"examples/configs/{args.model_family}_score.yaml",
                        args.model,
                        [],
                    )
                )
        for name, gpu, port, config, model, extra in configurations:
            with socket.socket() as probe:
                probe.settimeout(1)
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError(f"Validation port {port} is already in use")
            command = [
                sys.executable,
                "-m",
                "sglang_omni.cli",
                "serve",
                *(["--config", config] if config else []),
                "--model-path",
                model,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                *extra,
            ]
            with (output / f"{name}.log").open("w") as log:
                process = subprocess.Popen(
                    command,
                    cwd=args.omni,
                    env={**base_env, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": args.omni},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            servers.append((name, port, process))
        with httpx.Client(timeout=3, trust_env=False) as client:
            for name, port, process in servers:
                deadline = time.monotonic() + 600
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"{name} exited {process.returncode}; inspect its log")
                    try:
                        if client.get(f"http://127.0.0.1:{port}/health").is_success:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"{name} startup timed out")
                    time.sleep(3)
            info = client.post("http://127.0.0.1:18410/model_info", json={"stages": ["tts_engine"]})
            info.raise_for_status()
            (output / "initial_model_info.json").write_text(json.dumps(info.json(), indent=2) + "\n")
        train_env = {**base_env, "CUDA_VISIBLE_DEVICES": "3,4", "PYTHONPATH": args.megatron + ":" + str(snapshot)}
        common = [
            sys.executable,
            str(snapshot / "train.py"),
            "--hf-checkpoint",
            args.model,
            "--load",
            args.train_checkpoint,
            "--model-family",
            args.model_family,
            "--objective",
            args.objective,
            "--actor-num-gpus-per-node",
            "2",
            "--prompt-data",
            str(dataset),
            "--omni-endpoints",
            "http://127.0.0.1:18410",
            "--rollout-batch-size",
            "2",
            "--n-samples-per-prompt",
            "4",
            "--global-batch-size",
            "8",
            "--micro-batch-size",
            "1",
            "--rollout-max-response-len",
            str(args.rollout_max_response_len),
            "--omni-concurrency",
            "4",
            "--lr",
            "0.000003",
            "--lr-decay-style",
            "constant",
            "--weight-decay",
            "0",
            "--save-debug-rollout-data",
            str(output / "rollout_{rollout_id}.pt"),
            "--metrics-jsonl",
            str(output / "metrics.jsonl"),
            "--audio-output-dir",
            str(output / "audio"),
            "--no-gradient-accumulation-fusion",
            "--no-masked-softmax-fusion",
            "--no-rope-fusion",
            "--no-persist-layer-norm",
            "--attention-backend",
            "flash",
        ]
        if args.objective == "grpo":
            common += ["--asr-endpoint", "http://127.0.0.1:18411/v1/audio/transcriptions", "--asr-model", "qwen3-asr"]
        else:
            common += ["--mopd-teachers", "domain_a=http://127.0.0.1:18412", "domain_b=http://127.0.0.1:18413"]
        for name in ("logprob_parity_tolerance", "logprob_parity_mean_tolerance"):
            value = getattr(args, name)
            if value is not None:
                common += ["--" + name.replace("_", "-"), str(value)]
        phases = [("train", ["--num-rollout", "2"])]
        if not args.skip_resume:
            common += ["--save", str(output / "checkpoints"), "--save-interval", "2"]
            phases.append(("resume", ["--num-rollout", "3", "--load", str(output / "checkpoints")]))
        for phase, additional in phases:
            command = common + additional
            (output / f"{phase}_command.json").write_text(json.dumps(command, indent=2) + "\n")
            with (output / f"{phase}.log").open("w") as log:
                child = subprocess.Popen(command, env=train_env, cwd=snapshot, stdout=log, stderr=subprocess.STDOUT)
                code = child.wait(timeout=900)
            if code:
                raise RuntimeError(f"{phase} failed with {code}; inspect {phase}.log")
        metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        if not any("train/grad_norm" in item for item in metrics):
            raise RuntimeError("Validation did not produce training metrics")
        report = {
            "completed": True,
            "model_family": args.model_family,
            "objective": args.objective,
            "rollouts": 2 if args.skip_resume else 3,
            "metric_records": len(metrics),
        }
        if args.objective == "grpo":
            if not any("rollout/wer" in item for item in metrics):
                raise RuntimeError("Validation did not produce real WER metrics")
        else:
            import torch

            # The trusted local debug artifact retains teacher identities and
            # score tensors, so assert routing without a second scoring pass.
            saved = torch.load(output / "rollout_0.pt", weights_only=False)
            samples = saved["samples"]
            routes = {s["metadata"]["teacher_domain"] for s in samples}
            if routes != {"domain_a", "domain_b"} or any(s["teacher_scores"] is None for s in samples):
                raise RuntimeError("Both frozen-teacher routes must score the student's original actions")
            report["teacher_routes"] = sorted(routes)
            report["teacher_checkpoints"] = "two replicas of the same base; routing validation only"
        if not args.skip_resume:
            report["checkpoint_tracker"] = (
                (output / "checkpoints/latest_checkpointed_iteration.txt").read_text().strip()
            )
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for _, _, server in servers:
            server.terminate()
        for _, _, server in servers:
            try:
                server.wait(timeout=40)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
