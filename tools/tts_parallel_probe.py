"""Compare real-checkpoint CP/TP score and gradient behavior on fixed actions."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-checkpoint", required=True)
    parser.add_argument("--megatron", required=True)
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve()
    snapshot = output / "source"
    snapshot.mkdir(parents=True)
    for package in ("slime", "slime_plugins"):
        shutil.copytree(root / package, snapshot / package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(root / "tools/tts_megatron_probe.py", snapshot / "probe.py")
    manifest = {
        str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.rglob("*.py")
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = {}
    for label, workers, tp, cp in [("baseline", 1, 1, 1), ("cp2", 2, 1, 2), ("tp2", 2, 2, 1)]:
        directory = output / label
        directory.mkdir()
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={workers}",
            str(snapshot / "probe.py"),
            "--hf-checkpoint",
            args.model,
            "--load",
            args.train_checkpoint,
            "--debug-train-only",
            "--actor-num-gpus-per-node",
            str(workers),
            "--tensor-model-parallel-size",
            str(tp),
            "--context-parallel-size",
            str(cp),
            "--num-rollout",
            "1",
            "--rollout-batch-size",
            "1",
            "--n-samples-per-prompt",
            "2",
            "--global-batch-size",
            "1",
            "--micro-batch-size",
            "1",
            "--lr",
            "0.0001",
            "--lr-decay-style",
            "constant",
            "--weight-decay",
            "0",
            "--no-gradient-accumulation-fusion",
            "--no-masked-softmax-fusion",
            "--no-rope-fusion",
            "--no-persist-layer-norm",
            "--attention-backend",
            "flash",
            "--probe-rollout-file",
            args.rollout,
            "--probe-output-dir",
            str(directory),
            "--metrics-jsonl",
            str(directory / "metrics.jsonl"),
        ]
        env = {
            **os.environ,
            "PYTHONPATH": args.megatron + ":" + str(snapshot),
            "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(workers)),
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_CUMEM_ENABLE": "0",
            "NCCL_NVLS_ENABLE": "0",
            "HF_HUB_OFFLINE": "1",
            "OMP_NUM_THREADS": "8",
        }
        with (directory / "run.log").open("w") as log:
            subprocess.run(
                command, cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900
            )
        results[label] = json.loads((directory / "result.json").read_text())
        metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        results[label]["grad_norm"] = metrics[-1]["train/grad_norm"]
    baseline = results["baseline"]["grad_norm"]
    for label in ("cp2", "tp2"):
        relative = abs(results[label]["grad_norm"] - baseline) / max(abs(baseline), 1e-8)
        results[label]["gradient_norm_relative_difference"] = relative
        if relative > 0.02:
            raise RuntimeError(f"{label} gradient norm differs from baseline by {relative:.2%}")
    (output / "result.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
