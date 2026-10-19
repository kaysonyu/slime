"""Launch one frozen legacy-protocol ASR replica with per-replica telemetry."""

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allowed-root", required=True)
    args = parser.parse_args()
    if not (args.source / "voxrl/services/asr_vllm.py").is_file():
        raise FileNotFoundError(args.source)
    output = args.output / socket.gethostname()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        PYTHONPATH=str(args.source),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONNOUSERSITE="1",
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
    )
    for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK", "RAY_ADDRESS"):
        env.pop(key, None)
    command = [
        "/opt/venvs/asr/bin/python",
        "-m",
        "voxrl.services.asr_vllm",
        "--model-path",
        str(args.model),
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--max-num-seqs",
        "64",
        "--max-inflight",
        "256",
        "--execution-parallelism",
        "64",
        "--max-completion-tokens",
        "512",
        "--gpu-memory-utilization",
        "0.65",
        "--allowed-root",
        args.allowed_root,
    ]
    (output / "launch.json").write_text(json.dumps({"command": command, "started": time.time()}) + "\n")
    with (output / "gpu.csv").open("w") as gpu, (output / "service.log").open("w") as log:
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
                "-l",
                "2",
            ],
            stdout=gpu,
        )
        server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)

        def stop(signum, frame):
            server.terminate()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            return server.wait()
        finally:
            monitor.terminate()
            monitor.wait(timeout=15)


if __name__ == "__main__":
    raise SystemExit(main())
