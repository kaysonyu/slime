"""Bounded real-weight Omni speech/structured-action validation."""

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--omni", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--port", type=int, default=18410)
    parser.add_argument("--check-speech-api", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "8",
        "PYTHONPATH": args.omni,
    }
    command = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--config",
        "examples/configs/moss_tts_local.yaml",
        "--model-path",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--preprocessing.factory.codec_model_path",
        args.codec,
        "--vocoder.factory.codec_model_path",
        args.codec,
        "--tts_engine.engine.max_running_requests",
        "4",
        "--tts_engine.engine.cuda_graph_max_bs",
        "4",
    ]
    with (output / "omni.log").open("w") as log:
        server = subprocess.Popen(command, env=env, cwd=args.omni, stdout=log, stderr=subprocess.STDOUT)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=300, trust_env=False) as client:
            deadline = time.monotonic() + 600
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Omni exited {server.returncode}; inspect omni.log")
                try:
                    response = client.get("/health", timeout=3)
                    if response.is_success:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("Omni did not become healthy")
                time.sleep(3)
            prompt = "The sun is shining today. Let us take a walk in the park."
            payload = dict(
                prompt=prompt,
                output_modalities=["audio"],
                return_logprob=True,
                return_omni_rollout=True,
                stream=False,
                sampling_params=dict(temperature=1, top_p=1, top_k=-1, max_new_tokens=128, seed=42),
                stage_params={
                    "tts_engine": dict(
                        text_temperature=1,
                        audio_temperature=1,
                        text_top_p=1,
                        audio_top_p=1,
                        text_top_k=-1,
                        audio_top_k=-1,
                        audio_repetition_penalty=1,
                    )
                },
            )
            response = client.post("/generate", json=payload)
            if not response.is_success:
                raise RuntimeError(f"Generation failed: {response.status_code} {response.text[:4000]}")
            result = response.json()
            (output / "request.json").write_text(json.dumps(payload, indent=2) + "\n")
            (output / "rollout.json").write_text(json.dumps(result) + "\n")
            from slime_plugins.models.moss_tts_local.config import MossLocalConfig
            from slime_plugins.models.moss_tts_local.data import MossLocalTrajectory

            meta = result["meta_info"]
            trace = MossLocalTrajectory.from_omni(
                meta["omni_rollout"], meta, MossLocalConfig.from_pretrained(args.model)
            )
            data = base64.b64decode(result["audio"]["data"], validate=True)
            (output / "sample.wav").write_bytes(data)
            with wave.open(str(output / "sample.wav")) as audio:
                duration = audio.getnframes() / audio.getframerate()
                sample_rate = audio.getframerate()
            report = dict(
                frames=trace.num_frames,
                actions=trace.num_actions,
                finish_reason=trace.finish_reason,
                duration_seconds=duration,
                sample_rate=sample_rate,
                weight_version=trace.weight_version,
            )
            if not trace.num_frames or duration <= 0:
                raise RuntimeError("The real checkpoint emitted no audio")
            if args.check_speech_api:
                model_id = client.get("/v1/models").json()["data"][0]["id"]
                speech = {
                    "model": model_id,
                    "input": "Hello, this is a reference speech test.",
                    "ref_audio": "data:audio/wav;base64," + base64.b64encode(data).decode(),
                    "ref_text": prompt,
                    "seed": 43,
                    "max_new_tokens": 128,
                    "stream": False,
                    "response_format": "wav",
                }
                response = client.post("/v1/audio/speech", json=speech)
                response.raise_for_status()
                (output / "reference.wav").write_bytes(response.content)
                with wave.open(str(output / "reference.wav")) as audio:
                    report["reference_wav_seconds"] = audio.getnframes() / audio.getframerate()
                    if audio.getnframes() == 0:
                        raise RuntimeError("Reference-conditioned WAV is empty")
                speech.update(stream=True, response_format="pcm")
                with client.stream("POST", "/v1/audio/speech", json=speech) as response:
                    response.raise_for_status()
                    chunks = [chunk for chunk in response.iter_bytes() if chunk]
                    pcm = b"".join(chunks)
                    channels = int(response.headers["x-channels"])
                    bits = int(response.headers["x-bit-depth"])
                    rate = int(response.headers["x-sample-rate"])
                if not pcm or len(pcm) % (channels * bits // 8):
                    raise RuntimeError("Streaming PCM is empty or not frame-aligned")
                (output / "reference_stream.pcm").write_bytes(pcm)
                report["stream"] = dict(
                    bytes=len(pcm), chunks=len(chunks), sample_rate=rate, channels=channels, bit_depth=bits
                )
            (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=40)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


if __name__ == "__main__":
    main()
