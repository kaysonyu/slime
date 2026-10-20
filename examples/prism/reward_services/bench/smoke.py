"""Probe all four endpoints and exercise the actual Slime reward client.

Endpoint JSON maps qwen/whisper/redim/judge to either a URL or Serving status.
Run from the repository root with python -m; INF_API_KEY is inherited.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import httpx

from slime_plugins.audio_rewards.client import ServiceClient
from slime_plugins.prism.reward_adapter import PrismRewardAdapter


async def main(args):
    endpoints = json.loads(Path(args.endpoints).read_text())
    endpoints = {k: (v["endpoint"] if isinstance(v, dict) else v) for k, v in endpoints.items()}
    headers = {"Authorization": "Bearer " + os.environ["INF_API_KEY"]}
    results = {}
    async with httpx.AsyncClient(headers=headers, timeout=180, trust_env=False) as client:
        for name in ["qwen", "whisper"]:
            response = await client.post(
                endpoints[name] + "/v1/audio/transcriptions",
                data={
                    "model": "Qwen3-ASR-1.7B" if name == "qwen" else "whisper-large-v3",
                    "response_format": "json",
                    "temperature": "0",
                    "max_completion_tokens": "1024",
                },
                files={"file": (Path(args.audio).name, Path(args.audio).read_bytes(), "audio/wav")},
            )
            response.raise_for_status()
            body = response.json()
            assert isinstance(body.get("text"), str) and body["text"].strip()
            results[name] = body
            print(name, body["text"][:180], flush=True)
        for name in ["redim", "judge"]:
            response = await client.get(endpoints[name] + "/info")
            response.raise_for_status()
            identity = response.json()
            if name == "judge":
                assert len(identity["supported_categories"]) == 27
                assert "Loudness" not in identity["supported_categories"]
            payload = {
                "request_id": "smoke-" + name,
                "expected_embedding_version": identity["embedding_version"],
                "inputs": [{"id": "one", "audio": {"path": str(Path(args.audio).resolve())}}],
            }
            response = await client.post(endpoints[name] + "/embed_audio", json=payload)
            response.raise_for_status()
            body = response.json()
            assert body["results"][0]["status"] == "success"
            assert len(body["results"][0]["embedding"]) == identity["dimension"]
            results[name] = {"identity": identity, "response": body}
            print(name, "dimension", identity["dimension"], "version", identity["embedding_version"], flush=True)

    options = {"auth_token_env": "INF_API_KEY", "timeout_s": 180, "concurrency": 16}
    services = {
        "wer": {**options, "api": "openai_audio", "endpoint": endpoints["qwen"], "model": "Qwen3-ASR-1.7B"},
        "sim": {**options, "api": "voxrl_embedding", "endpoint": endpoints["redim"]},
        "judge_clap": {**options, "api": "voxrl_embedding", "endpoint": endpoints["judge"]},
    }
    if args.cache_dir:
        services["sim"]["embedding_cache"] = str(Path(args.cache_dir) / "references.sqlite")
        services["judge_clap"]["embedding_cache"] = str(Path(args.cache_dir) / "instruction-texts.sqlite")
    scorer = PrismRewardAdapter(services)
    whisper_scorer = None
    original_embeddings = ServiceClient.embeddings
    embedding_calls = []

    async def checked_embeddings(service, kind, inputs, request_id):
        if args.cache_dir and (kind == "text" or request_id.endswith("-reference")):
            raise AssertionError("the precomputed reference/text cache was not used")
        embedding_calls.append({"kind": kind, "request_id": request_id})
        return await original_embeddings(service, kind, inputs, request_id)

    ServiceClient.embeddings = checked_embeddings
    sample = SimpleNamespace(
        metadata={
            "omni_frames": 100,
            "omni_audio_path": str(Path(args.audio).resolve()),
            "language": args.language,
            "transcript_target": results["qwen"]["text"],
            "omni_prompt": {
                "modality": "speech",
                "references": {"audio1": str(Path(args.audio).resolve())},
                "reference_attributes": {"timbre": "audio1"},
                "global_instruction": {"Speed": "fast", "Gender": "female"},
            },
        }
    )
    try:
        components = await scorer.score_components(sample, ["wer", "sim", "judge_clap"])
        assert components["sim"]["raw"] > 0.999
        assert 0 <= components["judge_clap"]["raw"] <= 1
        results["client_components"] = components
        print("CLIENT_INTEGRATION_OK", {k: v["raw"] for k, v in components.items()}, flush=True)
        services["wer"] = {
            **options,
            "api": "openai_audio",
            "endpoint": endpoints["whisper"],
            "model": "whisper-large-v3",
        }
        whisper_scorer = PrismRewardAdapter(services)
        sample.metadata["transcript_target"] = results["whisper"]["text"]
        result = await whisper_scorer.score_components(sample, ["wer"])
        assert result["wer"]["raw"] == 0
        results["whisper_client"] = result
        results["client_embedding_calls"] = embedding_calls
        print("WHISPER_CLIENT_INTEGRATION_OK", flush=True)
    finally:
        ServiceClient.embeddings = original_embeddings
        await scorer.aclose()
        if whisper_scorer is not None:
            await whisper_scorer.aclose()
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoints", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--cache-dir")
    parser.add_argument("--output", required=True)
    asyncio.run(main(parser.parse_args()))
