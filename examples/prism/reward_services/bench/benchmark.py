"""Measure real Serving endpoints with full audio and explicit timing windows.

Warmup is continuous with measurement. Throughput counts successful audio items
completed within the measurement window only. Latency includes the drain of all
requests launched during that window, so late replies are not silently censored.
Credentials are read from an environment variable and never persisted.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import aiohttp
import numpy as np

from slime_plugins.audio_rewards.protocol import PROTOCOL


def load_samples(path, limit=0, seed=42, min_seconds=0, max_seconds=float("inf")):
    samples = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    samples = [x for x in samples if min_seconds <= x["duration_s"] <= max_seconds]
    random.Random(seed).shuffle(samples)
    if limit:
        samples = samples[:limit]
    if not samples:
        raise ValueError("no audio samples selected")
    for sample in samples:
        if not Path(sample["audio"]).is_file():
            raise ValueError(f"missing audio: {sample['audio']}")
    return samples


def validate_embeddings(body, request_id, ids, identity):
    if not isinstance(body, dict) or body.get("protocol") != PROTOCOL or body.get("request_id") != request_id:
        raise ValueError("embedding envelope differs")
    for name in ["embedding_version", "dimension", "normalized", "dtype"]:
        if body.get(name) != identity[name]:
            raise ValueError(f"embedding identity differs: {name}")
    results = body.get("results")
    if not isinstance(results, list) or [x.get("id") for x in results] != ids:
        raise ValueError("embedding response order/ids differ")
    ok, invalid = 0, Counter()
    for result in results:
        if result.get("status") == "invalid_audio":
            invalid[result.get("reason", "unknown")] += 1
            continue
        if result.get("status") != "success":
            raise ValueError("unknown embedding result status")
        vector = result.get("embedding")
        if (
            not isinstance(vector, list)
            or len(vector) != identity["dimension"]
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector)
        ):
            raise ValueError("invalid embedding vector")
        if identity["normalized"] and abs(math.fsum(x * x for x in vector) - 1) > 1e-3:
            raise ValueError("speaker vector is not normalized")
        ok += 1
    return ok, invalid


def summarize(records, begin, deadline):
    completed = [r for r in records if begin <= r["end"] < deadline]
    launched = [r for r in records if begin <= r["start"] < deadline]
    latencies = [1000 * (r["end"] - r["start"]) for r in launched]
    errors = Counter(r["error"] for r in completed if r["error"])
    errors_all = Counter(r["error"] for r in launched if r["error"])
    seconds = deadline - begin
    successful = sum(r["success_items"] for r in completed)
    return {
        "window_seconds": seconds,
        "completed_requests": len(completed),
        "successful_audio_items": successful,
        "successful_audio_per_s": successful / seconds,
        "audio_seconds_per_s": sum(r["success_audio_seconds"] for r in completed) / seconds,
        "request_per_s": len(completed) / seconds,
        "window_errors": dict(errors),
        "launched_requests": len(launched),
        "launched_errors": dict(errors_all),
        "warmup_errors": dict(Counter(r["error"] for r in records if r["start"] < begin and r["error"])),
        "completed_after_window": sum(r["end"] >= deadline for r in launched),
        "drain_seconds": max(0, max((r["end"] for r in launched), default=deadline) - deadline),
        "latency_ms": {f"p{p}": float(np.percentile(latencies, p)) if latencies else None for p in [50, 95, 99]},
        "invalid_audio_items": sum(r["invalid_items"] for r in completed),
        "worker_distribution": dict(Counter(r["worker_pid"] for r in completed if r.get("worker_pid") is not None)),
    }


async def run(args):
    samples = load_samples(args.manifest, args.limit, args.seed, args.min_seconds, args.max_seconds)
    endpoint = args.endpoint.rstrip("/")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    token = os.environ.get(args.token_env)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if args.require_auth and not token:
        raise ValueError(f"missing {args.token_env}; inject the API key into the benchmark process")
    encoded = {}
    if args.kind == "asr":
        if args.batch_size != 1:
            raise ValueError("OpenAI transcription takes one audio file per request")
        encoded = {x["audio"]: Path(x["audio"]).read_bytes() for x in samples}
    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(
        headers=headers, connector=connector, timeout=timeout, trust_env=False
    ) as session:
        route = "/v1/models" if args.kind == "asr" else "/info"
        async with session.get(endpoint + route) as response:
            response.raise_for_status()
            identity = await response.json()
        if args.kind == "asr":
            model_name = args.model or identity["data"][0]["id"]
        else:
            if identity.get("protocol") != PROTOCOL:
                raise ValueError("incompatible embedding service")
            model_name = identity["model"]

        async def send(batch, request_id):
            started = time.perf_counter()
            record = {
                "start": started,
                "success_items": 0,
                "success_audio_seconds": 0,
                "invalid_items": 0,
                "error": None,
                "request_id": request_id,
                "audio_ids": [x["id"] for x in batch],
            }
            try:
                if args.kind == "asr":
                    sample = batch[0]
                    form = aiohttp.FormData()
                    form.add_field(
                        "file",
                        encoded[sample["audio"]],
                        filename=Path(sample["audio"]).name,
                        content_type="audio/flac" if sample["audio"].endswith(".flac") else "audio/wav",
                    )
                    form.add_field("model", model_name)
                    form.add_field("response_format", "json")
                    form.add_field("temperature", "0")
                    if args.max_completion_tokens:
                        form.add_field("max_completion_tokens", str(args.max_completion_tokens))
                    request = session.post(endpoint + "/v1/audio/transcriptions", data=form)
                else:
                    payload = {
                        "request_id": request_id,
                        "expected_embedding_version": identity["embedding_version"],
                        "inputs": [{"id": str(i), "audio": {"path": x["audio"]}} for i, x in enumerate(batch)],
                    }
                    request = session.post(endpoint + "/embed_audio", json=payload)
                async with request as response:
                    if response.status != 200:
                        record["error"] = f"http_{response.status}"
                        record["error_detail"] = (await response.text())[:250]
                    else:
                        body = await response.json()
                        record["worker_pid"] = body.get("worker_pid")
                        if args.kind == "asr":
                            if not isinstance(body.get("text"), str):
                                raise ValueError("ASR response lacks text")
                            record["success_items"] = 1
                            record["transcript"] = body["text"]
                            record["success_audio_seconds"] = sample["duration_s"]
                        else:
                            ok, invalid = validate_embeddings(
                                body, request_id, [str(i) for i in range(len(batch))], identity
                            )
                            record["success_items"], record["invalid_items"] = ok, sum(invalid.values())
                            record["invalid_reasons"] = dict(invalid)
                            record["success_audio_seconds"] = sum(
                                x.get("audio_meta", {}).get("processed_seconds", sample["duration_s"])
                                for sample, x in zip(batch, body["results"], strict=True)
                                if x["status"] == "success"
                            )
            except asyncio.TimeoutError:
                record["error"] = "timeout"
            except aiohttp.ClientError as exc:
                record["error"] = type(exc).__name__
            except (ValueError, KeyError, TypeError) as exc:
                record["error"] = "contract_error"
                record["error_detail"] = str(exc)[:250]
            record["end"] = time.perf_counter()
            return record

        results = []
        for concurrency in args.concurrency:
            records = []
            counter = 0
            start = time.perf_counter()
            started_unix = time.time()
            begin = start + args.warmup
            deadline = begin + args.duration

            async def worker(deadline=deadline, records=records, concurrency=concurrency):
                nonlocal counter
                while time.perf_counter() < deadline:
                    position = counter
                    counter += args.batch_size
                    batch = [samples[(position + j) % len(samples)] for j in range(args.batch_size)]
                    records.append(await send(batch, f"bench-{concurrency}-{position}"))

            await asyncio.gather(*(worker() for _ in range(concurrency)))
            result = {
                "concurrency": concurrency,
                "http_batch_size": args.batch_size,
                "started_unix": started_unix,
                "measurement_started_unix": started_unix + args.warmup,
                "measurement_ended_unix": started_unix + args.warmup + args.duration,
                **summarize(records, begin, deadline),
            }
            results.append(result)
            with (out / f"requests-c{concurrency}.jsonl").open("w") as stream:
                for record in records:
                    record["start"] -= start
                    record["end"] -= start
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(result, ensure_ascii=False), flush=True)
            # Persist after every level so an interrupted experiment remains reviewable.
            report = {
                "endpoint": endpoint,
                "kind": args.kind,
                "model": model_name,
                "identity": identity,
                "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                "samples": len(samples),
                "duration_distribution_s": {
                    "min": min(x["duration_s"] for x in samples),
                    "max": max(x["duration_s"] for x in samples),
                    "mean": sum(x["duration_s"] for x in samples) / len(samples),
                    "p50": float(np.median([x["duration_s"] for x in samples])),
                },
                "warmup_s": args.warmup,
                "client_audio_preloaded": args.kind == "asr",
                "max_completion_tokens": args.max_completion_tokens,
                "transport": "multipart_upload" if args.kind == "asr" else "shared_path",
                "results": results,
            }
            (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if any(row["launched_errors"] or row["warmup_errors"] for row in results):
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--kind", choices=["asr", "embedding"], required=True)
    parser.add_argument("--model")
    parser.add_argument("--max-completion-tokens", type=int)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--warmup", type=float, default=5)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-seconds", type=float, default=0)
    parser.add_argument("--max-seconds", type=float, default=float("inf"))
    parser.add_argument("--token-env", default="INF_API_KEY")
    parser.add_argument("--require-auth", action="store_true")
    args = parser.parse_args()
    if min(args.concurrency) < 1 or args.batch_size < 1 or args.duration <= 0 or args.warmup < 0:
        parser.error("invalid concurrency, batch size or timing")
    raise SystemExit(asyncio.run(run(args)))
