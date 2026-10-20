"""Batch audio/text embedding API used by the frozen Prism reward clients."""

import argparse
import asyncio
import json
import logging
import os
import shutil
import socket
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from ..protocol import PROTOCOL
from .audio import InvalidAudio, decode_audio
from .batching import Batcher, ServiceBusy

LOG = logging.getLogger(__name__)


def create_app(
    factory,
    *,
    max_batch=32,
    max_cost=16384,
    window_ms=5,
    max_pending=None,
    decode_workers=4,
    max_items=64,
    max_seconds=120,
    processes=1,
    ready_dir=None,
):
    if max_pending is None:
        max_pending = max(max_items, 8 * max_batch)
    if max_pending < 1:
        raise ValueError("max_pending must be positive")

    @asynccontextmanager
    async def lifespan(app):
        backend = factory()
        backend.info["protocol"] = PROTOCOL
        backend.info["limits"] = {
            "max_items": max_items,
            "max_audio_seconds": max_seconds,
            "max_batch": max_batch,
            "max_batch_cost": max_cost,
            "max_pending": max_pending,
            "gpu_workers": 1,
            "processes": processes,
            "decode_workers": decode_workers,
            "batch_window_ms": window_ms,
        }
        backend.info["worker_pid"] = os.getpid()
        backend.info["worker_id"] = f"{socket.gethostname()}:{os.getpid()}"
        app.state.backend = backend
        app.state.decode = ThreadPoolExecutor(max_workers=decode_workers, thread_name_prefix="audio-decode")
        audio = Batcher(
            backend.encode_audio,
            max_batch=max_batch,
            max_cost=max_cost,
            window_ms=window_ms,
        )
        text = (
            Batcher(backend.encode_text, max_batch=64, max_cost=65536, window_ms=window_ms)
            if hasattr(backend, "encode_text")
            else None
        )
        app.state.audio, app.state.text = audio, text
        app.state.requests = 0
        app.state.active = 0
        app.state.peak_active = 0
        app.state.rejected_inputs = 0
        app.state.ready = False
        audio.start()
        if text:
            text.start()
        try:
            rng = np.random.default_rng(17)
            wave, key, cost, _ = await asyncio.get_running_loop().run_in_executor(
                app.state.decode, backend.prepare_audio, (rng.standard_normal(32000) * 0.03).astype(np.float32)
            )
            result = await audio.submit(wave, key, cost)
            _validate_vector(result, backend.info)
            if hasattr(backend, "warmup"):
                await asyncio.get_running_loop().run_in_executor(audio.pool, backend.warmup, max_batch, max_cost)
            if text:
                _validate_vector(await text.submit("Pitch: high", "text", 128), backend.info)
            app.state.ready = True
            if ready_dir:
                Path(ready_dir, str(os.getpid())).touch()
            LOG.info("READY %s", backend.info)
            yield
        finally:
            app.state.ready = False
            if ready_dir:
                Path(ready_dir, str(os.getpid())).unlink(missing_ok=True)
            await audio.close()
            if text:
                await text.close()
            app.state.decode.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="Prism frozen embeddings", lifespan=lifespan)

    @app.exception_handler(ServiceBusy)
    async def busy(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=503, headers={"Retry-After": "1"})

    @app.get("/health")
    async def health():
        if not getattr(app.state, "ready", False):
            raise HTTPException(503, "not ready")
        if ready_dir:
            ready = sum(Path("/proc", item.name).exists() for item in Path(ready_dir).iterdir() if item.name.isdigit())
            if ready < processes:
                raise HTTPException(503, "waiting for all embedding workers to warm up")
        return {"status": "ready"}

    @app.get("/info")
    async def info():
        await health()
        return app.state.backend.info

    @app.get("/metrics")
    async def metrics():
        await health()
        return {
            "requests": app.state.requests,
            "active": app.state.active,
            "active_limit": max_pending,
            "peak_active": app.state.peak_active,
            "rejected_inputs": app.state.rejected_inputs,
            "audio": app.state.audio.metrics(),
            "text": app.state.text.metrics() if app.state.text else None,
            "backend_timings": getattr(app.state.backend, "timings", {}),
            "worker_pid": os.getpid(),
            "worker_id": app.state.backend.info["worker_id"],
        }

    async def parse(request):
        await health()
        try:
            body = await request.json()
        except ValueError as exc:
            raise HTTPException(400, "invalid JSON") from exc
        if not isinstance(body, dict) or not isinstance(body.get("request_id"), str) or not body["request_id"]:
            raise HTTPException(400, "request_id must be a nonempty string")
        version = body.get("expected_embedding_version")
        if not isinstance(version, str) or not version:
            raise HTTPException(400, "expected_embedding_version is required; first query /info")
        if version != app.state.backend.info["embedding_version"]:
            raise HTTPException(409, "embedding version changed")
        inputs = body.get("inputs")
        if not isinstance(inputs, list) or not 0 < len(inputs) <= max_items:
            raise HTTPException(400, f"inputs must contain 1..{max_items} items")
        if any(not isinstance(x, dict) or not isinstance(x.get("id"), str) or not x["id"] for x in inputs):
            raise HTTPException(400, "every input requires a nonempty string id")
        if len({x["id"] for x in inputs}) != len(inputs):
            raise HTTPException(400, "input ids must be unique within a request")
        return body

    def response(body, results):
        identity = app.state.backend.info
        return {
            "protocol": PROTOCOL,
            "request_id": body["request_id"],
            "worker_pid": os.getpid(),
            **{k: identity[k] for k in ["embedding_version", "dimension", "dtype", "normalized"]},
            "results": results,
        }

    async def audio_item(item):
        started = time.monotonic()
        try:
            loop = asyncio.get_running_loop()
            try:
                wave = await loop.run_in_executor(
                    app.state.decode, partial(decode_audio, item.get("audio"), max_seconds)
                )
            except InvalidAudio:
                raise
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            wave, key, cost, meta = await loop.run_in_executor(app.state.decode, app.state.backend.prepare_audio, wave)
            if cost > app.state.audio.max_cost:
                raise HTTPException(413, "audio exceeds the configured inference budget")
            embedding = await app.state.audio.submit(wave, key, cost)
            _validate_vector(embedding, app.state.backend.info)
            return {
                "id": item["id"],
                "status": "success",
                "embedding": embedding.tolist(),
                "audio_meta": meta,
                "server_ms": (time.monotonic() - started) * 1000,
            }
        except InvalidAudio as exc:
            return {"id": item["id"], "status": "invalid_audio", "reason": exc.reason, **exc.details}

    async def run_items(body, operation):
        count = len(body["inputs"])
        # This is the only backlog limit. Reservation and increment contain no
        # await, so audio/text requests share one atomic event-loop budget.
        if app.state.active + count > max_pending:
            app.state.rejected_inputs += count
            raise ServiceBusy("request admission limit reached")
        app.state.active += count
        app.state.peak_active = max(app.state.peak_active, app.state.active)
        app.state.requests += 1
        try:
            # Finish all readers before returning an error: callers own their files.
            pending = asyncio.gather(*(operation(item) for item in body["inputs"]), return_exceptions=True)
            try:
                results = await asyncio.shield(pending)
            except asyncio.CancelledError:
                # Cancelling an executor Future does not stop its CPU/GPU work.
                # Keep the admission reservation until those readers finish.
                while not pending.done():
                    try:
                        await asyncio.shield(pending)
                    except asyncio.CancelledError:
                        pass
                pending.result()
                raise
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            # Avoid FastAPI walking thousands of already-validated scalar values.
            return JSONResponse(response(body, results))
        finally:
            app.state.active -= count

    @app.post("/embed_audio")
    async def embed_audio(request: Request):
        return await run_items(await parse(request), audio_item)

    @app.post("/embed_text")
    async def embed_text(request: Request):
        body = await parse(request)
        if app.state.text is None:
            raise HTTPException(404, "this model has no text encoder")
        categories = set(app.state.backend.info["supported_categories"])
        for item in body["inputs"]:
            text = item.get("text")
            if not isinstance(text, str) or not text or len(text) > 4096:
                raise HTTPException(422, "text must be nonempty and at most 4096 characters")
            category, separator, value = text.partition(":")
            if not separator or category not in categories or not value.strip() or "<|" in text or "<emb>" in text:
                raise HTTPException(422, "text must be a supported 'Category: value' query without special tokens")

        async def text_item(item):
            # Validate in the admitted CPU pool before this item can affect
            # unrelated requests sharing the GPU batch.
            try:
                await asyncio.get_running_loop().run_in_executor(
                    app.state.decode, app.state.backend.validate_text, item["text"]
                )
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            vector = await app.state.text.submit(item["text"], "text", 1024)
            _validate_vector(vector, app.state.backend.info)
            return {"id": item["id"], "status": "success", "embedding": vector.tolist()}

        return await run_items(body, text_item)

    return app


def _validate_vector(vector, identity):
    if vector.shape != (identity["dimension"],) or not np.isfinite(vector).all():
        raise RuntimeError("model returned an invalid embedding")
    if identity["normalized"] and abs(float(np.square(vector.astype(np.float64)).sum()) - 1) > 1e-3:
        raise RuntimeError("model returned a non-unit speaker embedding")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["redimnet2", "judge_clap"], required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--base")
    parser.add_argument("--adapter")
    parser.add_argument("--atom-stats")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--attention", choices=["sdpa", "original"], default="sdpa")
    parser.add_argument("--batching", choices=["exact", "masked"], default="masked")
    parser.add_argument("--merge-lora", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=None, help="Enabled by default for ReDimNet2"
    )
    parser.add_argument("--feature-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--preprocessing", choices=["batch", "per_audio"], default="per_audio")
    parser.add_argument("--adapter-precision", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--max-batch", type=int, help="Defaults to 32 for ReDimNet2, 16 for JudgeCLAP")
    parser.add_argument("--max-cost", type=int, default=16384)
    parser.add_argument("--batch-window-ms", type=float, help="Defaults to 5 ms for ReDimNet2, 10 ms for JudgeCLAP")
    parser.add_argument(
        "--max-pending", type=int, help="Per-process admitted items; defaults to eight batches, at least 64"
    )
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument(
        "--processes", type=int, help="Model processes on one GPU; defaults to 1 for ReDimNet2, 4 for JudgeCLAP"
    )
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--max-audio-seconds", type=float, default=120)
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    # Use the measured single-H200 settings unless explicitly overridden.
    judge = args.model == "judge_clap"
    if args.compile is None:
        args.compile = not judge
    if args.max_batch is None:
        args.max_batch = 16 if judge else 32
    if args.batch_window_ms is None:
        args.batch_window_ms = 10 if judge else 5
    if args.processes is None:
        args.processes = 4 if judge else 1
    if args.max_pending is None:
        args.max_pending = max(64, 8 * args.max_batch)
    for name in ["max_batch", "max_cost", "max_pending", "decode_workers", "processes", "cpu_threads"]:
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.batch_window_ms < 0 or args.max_audio_seconds <= 0:
        parser.error("invalid timing/length limit")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import uvicorn

    if args.model == "redimnet2" and not args.checkpoint:
        parser.error("--checkpoint is required")
    if args.model == "redimnet2" and args.batching == "masked" and args.max_cost & (args.max_cost - 1):
        parser.error("masked ReDimNet2 max-cost must be a power of two to bound padded GPU batches")
    if args.model == "judge_clap" and not all([args.base, args.adapter, args.atom_stats]):
        parser.error("JudgeCLAP requires --base, --adapter and --atom-stats")
    args.ready_dir = tempfile.mkdtemp(prefix="prism-embedding-ready-") if args.processes > 1 else None
    os.environ["PRISM_EMBEDDING_CONFIG"] = json.dumps(vars(args))
    try:
        uvicorn.run(
            "slime_plugins.audio_rewards.serving.server:application",
            factory=True,
            workers=args.processes,
            host=args.host,
            port=args.port,
            access_log=False,
        )
    finally:
        if args.ready_dir:
            shutil.rmtree(args.ready_dir, ignore_errors=True)


def application():
    """Uvicorn spawn-safe application factory; every process shares one GPU."""
    import torch

    from .backends import JudgeBackend, RedimBackend

    args = SimpleNamespace(**json.loads(os.environ["PRISM_EMBEDDING_CONFIG"]))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(args.cpu_threads)
    if args.model == "redimnet2":
        factory = partial(
            RedimBackend,
            args.checkpoint,
            device=args.device,
            precision=args.precision,
            compile_model=args.compile,
            attention=args.attention,
            batching=args.batching,
        )
    else:
        factory = partial(
            JudgeBackend,
            args.base,
            args.adapter,
            args.atom_stats,
            device=args.device,
            merge_lora=args.merge_lora,
            compile_model=args.compile,
            feature_device=args.feature_device,
            adapter_precision=args.adapter_precision,
            preprocessing=args.preprocessing,
        )
    return create_app(
        factory,
        max_batch=args.max_batch,
        max_cost=args.max_cost,
        window_ms=args.batch_window_ms,
        max_pending=args.max_pending,
        decode_workers=args.decode_workers,
        max_seconds=args.max_audio_seconds,
        processes=args.processes,
        ready_dir=args.ready_dir,
    )


if __name__ == "__main__":
    main()
