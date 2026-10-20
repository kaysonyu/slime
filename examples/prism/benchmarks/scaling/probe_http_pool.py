"""Exercise HTTP pool churn on loopback without GPUs or external services."""

import argparse
import asyncio
import collections
import importlib.metadata
import json
import time
import traceback
from pathlib import Path

import httpx
from httpcore._async.http11 import AsyncHTTP11Connection, HTTPConnectionState


async def probe(mode, origins, requests, concurrency):
    servers, tasks = [], set()
    closed_active, errors = [], collections.Counter()
    original_close = AsyncHTTP11Connection.aclose

    async def observe_close(connection):
        if connection._state == HTTPConnectionState.ACTIVE:
            closed_active.append(
                {
                    "state": str(connection._state),
                    "stack": [f"{frame.name}:{frame.lineno}" for frame in traceback.extract_stack()[-12:]],
                }
            )
        await original_close(connection)

    async def serve(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        try:
            while True:
                header = await reader.readuntil(b"\r\n\r\n")
                length = next(
                    (
                        int(line.split(b":", 1)[1])
                        for line in header.splitlines()
                        if line.lower().startswith(b"content-length:")
                    ),
                    0,
                )
                await reader.readexactly(length)
                await asyncio.sleep(0.002)
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()
            tasks.discard(task)

    AsyncHTTP11Connection.aclose = observe_close
    try:
        for _ in range(origins):
            servers.append(await asyncio.start_server(serve, "127.0.0.1", 0))
        urls = [f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/generate" for server in servers]
        keepalive = concurrency // 8 if mode == "shared" else concurrency * 2
        limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=keepalive)
        slots = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(limits=limits, timeout=30, trust_env=False) as client:

            async def request(index):
                async with slots:
                    try:
                        response = await client.post(urls[index % origins], content=b"x")
                        response.raise_for_status()
                        assert response.content == b"ok"
                    except httpx.HTTPError as error:
                        chain = []
                        while error is not None and len(chain) < 6:
                            chain.append(type(error).__name__)
                            error = error.__cause__ or error.__context__
                        errors[" -> ".join(chain)] += 1

            started = time.perf_counter()
            await asyncio.gather(*(request(index) for index in range(requests)))
            elapsed = time.perf_counter() - started
        return {
            "mode": mode,
            "origins": origins,
            "requests": requests,
            "concurrency": concurrency,
            "max_connections": concurrency * 2,
            "max_keepalive_connections": keepalive,
            "seconds": elapsed,
            "errors": dict(errors),
            "closed_while_active_count": len(closed_active),
            "closed_while_active_examples": closed_active[:3],
        }
    finally:
        AsyncHTTP11Connection.aclose = original_close
        for server in servers:
            server.close()
            await server.wait_closed()
        for task in list(tasks):
            task.cancel()
        await asyncio.gather(*list(tasks), return_exceptions=True)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origins", type=int, default=128)
    parser.add_argument("--requests", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.origins <= 128 or not 1 <= args.concurrency <= 512 or not 1 <= args.requests <= 32768:
        parser.error("Keep the loopback probe bounded")
    if args.output.exists():
        raise FileExistsError(args.output)
    result = {
        "started": time.time(),
        "versions": {name: importlib.metadata.version(name) for name in ("httpx", "httpcore", "anyio")},
        "runs": [
            await probe(mode, args.origins, args.requests, args.concurrency) for mode in ("shared", "retained-idle")
        ],
        "scope": "Loopback CPU reproduction with real HTTP sockets; no retries. Active-close observer records stacks without adding await points before the original close. Not a GPU throughput benchmark.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    asyncio.run(main())
