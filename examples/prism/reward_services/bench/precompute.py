"""Precompute frozen references or atomic query texts into a shared SQLite table.

Input JSONL: {"audio": "/path/reference.wav"} for audio, or {"text": "Pitch: high"}
for text. Cache keys are file-content SHA256 or the exact query string. Existing
rows can be resumed only with the same embedding version. No credentials persist.
"""

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
from pathlib import Path

import httpx
import numpy as np

from slime_plugins.audio_rewards.protocol import PROTOCOL


def audio_key(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


async def run(args):
    token = os.environ.get("INF_API_KEY")
    headers = {"Authorization": "Bearer " + token} if token else {}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(headers=headers, timeout=300, trust_env=False) as client:
        response = await client.get(args.endpoint.rstrip("/") + "/info")
        response.raise_for_status()
        identity = response.json()
        if identity.get("protocol") != PROTOCOL:
            raise ValueError("not a frozen embedding service")
        with sqlite3.connect(output) as database:
            database.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            database.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")
            saved = database.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
            if saved and json.loads(saved[0])["embedding_version"] != identity["embedding_version"]:
                raise ValueError("existing table belongs to a different embedding version")
            database.execute("INSERT OR REPLACE INTO metadata VALUES ('identity', ?)", (json.dumps(identity),))
            database.commit()
            pending = {}
            count = 0

            async def flush():
                nonlocal count
                if not pending:
                    return
                keys = list(pending)
                inputs = [{"id": str(i), **pending[key]} for i, key in enumerate(keys)]
                payload = {
                    "request_id": f"precompute-{count}",
                    "expected_embedding_version": identity["embedding_version"],
                    "inputs": inputs,
                }
                response = await client.post(args.endpoint.rstrip("/") + f"/embed_{args.kind}", json=payload)
                response.raise_for_status()
                body = response.json()
                if (
                    body.get("embedding_version") != identity["embedding_version"]
                    or body.get("request_id") != payload["request_id"]
                    or [x.get("id") for x in body.get("results", [])] != [x["id"] for x in inputs]
                ):
                    raise ValueError("embedding response version/ids differ")
                for key, result in zip(keys, body["results"], strict=True):
                    if result.get("status") != "success":
                        raise ValueError(f"cannot precompute input {key}: {result.get('reason')}")
                    vector = np.asarray(result["embedding"], dtype="<f4")
                    if vector.shape != (identity["dimension"],) or not np.isfinite(vector).all():
                        raise ValueError("invalid embedding")
                    database.execute("INSERT OR IGNORE INTO embeddings VALUES (?, ?)", (key, vector.tobytes()))
                database.commit()
                count += len(keys)
                pending.clear()
                print(f"precomputed {count} new embeddings", flush=True)

            with Path(args.manifest).open() as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if args.kind == "audio":
                        path = str(Path(row["audio"]).resolve(strict=True))
                        key, value = audio_key(path), {"audio": {"path": path}}
                    else:
                        key, value = row["text"], {"text": row["text"]}
                    if key in pending or database.execute("SELECT 1 FROM embeddings WHERE key=?", (key,)).fetchone():
                        continue
                    pending[key] = value
                    if len(pending) == args.batch_size:
                        await flush()
            await flush()
            print("total", database.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0], "output", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--kind", choices=["audio", "text"], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 64:
        parser.error("batch-size must be 1..64")
    asyncio.run(run(args))
