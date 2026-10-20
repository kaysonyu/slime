"""Validate a frozen eval package, preflight services, or rescore saved WAVs."""

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from examples.prism.eval.build import DOMAINS, digest, read_rows
from examples.prism.eval.metrics import units
from examples.prism.eval.runtime import EvalState, evaluate, export_results

from slime_plugins.prism.reward_adapter import load_reward_config


def validate(root):
    from types import SimpleNamespace

    import soundfile as sf

    from slime_plugins.omni_rollout import _prompt

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    for name, checksum in manifest["files"].items():
        if digest(root / name) != checksum:
            raise ValueError(f"Frozen file checksum mismatch: {name}")
    for name, asset in manifest["assets"].items():
        path = root / name
        if path.is_symlink() or digest(path) != asset["sha256"]:
            raise ValueError(f"Frozen asset checksum mismatch: {name}")
        info = sf.info(path)
        if info.frames < 1 or info.samplerate < 1:
            raise ValueError(f"Invalid reference/GT audio: {name}")
    rows = read_rows(root / "all.jsonl")
    counts, ids, metrics = Counter(), set(), Counter()
    for row in rows:
        metadata = row["metadata"]
        recipe = metadata["prism_eval"]
        if recipe["id"] in ids:
            raise ValueError(f"Duplicate ID: {recipe['id']}")
        ids.add(recipe["id"])
        counts[recipe["domain"]] += 1
        metrics.update(recipe["metrics"])
        _prompt(SimpleNamespace(metadata=metadata, prompt=row["text"]))
        if not units(metadata["transcript_target"], recipe["language"], recipe["normalization"]):
            raise ValueError(f"Empty WER target: {recipe['id']}")
        for audio in [*metadata["omni_prompt"]["references"].values(), recipe.get("gt_audio")]:
            if audio:
                path = Path(audio)
                if not path.is_relative_to(root) or path.relative_to(root).as_posix() not in manifest["assets"]:
                    raise ValueError(f"Unfrozen audio dependency: {audio}")
    if dict(counts) != DOMAINS:
        raise ValueError(f"Domain quotas differ: {counts}")
    for domain, expected in DOMAINS.items():
        subset = read_rows(root / f"{domain}.jsonl")
        if subset != [r for r in rows if r["metadata"]["prism_eval"]["domain"] == domain] or len(subset) != expected:
            raise ValueError(f"Subset mismatch: {domain}")
    smoke = read_rows(root / "smoke.jsonl")
    if len(smoke) != 28 or any(r not in rows for r in smoke):
        raise ValueError("Invalid smoke subset")
    return {
        "status": "passed",
        "rows": len(rows),
        "domains": counts,
        "metrics_expected": metrics,
        "assets": len(manifest["assets"]),
        "asset_bytes": sum(a["bytes"] for a in manifest["assets"].values()),
        "manifest_sha256": digest(root / "manifest.json"),
        "gpu_generation_tested": False,
        "live_service_scoring_tested": False,
    }


async def rescore(args):
    state = EvalState(load_reward_config(args.services))
    semaphore = asyncio.Semaphore(args.concurrency)
    records = (
        [json.loads(path.read_text()) for path in sorted(args.records.glob("*.json"))]
        if args.records.is_dir()
        else read_rows(args.records)
    )
    if not records:
        raise ValueError("No archived records to rescore")
    if args.output.exists():
        raise FileExistsError(f"Use a new rescore directory: {args.output}")
    try:
        await state.preflight()

        async def one(record):
            async with semaphore:
                metadata = {
                    "prism_eval": record["recipe"],
                    "omni_prompt": record["prompt"],
                    "transcript_target": record["text"],
                    "omni_audio_path": record["audio_path"],
                    "omni_frames": record["frames"],
                    "prism_eval_generation": record["generation_status"],
                }
                record["metrics"] = await evaluate(state, metadata)
            return record

        output = await asyncio.gather(*(one(r) for r in records))
        export_results(args.output, output)
    finally:
        await state.aclose()


async def preflight(path):
    state = EvalState(load_reward_config(path))
    try:
        await state.preflight()
        return {name: await state.client.service(name).info() for name in ("sim", "judge_clap")}
    finally:
        await state.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate")
    check.add_argument("root", type=Path)
    check.add_argument("--report", type=Path)
    probe = commands.add_parser("preflight")
    probe.add_argument("--services", required=True)
    replay = commands.add_parser("rescore")
    replay.add_argument("--services", required=True)
    replay.add_argument("--records", required=True, type=Path)
    replay.add_argument("--output", required=True, type=Path)
    replay.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    if args.command == "validate":
        result = validate(args.root)
        if args.report:
            args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "preflight":
        print(json.dumps(asyncio.run(preflight(args.services)), indent=2))
    else:
        if args.concurrency < 1:
            parser.error("--concurrency must be positive")
        asyncio.run(rescore(args))


if __name__ == "__main__":
    main()
