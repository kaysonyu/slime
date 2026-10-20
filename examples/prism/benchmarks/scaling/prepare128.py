"""Freeze the authorized 64/128-H200, B4096/B8192 comparison campaign."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

from prepare import write_json


def matrix():
    cases = []

    def add(name, gpus, batch, asr, workers, routers, **extra):
        cases.append(
            dict(
                name=name,
                gpus=gpus,
                batch=batch,
                asr_replicas=asr,
                workers=workers,
                router_processes=routers,
                router_policy="round_robin",
                asr_inflight=4,
                concurrency=16,
                steps=25,
                **extra,
            )
        )

    add("64-b4096-a16-w32-r8-start", 64, 4096, 16, 32, 8)
    add("128-b4096-a16-w32-r8", 128, 4096, 16, 32, 8)
    add("128-b8192-a16-w64-r16", 128, 8192, 16, 64, 16)
    add("128-b8192-a32-w64-r16", 128, 8192, 32, 64, 16)
    add("128-b8192-a32-w64-r8", 128, 8192, 32, 64, 8)
    add("128-b8192-a32-w128-r16", 128, 8192, 32, 128, 16)
    add("128-b8192-a32-w64-r16-c8", 128, 8192, 32, 64, 16)
    cases[-1]["concurrency"] = 8
    add("128-b8192-a32-w64-r16-i8", 128, 8192, 32, 64, 16)
    cases[-1]["asr_inflight"] = 8
    add("64-b4096-a16-w32-r8-end", 64, 4096, 16, 32, 8)
    for gpus, batch, replicas, workers, routers in ((64, 4096, 16, 32, 8), (128, 8192, 32, 64, 16)):
        add(f"{gpus}-confirm", gpus, batch, replicas, workers, routers, choose_best=True, save_interval=25)
        cases[-1]["steps"] = 100
        add(f"{gpus}-resume", gpus, batch, replicas, workers, routers, resume=f"{gpus}-confirm")
        cases[-1]["steps"] = 102
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--omni", type=Path, required=True)
    parser.add_argument("--asr-endpoint", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if root.exists():
        raise FileExistsError("Use an entirely new campaign root")
    previous = json.loads((args.previous / "manifest.json").read_text())
    source = Path(__file__).resolve().parents[4]
    heads = {}
    for name, path in (("source", source), ("omni", args.omni)):
        if subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip():
            raise RuntimeError(f"Freeze only committed code: {name}")
        heads[name] = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    for name, path in (("source", source), ("omni", args.omni)):
        destination = root / "source" / name
        destination.mkdir(parents=True)
        archive = subprocess.Popen(["git", "-C", str(path), "archive", "HEAD"], stdout=subprocess.PIPE)
        with tarfile.open(fileobj=archive.stdout, mode="r|") as stream:
            stream.extractall(destination, filter="data")
        if archive.wait():
            raise RuntimeError(f"Archive failed: {name}")
    for name in ("data", "configs", "runs", "analysis", "services", "platform"):
        (root / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.previous / "data/train.jsonl", root / "data/train.jsonl")
    shutil.copy2(args.previous / "configs/student.json", root / "configs/student.json")
    from slime_plugins.omni_rollout import _prompt

    references = set()
    with (root / "data/train.jsonl").open() as stream:
        for count, line in enumerate(stream, 1):
            row = json.loads(line)
            prompt = row["metadata"]["omni_prompt"]
            _prompt(SimpleNamespace(prompt=prompt, metadata={}))
            references.update(prompt["references"].values())
    assert count == 80000 and all(Path(path).is_file() for path in references)
    cases = matrix()
    entry = root / "source/source/slime/backends/sglang_utils/omni_router.py"
    for case in cases:
        case.update(
            router_python=str(root / "source/source/examples/prism/benchmarks/scaling/router_python.py"),
            router_source_commit=heads["source"],
            router_entry_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),
        )
    manifest = {
        key: previous[key]
        for key in (
            "hf_checkpoint",
            "native_checkpoint",
            "codec",
            "auth_file",
            "megatron",
            "workspace",
            "project",
            "group",
            "image",
            "seed",
            "max_frames",
            "max_tokens_per_gpu",
            "source_base",
        )
    }
    manifest.update(
        root=str(root),
        source=str(root / "source/source"),
        omni=str(root / "source/omni"),
        source_heads=heads,
        previous_campaign=str(args.previous),
        nodes=16,
        job_name="prism-grpo128-matrix-1020",
        asr_name="prism-grpo128-asr-1020",
        asr_endpoint=args.asr_endpoint,
        approved_asr_replicas=[16, 32],
        max_asr_gpus=32,
        dataset_count=count,
        reference_count=len(references),
        dataset_sha256=hashlib.sha256((root / "data/train.jsonl").read_bytes()).hexdigest(),
        cases=cases,
        target_weak_scaling_efficiency=0.9,
    )
    manifest["source_heads"]["megatron"] = subprocess.check_output(
        ["git", "-C", manifest["megatron"], "rev-parse", "HEAD"], text=True
    ).strip()
    controls = ["64-b4096-a16-w32-r8-start", "64-b4096-a16-w32-r8-end"]
    manifest["comparisons"] = [
        dict(
            name="matched_strong_B4096",
            controls=controls,
            candidate="128-b4096-a16-w32-r8",
            batch_ratio=1,
            gpu_ratio=2,
        )
    ]
    for case in cases:
        if case["name"].startswith("128-b8192-"):
            manifest["comparisons"].append(
                dict(name=case["name"], controls=controls, candidate=case["name"], batch_ratio=2, gpu_ratio=2)
            )
    write_json(root / "manifest.json", manifest)
    print(json.dumps({"root": str(root), "cases": len(cases), "source_heads": heads, "prompts": count}))


if __name__ == "__main__":
    main()
