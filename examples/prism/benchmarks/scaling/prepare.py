"""Materialize a fixed-data 32/64 GPU GRPO campaign before reserving GPUs."""

import argparse
import hashlib
import json
import subprocess
import tarfile
from itertools import islice
from pathlib import Path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--omni", type=Path, required=True)
    parser.add_argument("--megatron", type=Path, required=True)
    parser.add_argument("--asr-endpoint", required=True)
    parser.add_argument("--asr-name", required=True)
    parser.add_argument("--auth-file", type=Path, required=True)
    parser.add_argument("--job-name", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if (root / "manifest.json").exists():
        raise FileExistsError("Use a new campaign directory; never overwrite a run")
    source = Path(__file__).resolve().parents[4]
    for directory in ("data", "configs", "runs", "analysis", "services", "platform"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    references = set()
    count = 0
    with args.input.open() as stream, (root / "data/train.jsonl").open("w") as out:
        for count, line in enumerate(islice(stream, 80000), 1):
            row = json.loads(line)
            old = row["metadata"]["omni_prompt"]
            if "task_type" not in old:
                raise ValueError("Expected the fixed fullclone revision-6 source dataset")
            refs = {f"audio{i+1}": ref["uri"] for i, ref in enumerate(old["references"])}
            slots = {ref["id"]: f"audio{i+1}" for i, ref in enumerate(old["references"])}
            prompt = {
                "modality": "speech",
                "script": old["transcript_target"],
                "references": refs,
                "reference_attributes": {key: slots[value] for key, value in old["reference_targets"].items()},
                "global_instruction": {"Language": old["language"]} if old["show_language"] else None,
                "global_tokens": None,
            }
            row["metadata"] = {
                "id": f"ttsrl-fullclone-{count-1:06d}",
                "domain": "fullclone_chinese",
                "reward_category": "base",
                "transcript_target": row["script"],
                "language": "zh",
                "omni_prompt": prompt,
            }
            references.update(refs.values())
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    if count != 80000:
        raise ValueError(f"Expected 80000 fixed prompts, got {count}")
    for path in references:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    hf = args.assets / "prism/prism17b-v04-iter-0100000-r7/hf"
    native = args.assets / "prism/prism17b-v04-iter-0100000-r7/native/iter_0100000"
    codec = args.assets / "MOSS-Audio-Tokenizer-v2"
    for path in (hf / "config.json", native / "model", codec / "config.json"):
        if not path.exists():
            raise FileNotFoundError(path)
    cases = []

    def case(name, gpus, batch, replicas, workers, concurrency=16, steps=25, **extra):
        cases.append(
            dict(
                name=name,
                gpus=gpus,
                batch=batch,
                asr_replicas=replicas,
                asr_inflight=4,
                workers=workers,
                concurrency=concurrency,
                steps=steps,
                **extra,
            )
        )

    case("32-b2048-a4-w16-start", 32, 2048, 4, 16)
    case("64-b4096-a4-w16", 64, 4096, 4, 16)
    case("64-b4096-a8-w16", 64, 4096, 8, 16)
    case("64-b4096-a8-w32", 64, 4096, 8, 32)
    case("64-b4096-a16-w32", 64, 4096, 16, 32)
    case("64-b4096-a8-w32-c32", 64, 4096, 8, 32, concurrency=32)
    case("64-b4096-a8-w64", 64, 4096, 8, 64)
    case("64-b4096-a8-w32-i16", 64, 4096, 8, 32)
    cases[-1]["asr_inflight"] = 16
    case("32-b4096-a8-w32", 32, 4096, 8, 32)
    case("64-b2048-a4-w16", 64, 2048, 4, 16)
    case("32-b2048-a4-w16-end", 32, 2048, 4, 16)
    case("64-confirm", 64, 4096, 8, 32, steps=100, choose_best=True)
    case("64-resume", 64, 4096, 8, 32, steps=102, resume="64-confirm")
    manifest = dict(
        root=str(root),
        source=str(source),
        omni=str(args.omni),
        megatron=str(args.megatron),
        hf_checkpoint=str(hf),
        native_checkpoint=str(native),
        codec=str(codec),
        asr_endpoint=args.asr_endpoint,
        asr_name=args.asr_name,
        auth_file=str(args.auth_file),
        job_name=args.job_name,
        workspace="CQ-科研驾驶舱",
        project="多模态自进化科学模型",
        group="科研驾驶舱",
        image="slime-ttsrl:editable-v1",
        nodes=8,
        seed=6211,
        max_frames=1024,
        max_tokens_per_gpu=20480,
        cases=cases,
        dataset_count=count,
        reference_count=len(references),
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        dataset_sha256=hashlib.sha256((root / "data/train.jsonl").read_bytes()).hexdigest(),
        source_base="c73ed2730d480ba64e71739e7e1de7245e23618c",
    )
    manifest["source_heads"] = {
        name: subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        for name, path in (("slime", source), ("omni", args.omni), ("megatron", args.megatron))
    }
    for name, path in (("source", source), ("omni", args.omni)):
        dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True)
        if dirty.strip():
            raise ValueError(f"Commit the {name} checkout before freezing the campaign")
        destination = root / "source" / name
        destination.mkdir(parents=True, exist_ok=False)
        archive = subprocess.Popen(["git", "-C", str(path), "archive", "HEAD"], stdout=subprocess.PIPE)
        with tarfile.open(fileobj=archive.stdout, mode="r|") as stream:
            stream.extractall(destination, filter="data")
        if archive.wait():
            raise RuntimeError(f"Failed to archive {name}")
        manifest[name] = str(destination)
    original_prefix = hashlib.sha256()
    with args.input.open("rb") as stream:
        for line in islice(stream, 80000):
            original_prefix.update(line)
    manifest["historical_input_prefix_sha256"] = original_prefix.hexdigest()
    if original_prefix.hexdigest() != "307ecb34dad45b6654eb138d85bbfaa71813136a25f6689a60fc61afa08643b4":
        raise ValueError("The input does not match the measured historical 80000-prompt dataset")
    write_json(root / "manifest.json", manifest)
    write_json(
        root / "configs/student.json",
        {
            "config_cls": "MossTTSPrismPipelineConfig",
            "model_path": str(hf),
            "codec_model_path": str(codec),
            "enable_rl": True,
            "stages": {
                "tts_engine": {
                    "engine": {
                        "cuda_graph_backend_prefill": "disabled",
                        "max_running_requests": 32,
                        "max_total_tokens": 65536,
                        "cuda_graph_max_bs": 32,
                    }
                }
            },
        },
    )
    print(json.dumps({"root": str(root), "cases": len(cases), "prompts": count, "references": len(references)}))


if __name__ == "__main__":
    main()
