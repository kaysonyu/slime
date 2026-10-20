"""Build-only adapter: export full moss-eval manifests before selecting rows.

Run with the moss-eval metrics Python and --moss-eval pointing at its source.
All output is confined to --output; source datasets are read-only. CV3 uses an
already prepared directory, avoiding the platform's shared-cache writer.
"""

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

SOURCES = {
    "seed-tts": "seed-tts-eval/seedtts_testset",
    "moss-tts-voice-taxonomy": "moss-tts-voice-taxonomy-eval",
    "instruction-tts": "InstructTTSEval",
    "instruction-tts-v2": "InstructTTSEval-v2",
    "mint-bench": "MINT-Bench",
    "instruction-tts-mossv2": "MOSS-Instruction-Eval-v2",
    "instruction-voiceclone": "Instruction-VoiceClone-Eval",
    "breezeblue": "BreezeBlue-Voice-Design",
    "whisper-sing": "Whisper-Sing-Eval",
    "sound-event": "Sound-Event-Eval",
    "dialect-accent": "Dialect-Accent-Eval",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moss-eval", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--cv3-prepared", type=Path, required=True)
    parser.add_argument("--bench", action="append", help="Optional subset of canonical exports")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.moss_eval / "src"))
    from benchmarks.core.registry import benchmark_definition

    args.output.mkdir(parents=True, exist_ok=True)
    for name, subdir in {**SOURCES, "cv3": None}.items():
        if args.bench and name not in args.bench:
            continue
        out = args.output / f"{name}.jsonl"
        if out.exists():
            raise FileExistsError(f"Refusing to overwrite canonical export: {out}")
        root = args.datasets / subdir if subdir else args.cv3_prepared
        if name == "instruction-tts":
            root = root / "hf_data_profile"
        elif name == "mint-bench":
            root = root / "mint-bench-profile"
        definition = benchmark_definition(name)
        importer = definition.importer.materialize
        kwargs = {}
        if "importer_options" in inspect.signature(importer).parameters:
            kwargs["importer_options"] = {
                "auto_download": False,
                "validate_full_counts": True,
                "allow_unpinned_local": True,
            }
        # CV3 passing prepared directories bypasses _ensure_shared_cache.
        bundle = importer(root, args.output, None, **kwargs)
        rows = bundle.pop("rows")
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        bundle.update(source_root=str(root), canonical_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
        (args.output / f"{name}.stats.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2))
        print(f"{name}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
