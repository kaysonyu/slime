"""Isolate JudgeCLAP preprocessing, position and adapter-precision changes."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from slime_plugins.audio_rewards.serving.audio import decode_audio
from slime_plugins.audio_rewards.serving.backends import JudgeBackend, collate_audio_features
from .benchmark import load_samples
from .validate_gpu import QUERIES, differences, judge_scores


def run(args):
    torch.set_num_threads(1)
    backend = JudgeBackend(args.base, args.adapter, args.atom_stats, merge_lora=False, feature_device="cpu")
    rows = load_samples(args.manifest, args.samples, 17, 2, 15)
    waves = [backend.prepare_audio(decode_audio({"path": row["audio"]}))[0] for row in rows]

    def original(batch):
        with torch.inference_mode():
            return backend.model.encode(**{k: v.cuda() for k, v in batch.items()}).float().cpu().numpy()

    text = original(backend.build_text_batch(backend.processor, QUERIES))
    audio = np.concatenate(
        [original(backend.build_audio_batch(backend.processor, ["input.wav"], [wave])) for wave in waves]
    )
    scores = judge_scores(audio, text, backend.scale, backend.bias)
    report = {
        "samples": rows,
        "queries": QUERIES,
        "baseline_identity": backend.info,
        "baseline_adapter_dtypes": sorted({str(p.dtype) for n, p in backend.model.named_parameters() if "lora_" in n}),
        "variants": {},
    }

    variants = ["per_audio"] if args.pipeline_only else ["cpu_positions", "gpu_features", "bf16_adapter"]
    for name in variants:
        if name == "per_audio":
            reference_batch = backend.build_audio_batch(backend.processor, ["input.wav"] * len(waves), waves)
            reference_audio = original(reference_batch)
            backend.preprocessing = "per_audio"
            with ThreadPoolExecutor(max_workers=4) as pool:
                prepared = list(pool.map(lambda wave: backend.prepare_audio(wave)[0], waves))
            actual_batch = collate_audio_features(prepared, backend.processor.tokenizer.pad_token_id)
            tensor_errors = {
                key: float((reference_batch[key] - actual_batch[key]).abs().max()) for key in reference_batch
            }
            actual_audio = backend.encode_audio(prepared)
            actual_text = backend.encode_text(QUERIES)
            original_scores = judge_scores(reference_audio, text, backend.scale, backend.bias)
            actual_scores = judge_scores(actual_audio, actual_text, backend.scale, backend.bias)
            report["variants"][name] = {
                "input_tensor_max_abs": tensor_errors,
                "audio": differences(reference_audio, actual_audio),
                "score_max_abs_vs_same_batch": float(np.abs(original_scores - actual_scores).max()),
                "score_max_abs_vs_single": float(np.abs(scores - actual_scores).max()),
            }
            print(name, json.dumps(report["variants"][name]), flush=True)
            Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            continue
        if name == "gpu_features":
            backend.feature_device = "cuda"
        if name == "bf16_adapter":
            for param_name, param in backend.model.named_parameters():
                if "lora_" in param_name:
                    param.data = param.data.to(torch.bfloat16)
        actual_text = backend.encode_text(QUERIES)
        actual_audio = np.concatenate([backend.encode_audio([wave]) for wave in waves])
        actual_scores = judge_scores(actual_audio, actual_text, backend.scale, backend.bias)
        delta = np.abs(scores - actual_scores)
        variant = {
            "audio": differences(audio, actual_audio),
            "text": differences(text, actual_text),
            "score_max_abs": float(delta.max()),
            "score_p99_abs": float(np.percentile(delta, 99)),
            "score_mean_abs": float(delta.mean()),
            "scores": actual_scores.tolist(),
            "feature_device": backend.feature_device,
            "adapter_dtypes": sorted({str(p.dtype) for n, p in backend.model.named_parameters() if "lora_" in n}),
        }
        wave = np.resize(waves[0], 5 * 16000).astype(np.float32)
        profile = []
        for batch in [1, 4, 8]:
            for _ in range(2):
                backend.encode_audio([wave] * batch)
            start = time.perf_counter()
            for _ in range(5):
                backend.encode_audio([wave] * batch)
            profile.append(
                {"batch": batch, "audio_seconds": 5, "audio_per_s": 5 * batch / (time.perf_counter() - start)}
            )
        variant["fixed_shape_pipeline"] = profile
        report["variants"][name] = variant
        print(name, json.dumps({k: v for k, v in variant.items() if k != "scores"}), flush=True)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--atom-stats", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--pipeline-only", action="store_true")
    run(parser.parse_args())
