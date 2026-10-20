"""Compare optimized inference with the pinned upstream path on one GPU.

Writes complete numerical evidence and small fixed-length microbenchmarks.
Run inside an idle experiment Serving container, not alongside a load test.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from slime_plugins.audio_rewards.serving.audio import decode_audio
from slime_plugins.audio_rewards.serving.backends import JudgeBackend, RedimBackend
from .benchmark import load_samples

QUERIES = [
    "Gender: male",
    "Gender: female",
    "Pitch: high",
    "Pitch: low",
    "Speed: fast",
    "Speed: slow",
    "Emotion and psychological state: happy",
    "Phonation mode: whispering",
    "Speed: 正常",
    "Emotion and psychological state: 平静",
    "Pitch: 略高",
]


def differences(reference, actual):
    reference, actual = np.asarray(reference, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    delta = np.abs(reference - actual)
    cosine = (reference * actual).sum(-1) / (np.linalg.norm(reference, axis=-1) * np.linalg.norm(actual, axis=-1))
    return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean()), "min_cosine": float(cosine.min())}


def judge_scores(audio, text, scale, bias):
    a, t = torch.as_tensor(audio).float(), torch.as_tensor(text).float()
    return torch.sigmoid((a @ t.T - t.square().sum(-1)[None]) * scale + bias).numpy()


def run(args):
    torch.set_num_threads(2)
    samples = load_samples(args.manifest, args.samples, 17, 2, 15)
    waves = [decode_audio({"path": row["audio"]}) for row in samples]
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "samples": samples,
        "model": args.model,
        "measurements": {},
    }
    if args.model == "redimnet2":
        baseline = RedimBackend(args.checkpoint, precision="float32", attention="original")
        with torch.inference_mode():
            reference = np.concatenate(
                [F.normalize(baseline.model(torch.from_numpy(w)[None].cuda()), dim=-1).cpu().numpy() for w in waves]
            )
        del baseline
        gc.collect()
        torch.cuda.empty_cache()
        for precision in args.precisions:
            backend = RedimBackend(
                args.checkpoint,
                precision=precision,
                attention="sdpa",
                compile_model=args.compile,
                batching=args.batching,
            )
            actual = np.concatenate([backend.encode_audio([wave]) for wave in waves])
            comparison = differences(reference, actual)
            comparison["max_pairwise_cosine_error"] = float(np.abs(reference @ reference.T - actual @ actual.T).max())
            if args.batching == "masked":
                mixed = backend.encode_audio(waves[:4])
                comparison["mixed_length_batch"] = differences(reference[:4], mixed)
                comparison["mixed_length_batch"]["max_pairwise_cosine_error"] = float(
                    np.abs(reference[:4] @ reference[:4].T - mixed @ mixed.T).max()
                )
            # Different waveform lengths within one exact feature-length bucket.
            group = [waves[0][:48000], waves[1][:47840]]
            if min(map(len, group)) == 47840 and len(group[0]) == 48000:
                solo = np.concatenate([backend.encode_audio([w]) for w in group])
                batched = backend.encode_audio(group)
                comparison["batch_independence"] = differences(solo, batched)
            profile = []
            wave = np.resize(waves[0], 5 * 16000).astype(np.float32)
            for batch in [1, 4, 8, 16, 32]:
                for _ in range(2):
                    backend.encode_audio([wave] * batch)
                torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(5):
                    backend.encode_audio([wave] * batch)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                profile.append(
                    {
                        "batch": batch,
                        "audio_seconds": 5,
                        "audio_per_s": 5 * batch / seconds,
                        "mean_batch_ms": seconds / 5 * 1000,
                    }
                )
            report["measurements"][precision] = {
                "correctness": comparison,
                "fixed_shape_pipeline": profile,
                "identity": backend.info,
            }
            print(precision, json.dumps(comparison), flush=True)
            del backend
            gc.collect()
            torch.cuda.empty_cache()
    else:
        backend = JudgeBackend(args.base, args.adapter, args.atom_stats, merge_lora=False)
        prepared = [backend.prepare_audio(w)[0] for w in waves]
        text = backend.encode_text(QUERIES)
        audio = np.concatenate([backend.encode_audio([w]) for w in prepared])
        scores = judge_scores(audio, text, backend.scale, backend.bias)
        # Different lengths in a padded processor batch must preserve embeddings.
        batch_audio = backend.encode_audio(prepared)
        report["measurements"]["unmerged_batch"] = differences(audio, batch_audio)
        report["measurements"]["unmerged_batch"]["max_score_error"] = float(
            np.abs(scores - judge_scores(batch_audio, text, backend.scale, backend.bias)).max()
        )
        baseline_identity = backend.info
        del backend
        gc.collect()
        torch.cuda.empty_cache()
        merged = JudgeBackend(args.base, args.adapter, args.atom_stats, merge_lora=True)
        merged_text = merged.encode_text(QUERIES)
        merged_audio = np.concatenate([merged.encode_audio([w]) for w in prepared])
        merged_scores = judge_scores(merged_audio, merged_text, merged.scale, merged.bias)
        score_delta = np.abs(scores - merged_scores)
        report["measurements"]["merge"] = {
            "audio": differences(audio, merged_audio),
            "text": differences(text, merged_text),
            "max_score_error": float(score_delta.max()),
            "mean_score_error": float(score_delta.mean()),
            "p99_score_error": float(np.percentile(score_delta, 99)),
            "baseline_scores": scores.tolist(),
            "merged_scores": merged_scores.tolist(),
            "baseline_identity": baseline_identity,
            "merged_identity": merged.info,
        }
        print(
            "judge merge",
            json.dumps(
                {k: v for k, v in report["measurements"]["merge"].items() if "identity" not in k and "scores" not in k}
            ),
            flush=True,
        )
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("VALIDATION_WRITTEN", args.output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["redimnet2", "judge_clap"], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--checkpoint")
    parser.add_argument("--base")
    parser.add_argument("--adapter")
    parser.add_argument("--atom-stats")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--batching", choices=["exact", "masked"], default="exact")
    parser.add_argument("--precisions", nargs="+", default=["float32", "float16", "bfloat16"])
    run(parser.parse_args())
