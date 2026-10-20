"""CPU-only transport/partition evidence; never substitute for GPU step timing."""

import argparse
import importlib.util
import json
import statistics
import time
from functools import partial
from pathlib import Path

import torch

from prepare import write_json
from slime.utils.tensor_serialization import deserialize_tensor_payload, serialize_tensor_payload
from slime.utils.types import Sample


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def samples(count):
    result = []
    for index in range(count):
        frames = 64 + index % 64
        sample = Sample(index=index, tokens=list(range(frames + 40)), response_length=frames, reward=0.8)
        sample.policy_actions = {
            "rvq": {
                "actions": torch.zeros((frames, 8), dtype=torch.int64),
                "rollout_log_probs": torch.full((frames, 8), -0.5),
                "mask": torch.ones((frames, 8), dtype=torch.bool),
            },
            "stop": {
                "actions": torch.zeros(frames + 1, dtype=torch.int64),
                "rollout_log_probs": torch.full((frames + 1,), -0.3),
                "mask": torch.ones(frames + 1, dtype=torch.bool),
            },
        }
        sample.multimodal_train_inputs = {
            "input_ids": torch.zeros((frames + 40, 9), dtype=torch.int64),
            "item_kind": torch.zeros(frames + 40, dtype=torch.int64),
            "audio_role": torch.zeros(frames + 40, dtype=torch.int64),
            "successor_audio_mask": torch.ones(frames + 40, dtype=torch.bool),
            "successor_audio_codes": torch.zeros((frames + 40, 8), dtype=torch.int64),
            "target_history_retention_mask": torch.ones((frames + 40, 8), dtype=torch.bool),
        }
        sample.metadata = {"omni_frames": frames, "text": "传输微基准，不代表真实生成质量。"}
        result.append(sample)
    return result


def median_runtime(function, repeats):
    measurements = []
    for _ in range(repeats + 2):
        start = time.perf_counter()
        function()
        measurements.append(time.perf_counter() - start)
    return statistics.median(measurements[2:])


def benchmark(baseline_source, repeats):
    torch.set_num_threads(1)
    baseline = module_from(baseline_source / "slime/utils/seqlen_balancing.py", "baseline_kk")
    candidate = module_from(Path(__file__).parents[4] / "slime/utils/seqlen_balancing.py", "candidate_kk")
    output = {
        "scope": "Synthetic Prism-like CPU tensors and deterministic lengths, not a GPU result",
        "transport": {},
        "kk": {},
    }
    for count in (8, 64):
        value = samples(count)
        modes = {}
        for packed in (False, True):
            encoded = serialize_tensor_payload(value, packed=packed)
            decoded = deserialize_tensor_payload(encoded)
            assert all(
                torch.equal(a.policy_actions["rvq"]["actions"], b.policy_actions["rvq"]["actions"])
                for a, b in zip(value, decoded, strict=True)
            )
            modes[str(packed)] = {
                "bytes": encoded.nbytes,
                "serialize_median_seconds": median_runtime(
                    partial(serialize_tensor_payload, value, packed=packed), repeats
                ),
                "deserialize_median_seconds": median_runtime(partial(deserialize_tensor_payload, encoded), repeats),
            }
        output["transport"][str(count)] = modes
    for n, k in ((4096, 64), (8192, 128)):
        lengths = [80 + (i * 127) % 1024 for i in range(n)]
        expected = baseline.karmarkar_karp(lengths, k, False)
        assert candidate.karmarkar_karp(lengths, k, False) == expected
        output["kk"][f"{n}/{k}"] = {
            "exact_partition_equal": True,
            "baseline_median_seconds": median_runtime(partial(baseline.karmarkar_karp, lengths, k, False), 3),
            "candidate_median_seconds": median_runtime(partial(candidate.karmarkar_karp, lengths, k, False), 3),
        }
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=25)
    args = parser.parse_args()
    result = benchmark(args.baseline_source, args.repeats)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
