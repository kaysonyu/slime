"""Record rank packing and verify finite GRPO advantages at training admission."""

import json
import os
from pathlib import Path


def audit_train(args, rollout_id, rollout_data):
    import torch

    rank = torch.distributed.get_rank()
    output = Path(os.environ["PRISM_SCALE_RUN"]) / "validation"
    output.mkdir(parents=True, exist_ok=True)
    lengths = [len(tokens) for tokens in rollout_data["tokens"]]
    report = {
        "rollout": rollout_id,
        "rank": rank,
        "samples": len(lengths),
        "microbatches": rollout_data["num_microbatches"],
        "global_batch_sizes": rollout_data["global_batch_sizes"],
        "pack_lengths": [sum(lengths[i] for i in pack) for pack in rollout_data["micro_batch_indices"]],
    }
    if rollout_id in {0, args.num_rollout - 1}:
        for streams in rollout_data["policy_actions"]:
            for stream in streams.values():
                valid = stream["valid_mask"]
                if not torch.isfinite(stream["advantages"][valid]).all():
                    raise ValueError("Nonfinite frozen action advantages")
    (output / f"schedule-{rollout_id:04d}-{rank:03d}.json").write_text(json.dumps(report) + "\n")
