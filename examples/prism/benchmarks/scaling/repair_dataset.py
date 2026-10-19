"""Repair null revision-7 instructions, preserving the rejected dataset and provenance."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

from prepare import write_json

from slime_plugins.omni_rollout import _prompt


def repair(root):
    dataset = root / "data/train.jsonl"
    backup = root / "data/train-rejected-null-instruction.jsonl"
    temporary = root / "data/train-validated.tmp"
    if backup.exists() or temporary.exists():
        raise FileExistsError("Repair already attempted; inspect provenance before retrying")
    before = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if before != "93b0e4dc6671a3bd5d15bc96db7aa59b57e5aec221cc3c253ca2c7da120d66fe":
        raise ValueError("Unexpected dataset: refusing an unscoped rewrite")
    changed = count = 0
    with dataset.open() as stream, temporary.open("x") as out:
        for count, line in enumerate(stream, 1):
            row = json.loads(line)
            prompt = row["metadata"]["omni_prompt"]
            if prompt.get("global_instruction") is None:
                prompt["global_instruction"] = {}
                changed += 1
            _prompt(SimpleNamespace(prompt=prompt, metadata={}))
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    if count != 80000 or changed == 0:
        raise ValueError(f"Unexpected conversion counts: {count=}, {changed=}")
    after = hashlib.sha256(temporary.read_bytes()).hexdigest()
    shutil.copyfile(dataset, backup)
    temporary.replace(dataset)
    receipt = {
        "time": time.time(),
        "before_sha256": before,
        "after_sha256": after,
        "validated_prompts": count,
        "changed_null_to_empty_object": changed,
        "original_preserved": str(backup),
        "validator": "slime_plugins.omni_rollout._prompt",
        "scope": "Only null global_instruction becomes {}; text, references, order and language visibility unchanged.",
    }
    write_json(root / "data/repair-null-instruction.json", receipt)
    print(json.dumps(receipt))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    repair(parser.parse_args().root)
