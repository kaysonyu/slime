"""CPU-only deterministic reproduction of transient shared-slot rejection.

Loads just admission_shm.py from the specified trusted Omni checkout. No GPU,
network, production mmap, or runtime configuration is changed. This reproduces
the logged mechanism, not the scheduler/crash event that made a real slot odd.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def reproduce(omni):
    path = omni / "sglang_omni_router/python/admission_shm.py"
    spec = importlib.util.spec_from_file_location("_admission_reproduction", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    buf = bytearray(module.admission_file_size(8))
    now = [100.0]
    reader = module.SharedAdmission(
        buf, slots=8, own_index=0, max_inflight=512, generation=1, pid=100,
        clock=lambda: now[0],
    )
    writer = module.SharedAdmission(
        buf, slots=8, own_index=1, max_inflight=512, generation=1, pid=101,
        clock=lambda: now[0],
    )
    before = reader.to_dict()
    assert before["inflight"] == 0
    marker = writer._own.begin_write()  # Simulate a live writer paused mid-write.
    rejected_while_odd = not reader.try_acquire()
    writer._own.end_write(marker)  # Writer resumes; no reclaim or process death.
    rejected_after_writer_resumed = not reader.try_acquire()  # Cached 50-ms shed.
    now[0] += module._UNSTABLE_REPROBE_SECS + 0.001
    admitted_after_reprobe = reader.try_acquire()
    if admitted_after_reprobe:
        reader.release()
    after = reader.to_dict()
    assert rejected_while_odd and rejected_after_writer_resumed and admitted_after_reprobe
    assert after["inflight"] == 0 and after["rejected_total"] == 2
    return {
        "valid_reproduction": True,
        "source": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "before": before,
        "rejected_while_sibling_sequence_odd": rejected_while_odd,
        "cached_rejection_after_writer_resumed": rejected_after_writer_resumed,
        "admitted_after_reprobe": admitted_after_reprobe,
        "after": after,
        "scope": "Deterministic protocol reproduction below capacity. Does not prove why the field writer was unstable, test a fix, or validate GPU stability.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("omni", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = reproduce(args.omni)
    if args.output:
        from prepare import write_json

        write_json(args.output, result)
    print(json.dumps(result, indent=2))
