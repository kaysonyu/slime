"""CPU CI entry point for the opt-in scaling paths and their compatibility tests.

The native router cases require the separately installed, pinned Omni router;
pytest reports explicit skips when that optional dependency is unavailable.
All Ray instances in this suite use num_gpus=0.
"""

import sys
from pathlib import Path

import pytest

NUM_GPUS = 0
TEST_FILES = (
    "test_omni_arguments.py",
    "test_omni_router_diagnostics.py",
    "test_omni_router_pool_ownership.py",
    "test_rollout_http_pool_guard.py",
    "test_omni_router_multiprocess.py",
    "test_rollout_data_utils.py",
    "test_rollout_execution_pool.py",
    "test_rollout_group_serialization.py",
    "test_training_tensor_buffer.py",
    "test_weight_topology_cache.py",
    "test_seqlen_balancing_equivalence.py",
)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    raise SystemExit(pytest.main([*(str(directory / name) for name in TEST_FILES), *sys.argv[1:]]))
