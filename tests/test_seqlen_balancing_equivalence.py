"""Preserve exact partition order while sharing immutable empty KK partitions."""

import hashlib
import json
import random

import pytest

from slime.utils.seqlen_balancing import karmarkar_karp

NUM_GPUS = 0


def test_partitions_match_preoptimization_golden():
    # Digest produced by the unchanged 1d7c4b3e implementation. Include zeros,
    # ties and non-associative floating-point sums: partition order is a
    # contract, not merely preservation of membership or balanced loads.
    results = []
    for k in [1, 2, 3, 4, 8, 16]:
        for n in sorted({k, k + 1, 2 * k, 4 * k}):
            for mode in ["ones", "zeros", "ties", "integer", "float", "wide_float"]:
                rng = random.Random(6211 + n + k)
                values = {
                    "ones": lambda: 1,
                    "zeros": lambda: 0,
                    "ties": lambda rng=rng: rng.randint(0, 4),
                    "integer": lambda rng=rng: rng.randint(1, 10000),
                    "float": lambda rng=rng: rng.random(),
                    "wide_float": lambda rng=rng: rng.choice([1e20, 0.1, 0.01, 1e-20]),
                }
                lengths = [values[mode]() for _ in range(n)]
                for equal_size in [False, True]:
                    if equal_size and n % k:
                        continue
                    results.append([k, n, mode, equal_size, karmarkar_karp(lengths, k, equal_size)])
    assert len(results) == 246
    digest = hashlib.sha256(json.dumps(results, separators=(",", ":")).encode()).hexdigest()
    assert digest == "b66def16f9c2ddbe405c29fde0e4f9c63f2e09f43569fd9216bfdf5d2d9d02d4"


@pytest.mark.parametrize("n,k", [(4096, 64), (8192, 128), (8192, 256)])
def test_large_sparse_initial_states_keep_independent_partitions(n, k):
    lengths = [80 + (index * 127) % 1024 for index in range(n)]
    original = list(lengths)
    first = karmarkar_karp(lengths, k, False)
    assert len(first) == k and all(first)
    assert sorted(index for partition in first for index in partition) == list(range(n))
    expected = [list(partition) for partition in first]
    first[0].clear()
    assert karmarkar_karp(lengths, k, False) == expected
    assert lengths == original
