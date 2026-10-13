import gc
import subprocess
import sys

import pytest

from slime.rollout.gc_utils import configure_rollout_gc

NUM_GPUS = 0


def test_default_does_not_change_collection_policy():
    before = gc.get_threshold(), gc.isenabled()
    configure_rollout_gc(None)
    assert (gc.get_threshold(), gc.isenabled()) == before


@pytest.mark.parametrize("threshold", [0, -1])
def test_threshold_cannot_disable_collection(threshold):
    before = gc.get_threshold(), gc.isenabled()
    with pytest.raises(ValueError, match="must be positive"):
        configure_rollout_gc(threshold)
    assert (gc.get_threshold(), gc.isenabled()) == before


def test_configured_process_still_collects_temporary_cycles():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import gc
import weakref
from slime.rollout.gc_utils import configure_rollout_gc

class Cycle:
    pass

before = gc.get_threshold()
configure_rollout_gc(70000)
assert gc.get_threshold() == (70000, *before[1:])
assert gc.isenabled()
gc.collect()
value = Cycle()
value.self = value
reference = weakref.ref(value)
del value
# Exceed the threshold without manually collecting. The unreachable cycle
# must still disappear while the new objects remain live.
retained = [[index] for index in range(71000)]
assert reference() is None
""",
        ],
        check=True,
    )
