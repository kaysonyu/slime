"""CPU logging contracts: summaries/errors survive, diagnostics remain opt-in."""

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

NUM_GPUS = 0


@pytest.mark.parametrize("level", ["INFO", "DEBUG"])
def test_console_preserves_repeated_warnings_and_errors_and_respects_log_level(level):
    program = """
import logging
from slime.observability.logging_utils import configure_logger
configure_logger()
logging.getLogger("megatron").setLevel(logging.WARNING)
checkpoint_logger = logging.getLogger("megatron.core.utils")
for _ in range(2):
    checkpoint_logger.warning("make_tp_sharded_tensor_for_checkpoint received extra kwargs: ['allow_shape_mismatch']")
    checkpoint_logger.warning("unexpected checkpoint mismatch")
    checkpoint_logger.error("checkpoint read failed")
logger = logging.getLogger("slime")
logger.info("train 7: loss=0.5 reward=0.8")
logger.debug("rank memory diagnostic")
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": str(REPO), "SLIME_LOG_LEVEL": level, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr.count("received extra kwargs") == 2
    assert result.stderr.count("unexpected checkpoint mismatch") == 2
    assert result.stderr.count("checkpoint read failed") == 2
    assert "train 7: loss=0.5 reward=0.8" in result.stderr
    assert ("rank memory diagnostic" in result.stderr) == (level == "DEBUG")


def test_memory_diagnostics_preserve_return_value_cleanup_and_failure_logs(monkeypatch, caplog):
    from slime.utils import memory_utils

    memory = {"free_GB": 1.0}
    cleared = []
    monkeypatch.setattr(memory_utils, "available_memory", lambda: memory)
    monkeypatch.setattr(memory_utils.dist, "get_rank", lambda: 3)
    monkeypatch.setattr(memory_utils, "clear_memory", lambda: cleared.append(True))
    with caplog.at_level(logging.INFO):
        assert memory_utils.print_memory("before update", clear_before_print=True) is memory
        assert not caplog.records
        assert cleared == [True]
        assert memory_utils.print_memory("after torch distributed error", level=logging.ERROR) is memory
    assert len(caplog.records) == 1 and caplog.records[0].levelno == logging.ERROR
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        memory_utils.print_memory("before update")
    assert "[Rank 3] Memory-Usage before update" in caplog.text


def test_logging_settings_reach_ray_actors_without_overriding_explicit_training_options(monkeypatch):
    from slime.ray.utils import add_default_ray_env_vars

    for name in ("SLIME_LOG_LEVEL", "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_DEBUG_FILE", "RAY_DEDUP_LOGS"):
        monkeypatch.delenv(name, raising=False)
    normal = add_default_ray_env_vars()
    assert normal["NCCL_DEBUG"] == "WARN" and normal["SLIME_LOG_LEVEL"] == "INFO"
    monkeypatch.setenv("SLIME_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("NCCL_DEBUG", "INFO")
    monkeypatch.setenv("NCCL_DEBUG_FILE", "/shared/diagnostic/nccl.%h.%p.log")
    monkeypatch.setenv("RAY_DEDUP_LOGS", "0")
    diagnostic = add_default_ray_env_vars()
    assert diagnostic["SLIME_LOG_LEVEL"] == "DEBUG" and diagnostic["NCCL_DEBUG"] == "INFO"
    assert diagnostic["NCCL_DEBUG_FILE"] == "/shared/diagnostic/nccl.%h.%p.log"
    assert diagnostic["RAY_DEDUP_LOGS"] == "0"
    assert add_default_ray_env_vars({"NCCL_DEBUG": "WARN"})["NCCL_DEBUG"] == "WARN"
    assert os.environ["NCCL_DEBUG"] == "INFO"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
