"""Strict model initialization and native Megatron checkpoint resume."""

import logging
import os
import re
from pathlib import Path

from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron
from megatron.training.checkpointing import save_checkpoint
from megatron.training.global_vars import get_args

logger = logging.getLogger(__name__)
__all__ = ["save_checkpoint", "load_checkpoint"]


def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context):
    # ref: how megatron `load_checkpoint` gets directory
    args = get_args()
    load_path = args.load

    assert Path(load_path).exists() and _is_dir_nonempty(
        load_path
    ), f"{args.load=} does not exist or is an empty directory. Did you specify the wrong folder?"

    if not _is_megatron_checkpoint(load_path):
        raise ValueError("Megatron training loads torch_dist checkpoints; convert the Local HF artifact first")
    return _load_checkpoint_megatron(
        ddp_model=ddp_model,
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
        checkpointing_context=checkpointing_context,
        skip_load_to_model_and_opt=False,
    )


def _is_megatron_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "latest_checkpointed_iteration.txt").is_file() or bool(
        re.fullmatch(r"iter_\d{7}", Path(path).name)
    )


def _is_dir_nonempty(path):
    with os.scandir(path) as it:
        return any(it)
