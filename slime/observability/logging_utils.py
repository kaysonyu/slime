import json
import logging
from pathlib import Path

import wandb

from slime.observability import wandb_utils
from slime.observability.tensorboard_utils import _TensorboardAdapter

_LOGGER_CONFIGURED = False


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)


def finish_tracking(args):
    if args.use_tensorboard:
        instance = _TensorboardAdapter._instances.pop(_TensorboardAdapter, None)
        if instance is not None:
            instance.finish()
    if not args.use_wandb:
        return
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        logging.getLogger(__name__).exception("Failed to finish wandb run")


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    import torch

    metrics = {
        key: (
            (value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist())
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in metrics.items()
    }
    if getattr(args, "metrics_jsonl", None):
        path = Path(args.metrics_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
    if args.use_wandb:
        wandb.log(metrics)

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])
