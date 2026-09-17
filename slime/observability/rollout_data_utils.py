"""Save and replay exact native speech trajectories."""

import logging
from pathlib import Path
import torch
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def load_debug_rollout_data(path_template, *, rollout_id: int, subsample_ratio=None) -> list[Sample]:
    data = torch.load(path_template.format(rollout_id=rollout_id), weights_only=False)["samples"]
    data = [Sample.from_dict(sample) for sample in data]
    if subsample_ratio is not None:
        original_num_rows = len(data)
        rough_subsample_num_rows = int(original_num_rows * subsample_ratio)
        data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
        logger.info(
            "Subsample loaded debug rollout data using ratio=%s and change num rows %s -> %s",
            subsample_ratio,
            original_num_rows,
            len(data),
        )
    return data


def save_debug_rollout_data(path_template, data, *, rollout_id: int, evaluation: bool) -> None:
    if path_template is None:
        return

    path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
    logger.info(f"Save debug rollout data to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    if evaluation:
        samples = [sample.to_dict() for info in data.values() for sample in info["samples"]]
    else:
        samples = [sample.to_dict() for sample in data]

    torch.save({"rollout_id": rollout_id, "samples": samples}, path)
