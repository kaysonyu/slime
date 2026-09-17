"""Independent, named speech evaluation datasets and sampling settings."""

from dataclasses import dataclass, field
import math
from pathlib import Path
import re

import yaml


@dataclass(frozen=True)
class EvalDatasetConfig:
    name: str
    path: str
    n_samples_per_eval_prompt: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_response_len: int = 512
    language: str | None = None
    reward_config: str | None = None
    metadata_overrides: dict = field(default_factory=dict)

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name):
            raise ValueError("Evaluation dataset name must be a path-safe identifier")
        if not self.path or not Path(self.path).is_file():
            raise ValueError(f"Evaluation dataset {self.name} requires an existing JSONL path")
        if type(self.n_samples_per_eval_prompt) is not int or self.n_samples_per_eval_prompt < 1:
            raise ValueError("Evaluation fanout must be a positive integer")
        if type(self.max_response_len) is not int or self.max_response_len < 1:
            raise ValueError("Evaluation response length must be a positive integer")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Evaluation temperature must be finite and positive")
        if self.top_p != 1 or self.top_k != -1:
            raise ValueError("Omni speech action replay requires evaluation top_p=1 and top_k=-1")
        if not isinstance(self.metadata_overrides, dict) or "id" in self.metadata_overrides:
            raise ValueError("metadata_overrides must be an object without sample id overrides")


def resolve_eval_datasets(args):
    config_path = getattr(args, "eval_config", None)
    pairs = getattr(args, "eval_prompt_data", None)
    legacy = getattr(args, "eval_data", None)
    if sum(bool(value) for value in (config_path, pairs, legacy)) > 1:
        raise ValueError("Select one of --eval-config, --eval-prompt-data, or --eval-data")
    defaults = dict(
        n_samples_per_eval_prompt=getattr(args, "n_samples_per_eval_prompt", 1),
        temperature=args.rollout_temperature
        if getattr(args, "eval_temperature", None) is None
        else args.eval_temperature,
        max_response_len=args.rollout_max_response_len
        if getattr(args, "eval_max_response_len", None) is None
        else args.eval_max_response_len,
    )
    base = Path.cwd()
    if config_path:
        base = Path(config_path).resolve().parent
        raw = yaml.safe_load(Path(config_path).read_text())
        if not isinstance(raw, dict) or set(raw) != {"eval"}:
            raise ValueError("Evaluation YAML must contain an eval object")
        config = raw["eval"]
        if not isinstance(config, dict) or set(config) - {"defaults", "datasets"}:
            raise ValueError("eval accepts defaults and datasets")
        overrides = config.get("defaults", {})
        if not isinstance(overrides, dict) or set(overrides) - (
            set(EvalDatasetConfig.__dataclass_fields__) - {"name", "path"}
        ):
            raise ValueError("Unknown evaluation defaults")
        defaults.update(overrides)
        rows = config.get("datasets")
        if not isinstance(rows, list) or not rows:
            raise ValueError("eval.datasets must be a nonempty list")
    elif pairs:
        if len(pairs) % 2:
            raise ValueError("--eval-prompt-data requires NAME PATH pairs")
        rows = [dict(name=name, path=path) for name, path in zip(pairs[::2], pairs[1::2], strict=True)]
    else:
        rows = [dict(name="tts", path=legacy)] if legacy else []
    datasets = []
    for row in rows:
        if not isinstance(row, dict) or set(row) - set(EvalDatasetConfig.__dataclass_fields__):
            raise ValueError("Unknown evaluation dataset field")
        values = defaults | row
        if not values.get("name") or not values.get("path"):
            raise ValueError("Every evaluation dataset requires name and path")
        for key in ("path", "reward_config"):
            if values.get(key):
                values[key] = str((base / values[key]).resolve())
        datasets.append(EvalDatasetConfig(**values))
    if len({dataset.name for dataset in datasets}) != len(datasets):
        raise ValueError("Evaluation dataset names must be unique")
    return datasets
