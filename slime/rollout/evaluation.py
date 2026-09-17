"""Per-dataset evaluation arguments without changing training sampler state."""

import copy

from slime.rollout.data_source import RolloutDataSource
from slime.rollout.rm_hub.config import get_reward_config


def evaluation_source(args, dataset):
    evaluation = copy.copy(args)
    evaluation.prompt_data = dataset.path
    evaluation.eval_dataset_name = dataset.name
    evaluation.artifact_namespace = f"eval/{dataset.name}"
    evaluation.objective = "grpo"
    evaluation.n_samples_per_prompt = dataset.n_samples_per_eval_prompt
    evaluation.rollout_temperature = dataset.temperature
    evaluation.rollout_max_response_len = dataset.max_response_len
    evaluation.wer_language = dataset.language or args.wer_language
    evaluation.metadata_overrides = dataset.metadata_overrides
    evaluation.rollout_shuffle = False
    if dataset.reward_config:
        evaluation.reward_config = dataset.reward_config
        evaluation.reward_configuration = None
    if evaluation.custom_rm_path in (None, "slime.rollout.rm_hub.wer.reward_func"):
        get_reward_config(evaluation)
    source = RolloutDataSource(evaluation)
    evaluation.rollout_batch_size = len(source)
    return evaluation, source
