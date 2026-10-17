import asyncio
import copy
import inspect
import json
from functools import lru_cache
from pathlib import Path

import aiohttp
import torch
import yaml

from slime.rollout.teacher_client import BatchedTeacherClient, TeacherClientOptions
from slime.utils.misc import load_function
from slime.utils.policy import validate_policy_actions
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample


async def reward_func(args, sample, **kwargs):
    payload = {
        # "text": sample.prompt + sample.response,
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": args.rollout_temperature,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(kwargs.get("endpoint") or args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response (which contains sglang's logprob output)
    2. Trims them to match the response length
    3. Stores them in sample.teacher_log_probs for OPD KL penalty computation
    4. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO

    Note: The reward_func calls the teacher server which returns token-level log-probs.
    For pure on-policy distillation without task rewards, we return 0.0 for each sample.
    The actual learning signal comes from the OPD KL penalty applied in compute_advantages_and_returns.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    for sample, response, length in zip(samples, raw_rewards, response_lengths, strict=True):
        sample.teacher_log_probs = response_log_probs(response, length)

    # Return scalar rewards for GRPO/PPO advantage estimator
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards


@lru_cache(maxsize=32)
def _read_teacher_config(value):
    if value.lstrip().startswith("{"):
        config = json.loads(value)
    else:
        config = yaml.safe_load(Path(value).read_text())
    if (
        not isinstance(config, dict)
        or not isinstance(config.get("models"), dict)
        or not isinstance(config.get("by_domain", {}), dict)
    ):
        raise ValueError("Teacher configuration requires models and by_domain mappings")
    return config


def load_teacher_config(value):
    """Load model-independent teacher bindings without owning model scoring protocols."""
    if isinstance(value, dict):
        value = json.dumps(value, sort_keys=True)
    if not isinstance(value, str) or not value:
        raise ValueError("Teacher configuration must be a mapping, JSON string or YAML path")
    return _read_teacher_config(value)


def select_teacher(config, sample):
    domain = (sample.metadata or {}).get("domain", sample.policy_domain)
    key = config.get("default_teacher") if domain is None else config.get("by_domain", {}).get(domain)
    models = config["models"]
    if key is None or key not in models or not isinstance(models[key], dict):
        raise ValueError(f"Unmapped teacher domain {domain!r}")
    sample.policy_domain = domain
    sample.policy_teacher = key
    return key, models[key]


def response_log_probs(response, response_length):
    scores = torch.tensor([item[0] for item in response["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
    if scores.numel() < response_length:
        raise ValueError("Teacher scores do not cover the student's response")
    return scores[-response_length:] if response_length else scores[:0]


async def score_text_teacher(args, sample, *, binding, client, evaluation=False):
    if sample.policy_actions is not None:
        raise ValueError("Explicit action streams require a model-specific teacher score_function")
    response = (await client.score(binding["endpoint"], sample)).value
    sample.teacher_log_probs = response_log_probs(response, sample.response_length)
    return sample


class TextTeacherClient(BatchedTeacherClient):
    def __init__(self, args, configuration):
        super().__init__(TeacherClientOptions.from_config(configuration))
        self.args = copy.copy(args)

    async def send_batch(self, endpoint, samples):
        results = await asyncio.gather(
            *(reward_func(self.args, sample, endpoint=endpoint) for sample in samples), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return results


score_text_teacher.create_client = TextTeacherClient


class TeacherRuntime:
    """Own a scorer and its client for one rollout executor's lifetime."""

    def __init__(self, args):
        self.args = args
        self.configuration = copy.deepcopy(load_teacher_config(args.opd_teacher_config))
        path = self.configuration.get("score_function", "slime.rollout.on_policy_distillation.score_text_teacher")
        self.scorer = load_function(path)
        factory = getattr(self.scorer, "create_client", None)
        self.client = factory(args, self.configuration) if factory is not None else None
        self.closed = False

    async def score(self, sample, *, evaluation=False):
        if sample.status == Sample.Status.ABORTED:
            return sample
        if self.closed:
            raise RuntimeError("Teacher runtime is closed")
        if sample.teacher_log_probs is not None or any(
            "teacher_log_probs" in stream for stream in (sample.policy_actions or {}).values()
        ):
            raise ValueError("Teacher scores already exist; configure only one teacher scoring path")
        teacher_key, binding = select_teacher(self.configuration, sample)
        domain = sample.policy_domain
        kwargs = {"binding": binding, "evaluation": evaluation}
        if self.client is not None:
            kwargs["client"] = self.client
        result = self.scorer(self.args, sample, **kwargs)
        result = await result if inspect.isawaitable(result) else result
        if result is not None and not isinstance(result, Sample):
            raise TypeError("An OPD score function must return Sample or None")
        sample = sample if result is None else result
        sample.policy_domain = domain
        sample.policy_teacher = teacher_key
        if sample.policy_actions is not None:
            validate_policy_actions(sample.policy_actions, len(sample.tokens))
            if not all("teacher_log_probs" in stream for stream in sample.policy_actions.values()):
                raise ValueError("Teacher scorer must fill every declared action stream")
        elif sample.teacher_log_probs is None:
            raise ValueError("Teacher scorer did not attach teacher log probabilities")
        return sample

    async def flush(self):
        if self.client is not None:
            await self.client.flush()
        flush = getattr(self.scorer, "flush", None)
        if flush is not None:
            result = flush(self.args)
            if inspect.isawaitable(result):
                await result

    async def close(self):
        if self.closed:
            return
        self.closed = True
        if self.client is not None:
            await self.client.close()
        close = getattr(self.scorer, "close", None)
        if close is not None:
            result = close(self.args)
            if inspect.isawaitable(result):
                await result


async def score_teacher(args, sample, *, evaluation=False, runtime=None):
    """Attach external OPD scores independently of the task reward function.

    Executors supply their owned runtime. A standalone call owns and closes a
    temporary runtime, including all requests when the input is a list.
    """
    if not getattr(args, "use_opd", False) or getattr(args, "opd_teacher_config", None) is None:
        return sample
    if evaluation and not getattr(args, "opd_eval_teacher_scoring", True):
        return sample
    if isinstance(sample, Sample) and sample.status == Sample.Status.ABORTED:
        return sample
    if isinstance(sample, list) and all(
        isinstance(item, Sample) and item.status == Sample.Status.ABORTED for item in sample
    ):
        return sample
    owned = runtime is None
    runtime = TeacherRuntime(args) if owned else runtime
    try:
        if isinstance(sample, list):
            results = await asyncio.gather(
                *(score_teacher(args, item, evaluation=evaluation, runtime=runtime) for item in sample),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            return results
        return await runtime.score(sample, evaluation=evaluation)
    finally:
        if owned:
            await runtime.close()
