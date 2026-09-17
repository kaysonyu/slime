"""Synchronous policy iterations with concurrent speech generation and scoring."""

import asyncio
import copy

import httpx

from slime.backends.sglang_omni_utils.client import OmniClient
from slime.rollout.audio_artifacts import persist_audio
from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.failures import RecoverableFailureBudget, RecoverableRolloutError, mark_sample_failed
from slime.rollout.rm_hub import batched_async_rm
from slime.rollout.rm_hub.composite import reward_batch
from slime.rollout.rm_hub.config import get_reward_config
from slime.rollout.rm_hub.runtime import reward_http_runtime_scope
from slime.utils.types import Sample


def build_request(args, sample):
    temperature = args.rollout_temperature
    from importlib import import_module

    adapter = import_module(f"slime_plugins.models.{args.model_family}.data")
    prompt, parameters = adapter.generation_inputs(sample, temperature)
    return dict(
        prompt=prompt,
        output_modalities=["audio"],
        stream=False,
        return_logprob=True,
        return_omni_rollout=True,
        sampling_params=dict(
            temperature=temperature,
            top_p=1,
            top_k=-1,
            max_new_tokens=args.rollout_max_response_len,
            seed=args.rollout_seed + sample.index + sample.metadata.get("generation_attempt", 0) * 1000003,
        ),
        stage_params={args.omni_stage: parameters},
        metadata={"tts_params": parameters},
    )


def apply_response(args, sample, response, rollout_id):
    meta = response["meta_info"]
    if args.model_family == "moss_tts_local":
        from slime_plugins.models.moss_tts_local.data import MossLocalTrajectory

        trace = MossLocalTrajectory.from_omni(meta["omni_rollout"], meta, args.policy_config)
    else:
        from slime_plugins.models.higgs_tts.data import HiggsTrajectory

        trace = HiggsTrajectory.from_omni(meta["omni_rollout"], meta, args.policy_config)
    sample.trajectory = trace
    sample.status = Sample.Status.COMPLETED if trace.finish_reason == "stop" else Sample.Status.TRUNCATED
    audio = response.get("audio")
    if audio and audio.get("data"):
        sample.audio_path, details = persist_audio(
            audio,
            root=args.audio_output_dir,
            namespace=getattr(args, "artifact_namespace", "train"),
            rollout_id=rollout_id,
            sample_id=sample.index,
            attempt=sample.metadata.get("generation_attempt", 0),
        )
        sample.metadata.update(details)
    elif trace.num_frames:
        raise ValueError("Non-empty generated actions have no corresponding audio")
    sample.metadata["frames"] = trace.num_frames


async def _collect(args, rollout_id, groups):
    from slime.rollout.on_policy_distillation import TeacherScorer

    clients = [OmniClient(endpoint, args.omni_stage, args.omni_timeout) for endpoint in args.omni_endpoints]
    teacher = TeacherScorer(args) if args.objective == "mopd" else None
    semaphore = asyncio.Semaphore(args.omni_concurrency)
    budget = RecoverableFailureBudget(getattr(args, "max_recoverable_rollout_failures", 32))
    observed_version = getattr(args, "expected_weight_version", None)
    retries = getattr(args, "rollout_group_max_retries", 2)
    custom = getattr(args, "custom_rm_path", None) not in (None, "slime.rollout.rm_hub.wer.reward_func")

    async def generate_one(sample):
        nonlocal observed_version
        try:
            # Generation permits are released before any reward-service request.
            async with semaphore:
                response = await clients[sample.index % len(clients)].generate(build_request(args, sample))
        except httpx.HTTPStatusError as error:
            if error.response.status_code != 429 and error.response.status_code < 500:
                raise
            raise RecoverableRolloutError("generation.http_transient") from error
        except httpx.TransportError as error:
            raise RecoverableRolloutError("generation.transport") from error
        apply_response(args, sample, response, rollout_id)
        version = sample.trajectory.weight_version
        if observed_version is None:
            observed_version = version
        if version != observed_version:
            raise ValueError("Generation/retry crossed the published student weight version")

    async def run_group(original):
        for attempt in range(retries + 1):
            group = copy.deepcopy(original)
            for sample in group:
                sample.rollout_id = rollout_id
                sample.metadata["generation_attempt"] = attempt
            generated = await asyncio.gather(*(generate_one(sample) for sample in group), return_exceptions=True)
            for sample, result in zip(group, generated, strict=True):
                if isinstance(result, RecoverableRolloutError):
                    mark_sample_failed(sample, result.failure)
                elif isinstance(result, BaseException):
                    raise result
            if not any(sample.remove_sample for sample in group):
                if teacher is not None:
                    await asyncio.gather(*(teacher.score(sample) for sample in group))
                else:
                    if custom:
                        rewards = await batched_async_rm(args, group)
                    else:
                        rewards = await reward_batch(args, group)
                    for sample, reward in zip(group, rewards, strict=True):
                        sample.reward = reward
            if not any(sample.remove_sample for sample in group):
                return group
            budget.record(group)
            if attempt == retries:
                raise RuntimeError("Complete prompt group retry budget exhausted")
        raise AssertionError("Group retry loop exited unexpectedly")

    try:
        async with reward_http_runtime_scope() as runtime:
            if args.objective == "grpo" and not custom:
                runtime.add_services(get_reward_config(args).services)
            tasks = [asyncio.create_task(run_group(group)) for group in groups]
            try:
                completed = await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            metrics = {**runtime.collect_metrics(), **budget.collect_metrics()}
            metrics["rollout/retried_groups"] = sum(group[0].metadata["generation_attempt"] > 0 for group in completed)
            return completed, metrics
    finally:
        await asyncio.gather(*(client.close() for client in clients))
        if teacher is not None:
            await teacher.close()


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    groups = data_source.get_samples(args.rollout_batch_size)
    groups, metrics = asyncio.run(_collect(args, rollout_id, groups))
    if evaluation:
        samples = [sample for group in groups for sample in group]
        return RolloutFnEvalOutput(
            data={
                getattr(args, "eval_dataset_name", "tts"): {
                    "samples": samples,
                    "rewards": [sample.reward for sample in samples],
                }
            },
            metrics=metrics,
        )
    return RolloutFnTrainOutput(samples=groups, metrics=metrics)
