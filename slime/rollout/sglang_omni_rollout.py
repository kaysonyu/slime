"""Synchronous policy iterations with concurrent speech generation and scoring."""

import asyncio
import base64
import hashlib
from pathlib import Path

import httpx

from slime.backends.sglang_omni_utils.client import OmniClient
from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.rm_hub.wer import reward_func
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
            seed=args.rollout_seed + sample.index,
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
        directory = Path(args.audio_output_dir) / f"rollout_{rollout_id:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"sample_{sample.index:08d}.wav"
        data = base64.b64decode(audio["data"], validate=True)
        path.write_bytes(data)
        sample.audio_path = str(path)
        sample.metadata["audio_sha256"] = hashlib.sha256(data).hexdigest()
    elif trace.num_frames:
        raise ValueError("Non-empty generated actions have no corresponding audio")
    sample.metadata["frames"] = trace.num_frames


async def _collect(args, rollout_id, groups):
    clients = [OmniClient(endpoint, args.omni_stage, args.omni_timeout) for endpoint in args.omni_endpoints]
    from slime.rollout.on_policy_distillation import TeacherScorer

    teacher = TeacherScorer(args) if args.objective == "mopd" else None
    semaphore = asyncio.Semaphore(args.omni_concurrency)
    samples = [sample for group in groups for sample in group]
    try:
        async with httpx.AsyncClient(timeout=args.omni_timeout, trust_env=False) as asr:

            async def generate(index, sample):
                async with semaphore:
                    response = await clients[index % len(clients)].generate(build_request(args, sample))
                    apply_response(args, sample, response, rollout_id)
                    if args.objective == "grpo":
                        if args.custom_rm_path == "slime.rollout.rm_hub.wer.reward_func":
                            sample.reward = await reward_func(args, sample, client=asr)
                        else:
                            from slime.rollout.rm_hub import async_rm

                            sample.reward = await async_rm(args, sample)
                    else:
                        await teacher.score(sample)

            tasks = [asyncio.create_task(generate(i, sample)) for i, sample in enumerate(samples)]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
    finally:
        await asyncio.gather(*(client.close() for client in clients))
        if teacher is not None:
            await teacher.close()
    versions = {sample.trajectory.weight_version for sample in samples}
    if len(versions) != 1:
        raise ValueError(f"One synchronous rollout batch contains different policy versions: {versions}")
    return groups


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    groups = data_source.get_samples(args.rollout_batch_size)
    groups = asyncio.run(_collect(args, rollout_id, groups))
    if evaluation:
        samples = [s for group in groups for s in group]
        return RolloutFnEvalOutput(data={"tts": {"samples": samples, "rewards": [s.reward for s in samples]}})
    return RolloutFnTrainOutput(samples=groups)
