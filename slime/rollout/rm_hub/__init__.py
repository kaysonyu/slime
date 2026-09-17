"""Small reward-function boundary for native speech samples."""

import asyncio
import inspect
import math

from slime.utils.misc import load_function


async def async_rm(args, sample, **kwargs):
    function = load_function(args.custom_rm_path or "slime.rollout.rm_hub.wer.reward_func")
    result = function(args, sample, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    if isinstance(result, dict):
        if not args.reward_key:
            raise ValueError("Dictionary rewards require --reward-key")
        value = result[args.reward_key]
    else:
        value = result
    if not math.isfinite(float(value)):
        raise ValueError("A reward function must return a finite score")
    return result


async def batched_async_rm(args, samples):
    if getattr(args, "group_rm", False):
        function = load_function(args.custom_rm_path)
        results = function(args, samples)
        if inspect.isawaitable(results):
            results = await results
        if not isinstance(results, (list, tuple)) or len(results) != len(samples):
            raise ValueError("Batch reward must return one result per sample in input order")
        for result in results:
            value = result[args.reward_key] if isinstance(result, dict) and args.reward_key else result
            if not math.isfinite(float(value)):
                raise ValueError("Batch reward must return finite scores")
        return results
    return await asyncio.gather(*(async_rm(args, sample) for sample in samples))
