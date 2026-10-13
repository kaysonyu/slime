from __future__ import annotations

import inspect
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from slime.utils.misc import load_function
from slime.utils.types import Sample

_current_rollout_id: int | None = None
_task_rollout_id = ContextVar("rollout_sample_hook_id", default=None)
logger = logging.getLogger(__name__)


def set_current_rollout_id(rollout_id: int | None) -> None:
    global _current_rollout_id
    _current_rollout_id = rollout_id


def get_current_rollout_id() -> int | None:
    rollout_id = _task_rollout_id.get()
    return rollout_id if rollout_id is not None else _current_rollout_id


@contextmanager
def rollout_context(rollout_id):
    """Keep concurrent worker requests' hook contexts independent."""
    token = _task_rollout_id.set(rollout_id)
    try:
        yield
    finally:
        _task_rollout_id.reset(token)


def _accepted_kwargs(function, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(function)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


async def _apply_to_sample(args, sample: Sample, paths: list[str], **kwargs) -> Sample:
    for path in paths:
        hook = load_function(path)
        result = hook(args, sample, **_accepted_kwargs(hook, kwargs))
        if inspect.isawaitable(result):
            result = await result
        if result is not None:
            if not isinstance(result, Sample):
                raise TypeError(
                    f"Rollout sample hook {path!r} returned {type(result).__name__}, expected Sample or None."
                )
            sample = result
    return sample


async def apply_rollout_sample_hooks(args, value, **kwargs):
    """Apply user hooks; the owning queue publication persists replay tensors."""

    paths = getattr(args, "rollout_sample_hook_path", None) or []
    if not paths:
        return value
    kwargs.setdefault("rollout_id", get_current_rollout_id())
    if isinstance(value, Sample):
        return await _apply_to_sample(args, value, paths, **kwargs)
    if isinstance(value, list):
        return [await apply_rollout_sample_hooks(args, item, **kwargs) for item in value]
    raise TypeError(f"Rollout sample hooks expected Sample or list, got {type(value).__name__}.")


async def notify_sample_observers(args, value, *, event, stage, error=None, evaluation=False):
    """Report lifecycle events without changing reward routing or sample ownership."""
    paths = getattr(args, "rollout_sample_observer_path", None) or []
    if not paths:
        return
    if isinstance(value, list):
        for item in value:
            await notify_sample_observers(args, item, event=event, stage=stage, error=error, evaluation=evaluation)
        return
    for path in paths:
        observer = load_function(path)
        try:
            result = observer(
                args,
                value,
                event=event,
                stage=stage,
                error=error,
                evaluation=evaluation,
                rollout_id=get_current_rollout_id(),
            )
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                raise TypeError(f"Sample observer {path!r} must return None")
        except BaseException:
            if error is None:
                raise
            logger.exception("Sample observer failed while recording %s", type(error).__name__)


async def flush_sample_observers(args):
    paths = [
        *(getattr(args, "rollout_sample_hook_path", None) or []),
        *(getattr(args, "rollout_sample_observer_path", None) or []),
    ]
    for path in paths:
        flush = getattr(load_function(path), "flush", None)
        if flush is not None:
            result = flush(args)
            if inspect.isawaitable(result):
                await result


async def close_sample_hooks(args):
    paths = [
        *(getattr(args, "rollout_sample_hook_path", None) or []),
        *(getattr(args, "rollout_sample_observer_path", None) or []),
    ]
    for path in paths:
        close = getattr(load_function(path), "close", None)
        if close is not None:
            result = close(args)
            if inspect.isawaitable(result):
                await result


async def finalize_sample_representation(args, value, generate_function=None):
    """Let a generator discard wire-only copies after hooks and scoring finish."""
    if isinstance(value, list):
        for sample in value:
            await finalize_sample_representation(args, sample, generate_function)
        return
    if generate_function is None:
        path = value.generate_function_path or args.custom_generate_function_path
        generate_function = load_function(path) if path else None
    finalize = getattr(generate_function, "finalize", None)
    if finalize is not None:
        result = finalize(args, value)
        if inspect.isawaitable(result):
            result = await result
        if result is not None:
            raise TypeError("A generator finalizer must preserve the existing Sample and return None")
