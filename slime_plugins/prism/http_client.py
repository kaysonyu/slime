"""Bounded HTTP ownership shared by generation and frozen action scoring."""

import asyncio
import math
from slime.utils.http_utils import post


async def _request(args, url, payload, *, headers=None):
    """Drain a locally owned HTTP task on cancellation, within a fixed bound.

    Draining the client is not a claim that the server stopped its GPU request.
    Generation/teacher POST is not known to be idempotent, so use one attempt.
    """
    timeout = float(getattr(args, "omni_request_timeout_s", 600.0))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("omni_request_timeout_s must be finite and positive")
    drain = float(getattr(args, "omni_cancel_drain_timeout_s", min(timeout, 30.0)))
    if not math.isfinite(drain) or drain <= 0:
        raise ValueError("omni_cancel_drain_timeout_s must be finite and positive")
    task = asyncio.create_task(post(url, payload, max_retries=1, headers=headers, timeout=timeout))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=drain)
        except Exception:
            task.cancel()
            # asyncio.wait does not wait indefinitely for a transport/remote
            # actor that ignores cancellation. Consume its eventual exception.
            await asyncio.wait({task}, timeout=min(drain, 1.0))
            task.add_done_callback(lambda finished: None if finished.cancelled() else finished.exception())
        raise
