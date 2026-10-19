"""Standalone Omni router entry point, executed by ``--omni-python``.

This file must not import Slime: the serving interpreter may only contain Omni.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import httpx


class WorkerTransport(httpx.AsyncBaseTransport):
    """Keep each replica's upstream connection pool independent of other replicas."""

    def __init__(self, connections_per_worker):
        self.connections_per_worker = connections_per_worker
        self.pools = {}

    async def handle_async_request(self, request):
        origin = (request.url.scheme, request.url.host, request.url.port)
        if origin not in self.pools:
            self.pools[origin] = httpx.AsyncHTTPTransport(
                trust_env=False,
                limits=httpx.Limits(
                    max_connections=self.connections_per_worker,
                    max_keepalive_connections=self.connections_per_worker,
                ),
            )
        return await self.pools[origin].handle_async_request(request)

    async def aclose(self):
        await asyncio.gather(*(pool.aclose() for pool in self.pools.values()))


def create_router_app(configuration, *, disabled=False):
    from sglang_omni_router.python.app import create_app
    from sglang_omni_router.python.config import RouterConfig

    configuration = dict(configuration)
    connections_per_worker = configuration.pop("connections_per_worker", 128)
    configuration.pop("log_level", None)
    configuration.pop("log_level_http", None)
    configuration.pop("router_processes", None)
    router_config = RouterConfig(**configuration)
    client = httpx.AsyncClient(
        transport=WorkerTransport(connections_per_worker),
        timeout=httpx.Timeout(router_config.request_timeout_secs),
        trust_env=False,
    )
    health_client = httpx.AsyncClient(
        timeout=httpx.Timeout(router_config.health_check_timeout_secs),
        limits=httpx.Limits(max_connections=max(1, len(router_config.workers))),
        trust_env=False,
    )
    app = create_app(router_config, client=client, health_client=health_client)
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        async with client, health_client, original_lifespan(application):
            if disabled:
                for worker in application.state.workers:
                    worker.set_disabled(True)
            yield

    app.router.lifespan_context = lifespan
    return app


def run_multiprocess_router(configuration, *, disabled=False):
    """Use Omni's shared admission and control plane, not independent Uvicorn apps."""
    from sglang_omni_router.python.config import RouterConfig
    from sglang_omni_router.python.supervisor import RouterSupervisor

    if disabled:
        raise ValueError("Initially disabled routers require single-process mode")
    configuration = dict(configuration)
    processes = configuration.pop("router_processes")
    # Native Omni owns the aggregate HTTP pool in multi-process mode.
    configuration.pop("connections_per_worker", None)
    level = configuration.pop("log_level", "warning")
    configuration.pop("log_level_http", None)
    os.environ["SGLANG_OMNI_ROUTER_LOG_LEVEL"] = "warning" if level == "warn" else level
    # Like WorkerTransport(trust_env=False), all router traffic is internal.
    # Native CP/DP clients otherwise inherit developer-machine HTTP proxies.
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)

    def spawn_control_plane(context):
        return subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--control-plane"],
            env=context.child_env(),
            pass_fds=(context.death_pipe_fd,),
        )

    supervisor = RouterSupervisor(
        RouterConfig(**configuration), router_processes=processes, spawn_cp=spawn_control_plane
    )
    supervisor.start()
    supervisor.run_forever()  # Owns SIGTERM, child drain/reap and shared-memory cleanup.


class WorkerUpdateAcknowledgement:
    """Do not acknowledge Slime's worker enable/disable before every DP sees it.

    Omni's disk-update endpoint already uses this barrier. Slime publishes over
    NCCL and controls admission with worker PUTs, so it needs the same fence on
    those successful responses. No sleeps or generation retries hide a race.
    """

    def __init__(self, app, *, control_app):
        self.app = app
        self.control_app = control_app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "PUT" or not scope["path"].startswith("/workers/"):
            return await self.app(scope, receive, send)
        rejected = False

        async def fenced_send(message):
            nonlocal rejected
            if message["type"] == "http.response.start" and 200 <= message["status"] < 300:
                accepted, pending = await self.control_app.state.dp_snapshot_ack_barrier()
                if not accepted:
                    from starlette.responses import JSONResponse

                    rejected = True
                    response = JSONResponse(
                        {"error": "worker update not acknowledged by all data planes", "pending": pending},
                        status_code=503,
                    )
                    await response(scope, receive, send)
            if not rejected:
                await send(message)

        await self.app(scope, receive, fenced_send)


def run_control_plane():
    from sglang_omni_router.python import cp_runner

    original_factory = cp_runner.create_control_plane_app

    def factory(*args, **kwargs):
        app = original_factory(*args, **kwargs)
        app.add_middleware(WorkerUpdateAcknowledgement, control_app=app)
        return app

    cp_runner.create_control_plane_app = factory
    cp_runner.main()


def main():
    if sys.argv[1:] == ["--control-plane"]:
        run_control_plane()
        return
    import uvicorn
    from sglang_omni.serve.access_logging import configure_access_logging

    configuration, disabled = json.loads(sys.argv[1])
    log_level = configuration.get("log_level", "warning")
    log_level = "warning" if log_level == "warn" else log_level
    logging.basicConfig(level=log_level.upper())
    processes = configuration.get("router_processes", 1)
    if type(processes) is not int or processes < 1:
        raise ValueError("router_processes must be a positive integer")
    if processes > 1:
        run_multiprocess_router(configuration, disabled=disabled)
        return
    app = create_router_app(configuration, disabled=disabled)
    # Rollout CPU processing can outlive Uvicorn's short idle default.
    server_config = uvicorn.Config(
        app,
        host=configuration["host"],
        port=configuration["port"],
        workers=1,
        timeout_keep_alive=120,
        log_level=log_level,
    )
    configure_access_logging(configuration.get("log_level_http", "warning"))
    uvicorn.Server(server_config).run()


if __name__ == "__main__":
    main()
