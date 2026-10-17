"""Standalone Omni router entry point, executed by ``--omni-python``.

This file must not import Slime: the serving interpreter may only contain Omni.
"""

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager

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


def main():
    import uvicorn
    from sglang_omni.serve.access_logging import configure_access_logging

    configuration, disabled = json.loads(sys.argv[1])
    log_level = configuration.get("log_level", "warning")
    log_level = "warning" if log_level == "warn" else log_level
    logging.basicConfig(level=log_level.upper())
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
