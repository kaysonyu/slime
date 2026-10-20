"""Standalone Omni router entry point, executed by ``--omni-python``.

This file must not import Slime: the serving interpreter may only contain Omni.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import httpx


def install_transport_error_diagnostics(*, component="omni_router"):
    """Observe native DP transport failures without retrying or changing pools.

    Install only in the standalone data-plane process, before clients exist.
    Native proxy handlers otherwise retain only the outer exception class.
    Never record headers, bodies, query strings or exception message text.
    """
    original_send = httpx.AsyncClient.send
    if getattr(original_send, "_prism_transport_diagnostics", False):
        return

    async def send(client, request, *args, **kwargs):
        try:
            return await original_send(client, request, *args, **kwargs)
        except httpx.TransportError as error:
            chain, seen = [], set()
            current = error
            while current is not None and id(current) not in seen and len(chain) < 8:
                seen.add(id(current))
                chain.append(
                    {
                        "type": f"{type(current).__module__}.{type(current).__name__}",
                        "errno": getattr(current, "errno", None),
                        "frames": [
                            {"file": frame.filename, "line": frame.lineno, "function": frame.name}
                            for frame in traceback.extract_tb(current.__traceback__)
                        ],
                    }
                )
                current = current.__cause__ or current.__context__
            logging.getLogger(__name__).error(
                "%s_transport_failure %s",
                component,
                json.dumps(
                    {
                        "time": time.time(),
                        "pid": os.getpid(),
                        "method": request.method,
                        "host": request.url.host,
                        "port": request.url.port,
                        "path": request.url.path,
                        "chain": chain,
                        "retried": False,
                    }
                ),
            )
            raise

    send._prism_transport_diagnostics = True
    httpx.AsyncClient.send = send


def install_active_close_diagnostics(*, component="omni_router"):
    """Record the caller closing an active HTTP/1 stream, without delaying it."""
    from httpcore._async.http11 import AsyncHTTP11Connection, HTTPConnectionState

    original_close = AsyncHTTP11Connection.aclose
    if getattr(original_close, "_prism_active_close_diagnostics", False):
        return

    async def close(connection):
        if connection._state == HTTPConnectionState.ACTIVE:
            logging.getLogger(__name__).error(
                "%s_active_connection_close %s",
                component,
                json.dumps(
                    {
                        "time": time.time(),
                        "pid": os.getpid(),
                        "host": connection._origin.host.decode("ascii", errors="replace"),
                        "port": connection._origin.port,
                        "request_count": connection._request_count,
                        "connection_id": id(connection),
                        "frames": [
                            {"file": frame.filename, "line": frame.lineno, "function": frame.name}
                            for frame in traceback.extract_stack(limit=20)
                        ],
                    }
                ),
            )
        return await original_close(connection)

    close._prism_active_close_diagnostics = True
    AsyncHTTP11Connection.aclose = close


def install_reserved_connection_guard():
    """Keep HTTP/1 pool assignments alive until their request releases them.

    httpcore 1.0.9 can evict a connection between assigning it and starting
    its request. Socket closure then races that request. See upstream #1110
    and #1111; neither a larger pool nor a retry establishes safe ownership.
    This process-local compatibility guard preserves limits, expiry, and
    errors. It does not modify installed packages or retry requests.
    """
    import httpcore
    from httpcore._async.connection_pool import AsyncConnectionPool

    original = AsyncConnectionPool._assign_requests_to_connections
    if getattr(original, "_prism_reserved_connection_guard", False):
        return
    if httpcore.__version__ != "1.0.9":
        raise RuntimeError("Revalidate the router pool compatibility guard for this httpcore version")

    def assign(pool):
        if pool._http2:
            return original(pool)
        reserved = {request.connection for request in pool._requests if request.connection is not None}
        closing = []
        for connection in list(pool._connections):
            if connection.is_closed():
                pool._connections.remove(connection)
            elif connection not in reserved and (
                connection.has_expired()
                or (connection.is_idle() and len(pool._connections) > pool._max_keepalive_connections)
            ):
                # Keep 1.0.9's existing retention policy, changing ownership
                # only. In particular, do not expand the connection budget.
                pool._connections.remove(connection)
                closing.append(connection)
        for request in pool._requests:
            if not request.is_queued():
                continue
            origin = request.request.url.origin
            available = next(
                (
                    item
                    for item in pool._connections
                    if item not in reserved and item.can_handle_request(origin) and item.is_available()
                ),
                None,
            )
            if available is None and len(pool._connections) >= pool._max_connections:
                idle = next(
                    (item for item in pool._connections if item not in reserved and item.is_idle()), None
                )
                if idle is None:
                    continue
                pool._connections.remove(idle)
                closing.append(idle)
            if available is None:
                available = pool.create_connection(origin)
                pool._connections.append(available)
            request.assign_to_connection(available)
            reserved.add(available)
        return closing

    assign._prism_reserved_connection_guard = True
    AsyncConnectionPool._assign_requests_to_connections = assign


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

    def spawn_data_plane(context, index, generation):
        from sglang_omni_router.python.supervisor import DP_GENERATION_ENV, DP_INDEX_ENV, SOCKET_FD_ENV

        environment = context.child_env()
        environment[SOCKET_FD_ENV] = str(context.socket_fd)
        environment[DP_INDEX_ENV] = str(index)
        environment[DP_GENERATION_ENV] = str(generation)
        return subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--data-plane"],
            env=environment,
            pass_fds=(context.socket_fd, context.death_pipe_fd),
        )

    supervisor = RouterSupervisor(
        RouterConfig(**configuration),
        router_processes=processes,
        spawn_cp=spawn_control_plane,
        spawn_dp=spawn_data_plane,
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


def run_data_plane():
    from sglang_omni_router.python import data_plane, dp_runner
    from sglang_omni_router.python.app_factory import load_config_from_env

    install_transport_error_diagnostics()
    install_active_close_diagnostics()
    install_reserved_connection_guard()
    config = load_config_from_env()
    # Slime enables/disables every replica concurrently. Shared-socket accept
    # is not guaranteed fair: a DP may receive an entire publication burst.
    # Keep body and HTTP-pool bounds, but size the bounded admin admission for
    # the replica fanout instead of Omni's fixed 16-request per-DP default.
    data_plane._MAX_CONCURRENT_FORWARDS = max(128, 2 * len(config.workers))
    dp_runner.main()


def main():
    if sys.argv[1:] == ["--control-plane"]:
        run_control_plane()
        return
    if sys.argv[1:] == ["--data-plane"]:
        run_data_plane()
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
