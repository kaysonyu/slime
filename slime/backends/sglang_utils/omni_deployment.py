"""Start complete Omni pipelines and connect their routers."""

import json
import multiprocessing
import os
import random
import time
from collections.abc import Callable
from pathlib import Path

import ray
import requests

from slime.backends.sglang_utils.engine_group import RolloutServer, ServerGroupPlacement, shutdown_rollout_resources
from slime.backends.sglang_utils.omni_config import resolve_omni_deployment
from slime.utils.cleanup import Cleanup
from slime.utils.http_utils import _wrap_ipv6, find_available_port, get_host_info


def launch_omni_router(python: str, configuration: dict, disabled: bool, admin_key: str | None) -> None:
    """Run the standalone router with the serving interpreter and its admin key."""
    environment = os.environ.copy()
    environment.pop("SGLANG_OMNI_ADMIN_KEY", None)
    if admin_key:
        environment["SGLANG_OMNI_ADMIN_KEY"] = admin_key
    script = str(Path(__file__).with_name("omni_router.py").resolve())
    command = [python, script, json.dumps([configuration, disabled])]
    os.execvpe(python, command, environment)


def wait_for_omni_router(process, client, url, *, deadline):
    """Wait for the requested lifecycle endpoint within the startup deadline."""
    while time.monotonic() < deadline:
        if not process.is_alive():
            raise RuntimeError(f"Omni router exited with code {process.exitcode}")
        try:
            response = client.get(url, timeout=min(1, max(0.01, deadline - time.monotonic())))
            response.raise_for_status()
            return
        except requests.RequestException:
            time.sleep(0.1)
    raise TimeoutError(f"Omni router startup timed out waiting for {url}")


def start_omni_rollout_servers(
    args,
    placement: ServerGroupPlacement,
    *,
    on_router_started: Callable[[multiprocessing.Process], None] | None = None,
) -> tuple[dict[str, RolloutServer], list]:
    """Preflight all models, overlap engine initialization, then establish routing."""
    plans = resolve_omni_deployment(args)
    servers = {}
    port_cursors = {}
    created_groups, router_processes, init_handles = [], [], []
    deadline = time.monotonic() + getattr(args, "distributed_timeout_minutes", 10) * 60
    try:
        for plan in plans:
            groups = []
            servers[plan.name] = RolloutServer(
                server_groups=groups, model_name=plan.name, update_weights=plan.update_weights
            )
            for group_config in plan.server_groups:
                group = placement.create(group_config, None, None)
                created_groups.append(group)
                groups.append(group)
                pending, port_cursors = group.start_engines(port_cursors)
                init_handles.extend(pending)
        ray.get(init_handles, timeout=max(0, deadline - time.monotonic()))

        router_ports = set()
        for model_index, plan in enumerate(plans):
            server = servers[plan.name]
            workers = [
                {"url": group.engine_urls[index], "capabilities": group_config.overrides["worker_capabilities"]}
                for group_config, group in zip(plan.server_groups, server.server_groups, strict=True)
                if group_config.worker_type == "regular"
                for index in range(0, len(group.all_engines), group.nodes_per_engine)
            ]
            router_ip = _wrap_ipv6(get_host_info()[1])
            router_port = args.sglang_router_port if model_index == 0 else None
            router_port = router_port or find_available_port(random.randint(3000, 4000))
            # Earlier routers may not have bound their ports yet.
            while router_port in router_ports:
                router_port = find_available_port(router_port + 1)
            router_ports.add(router_port)
            configuration = {
                "host": router_ip.strip("[]"),
                "port": router_port,
                "workers": workers,
                "policy": args.omni_router_policy,
                "router_processes": getattr(args, "omni_router_processes", 1),
                "request_timeout_secs": args.sglang_router_request_timeout_secs,
                "log_level": getattr(args, "router_log_level", "warn"),
                "log_level_http": getattr(args, "sglang_log_level_http", None) or "warning",
            }
            process = multiprocessing.Process(
                target=launch_omni_router, args=(args.omni_python, configuration, False, args.omni_admin_key)
            )
            process.start()
            router_processes.append(process)
            if on_router_started is not None:
                on_router_started(process)
            server.router_ip, server.router_port = router_ip, router_port

        with requests.Session() as client:
            client.trust_env = False
            attach_handles = []
            for server, process in zip(servers.values(), router_processes, strict=True):
                router_ip, router_port = server.router_ip, server.router_port
                url = f"http://{router_ip}:{router_port}"
                wait_for_omni_router(process, client, f"{url}/live", deadline=deadline)
                for group in server.server_groups:
                    group.router_ip, group.router_port = router_ip, router_port
                attach_handles.extend(engine.attach_router.remote(router_ip, router_port) for engine in server.engines)
            ray.get(attach_handles, timeout=max(0, deadline - time.monotonic()))
            # HTTP liveness and admission updates do not make UNKNOWN workers
            # routable. All routers probe their workers concurrently.
            for server, process in zip(servers.values(), router_processes, strict=True):
                url = f"http://{server.router_ip}:{server.router_port}"
                wait_for_omni_router(process, client, f"{url}/ready", deadline=deadline)
    except BaseException:
        engines = [engine for group in created_groups for engine in group.all_engines if engine is not None]
        shutdown_rollout_resources(engines, router_processes, Cleanup(args.rollout_cleanup_timeout))
        raise
    first = next(iter(servers.values()))
    args.sglang_router_ip, args.sglang_router_port = first.router_ip, first.router_port
    args.sglang_model_routers = {name: (server.router_ip, server.router_port) for name, server in servers.items()}
    return servers, []
