"""CPU lifecycle tests for workers-first Omni deployment startup."""

import asyncio
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import httpx
import pytest
import requests
import yaml

from slime.backends.sglang_utils import deployment, engine_group, omni_config, omni_deployment, omni_router
from slime.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, SglangConfig

NUM_GPUS = 0
pytestmark = pytest.mark.unit


@dataclass
class _Handle:
    operation: str
    engine: object
    error: Exception | None = None


class _Engine:
    def __init__(self, harness, number):
        self.harness = harness
        self.number = number
        self.url = f"http://worker-{number}:15000"
        self.shutdown = SimpleNamespace(remote=self._shutdown)
        self.attach_router = SimpleNamespace(remote=self._attach_router)

    def _shutdown(self):
        self.harness.events.append(("shutdown", self.number))
        if self.number in self.harness.shutdown_errors:
            raise RuntimeError("shutdown RPC unavailable")
        return _Handle("shutdown", self)

    def _attach_router(self, host, port):
        self.harness.events.append(("attach", self.number, host, port))
        return _Handle("attach", self, self.harness.attach_error)


class _Group:
    def __init__(self, harness, configuration):
        self.harness = harness
        self.configuration = configuration
        self.sglang_overrides = configuration.overrides
        self.router_ip = self.router_port = None
        self.nodes_per_engine = 1
        self.all_engines = []
        self.engine_urls = {}

    @property
    def engines(self):
        return self.all_engines

    def start_engines(self, port_cursors):
        self.harness.events.append(("start_engines",))
        if self.configuration.worker_type == "placeholder":
            return [], port_cursors
        count = self.configuration.num_gpus // self.configuration.num_gpus_per_engine
        for _ in range(count):
            engine = _Engine(self.harness, len(self.harness.engines))
            self.harness.engines.append(engine)
            self.all_engines.append(engine)
            self.engine_urls[len(self.all_engines) - 1] = engine.url
            if self.harness.start_error is not None:
                self.all_engines.append(None)
                raise self.harness.start_error
        return [_Handle("init", engine, self.harness.init_error) for engine in self.engines], {0: 16000}


class _Process:
    def __init__(self, harness, *, target, args):
        self.harness = harness
        self.number = len(harness.processes)
        self.target, self.args = target, args
        self.alive = False
        self.exitcode = 23
        harness.processes.append(self)

    def start(self):
        self.harness.events.append(("router_start", self.number))
        self.alive = self.number not in self.harness.exited_routers

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.harness.events.append(("router_terminate", self.number))
        if not self.harness.stubborn_router:
            self.alive = False

    def join(self, *, timeout):
        self.harness.events.append(("router_join", self.number, timeout))

    def kill(self):
        self.harness.events.append(("router_kill", self.number))
        self.alive = False


class _Session:
    def __init__(self, harness):
        self.harness = harness
        self.trust_env = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get(self, url, *, timeout):
        self.harness.events.append(
            ("router_ready" if url.endswith("/ready") else "router_live", url, timeout, self.trust_env)
        )
        if self.harness.http_error is not None:
            raise self.harness.http_error
        if url.endswith("/ready"):
            if self.harness.ready_error is not None:
                raise self.harness.ready_error
            if self.harness.ready_failures:
                self.harness.ready_failures -= 1
                raise requests.HTTPError("workers not ready")
        return SimpleNamespace(raise_for_status=lambda: None)


class _Harness:
    def __init__(self, monkeypatch):
        self.args = Namespace(
            rollout_backend="sglang-omni",
            rollout_external=False,
            colocate=False,
            offload_rollout=False,
            debug_train_only=False,
            debug_rollout_only=False,
            actor_num_nodes=1,
            actor_num_gpus_per_node=2,
            num_gpus_per_node=8,
            rollout_num_gpus_per_engine=1,
            hf_checkpoint="policy",
            sglang_router_ip=None,
            sglang_router_port=3001,
            sglang_router_request_timeout_secs=30,
            sglang_server_concurrency=4,
            omni_config=str(Path(__file__).resolve().parents[1] / "examples/prism/configs/student.yaml"),
            omni_update_stage="tts_engine",
            omni_worker_capabilities=["chat", "audio_input", "audio_output"],
            omni_router_policy="round_robin",
            omni_python="/python/omni",
            omni_admin_key="test-admin-key",
            rollout_health_check_timeout=0.3,
            rollout_cleanup_timeout=5,
        )
        self.models = [self.model("actor", replicas=2)]
        self.events, self.engines, self.groups, self.processes = [], [], [], []
        self.init_error = self.attach_error = self.start_error = self.http_error = None
        self.wait_error = self.ready_error = None
        self.ready_failures = 0
        self.get_timeouts = []
        self.shutdown_errors, self.exited_routers = set(), set()
        self.stubborn_router = False
        self.now = 0
        monkeypatch.setattr(omni_config, "resolve_sglang_config", lambda args: SglangConfig(self.models))
        monkeypatch.setattr(deployment, "ServerGroupPlacement", lambda **kwargs: self)
        monkeypatch.setattr(omni_deployment, "get_host_info", lambda: ("hostname", "127.0.0.1"))
        monkeypatch.setattr(omni_deployment.random, "randint", lambda *args: 3000)
        monkeypatch.setattr(omni_deployment, "find_available_port", lambda start: max(3002, start))
        monkeypatch.setattr(omni_deployment, "multiprocessing", SimpleNamespace(Process=self.process))
        monkeypatch.setattr(
            omni_deployment,
            "requests",
            SimpleNamespace(Session=lambda: _Session(self), RequestException=requests.RequestException),
        )
        fake_ray = SimpleNamespace(get=self.get, wait=self.wait, kill=self.kill)
        monkeypatch.setattr(omni_deployment, "ray", fake_ray)
        monkeypatch.setattr(engine_group, "ray", fake_ray)
        monkeypatch.setattr(omni_deployment, "time", SimpleNamespace(monotonic=lambda: self.now, sleep=self.sleep))

    @staticmethod
    def model(name, *, replicas=1, update_weights=True):
        return ModelConfig(
            name=name,
            update_weights=update_weights,
            server_groups=[ServerGroupConfig("regular", replicas, num_gpus_per_engine=1)],
        )

    def create(self, configuration, router_ip, router_port, *, overrides_extra=None):
        assert router_ip is None and router_port is None
        group = _Group(self, configuration)
        group.sglang_overrides = {**(overrides_extra or {}), **configuration.overrides}
        self.groups.append(group)
        return group

    def process(self, **kwargs):
        return _Process(self, **kwargs)

    def sleep(self, duration):
        self.now += duration

    def get(self, handles, *, timeout=None):
        self.get_timeouts.append(timeout)
        for handle in handles:
            self.events.append(("get", handle.operation, handle.engine.number))
            if handle.error is not None:
                raise handle.error
        return [None] * len(handles)

    def wait(self, handles, *, num_returns, timeout):
        self.events.append(("wait_shutdown", tuple(handle.engine.number for handle in handles), num_returns, timeout))
        if self.wait_error is not None:
            raise self.wait_error
        return handles, []

    def kill(self, engine, *, no_restart):
        self.events.append(("kill", engine.number, no_restart))

    def assert_cleaned(self):
        assert [event[1] for event in self.events if event[0] == "shutdown"] == list(range(len(self.engines)))
        assert [event[1:] for event in self.events if event[0] == "kill"] == [
            (number, True) for number in range(len(self.engines))
        ]
        assert all(not process.is_alive() for process in self.processes)
        assert not hasattr(self.args, "sglang_model_routers")


@pytest.fixture
def harness(monkeypatch):
    return _Harness(monkeypatch)


def test_workers_are_ready_before_named_routers_are_started(harness):
    harness.models.append(harness.model("teacher", update_weights=False))
    servers, handles = deployment.start_rollout_servers(harness.args, pg=object())

    assert handles == []
    assert list(servers) == ["actor", "teacher"]
    assert harness.args.sglang_model_routers == {"actor": ("127.0.0.1", 3001), "teacher": ("127.0.0.1", 3002)}
    assert servers["actor"].update_weights is True
    assert servers["teacher"].update_weights is False
    for process, numbers in [(harness.processes[0], [0, 1]), (harness.processes[1], [2])]:
        _, configuration, initially_disabled, admin_key = process.args
        assert initially_disabled is False
        assert admin_key == "test-admin-key"
        assert configuration["log_level"] == "warn"
        assert configuration["log_level_http"] == "warning"
        assert configuration["workers"] == [
            {"url": harness.engines[number].url, "capabilities": harness.args.omni_worker_capabilities}
            for number in numbers
        ]
        router_started = harness.events.index(("router_start", process.number))
        for number in numbers:
            assert harness.events.index(("get", "init", number)) < router_started
            assert (
                harness.events.index(("attach", number, configuration["host"], configuration["port"])) > router_started
            )
    assert all(event[-1] is False for event in harness.events if event[0] == "router_live")
    assert all(group.router_ip == "127.0.0.1" for group in harness.groups)
    assert not any(event[0] in {"shutdown", "kill"} for event in harness.events)
    starts = [i for i, event in enumerate(harness.events) if event[0] == "start_engines"]
    first_wait = next(i for i, event in enumerate(harness.events) if event[:2] == ("get", "init"))
    assert max(starts) < first_wait
    router_starts = [i for i, event in enumerate(harness.events) if event[0] == "router_start"]
    first_router_probe = next(i for i, event in enumerate(harness.events) if event[0] == "router_live")
    assert max(router_starts) < first_router_probe
    assert all(timeout is not None and 0 < timeout <= 600 for timeout in harness.get_timeouts)


def test_model_roles_resolve_concrete_offload_options_before_actor_creation(harness):
    harness.models.append(harness.model("teacher", update_weights=False))
    deployment.start_rollout_servers(harness.args, pg=object())

    student, teacher = harness.groups
    assert "tts_engine.engine.enable_weights_cpu_backup" not in student.sglang_overrides
    assert teacher.sglang_overrides["tts_engine.engine.enable_weights_cpu_backup"] is True
    assert all("update_weights" not in group.sglang_overrides for group in harness.groups)


def test_pipeline_validation_uses_resolved_model_gpu_count(harness, tmp_path):
    source = yaml.safe_load(Path(harness.args.omni_config).read_text())
    source["stages"] = {"tts_engine": {"gpu": 0}, "vocoder": {"gpu": 1}}
    path = tmp_path / "two-gpu-pipeline.yaml"
    path.write_text(yaml.safe_dump(source))
    harness.args.omni_config = str(path)
    harness.models = [
        ModelConfig(name="actor", num_gpus_per_engine=2, server_groups=[ServerGroupConfig("regular", 2)])
    ]

    servers, _ = deployment.start_rollout_servers(harness.args, pg=object())

    assert len(servers["actor"].engines) == 1
    assert harness.groups[0].configuration.num_gpus_per_engine == 2


def test_placeholder_reserves_slots_without_loading_a_pipeline(harness, tmp_path):
    source = yaml.safe_load(Path(harness.args.omni_config).read_text())
    source["stages"] = {"tts_engine": {"gpu": 0}, "vocoder": {"gpu": 1}}
    path = tmp_path / "two-gpu-pipeline.yaml"
    path.write_text(yaml.safe_dump(source))
    harness.args.omni_config = str(path)
    harness.models = [
        ModelConfig(
            name="actor",
            num_gpus_per_engine=2,
            server_groups=[
                ServerGroupConfig("placeholder", 1, 1, {"omni_config": str(tmp_path / "unused.yaml")}),
                ServerGroupConfig("regular", 2),
            ],
        )
    ]

    servers, _ = deployment.start_rollout_servers(harness.args, pg=object())

    assert len(servers["actor"].engines) == 1
    assert harness.groups[0].engines == []
    assert len(harness.processes[0].args[1]["workers"]) == 1


@pytest.mark.parametrize("reserved", [1, 9])
def test_placeholder_can_inherit_replica_size_without_divisibility_constraints(harness, reserved):
    harness.models = [
        ModelConfig(
            name="actor",
            num_gpus_per_engine=2,
            server_groups=[ServerGroupConfig("placeholder", reserved), ServerGroupConfig("regular", 2)],
        )
    ]
    servers, _ = deployment.start_rollout_servers(harness.args, object())
    assert harness.groups[0].configuration.num_gpus_per_engine == 2
    assert harness.groups[0].configuration.num_gpus == reserved
    assert len(servers["actor"].engines) == 1


@pytest.mark.parametrize("invalid", ["worker_type", "gpus", "pipeline", "no_workers"])
def test_all_models_are_validated_before_any_resources_are_created(harness, tmp_path, invalid):
    teacher = harness.model("teacher", update_weights=False)
    if invalid == "worker_type":
        teacher.server_groups = [ServerGroupConfig("prefill", 1)]
    elif invalid == "gpus":
        teacher.server_groups = [ServerGroupConfig("regular", 3, 2)]
    elif invalid == "pipeline":
        teacher.server_groups[0].overrides["omni_config"] = str(tmp_path / "missing.yaml")
    else:
        teacher.server_groups = [ServerGroupConfig("placeholder", 1)]
    harness.models.append(teacher)
    with pytest.raises((ValueError, FileNotFoundError)):
        deployment.start_rollout_servers(harness.args, object())
    assert not harness.groups and not harness.engines and not harness.processes
    assert harness.args.sglang_router_ip is None


def test_omni_rejects_an_external_router_before_creating_resources(harness):
    harness.args.sglang_router_ip = "192.0.2.7"
    with pytest.raises(ValueError, match="sglang-router-ip"):
        deployment.start_rollout_servers(harness.args, object())
    assert not harness.groups and not harness.processes
    assert harness.args.sglang_router_ip == "192.0.2.7"


def test_router_readiness_follows_attachment_and_is_retried(harness):
    harness.ready_failures = 2
    owned = []
    deployment.start_rollout_servers(harness.args, object(), on_router_started=owned.append)
    assert owned == harness.processes
    assert len([event for event in harness.events if event[0] == "router_live"]) == 1
    probes = [i for i, event in enumerate(harness.events) if event[0] == "router_ready"]
    attached = [i for i, event in enumerate(harness.events) if event[:2] == ("get", "attach")]
    assert len(probes) == 3 and min(probes) > max(attached)
    assert harness.now == pytest.approx(0.2)


def test_readiness_timeout_rolls_back_registered_processes(harness):
    harness.args.distributed_timeout_minutes = 0.01
    harness.ready_error = requests.HTTPError("no routable workers")
    owned = []
    with pytest.raises(TimeoutError, match="/ready"):
        deployment.start_rollout_servers(harness.args, object(), on_router_started=owned.append)
    assert owned == harness.processes
    assert harness.args.sglang_router_ip is None
    harness.assert_cleaned()


def test_router_registration_failure_keeps_the_original_error(harness):
    error = RuntimeError("owner rejected router")

    def register(process):
        assert process.is_alive()
        raise error

    with pytest.raises(RuntimeError) as caught:
        deployment.start_rollout_servers(harness.args, object(), on_router_started=register)
    assert caught.value is error
    harness.assert_cleaned()


def test_concurrent_routers_do_not_reuse_unbound_ports(harness, monkeypatch):
    harness.models.extend(
        [harness.model("teacher-a", update_weights=False), harness.model("teacher-b", update_weights=False)]
    )
    requested = []

    def available(start):
        requested.append(start)
        return start

    monkeypatch.setattr(omni_deployment, "find_available_port", available)
    monkeypatch.setattr(omni_deployment.random, "randint", lambda *args: 3001)
    servers, _ = deployment.start_rollout_servers(harness.args, object())
    assert [server.router_port for server in servers.values()] == [3001, 3002, 3003]
    assert requested[-1] == 3003


def test_frozen_teacher_uses_the_standalone_scoring_config(harness):
    path = Path(harness.args.omni_config).with_name("teacher-score.yaml")
    original = path.read_bytes()
    teacher = harness.model("teacher", update_weights=False)
    teacher.server_groups[0].overrides.update(omni_config=str(path), worker_capabilities=["audio_input"])
    harness.models.append(teacher)

    servers, _ = deployment.start_rollout_servers(harness.args, pg=object())

    assert servers["teacher"].update_weights is False
    assert harness.groups[-1].sglang_overrides["omni_config"] == str(path)
    assert harness.processes[-1].args[1]["workers"] == [
        {"url": harness.engines[-1].url, "capabilities": ["audio_input"]}
    ]
    assert path.read_bytes() == original


def test_missing_update_stage_fails_before_starting_resources(harness):
    harness.args.omni_update_stage = None
    with pytest.raises(ValueError, match="omni-update-stage"):
        deployment.start_rollout_servers(harness.args, pg=object())
    assert not harness.engines
    assert not harness.processes


def test_router_receives_explicit_diagnostic_log_levels(harness):
    harness.args.router_log_level = "debug"
    harness.args.sglang_log_level_http = "info"
    deployment.start_rollout_servers(harness.args, pg=object())
    configuration = harness.processes[0].args[1]
    assert configuration["log_level"] == "debug"
    assert configuration["log_level_http"] == "info"


@pytest.mark.parametrize("http_log_level", ["warning", "info"])
def test_python_router_access_logging_with_real_http(tmp_path, http_log_level):
    if importlib.util.find_spec("sglang_omni") is None:
        pytest.skip("Install the paired Omni checkout to test the Python router process")
    if importlib.util.find_spec("sglang_omni.serve.access_logging") is None:
        pytest.skip("The paired Omni checkout must include status-aware access logging")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    configuration = {
        "host": "127.0.0.1",
        "port": port,
        "workers": [{"url": f"http://127.0.0.1:{port}", "capabilities": ["audio_output"]}],
        "log_level": "warn",
        "log_level_http": http_log_level,
    }
    output = tmp_path / "router.log"
    with output.open("w") as stream, requests.Session() as client:
        client.trust_env = False
        process = subprocess.Popen(
            [sys.executable, str(Path(omni_router.__file__).resolve()), json.dumps([configuration, True])],
            stdout=stream,
            stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            cwd=tmp_path,
        )
        try:
            deadline = time.monotonic() + 30
            while True:
                assert process.poll() is None, output.read_text()
                try:
                    response = client.get(f"http://127.0.0.1:{port}/live", timeout=1)
                    if response.status_code == 200:
                        break
                except requests.ConnectionError:
                    pass
                assert time.monotonic() < deadline, output.read_text()
                time.sleep(0.05)
            assert client.get(f"http://127.0.0.1:{port}/missing", timeout=2).status_code == 404
            assert client.post(f"http://127.0.0.1:{port}/generate", json={}, timeout=2).status_code == 503
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    logs = output.read_text()
    assert ('"GET /live HTTP/1.1" 200' in logs) == (http_log_level == "info")
    assert any('"GET /missing HTTP/1.1" 404' in line and "WARNING" in line for line in logs.splitlines())
    assert any('"POST /generate HTTP/1.1" 503' in line and "ERROR" in line for line in logs.splitlines())


@pytest.mark.parametrize("admin_key", [None, "child-key"])
def test_router_launcher_uses_an_independent_script_and_private_environment(monkeypatch, admin_key):
    monkeypatch.setenv("SGLANG_OMNI_ADMIN_KEY", "parent-key")
    calls = []
    monkeypatch.setattr(omni_deployment.os, "execvpe", lambda *args: calls.append(args))
    configuration = {"host": "127.0.0.1", "port": 3001}
    omni_deployment.launch_omni_router("/serving/python", configuration, True, admin_key)
    executable, command, environment = calls[0]
    assert executable == command[0] == "/serving/python"
    assert Path(command[1]).resolve() == Path(omni_router.__file__).resolve()
    assert json.loads(command[2]) == [configuration, True]
    assert environment.get("SGLANG_OMNI_ADMIN_KEY") == admin_key
    assert os.environ["SGLANG_OMNI_ADMIN_KEY"] == "parent-key"


@pytest.mark.parametrize("disabled", [False, True])
def test_router_liveness_and_admission_do_not_replace_worker_readiness(monkeypatch, tmp_path, disabled):
    pytest.importorskip("sglang_omni_router.python.app")
    native_client = httpx.AsyncClient
    clients = []

    def make_client(**kwargs):
        client = native_client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"running": True, "text": "ok"})),
            timeout=kwargs["timeout"],
        )
        clients.append(client)
        return client

    monkeypatch.setattr(omni_router.httpx, "AsyncClient", make_client)

    async def exercise():
        config = {
            "host": "127.0.0.1",
            "port": 19887,
            "workers": [{"url": "http://worker:19888", "capabilities": ["chat"]}],
            "router_state_dir": str(tmp_path),
        }
        app = omni_router.create_router_app(config, disabled=disabled)
        async with app.router.lifespan_context(app):
            assert len(clients) == 2
            assert app.state.http_client is clients[0]
            assert app.state.health_http_client is clients[1]
            assert all(not client.is_closed for client in clients)
            worker = app.state.workers[0]
            assert worker.disabled is disabled
            for _ in range(100):
                if worker.consecutive_successes:
                    break
                await asyncio.sleep(0)
            assert worker.consecutive_successes == 1
            async with native_client(transport=httpx.ASGITransport(app=app), base_url="http://router") as api:
                assert (await api.get("/live")).status_code == 200
                assert (
                    await api.put("/workers/" + quote(worker.url, safe=""), json={"disabled": False})
                ).status_code == 200
                assert (await api.get("/ready")).status_code == 503
                assert (await api.post("/generate", json={})).status_code == 503
                await app.state.health_checker.check_all_workers_health()
                assert (await api.get("/ready")).status_code == 200
                assert (await api.post("/generate", json={})).status_code == 200
        assert all(client.is_closed for client in clients)
        assert config["workers"][0]["url"] == "http://worker:19888"

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["partial_start", "init", "router_timeout"])
def test_startup_failure_shuts_down_and_kills_all_created_actors(harness, failure):
    error = RuntimeError(f"original {failure} failure")
    if failure == "partial_start":
        harness.start_error = error
    elif failure == "init":
        harness.init_error = error
    else:
        harness.http_error = requests.ConnectionError("router not ready")

    expected_type = TimeoutError if failure == "router_timeout" else RuntimeError
    with pytest.raises(expected_type) as caught:
        deployment.start_rollout_servers(harness.args, pg=object())

    if failure in {"partial_start", "init"}:
        assert caught.value is error
    else:
        assert "startup timed out" in str(caught.value)
    harness.assert_cleaned()
    waits = [event for event in harness.events if event[0] == "wait_shutdown"]
    assert len(waits) == 1
    assert waits[0][2] == len(harness.engines)
    assert 0 < waits[0][3] <= harness.args.rollout_cleanup_timeout / 2
    if failure in {"partial_start", "init"}:
        assert harness.processes == []


def test_later_named_router_failure_cleans_up_earlier_model(harness):
    harness.models.append(harness.model("teacher", update_weights=False))
    harness.exited_routers.add(1)

    with pytest.raises(RuntimeError, match="exited with code 23"):
        deployment.start_rollout_servers(harness.args, pg=object())

    assert len(harness.engines) == 3
    assert len(harness.processes) == 2
    harness.assert_cleaned()
    assert ("router_terminate", 0) in harness.events
    assert ("router_terminate", 1) not in harness.events


def test_cleanup_errors_preserve_startup_failure_and_force_router_kill(harness):
    error = RuntimeError("attach failed")
    harness.attach_error = error
    harness.shutdown_errors.add(0)
    harness.wait_error = RuntimeError("shutdown wait failed")
    harness.stubborn_router = True

    with pytest.raises(RuntimeError) as caught:
        deployment.start_rollout_servers(harness.args, pg=object())

    assert caught.value is error
    harness.assert_cleaned()
    assert ("router_terminate", 0) in harness.events
    assert ("router_kill", 0) in harness.events
    joins = [event for event in harness.events if event[0] == "router_join"]
    assert len(joins) == 2
    assert all(0 < event[2] <= harness.args.rollout_cleanup_timeout for event in joins)


def test_pipeline_replicas_use_physical_gpus_and_actual_node_ports(monkeypatch):
    from slime.backends.sglang_utils import engine_group

    actor_locations = [("10.0.0.2", 18004), ("10.0.0.2", 18010), ("2001:db8::5", 18100)]
    actors, actor_requests, allocations, initializations = [], [], [], []

    def create_actor(args, **kwargs):
        host, port = actor_locations[len(actors)]
        handle = object()

        def allocate(*, start_port, consecutive):
            allocations.append((host, start_port, consecutive))
            return host, port

        def initialize(**address):
            initializations.append(address)
            return handle

        actor = SimpleNamespace(
            _get_current_node_ip_and_free_port=SimpleNamespace(remote=allocate),
            init=SimpleNamespace(remote=initialize),
            init_handle=handle,
        )
        actors.append(actor)
        actor_requests.append(kwargs)
        return actor

    scheduling_requests = []

    def schedule(**kwargs):
        scheduling_requests.append(kwargs)
        return kwargs

    remote_actor = SimpleNamespace(options=lambda **options: SimpleNamespace(remote=create_actor))
    monkeypatch.setattr(engine_group, "ray", SimpleNamespace(remote=lambda cls: remote_actor, get=lambda ref: ref))
    monkeypatch.setattr(engine_group, "PlacementGroupSchedulingStrategy", schedule)
    args = Namespace(
        rollout_backend="sglang-omni",
        debug_train_only=False,
        num_gpus_per_node=8,
        rollout_num_gpus=6,
        rollout_num_gpus_per_engine=2,
    )
    placement_group = object()
    group = engine_group.ServerGroup(
        args=args,
        pg=(placement_group, [7, 8, 13, 14, 2, 3], [2, 3, 4, 5, 0, 1]),
        all_engines=[None] * 3,
        num_gpus_per_engine=2,
        num_new_engines=0,
        rank_offset=3,
    )
    handles, cursors = group.start_engines({"10.0.0.9": 18000})

    assert handles == [actor.init_handle for actor in actors]
    assert [request["gpu_ids"] for request in actor_requests] == [[2, 3], [4, 5], [0, 1]]
    assert [request["base_gpu_id"] for request in actor_requests] == [2, 4, 0]
    assert [request["rank"] for request in actor_requests] == [3, 4, 5]
    assert [request["placement_group_bundle_index"] for request in scheduling_requests] == [7, 13, 2]
    assert all(request["placement_group"] is placement_group for request in scheduling_requests)
    assert allocations == [("10.0.0.2", 18000, 3), ("10.0.0.2", 18007, 3), ("2001:db8::5", 18013, 3)]
    assert cursors == {"10.0.0.9": 18000, "10.0.0.2": 18013, "2001:db8::5": 18103}
    assert initializations == [
        {
            "host": host,
            "port": port,
            "nccl_port": port + 1,
            "dist_init_addr": f"{host}:{port + 2}",
            "router_ip": None,
            "router_port": None,
        }
        for host, port in actor_locations
    ]
    assert group.engine_urls == {
        0: "http://10.0.0.2:18004",
        1: "http://10.0.0.2:18010",
        2: "http://[2001:db8::5]:18100",
    }
    assert group.num_new_engines == 3
    assert group.start_engines(cursors) == ([], cursors)
    assert group.num_new_engines == 0
    assert len(actors) == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
