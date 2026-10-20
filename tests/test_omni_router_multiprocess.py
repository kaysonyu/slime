"""Real CPU HTTP control/data-plane and process cleanup contract."""

import json
import os
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

import psutil
import pytest
import requests

NUM_GPUS = 0
pytestmark = pytest.mark.unit


@pytest.mark.parametrize("processes,updates", [(2, 1), (4, 64), (8, 128), (16, 128)])
def test_native_multiprocess_router_control_forwarding_and_cleanup(tmp_path, processes, updates):
    pytest.importorskip("sglang_omni_router.python.supervisor")
    payload = json.dumps({"echo": "x" * 100000}).encode()

    class Backend(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
    threading.Thread(target=backend.serve_forever, daemon=True).start()
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    upstream = f"http://127.0.0.1:{backend.server_port}"
    configuration = dict(
        host="127.0.0.1",
        port=port,
        workers=[dict(url=upstream, capabilities=["chat"])],
        router_processes=processes,
        policy="round_robin",
        health_check_interval_secs=1,
        request_timeout_secs=10,
        shutdown_drain_secs=2,
        router_state_dir=str(tmp_path / "state"),
    )
    entry = Path(__file__).resolve().parents[1] / "slime/backends/sglang_utils/omni_router.py"
    logfile = tmp_path / "router.log"
    children = []
    with logfile.open("w") as log, requests.Session() as client:
        client.trust_env = False
        process = subprocess.Popen(
            [sys.executable, str(entry), json.dumps([configuration, False])],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            stdout=log,
            stderr=log,
        )
        url = f"http://127.0.0.1:{port}"

        def wait_status(path, status):
            deadline = time.monotonic() + 30
            consecutive = 0
            while time.monotonic() < deadline:
                assert process.poll() is None, logfile.read_text()
                try:
                    # A persistent connection sees only one DP. Probe fresh
                    # connections through a complete snapshot refresh window.
                    with requests.Session() as probe:
                        probe.trust_env = False
                        if probe.get(url + path, timeout=1).status_code == status:
                            consecutive += 1
                            if consecutive >= 12:
                                return
                        else:
                            consecutive = 0
                except requests.RequestException:
                    consecutive = 0
                time.sleep(0.05)
            pytest.fail(logfile.read_text())

        try:
            wait_status("/ready", 200)
            children = psutil.Process(process.pid).children(recursive=True)
            assert len(children) == processes + 1  # one CP plus all DPs

            def generate(_):
                with requests.Session() as local:
                    local.trust_env = False
                    response = local.post(url + "/generate", json={}, timeout=10)
                    assert response.status_code == 200, response.text
                    assert response.content == payload

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(generate, range(32)))

            def change_admission(disabled):
                start = threading.Barrier(updates)

                def update(_):
                    with requests.Session() as local:
                        local.trust_env = False
                        start.wait(timeout=10)
                        response = local.put(
                            url + "/workers/" + quote(upstream, safe=""), json={"disabled": disabled}, timeout=15
                        )
                        return response.status_code, response.text

                with ThreadPoolExecutor(max_workers=updates) as executor:
                    results = list(executor.map(update, range(updates)))
                failures = [result for result in results if result[0] != 200]
                assert not failures, failures

            change_admission(True)
            wait_status("/ready", 503)
            assert client.post(url + "/generate", json={}, timeout=5).status_code == 503
            change_admission(False)
            # The PUT itself must fence every DP, not just the one owning this
            # HTTP connection. No sleeps or request retries after publication.
            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(generate, range(32)))
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            backend.shutdown()
            backend.server_close()
        assert process.returncode == 0, logfile.read_text()
        assert not any(child.is_running() and child.status() != psutil.STATUS_ZOMBIE for child in children)
