"""Failure-only native-router diagnostics must preserve HTTP behavior."""

import asyncio
import errno
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

NUM_GPUS = 0
pytestmark = pytest.mark.unit
path = Path(__file__).resolve().parents[1] / "slime/backends/sglang_utils/omni_router.py"
spec = importlib.util.spec_from_file_location("router_diagnostics_under_test", path)
router = importlib.util.module_from_spec(spec)
spec.loader.exec_module(router)


def test_transport_diagnostic_preserves_exception_and_never_retries_or_leaks_payload(monkeypatch, caplog):
    request = httpx.Request(
        "POST", "http://user:password@127.0.0.1:9999/generate?token=querysecret",
        headers={"Authorization": "Bearer headersecret"}, content=b"bodysecret",
    )
    failure = httpx.ReadError("exceptionsecret", request=request)
    calls = []

    async def original(client, candidate, **kwargs):
        calls.append((candidate, kwargs))
        try:
            raise ConnectionResetError(errno.ECONNRESET, "causesecret")
        except ConnectionResetError as cause:
            raise failure from cause

    monkeypatch.setattr(httpx.AsyncClient, "send", original)
    router.install_transport_error_diagnostics()
    installed = httpx.AsyncClient.send
    router.install_transport_error_diagnostics()
    assert httpx.AsyncClient.send is installed
    with pytest.raises(httpx.ReadError) as raised:
        asyncio.run(httpx.AsyncClient.send(object(), request, stream=True))
    assert raised.value is failure
    assert calls == [(request, {"stream": True})]
    payload = json.loads(caplog.records[-1].message.split("omni_router_transport_failure ", 1)[1])
    assert payload["path"] == "/generate"
    assert payload["chain"][1]["errno"] == errno.ECONNRESET
    assert not payload["retried"]
    for secret in ("password", "querysecret", "headersecret", "bodysecret", "exceptionsecret", "causesecret"):
        assert secret not in caplog.text


def test_transport_diagnostic_success_is_silent_and_unchanged(monkeypatch, caplog):
    request = httpx.Request("GET", "http://127.0.0.1/health")
    response = httpx.Response(200, request=request)
    calls = []

    async def original(client, candidate, **kwargs):
        calls.append((candidate, kwargs))
        return response

    monkeypatch.setattr(httpx.AsyncClient, "send", original)
    router.install_transport_error_diagnostics()
    assert asyncio.run(httpx.AsyncClient.send(object(), request, stream=True)) is response
    assert calls == [(request, {"stream": True})]
    assert not caplog.records
