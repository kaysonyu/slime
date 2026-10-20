"""Owned reward HTTP connections; request admission belongs to ServiceClient."""

import asyncio

import httpx


class RewardHTTPClient:
    """Reuse sockets without imposing another active-request limit.

    ServiceClient bounds requests before preparing their audio payloads. An
    independent pool limit would only move that wait into HTTP duration and
    couple reward capacity to another client's settings. Idle sockets remain
    bounded by the local request budget.
    """

    def __init__(self, concurrency):
        self.concurrency = concurrency
        self._client = None
        self._closed = False

    @property
    def client(self):
        if self._closed:
            raise RuntimeError("reward HTTP client is closed")
        if self._client is None:
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=None, max_keepalive_connections=self.concurrency),
                timeout=httpx.Timeout(None),
                trust_env=False,
            )
        return self._client

    async def _request(self, method, url, *, attempts=1, **kwargs):
        for attempt in range(attempts):
            response = None
            try:
                # request() builds a fresh multipart stream on every retry.
                response = await self.client.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as error:
                if isinstance(error, httpx.HTTPStatusError) and error.response.status_code not in {
                    408,
                    429,
                    500,
                    502,
                    503,
                    504,
                }:
                    raise
                if attempt + 1 == attempts:
                    raise
                await asyncio.sleep(min(2**attempt, 5))
            finally:
                if response is not None:
                    await response.aclose()

    async def get(self, url, **kwargs):
        return await self._request("GET", url, **kwargs)

    async def post(self, url, payload, **kwargs):
        return await self._request("POST", url, json=payload, **kwargs)

    async def post_file(self, url, *, fields, filename, content, **kwargs):
        return await self._request(
            "POST", url, data=fields, files={"file": (filename, content, "application/octet-stream")}, **kwargs
        )

    async def aclose(self):
        self._closed = True
        if self._client is not None:
            await self._client.aclose()
