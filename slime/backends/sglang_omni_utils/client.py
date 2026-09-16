"""Stage-aware Omni protocol with explicit validation of control operations."""

import httpx


class OmniClient:
    def __init__(self, endpoint, stage="tts_engine", timeout=600):
        self.endpoint = endpoint.rstrip("/")
        self.stage = stage
        self.http = httpx.AsyncClient(base_url=self.endpoint, timeout=timeout, trust_env=False)

    async def generate(self, payload):
        response = await self.http.post("/generate", json=payload)
        response.raise_for_status()
        return response.json()

    async def score_actions(self, samples):
        response = await self.http.post("/score_actions", json={"samples": samples})
        response.raise_for_status()
        result = response.json()
        if result.get("version") != 1 or len(result.get("results", [])) != len(samples):
            raise ValueError("Omni scoring response omitted requested trajectories")
        return result["results"]

    async def _admin(self, operation, payload=None):
        response = await self.http.post("/" + operation, json={**(payload or {}), "stages": [self.stage]})
        response.raise_for_status()
        result = response.json()
        if result.get("success") is not True:
            raise RuntimeError(f"Omni {operation} failed: {result}")
        results = result.get("stages", result.get("results", []))
        selected = [r for r in results if r.get("stage") == self.stage]
        if len(selected) != 1 or selected[0].get("success") is False:
            raise RuntimeError(f"Omni {operation} did not execute on {self.stage}: {result}")
        data = selected[0].get("data", {})
        if data.get("unsupported") or data.get("skipped") or data.get("success") is False:
            raise RuntimeError(f"Omni stage rejected {operation}: {data}")
        return data

    async def get_model_info(self):
        return await self._admin("model_info")

    async def pause_generation(self):
        return await self._admin("pause_generation", {"mode": "abort"})

    async def continue_generation(self):
        return await self._admin("continue_generation")

    async def init_weights_update_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend
    ):
        return await self._admin(
            "init_weights_update_group",
            dict(
                master_address=master_address,
                master_port=master_port,
                rank_offset=rank_offset,
                world_size=world_size,
                group_name=group_name,
                backend=backend,
            ),
        )

    async def destroy_weights_update_group(self, group_name):
        return await self._admin("destroy_weights_update_group", {"group_name": group_name})

    async def update_weights_from_distributed(self, names, dtypes, shapes, group_name, weight_version, **kwargs):
        return await self._admin(
            "update_weights_from_distributed",
            dict(
                names=names,
                dtypes=[str(dtype).removeprefix("torch.") for dtype in dtypes],
                shapes=[list(shape) for shape in shapes],
                group_name=group_name,
                weight_version=str(weight_version),
                **kwargs,
            ),
        )

    async def close(self):
        await self.http.aclose()
