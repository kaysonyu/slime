"""Domain-routed frozen-teacher scoring on the student's exact actions."""

import asyncio
import hashlib
import json
from importlib import import_module

from slime.backends.sglang_omni_utils.client import OmniClient


class TeacherScorer:
    def __init__(self, args):
        self.args = args
        self.clients = {}
        for entry in args.mopd_teachers:
            domain, separator, endpoint = entry.partition("=")
            if (
                not separator
                or not domain
                or domain in self.clients
                or not endpoint.startswith(("http://", "https://"))
            ):
                raise ValueError("Teachers must be unique DOMAIN=URL routes")
            self.clients[domain] = OmniClient(
                endpoint.removesuffix("/score_actions"), args.omni_stage, args.omni_timeout
            )

    async def score(self, sample):
        domain = sample.metadata.get("domain")
        if domain not in self.clients:
            raise ValueError(f"No teacher route for domain {domain!r}")
        request = sample.trajectory.score_request(sample.index, self.args.rollout_temperature)
        (result,) = await self.clients[domain].score_actions([request])
        expected = hashlib.sha256(
            json.dumps(
                {key: value for key, value in request.items() if key != "sample_id"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if result.get("sample_id") != str(sample.index) or result.get("input_sha256") != expected:
            raise ValueError("Teacher did not score the exact student conditioning and actions")
        if (
            result.get("temperature") != self.args.rollout_temperature
            or result.get("logprob_semantics") != "temperature_scaled_full_vocab_v1"
        ):
            raise ValueError("Teacher/student scoring distributions differ")
        if not result.get("teacher_weight_sha256") or result.get("weight_version") is None:
            raise ValueError("A frozen teacher must identify its weights and version")
        identity = {"version": str(result["weight_version"]), "weights": result["teacher_weight_sha256"]}
        if self.args.teacher_versions.setdefault(domain, identity) != identity:
            raise ValueError(f"Frozen teacher identity changed for domain {domain}")
        adapter = import_module(f"slime_plugins.models.{self.args.model_family}.data")
        sample.teacher_scores = adapter.teacher_scores(sample.trajectory, result, self.args.policy_config)
        sample.metadata.update(
            teacher_domain=domain, teacher_version=identity["version"], teacher_weight_sha256=identity["weights"]
        )
        sample.reward = 0.0

    async def close(self):
        await asyncio.gather(*(client.close() for client in self.clients.values()))
