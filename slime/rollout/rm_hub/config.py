"""Model-independent speech reward configuration, resolved before rollout."""

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import yaml


@dataclass(frozen=True)
class ComponentReward:
    name: str
    reward: float
    raw_value: float


@dataclass(frozen=True)
class RuntimeConfig:
    timeout_seconds: float = 120
    concurrency: int = 8
    max_retries: int = 2

    def __post_init__(self):
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Reward timeout_seconds must be finite and positive")
        for name, minimum in (("concurrency", 1), ("max_retries", 0)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"Reward {name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class ServiceConfig:
    endpoint: str
    auth_token_env: str | None = None
    model: str | None = None
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    protocol: str | None = None
    tokenizer_path: str | None = None

    def __post_init__(self):
        parsed = urlsplit(self.endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Reward endpoint must be an HTTP(S) URL without embedded credentials")
        if self.protocol not in (
            None,
            "openai_audio_transcriptions",
            "qwen3_asr_chat_path",
            "openai_chat",
            "openai_chat_path",
        ):
            raise ValueError(f"Unsupported reward protocol: {self.protocol}")
        if self.auth_token_env is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.auth_token_env):
            raise ValueError("auth_token_env must name an environment variable")


@dataclass(frozen=True)
class RewardComponentConfig:
    name: str
    weight: float

    def __post_init__(self):
        if self.name not in ("wer", "reference_similarity", "judge", "noop"):
            raise ValueError(f"Unsupported reward component: {self.name}")
        if isinstance(self.weight, bool) or not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("Reward weights must be finite and non-negative")


@dataclass(frozen=True)
class RewardConfig:
    components: tuple[RewardComponentConfig, ...]
    services: dict[str, ServiceConfig] = field(default_factory=dict)

    def __post_init__(self):
        names = [component.name for component in self.components]
        if len(names) != len(set(names)) or not any(component.weight > 0 for component in self.components):
            raise ValueError("Reward components must be unique with at least one positive weight")
        if set(self.services) - {"wer", "timbre_sim", "judge"}:
            raise ValueError("Unknown reward service")
        for name in ("wer", "judge"):
            if self.weight(name) > 0 and name not in self.services:
                raise ValueError(f"Active {name} reward requires services.{name}")
        protocols = {
            "wer": (None, "openai_audio_transcriptions", "qwen3_asr_chat_path"),
            "judge": (None, "openai_chat", "openai_chat_path"),
            "timbre_sim": (None,),
        }
        for name, service in self.services.items():
            if service.protocol not in protocols[name]:
                raise ValueError(f"Protocol {service.protocol} is not valid for {name}")
        if self.weight("judge") > 0:
            service = self.services["judge"]
            if not service.model or not service.tokenizer_path:
                raise ValueError("Judge requires model and a local tokenizer_path")

    def weight(self, name: str) -> float:
        return next((component.weight for component in self.components if component.name == name), 0.0)

    @property
    def identity(self) -> str:
        # Include the scoring definition and service/model identities, never secret values.
        payload = {
            "wer_definition": "delay_multilingual_bounded_v1",
            "components": [asdict(component) for component in self.components],
            "services": {
                name: {"model": service.model, "protocol": service.protocol, "tokenizer_path": service.tokenizer_path}
                for name, service in self.services.items()
            },
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _mapping(value, allowed, label):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f"{label} must be an object containing only {sorted(allowed)}")
    return value


def _expand(value):
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if re.search(r"\$\{?[A-Za-z_]", expanded):
            raise ValueError("Reward configuration contains an unset environment variable")
        return expanded
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def load_reward_config(path: str) -> RewardConfig:
    data = _mapping(_expand(yaml.safe_load(Path(path).read_text())), {"reward", "services"}, "Reward config")
    reward = _mapping(data.get("reward"), {"components"}, "reward")
    components = reward.get("components")
    if not isinstance(components, list):
        raise ValueError("reward.components must be a list")
    parsed = tuple(RewardComponentConfig(**_mapping(item, {"name", "weight"}, "component")) for item in components)
    services = {}
    for name, raw in _mapping(data.get("services", {}), {"wer", "timbre_sim", "judge"}, "services").items():
        values = dict(
            _mapping(raw, {"endpoint", "auth_token_env", "model", "runtime", "protocol", "tokenizer_path"}, name)
        )
        runtime = _mapping(values.pop("runtime", {}), {"timeout_seconds", "concurrency", "max_retries"}, "runtime")
        services[name] = ServiceConfig(**values, runtime=RuntimeConfig(**runtime))
    return RewardConfig(parsed, services)


def get_reward_config(args) -> RewardConfig:
    config = getattr(args, "reward_configuration", None)
    if config is not None:
        return config
    if getattr(args, "reward_config", None):
        config = load_reward_config(args.reward_config)
    else:
        endpoint = getattr(args, "asr_endpoint", None)
        if not endpoint:
            raise ValueError("WER requires --asr-endpoint or --reward-config")
        config = RewardConfig(
            (RewardComponentConfig("wer", 1.0),),
            {
                "wer": ServiceConfig(
                    endpoint=endpoint,
                    model=getattr(args, "asr_model", "qwen3-asr"),
                    protocol=getattr(args, "asr_protocol", "openai_audio_transcriptions"),
                    auth_token_env=getattr(args, "asr_auth_token_env", None),
                    runtime=RuntimeConfig(
                        timeout_seconds=getattr(args, "reward_timeout", 120),
                        concurrency=getattr(args, "reward_concurrency", 8),
                        max_retries=getattr(args, "reward_max_retries", 2),
                    ),
                )
            },
        )
    args.reward_configuration = config
    return config
