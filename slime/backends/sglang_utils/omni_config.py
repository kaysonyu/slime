"""Slime deployment options for native Omni pipeline configurations."""

from dataclasses import replace
from pathlib import Path

from slime.backends.sglang_utils.sglang_config import ModelConfig, resolve_sglang_config


def resolve_omni_update_stage(args, overrides) -> str:
    stage = overrides.get("omni_update_stage", args.omni_update_stage)
    if not isinstance(stage, str) or not stage or any(char.isspace() for char in stage):
        raise ValueError("Omni requires --omni-update-stage or a server group omni_update_stage naming one stage")
    return stage


def resolve_omni_worker_capabilities(args, overrides) -> list[str]:
    capabilities = overrides.get("worker_capabilities", args.omni_worker_capabilities)
    if (
        not isinstance(capabilities, list)
        or not capabilities
        or any(not isinstance(name, str) or not name or any(char.isspace() for char in name) for name in capabilities)
        or len(set(capabilities)) != len(capabilities)
    ):
        raise ValueError(
            "Omni requires --omni-worker-capabilities or server group worker_capabilities as unique names"
        )
    return list(capabilities)


def resolve_omni_deployment(args) -> list[ModelConfig]:
    """Resolve Slime placement and options; Omni owns pipeline parsing and validation."""
    if args.rollout_external:
        raise ValueError("Omni rollout requires internally owned pipeline replicas")
    if args.sglang_router_ip is not None:
        raise ValueError("Omni owns its router; --sglang-router-ip is unsupported")
    models = resolve_sglang_config(args).models
    if not models:
        raise ValueError("Omni deployment requires at least one model")
    if len({model.name for model in models}) != len(models):
        raise ValueError("Omni model names must be unique")
    plans = []
    for model in models:
        model.resolve(args)
        groups = []
        for group in model.server_groups:
            if group.worker_type == "placeholder":
                groups.append(group)
                continue
            if group.worker_type != "regular":
                raise ValueError("Omni uses complete regular pipeline replicas, not PD/EPD server groups")
            replica_gpus = group.num_gpus_per_engine
            if (
                not replica_gpus
                or replica_gpus < 1
                or replica_gpus > args.num_gpus_per_node
                or group.num_gpus % replica_gpus
            ):
                raise ValueError("Omni replica GPUs must fit one node and divide the group's GPU allocation")
            config = group.overrides.get("omni_config", args.omni_config)
            if config is None:
                raise ValueError("Each Omni server group requires --omni-config or an omni_config override")
            path = Path(config).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            overrides = dict(
                group.overrides,
                omni_config=str(path),
                omni_update_stage=resolve_omni_update_stage(args, group.overrides),
                worker_capabilities=resolve_omni_worker_capabilities(args, group.overrides),
            )
            if not model.update_weights:
                # Frozen engines restore their weights without a learner publication.
                overrides.setdefault(f"{overrides['omni_update_stage']}.engine.enable_weights_cpu_backup", True)
            groups.append(replace(group, overrides=overrides))
        if not any(group.worker_type == "regular" for group in groups):
            raise ValueError("The Omni Python router requires at least one complete pipeline worker")
        plans.append(replace(model, server_groups=groups))
    return plans
