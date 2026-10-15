"""Task-vector decisions and FP32 Megatron shard updates at rollout boundaries."""

import hashlib
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist

from slime.data.checkpoint_boundary import write_checkpoint_json
from slime_plugins.prism.task_vectors import TaskVectorStore, file_sha256


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get("version") != 1 or not plan.get("vectors"):
        raise ValueError("IM-MOPD requires a version-1 pinned vector plan")
    for key, entry in plan["vectors"].items():
        if file_sha256(Path(entry["path"]) / "index.json") != entry["sha256"]:
            raise ValueError(f"Task vector changed: {key}")
    initial = plan.get("initial", {})
    for domain, value in initial.items():
        if domain not in plan["vectors"] or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid initial task-vector coefficients")
    steps = plan.get("steps", [])
    if steps != sorted(set(steps)) or any(type(step) is not int or step < 1 for step in steps):
        raise ValueError("IM-MOPD intervention steps must be ordered positive updates")
    if plan.get("mode", "recovery") not in ("recovery", "fixed"):
        raise ValueError("Unknown IM-MOPD decision mode")
    if not math.isfinite(plan.get("delta", 0)) or plan.get("delta", 0) <= 0:
        raise ValueError("IM-MOPD delta must be positive")
    if plan.get("mode", "recovery") == "recovery":
        if not math.isfinite(plan["gamma"]):
            raise ValueError("Recovery threshold must be finite")
        for domain, anchor in plan["anchors"].items():
            if domain not in plan["vectors"] or not all(math.isfinite(anchor[k]) for k in ("base", "teacher")):
                raise ValueError("Invalid reference scores")
            if anchor["teacher"] - anchor["base"] <= plan.get("min_anchor_gap", 0.01):
                raise ValueError(f"No positive, resolved teacher recovery gap: {domain}")
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return plan, digest


def decide(plan, completed_updates, scores):
    if completed_updates not in plan.get("steps", []):
        return {"event": f"update-{completed_updates}", "adds": {}, "recovery": {}}
    if plan.get("mode", "recovery") == "fixed":
        selected = plan["schedule"][str(completed_updates)]
        recovery = {}
    else:
        recovery = {}
        for domain, anchor in plan["anchors"].items():
            dataset = plan["datasets"][domain]
            score = scores[dataset]
            if not math.isfinite(score):
                raise ValueError(f"Invalid validation score: {dataset}")
            recovery[domain] = (score - anchor["base"]) / (anchor["teacher"] - anchor["base"])
        selected = [domain for domain, value in recovery.items() if value <= plan["gamma"]]
    if any(domain not in plan["vectors"] for domain in selected):
        raise ValueError("Merge decision names an unknown teacher")
    return {
        "event": f"update-{completed_updates}",
        "adds": {domain: plan["delta"] for domain in selected},
        "scores": scores,
        "recovery": recovery,
    }


def restore_state(args, loaded_iteration, digest):
    if not args.load or args.finetune:
        return {"version": 1, "plan_sha256": digest, "events": []}
    root = Path(args.load)
    path = (
        root if root.name.startswith("iter_") else root / f"iter_{loaded_iteration:07d}"
    ) / "task_vector_state.json"
    state = json.loads(path.read_text())
    if state.get("version") != 1 or state.get("plan_sha256") != digest:
        raise ValueError("Resumed IM-MOPD plan differs from the committed model")
    return state


def _model_parameters(models):
    if len(models) != 1:
        raise ValueError("Prism task-vector merging requires one PP=1 model")
    model = models[0]
    while hasattr(model, "module"):
        model = model.module
    return {id(parameter): (name, parameter) for name, parameter in model.named_parameters()}


@torch.no_grad()
def apply_task_vectors(models, optimizer, stores, coefficients):
    """Update owned FP32 master shards; preserve moments and synchronize model replicas."""
    parameters = _model_parameters(models)
    names = {name for name, parameter in parameters.values() if parameter.requires_grad}
    for store in stores.values():
        if set(store.tensors) != names:
            raise ValueError(f"Vector/model parameters differ: {sorted(set(store.tensors) ^ names)[:8]}")
    shards = []
    optimizers = getattr(optimizer, "chained_optimizers", [optimizer])
    for item in optimizers:
        if item.__class__.__name__ != "DistributedOptimizer" or item.config.use_precision_aware_optimizer:
            raise ValueError("Task-vector merging requires standard Megatron DistributedOptimizer")
        if item.ddp_config.overlap_param_gather or item.ddp_config.use_megatron_fsdp:
            raise ValueError("Task-vector merging requires synchronous DP parameter gather")
        for model_groups, main_groups in (
            (item.model_float16_groups, item.shard_fp32_from_float16_groups),
            (item.model_fp32_groups, item.shard_fp32_groups),
        ):
            for group, main_group in zip(model_groups, main_groups, strict=True):
                for parameter, main in zip(group, main_group, strict=True):
                    name, _ = parameters[id(parameter)]
                    span = item._get_model_param_range_map(parameter)["param"]
                    if main.dtype != torch.float32 or main.numel() != span.end - span.start:
                        raise ValueError(f"Unexpected optimizer master shard: {name}")
                    shards.append((name, parameter.shape, span.start, span.end, main))
    # Validate every local slice before touching any master parameter.
    for name, shape, start, end, _ in shards:
        for domain in coefficients:
            entry = stores[domain].tensors[name]
            if list(shape) != entry["shape"] or not 0 <= start <= end <= math.prod(shape):
                raise ValueError(f"Invalid task-vector shard: {name}")
    squared_delta = torch.zeros((), dtype=torch.float64, device=next(iter(parameters.values()))[1].device)
    for name, shape, start, end, main in shards:
        delta = torch.zeros_like(main)
        for domain, coefficient in coefficients.items():
            delta.add_(
                stores[domain].slice(name, shape, start, end).to(main.device).reshape_as(main), alpha=coefficient
            )
        if not torch.isfinite(delta).all():
            raise ValueError(f"Non-finite merge: {name}")
        main.add_(delta)
        squared_delta += delta.double().square().sum()
    for item in optimizers:
        item._copy_main_params_to_model_params()
        item.start_param_sync_for_bucket_group_subset(force_sync=True)
    for name, _shape, start, end, main in shards:
        parameter = next(value for parameter_name, value in parameters.values() if parameter_name == name)
        if not torch.equal(parameter.view(-1)[start:end], main.to(parameter.dtype).view(-1)):
            raise RuntimeError(f"Master/model weights differ after task-vector synchronization: {name}")
    dist.all_reduce(squared_delta)
    return float(squared_delta.sqrt())


class MergeOwner:
    def __init__(self, args, loaded_iteration):
        self.plan, self.digest = load_plan(args.im_mopd_config)
        self.state = restore_state(args, loaded_iteration, self.digest)
        self.stores = {
            key: TaskVectorStore(value["path"], value["sha256"]) for key, value in self.plan["vectors"].items()
        }
        error = None
        if dist.get_rank() == 0:
            try:
                for store in self.stores.values():
                    for entry in store.tensors.values():
                        if file_sha256(store.root / entry["file"]) != entry["sha256"]:
                            raise ValueError(f"Task-vector payload changed: {entry['file']}")
            except (ValueError, OSError) as failure:
                error = str(failure)
        results = [error]
        dist.broadcast_object_list(results, src=0)
        if results[0] is not None:
            raise ValueError(results[0])

    def apply(self, models, optimizer, event):
        previous = next((item for item in self.state["events"] if item["event"] == event["event"]), None)
        if previous is not None:
            if previous["adds"] != event["adds"]:
                raise ValueError("A committed merge event cannot be replayed with different coefficients")
            return self.state
        if any(
            key not in self.stores or not math.isfinite(value) or value < 0 for key, value in event["adds"].items()
        ):
            raise ValueError("Invalid task-vector event")
        norm = apply_task_vectors(models, optimizer, self.stores, event["adds"]) if event["adds"] else 0.0
        self.state["events"].append({**event, "delta_l2": norm})
        return self.state

    def save(self, root, rollout_id):
        write_checkpoint_json(Path(root) / f"iter_{rollout_id:07d}" / "task_vector_state.json", self.state)
