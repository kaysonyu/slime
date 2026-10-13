"""Prism sample archives and reward monitoring through existing slime hooks.

Waveforms are grouped before scoring. Each candidate has an atomic JSON record
even if it is later filtered, cancelled, or its scoring service fails. Batch
logs compact these records into JSONL without serializing replay tensors.
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import statistics
import uuid
from collections import defaultdict
from pathlib import Path

from examples.prism.experiment import path_component

logger = logging.getLogger(__name__)
REWARD_HOOK = "examples.prism.artifacts.reward"


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _record(sample, *, reward_value=None, reward_status="pending", error=None):
    metadata = sample.metadata
    artifact = metadata["prism_artifact"]
    trace = metadata.get("omni_rollout") or {}
    prompt = metadata.get("omni_prompt") or {}
    return {
        "schema_version": 1,
        **artifact,
        "sample_index": sample.index,
        "group_index": sample.group_index,
        "prompt": prompt,
        "text": prompt.get("transcript_target", sample.prompt),
        "language": prompt.get("language", metadata.get("language")),
        "domain": metadata.get("domain"),
        "reward_category": metadata.get("reward_category"),
        "generation_status": sample.status.value,
        "finish_reason": metadata.get("omni_finish_reason"),
        "frames": metadata.get("omni_frames"),
        "audio_path": metadata.get("omni_audio_path"),
        "policy_version": metadata.get("omni_policy_version", metadata.get("policy_version")),
        "model_identity": trace.get("model_identity"),
        "sampling": trace.get("sampling"),
        "request_id": trace.get("request_id"),
        "teacher": {
            name.removeprefix("omni_teacher_"): value
            for name, value in metadata.items()
            if name.startswith("omni_teacher_")
        },
        "reward": reward_value,
        "reward_status": reward_status,
        "reward_details": metadata.get("omni_reward"),
        "error": error,
        "remove_sample": sample.remove_sample,
    }


def _prepare_sample(args, sample, rollout_id, evaluation):
    if rollout_id is None:
        raise ValueError("Prism archive hook requires the rollout context")
    metadata = sample.metadata
    prompt = metadata.get("omni_prompt") or {}
    fingerprint = hashlib.sha256(
        json.dumps(prompt or sample.prompt, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]
    data_id = next(
        (metadata[key] for key in ("id", "sample_id", "rollout_audio_id") if metadata.get(key) is not None),
        fingerprint,
    )
    split = "eval" if evaluation else "train"
    audio_root = Path(args.omni_audio_dir).resolve()
    directory = audio_root / split / f"rollout_{rollout_id:07d}"
    dataset = metadata.get("eval_dataset_name") if evaluation else str(args.prompt_data)
    if evaluation:
        if dataset is None or any(
            key not in metadata for key in ("eval_prompt_index", "eval_sample_index", "eval_run_id")
        ):
            raise ValueError("Eval archives require dataset, prompt and candidate indices")
        directory /= path_component(dataset)
        directory /= path_component(metadata["eval_run_id"])
        group_index = metadata["eval_prompt_index"]
        candidate_index = metadata["eval_sample_index"]
    else:
        group_index, candidate_index = sample.group_index, sample.index
    directory /= f"group_{path_component(group_index)}_{path_component(data_id)}"
    directory.mkdir(parents=True, exist_ok=True)
    record_id = uuid.uuid4().hex
    stem = f"gen_audio_s{path_component(candidate_index)}_{record_id}"
    source = metadata.get("omni_audio_path")
    if source:
        source = Path(source).resolve(strict=True)
        if not source.is_relative_to(audio_root):
            raise ValueError("Generated audio must belong to omni_audio_dir before archival")
        target = directory / f"{stem}.wav"
        os.replace(source, target)
        metadata["omni_audio_path"] = str(target)
    _write_json(directory / "prompt.json", prompt)
    text = prompt.get("transcript_target", sample.prompt)
    if isinstance(text, str):
        # Identical candidates share this prompt. Atomic JSON records are the
        # authoritative source; this text file is a convenience for listening.
        (directory / "prompt_text.txt").write_text(text)
    for reference in prompt.get("references", []):
        uri = reference.get("uri")
        if isinstance(uri, str) and "://" not in uri and Path(uri).is_file():
            reference_path = Path(uri).resolve()
            link = directory / f"ref_{path_component(reference['id'])}{reference_path.suffix}"
            try:
                link.symlink_to(reference_path)
            except FileExistsError:
                if link.resolve() != reference_path:
                    raise ValueError(f"Reference changed within an archived prompt group: {link}") from None

    reward_path = sample.custom_rm_path or args.prism_reward_function_path
    if reward_path == REWARD_HOOK:
        reward_path = metadata.get("prism_artifact", {}).get("reward_function_path", args.prism_reward_function_path)
    metadata["prism_artifact"] = {
        "record_id": record_id,
        "record_path": str(directory / f"{stem}.json"),
        "run_dir": args.prism_artifact_dir,
        "generation_rollout_id": rollout_id,
        "split": split,
        "dataset": dataset,
        "data_id": data_id,
        "prompt_index": group_index,
        "candidate_index": candidate_index,
        "evaluation_id": metadata.get("eval_run_id") if evaluation else None,
        "reward_function_path": reward_path,
    }
    # Per-dataset rewards retain precedence. The wrapper owns persistence only,
    # and invokes the original function with its original scalar/batch shape.
    sample.custom_rm_path = REWARD_HOOK
    _write_json(metadata["prism_artifact"]["record_path"], _record(sample))


async def prepare_sample(args, sample, *, rollout_id=None, evaluation=False):
    await asyncio.to_thread(_prepare_sample, args, sample, rollout_id, evaluation)


async def reward(args, sample, **kwargs):
    from slime.utils.misc import load_function

    samples = sample if isinstance(sample, list) else [sample]
    paths = {item.metadata["prism_artifact"]["reward_function_path"] for item in samples}
    if len(paths) != 1:
        # Normal eval scores single samples. A mixed batched invocation still
        # honors each sample's explicit reward selection and preserves order.
        results = await asyncio.gather(*(reward(args, item, **kwargs) for item in samples), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return results
    try:
        result = await load_function(paths.pop())(args, sample, **kwargs)
        values = result if isinstance(sample, list) else [result]
        if len(values) != len(samples):
            raise ValueError("Reward result count differs from archived candidate count")
    except BaseException as error:
        for item in samples:
            try:
                await asyncio.to_thread(
                    _write_json,
                    item.metadata["prism_artifact"]["record_path"],
                    _record(
                        item,
                        reward_status="cancelled" if isinstance(error, asyncio.CancelledError) else "error",
                        error=type(error).__name__,
                    ),
                )
            except Exception:
                logger.exception("Failed to record a scoring failure")
        raise
    for item, value in zip(samples, values, strict=True):
        await asyncio.to_thread(
            _write_json,
            item.metadata["prism_artifact"]["record_path"],
            _record(item, reward_value=value, reward_status="success"),
        )
    return result


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _statistics(values):
    return {"mean": statistics.fmean(values), "std": statistics.pstdev(values), "min": min(values), "max": max(values)}


def _percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def compute_metrics(records, group_size):
    """Use scored/active populations; missing components are never zero scores."""
    if not records:
        return {}
    metrics = {"sample_count": len(records)}
    for key, predicate in {
        "zero_frame_ratio": lambda r: r["frames"] == 0,
        "truncated_ratio": lambda r: r["generation_status"] == "truncated",
        "scoring_error_ratio": lambda r: r["reward_status"] == "error",
        "pending_ratio": lambda r: r["reward_status"] == "pending",
        "cancelled_ratio": lambda r: r["reward_status"] == "cancelled",
    }.items():
        metrics[key] = sum(predicate(record) for record in records) / len(records)
    scored = [record for record in records if record["reward_status"] == "success"]
    metrics["scored_count"] = len(scored)
    values = [r["reward"] for r in scored if _number(r["reward"])]
    if values:
        metrics.update({f"reward/{key}": value for key, value in _statistics(values).items()})
    components = defaultdict(list)
    groups = defaultdict(list)
    for record in scored:
        for name, component in (record.get("reward_details") or {}).get("components", {}).items():
            components[name].append(component)
        groups[(record["dataset"], record["prompt_index"])].append(record)
    for name, items in components.items():
        prefix = f"component/{path_component(name)}"
        metrics[f"{prefix}/active_count"] = len(items)
        metrics[f"{prefix}/active_ratio"] = len(items) / len(records)
        metrics[f"{prefix}/invalid_audio_ratio"] = sum(item.get("status") == "invalid_audio" for item in items) / len(
            items
        )
        for field in ("raw", "transformed"):
            values = [item[field] for item in items if _number(item.get(field))]
            if values:
                metrics.update({f"{prefix}/{field}/{key}": value for key, value in _statistics(values).items()})
    complete = [group for group in groups.values() if len(group) == group_size]
    metrics["complete_group_count"] = len(complete)
    metrics["incomplete_group_count"] = len(groups) - len(complete)
    for name in ("total", *sorted(components)):
        values_by_group = []
        for group in complete:
            values = [
                (
                    record["reward"]
                    if name == "total"
                    else (record.get("reward_details") or {}).get("components", {}).get(name, {}).get("transformed")
                )
                for record in group
            ]
            if all(_number(value) for value in values):
                values_by_group.append(values)
        if values_by_group:
            deviations = [statistics.pstdev(values) for values in values_by_group]
            prefix = f"group_reward/{path_component(name)}"
            metrics.update(
                {
                    f"{prefix}/mean": statistics.fmean(statistics.fmean(values) for values in values_by_group),
                    f"{prefix}/std_mean": statistics.fmean(deviations),
                    f"{prefix}/std_p50": _percentile(deviations, 0.5),
                    f"{prefix}/std_p90": _percentile(deviations, 0.9),
                    f"{prefix}/zero_std_ratio": sum(value <= 1e-12 for value in deviations) / len(deviations),
                }
            )
    return metrics


def _summarize(args, rollout_id, samples, *, dataset=None, group_size=None):
    evaluation = dataset is not None
    split = "eval" if evaluation else "train"
    source = Path(args.omni_audio_dir) / split / f"rollout_{rollout_id:07d}"
    if evaluation:
        source /= path_component(dataset)
        evaluation_ids = {sample.metadata["prism_artifact"]["evaluation_id"] for sample in samples}
        if len(evaluation_ids) != 1:
            raise ValueError("Eval summary must belong to one evaluation invocation")
        evaluation_id = evaluation_ids.pop()
        source /= path_component(evaluation_id)
    records = {
        record["record_id"]: record
        for path in source.glob("*/gen_audio_s*.json")
        for record in [json.loads(path.read_text())]
    }
    selected = set()
    for sample in samples:
        if "prism_artifact" not in sample.metadata:
            raise ValueError("Prism artifact logging requires prepare_sample on every candidate")
        record = _record(sample, reward_value=sample.reward, reward_status="success")
        records[record["record_id"]] = record
        selected.add(record["record_id"])
    rows = sorted(
        records.values(), key=lambda row: (str(row["prompt_index"]), str(row["candidate_index"]), row["record_id"])
    )
    for row in rows:
        row["selected_for_training"] = None if evaluation else row["record_id"] in selected
        row["training_rollout_id"] = rollout_id if row["selected_for_training"] else None
    destination = Path(args.prism_artifact_dir) / "records" / split
    destination.mkdir(parents=True, exist_ok=True)
    name = f"rollout_{rollout_id:07d}" + (
        f"_{path_component(dataset)}_{path_component(evaluation_id)}" if evaluation else ""
    )
    path = destination / f"{name}.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
    os.replace(temporary, path)

    prefix = f"eval/{path_component(dataset)}/prism" if evaluation else "rollout/prism"
    subsets = {"all": rows}
    if not evaluation:
        subsets.update(
            selected=[r for r in rows if r["selected_for_training"]],
            not_selected=[r for r in rows if not r["selected_for_training"]],
        )
    selected_rows = rows if evaluation else subsets["selected"]
    for field in ("reward_category", "language", "domain"):
        for value in sorted({str(row[field]) for row in selected_rows if row.get(field) is not None}):
            subsets[f"{field}/{path_component(value)}"] = [r for r in selected_rows if str(r.get(field)) == value]
    metrics = {
        f"{prefix}/{name}/{key}": value
        for name, subset in subsets.items()
        for key, value in compute_metrics(subset, group_size or args.n_samples_per_prompt).items()
    }
    _write_json(destination / f"{name}.metrics.json", metrics)
    return metrics


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    from slime.observability import logging_utils
    from slime.observability.metric_utils import compute_rollout_step
    from slime.utils.misc import load_function

    metrics = _summarize(args, rollout_id, samples)
    metrics["rollout/step"] = compute_rollout_step(args, rollout_id)
    logging_utils.log(args, metrics, step_key="rollout/step")
    logger.info("Prism artifacts for rollout %s: %s", rollout_id, args.prism_artifact_dir)
    hook = args.prism_rollout_log_function_path
    return bool(hook and load_function(hook)(rollout_id, args, samples, rollout_extra_metrics, rollout_time))


def log_eval_rollout_data(rollout_id, args, data, extra_metrics):
    from slime.observability import logging_utils
    from slime.observability.metric_utils import compute_rollout_step
    from slime.utils.misc import load_function

    sizes = {config.name: config.n_samples_per_eval_prompt for config in args.eval_datasets}
    metrics = {}
    for name, result in data.items():
        metrics.update(_summarize(args, rollout_id, result["samples"], dataset=name, group_size=sizes[name]))
    metrics["eval/step"] = compute_rollout_step(args, rollout_id)
    logging_utils.log(args, metrics, step_key="eval/step")
    hook = args.prism_eval_rollout_log_function_path
    return bool(hook and load_function(hook)(rollout_id, args, data, extra_metrics))
