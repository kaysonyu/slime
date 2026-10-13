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
import socket
import statistics
import threading
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from examples.prism.experiment import path_component

logger = logging.getLogger(__name__)
REWARD_HOOK = "examples.prism.artifacts.reward"
INDEX_LOCK = threading.Lock()


@lru_cache(maxsize=128)
def _resolved_directory(path):
    return Path(path).resolve()


def _resolved_leaf(path):
    path = Path(path)
    result = _resolved_directory(os.path.abspath(path.parent)) / path.name
    return result.resolve() if result.is_symlink() else result


@lru_cache(maxsize=1)
def _io_executor():
    # Bound concurrent metadata operations on shared filesystems.
    return ThreadPoolExecutor(max_workers=4, thread_name_prefix="prism-archive")


async def _run_io(function, *args):
    pending = asyncio.get_running_loop().run_in_executor(_io_executor(), function, *args)
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        # Draining a rollout must also finish its already submitted durable writes.
        await asyncio.shield(pending)
        raise


def audio_archive_root(args):
    """Isolate attempts even when an external audio storage root is reused."""
    storage = _resolved_directory(os.path.abspath(args.omni_audio_dir))
    run = _resolved_directory(os.path.abspath(args.prism_artifact_dir))
    if storage.is_relative_to(run):
        return storage
    digest = hashlib.sha256(str(run).encode()).hexdigest()[:12]
    return storage.joinpath(
        *(path_component(part) for part in run.parts[-3:-1]),
        f"{path_component(run.name)}-{digest}",
    )


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _record(sample, *, reward_value=None, reward_status="pending", error=None):
    metadata = sample.metadata
    artifact = metadata["prism_artifact"]
    trace = metadata.get("omni_rollout") or {}
    prompt = metadata.get("omni_prompt") or {}
    outcome = metadata.get("omni_reward") or {}
    return {
        "schema_version": 1,
        **artifact,
        "sample_index": sample.index,
        "group_index": sample.group_index,
        "prompt": prompt,
        "text": prompt.get("transcript_target", sample.prompt),
        "language": prompt.get("language", metadata.get("language")),
        "domain": metadata.get("domain"),
        "teacher_key": metadata.get("omni_teacher_key"),
        "reward_category": outcome.get("category") or metadata.get("reward_category"),
        "requested_reward_category": metadata.get("reward_category"),
        "generation_status": sample.status.value,
        "finish_reason": metadata.get("omni_finish_reason"),
        "frames": metadata.get("omni_frames"),
        "audio_seconds": metadata.get("omni_audio_seconds"),
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
        (
            metadata[key]
            for key in ("id", "sample_id", "prompt_id", "rollout_audio_id")
            if metadata.get(key) is not None
        ),
        fingerprint,
    )
    split = "eval" if evaluation else "train"
    audio_root = _resolved_directory(os.path.abspath(args.omni_audio_dir))
    directory = audio_archive_root(args) / split / f"rollout_{rollout_id:07d}"
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
    round_directory = directory
    directory /= f"group_{path_component(group_index)}_{path_component(data_id)}"
    directory.mkdir(parents=True, exist_ok=True)
    record_id = uuid.uuid4().hex
    stem = f"gen_audio_s{path_component(candidate_index)}_{record_id}"
    source = metadata.get("omni_audio_path")
    if source:
        source = _resolved_leaf(source)
        if not source.is_relative_to(audio_root):
            raise ValueError("Generated audio must belong to omni_audio_dir before archival")
        target = directory / f"{stem}.wav"
        os.replace(source, target)
        metadata["omni_audio_path"] = str(target)
    if not (directory / "prompt.json").exists():
        _write_json(directory / "prompt.json", prompt)
    text = prompt.get("transcript_target", sample.prompt)
    if isinstance(text, str):
        # Identical candidates share this prompt. Atomic JSON records are the
        # authoritative source; this text file is a convenience for listening.
        if not (directory / "prompt_text.txt").exists():
            (directory / "prompt_text.txt").write_text(text)
    for reference in prompt.get("references", []):
        uri = reference.get("uri")
        if isinstance(uri, str) and "://" not in uri:
            reference_path = _resolved_leaf(uri)
            if not reference_path.is_file():
                continue
            link = directory / f"ref_{path_component(reference['id'])}{reference_path.suffix}"
            try:
                link.symlink_to(reference_path)
            except FileExistsError:
                if not link.is_symlink() or (link.readlink() != reference_path and link.resolve() != reference_path):
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
    # One append-only index per process avoids walking thousands of group directories.
    index = round_directory / f"{socket.gethostname()}-{os.getpid()}.candidates.jsonl"
    with INDEX_LOCK, index.open("a") as stream:
        stream.write(
            json.dumps(str(Path(metadata["prism_artifact"]["record_path"]).relative_to(round_directory))) + "\n"
        )


async def prepare_sample(args, sample, *, rollout_id=None, evaluation=False):
    await _run_io(_prepare_sample, args, sample, rollout_id, evaluation)


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
                await _run_io(
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
        await _run_io(
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
    metrics["scored_ratio"] = len(scored) / len(records)
    for field in ("frames", "audio_seconds"):
        values = [record[field] for record in records if _number(record.get(field))]
        metrics[f"{field}/coverage_ratio"] = len(values) / len(records)
        if values:
            metrics.update({f"{field}/{key}": value for key, value in _statistics(values).items()})
            metrics[f"{field}/p50"] = _percentile(values, 0.5)
            metrics[f"{field}/p95"] = _percentile(values, 0.95)
    values = [r["reward"] for r in scored if _number(r["reward"])]
    if values:
        metrics.update({f"reward/{key}": value for key, value in _statistics(values).items()})
    components = defaultdict(list)
    groups = defaultdict(list)
    for record in scored:
        groups[(record["dataset"], record["prompt_index"])].append(record)
    # Retain completed components even when a different service/aggregation failed.
    for record in records:
        outcome = record.get("reward_details") or {}
        returned = outcome.get("components", {})
        for name, component in returned.items():
            components[name].append(component)
        for name in set(outcome.get("errors", {})) - returned.keys():
            if name in {"wer", "sim", "judge_clap"}:
                components[name].append({"status": "error"})
    for name, items in components.items():
        prefix = f"component/{path_component(name)}"
        metrics[f"{prefix}/active_count"] = len(items)
        metrics[f"{prefix}/active_ratio"] = len(items) / len(records)
        valid = [item for item in items if item.get("status") == "success" and _number(item.get("raw"))]
        metrics[f"{prefix}/valid_count"] = len(valid)
        metrics[f"{prefix}/coverage_ratio"] = len(valid) / len(records)
        metrics[f"{prefix}/valid_ratio"] = len(valid) / len(items)
        metrics[f"{prefix}/error_ratio"] = sum(item.get("status") == "error" for item in items) / len(items)
        metrics[f"{prefix}/invalid_audio_ratio"] = sum(item.get("status") == "invalid_audio" for item in items) / len(
            items
        )
        for field in ("raw", "transformed"):
            values = [item[field] for item in items if _number(item.get(field))]
            if values:
                metrics.update({f"{prefix}/{field}/{key}": value for key, value in _statistics(values).items()})
        if name == "wer":
            # Never pool character and word denominators, or different languages.
            languages = defaultdict(list)
            for item in valid:
                detail = item.get("details") or {}
                errors, tokens = detail.get("errors"), detail.get("reference_tokens")
                language = detail.get("language")
                if language and _number(errors) and errors >= 0 and _number(tokens) and tokens > 0:
                    languages[str(language)].append((errors, tokens))
            for language, counts in languages.items():
                key = f"{prefix}/language/{path_component(language)}"
                errors = sum(value[0] for value in counts)
                tokens = sum(value[1] for value in counts)
                metrics.update(
                    {
                        f"{key}/errors": errors,
                        f"{key}/reference_tokens": tokens,
                        f"{key}/corpus_error_rate": errors / tokens,
                        f"{key}/scored_count": len(counts),
                    }
                )
        elif name == "judge_clap":
            attributes = defaultdict(list)
            for item in valid:
                detail = item.get("details") or {}
                for query, score in zip(detail.get("queries", []), detail.get("attribute_scores", []), strict=True):
                    if isinstance(query, str) and _number(score):
                        attributes[query.partition(":")[0]].append(score)
            for attribute, scores in attributes.items():
                key = f"{prefix}/attribute/{path_component(attribute)}"
                metrics[f"{key}/mean"] = statistics.fmean(scores)
                metrics[f"{key}/coverage_ratio"] = len(scores) / len(records)
    complete = [group for group in groups.values() if len(group) == group_size]
    metrics["complete_group_count"] = len(complete)
    metrics["incomplete_group_count"] = len(groups) - len(complete)
    if group_size == 1:
        return metrics
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
    source = audio_archive_root(args) / split / f"rollout_{rollout_id:07d}"
    if evaluation:
        source /= path_component(dataset)
        evaluation_ids = {sample.metadata["prism_artifact"]["evaluation_id"] for sample in samples}
        if len(evaluation_ids) != 1:
            raise ValueError("Eval summary must belong to one evaluation invocation")
        evaluation_id = evaluation_ids.pop()
        source /= path_component(evaluation_id)
    # Selected samples already carry the final scores; only unselected records
    # need a disk read to retain pending, failed, or filtered candidates.
    selected_paths = set()
    for sample in samples:
        if "prism_artifact" not in sample.metadata:
            raise ValueError("Prism artifact logging requires prepare_sample on every candidate")
        selected_paths.add(sample.metadata["prism_artifact"]["record_path"])
    indexes = list(source.glob("*.candidates.jsonl"))
    if indexes:
        paths = [source / json.loads(line) for index in indexes for line in index.read_text().splitlines()]
    else:
        paths = source.glob("*/gen_audio_s*.json")
    records = {
        record["record_id"]: record
        for path in paths
        if str(path) not in selected_paths
        for record in [json.loads(path.read_text())]
        if record["run_dir"] == args.prism_artifact_dir
    }
    selected = set()
    for sample in samples:
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
    for field in ("reward_category", "language", "domain", "teacher_key"):
        for value in sorted({str(row[field]) for row in selected_rows if row.get(field) is not None}):
            namespace = "teacher" if field == "teacher_key" else field
            subsets[f"{namespace}/{path_component(value)}"] = [r for r in selected_rows if str(r.get(field)) == value]
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
