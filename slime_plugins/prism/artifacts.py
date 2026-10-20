"""Prism sample archives and reward monitoring through existing slime hooks.

Waveforms are grouped before scoring. Each candidate has an atomic JSON record
even if it is later filtered, cancelled, or its scoring service fails. Batch
logs compact these records into JSONL without serializing replay tensors.
"""

import asyncio
import hashlib
import json
import logging
import os
import socket
import threading
import uuid
from _thread import LockType
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from functools import lru_cache
from pathlib import Path

from slime.rollout.sample_hooks import get_current_rollout_id
from slime_plugins.prism.metrics import compute_metrics, path_component

logger = logging.getLogger(__name__)
INDEX_LOCK = threading.Lock()
GROUP_LOCK = threading.Lock()
INDEX_STREAMS = OrderedDict()
INDEX_SYNC_SAMPLES = 32


@dataclass
class _ArchiveGroup:
    lock: LockType = dataclass_field(default_factory=threading.Lock)
    fingerprint: str | None = None


@lru_cache(maxsize=4096)
def _archive_group(directory):
    return _ArchiveGroup()


@lru_cache(maxsize=128)
def _resolved_directory(path):
    return Path(path).resolve()


def _resolved_leaf(path):
    path = Path(path)
    result = _resolved_directory(os.path.abspath(path.parent)) / path.name
    return result.resolve() if result.is_symlink() else result


@lru_cache(maxsize=2)
def _io_executor(preparing):
    # Score completion must not queue behind all outstanding preparations.
    return ThreadPoolExecutor(max_workers=8 if preparing else 4, thread_name_prefix="prism-archive")


async def _run_io(function, *args, preparing=False):
    pending = asyncio.get_running_loop().run_in_executor(_io_executor(preparing), function, *args)
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        # Draining a rollout must also finish its already submitted durable writes.
        await asyncio.shield(pending)
        raise


@lru_cache(maxsize=4096)
def _make_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _candidate_index(path):
    if path not in INDEX_STREAMS:
        if len(INDEX_STREAMS) >= 16:
            _, (stream, _) = INDEX_STREAMS.popitem(last=False)
            os.fsync(stream.fileno())
            stream.close()
        INDEX_STREAMS[path] = [Path(path).open("a", buffering=1), 0]
    INDEX_STREAMS.move_to_end(path)
    return INDEX_STREAMS[path]


def _flush_indexes():
    with INDEX_LOCK:
        for stream, _ in INDEX_STREAMS.values():
            os.fsync(stream.fileno())
            stream.close()
        INDEX_STREAMS.clear()


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


def _prompt_text(sample):
    metadata = sample.metadata
    prompt = metadata.get("omni_prompt") or {}
    text = metadata.get("transcript_target", prompt.get("script", sample.prompt))
    if isinstance(text, list):
        return "".join(segment["text"] for segment in text)
    return text


def _record(sample, *, reward_value=None, reward_status="pending", error=None, failure_stage=None):
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
        "text": _prompt_text(sample),
        "task_type": metadata.get("task_type"),
        "language": metadata.get("language"),
        "show_language": metadata.get("show_language"),
        "domain": sample.policy_domain or metadata.get("domain"),
        "teacher_key": sample.policy_teacher,
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
        "failure_stage": failure_stage,
        "remove_sample": sample.remove_sample,
    }


def _archive_location(args, sample, rollout_id, evaluation):
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
    audio_requested = getattr(args, "omni_return_audio", True)
    if evaluation and getattr(args, "omni_eval_return_audio", None) is not None:
        audio_requested = args.omni_eval_return_audio
    directory /= (
        f"group_{path_component(group_index)}_{path_component(data_id)}"
        if audio_requested
        else f"_records_{socket.gethostname()}_{os.getpid()}"
    )
    record_id = uuid.uuid4().hex
    stem = f"gen_audio_s{path_component(candidate_index)}_{record_id}"
    artifact = {
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
    }
    return directory, round_directory, fingerprint, audio_requested, artifact


def audio_path(args, sample, *, evaluation=False):
    directory, _, _, _, artifact = _archive_location(args, sample, get_current_rollout_id(), evaluation)
    _make_directory(str(directory))
    sample.metadata["prism_audio_record"] = artifact
    return Path(artifact["record_path"]).with_suffix(".wav")


def _prepare_sample(args, sample, rollout_id, evaluation):
    directory, round_directory, fingerprint, audio_requested, artifact = _archive_location(
        args, sample, rollout_id, evaluation
    )
    metadata = sample.metadata
    prompt = metadata.get("omni_prompt") or {}
    audio_root = _resolved_directory(os.path.abspath(args.omni_audio_dir))
    _make_directory(str(directory))
    direct_record = metadata.pop("prism_audio_record", None)
    source = metadata.get("omni_audio_path")
    if source:
        source = _resolved_leaf(source)
        if not source.is_relative_to(audio_root):
            raise ValueError("Generated audio must belong to omni_audio_dir before archival")
        if direct_record is not None:
            expected = Path(direct_record["record_path"]).with_suffix(".wav")
            if source != expected or any(
                direct_record[key] != artifact[key]
                for key in (
                    "run_dir",
                    "generation_rollout_id",
                    "split",
                    "dataset",
                    "data_id",
                    "prompt_index",
                    "candidate_index",
                    "evaluation_id",
                )
            ):
                raise ValueError("Generated audio does not match its archive context")
            artifact = direct_record
        else:
            target = Path(artifact["record_path"]).with_suffix(".wav")
            os.replace(source, target)
            metadata["omni_audio_path"] = str(target)
    if audio_requested:
        with GROUP_LOCK:
            group = _archive_group(str(directory))
        with group.lock:
            if group.fingerprint != fingerprint:
                if not (directory / "prompt.json").exists():
                    _write_json(directory / "prompt.json", prompt)
                text = _prompt_text(sample)
                if isinstance(text, str) and not (directory / "prompt_text.txt").exists():
                    (directory / "prompt_text.txt").write_text(text)
                for slot, uri in prompt.get("references", {}).items():
                    if isinstance(uri, str) and "://" not in uri:
                        reference_path = _resolved_leaf(uri)
                        if not reference_path.is_file():
                            continue
                        link = directory / f"ref_{path_component(slot)}{reference_path.suffix}"
                        try:
                            link.symlink_to(reference_path)
                        except FileExistsError:
                            if not link.is_symlink() or (
                                link.readlink() != reference_path and link.resolve() != reference_path
                            ):
                                raise ValueError(
                                    f"Reference changed within an archived prompt group: {link}"
                                ) from None
                group.fingerprint = fingerprint

    reward_path = sample.custom_rm_path or args.custom_rm_path
    metadata["prism_artifact"] = {**artifact, "reward_function_path": reward_path}
    _write_json(metadata["prism_artifact"]["record_path"], _record(sample))
    # One append-only index per process avoids walking thousands of group directories.
    index = round_directory / f"{socket.gethostname()}-{os.getpid()}.candidates.jsonl"
    with INDEX_LOCK:
        entry = _candidate_index(str(index))
        stream = entry[0]
        stream.write(
            json.dumps(str(Path(metadata["prism_artifact"]["record_path"]).relative_to(round_directory))) + "\n"
        )
        entry[1] += 1
        if entry[1] >= INDEX_SYNC_SAMPLES:
            os.fsync(stream.fileno())
            entry[1] = 0


async def observe(args, sample, *, event, stage, error=None, evaluation=False, rollout_id=None):
    if sample.metadata is None:
        sample.metadata = {}
    if "prism_artifact" not in sample.metadata:
        await _run_io(_prepare_sample, args, sample, rollout_id, evaluation, preparing=True)
    if event == "generated":
        return
    cancelled = isinstance(error, asyncio.CancelledError) or sample.status.value == "aborted"
    status = "cancelled" if cancelled else "error" if error is not None else "success"
    await _run_io(
        _write_json,
        sample.metadata["prism_artifact"]["record_path"],
        _record(
            sample,
            reward_value=sample.reward,
            reward_status=status,
            error=type(error).__name__ if error is not None else None,
            failure_stage=stage if error is not None else None,
        ),
    )


async def flush(args):
    await _run_io(_flush_indexes)


observe.flush = flush


def _summarize(args, rollout_id, samples, *, dataset=None, group_size=None):
    _flush_indexes()
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
            raise ValueError("Prism artifact logging requires the observer on every candidate")
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
    metrics = {}
    populations = {}
    for subset_name, subset in subsets.items():
        identity = tuple(row["record_id"] for row in subset)
        if identity not in populations:
            populations[identity] = compute_metrics(subset, group_size or args.n_samples_per_prompt)
        metrics.update({f"{prefix}/{subset_name}/{key}": value for key, value in populations[identity].items()})
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
    hook = getattr(args, "prism_rollout_log_function_path", None)
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
    hook = getattr(args, "prism_eval_rollout_log_function_path", None)
    return bool(hook and load_function(hook)(rollout_id, args, data, extra_metrics))
