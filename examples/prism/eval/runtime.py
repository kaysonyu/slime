"""Slime hooks for the frozen eval package; never compose a training reward."""

import asyncio
import copy
import csv
import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from examples.prism.eval.metrics import judge_record, summarize, wer_record

from slime_plugins.audio_rewards.client import AudioRewardClient, _drain_on_cancel
from slime_plugins.prism.reward_adapter import QUERY_VERSION, RewardStateCache, reference_audio

logger = logging.getLogger(__name__)


class EvalState:
    def __init__(self, config, *, worker_index=0, worker_count=1):
        # Reuse ONLY service bindings. Training weights/scorers never affect eval.
        services = config.get("services", {})
        for name in ("wer", "sim", "judge_clap"):
            endpoint = services.get(name, {}).get("endpoint", "")
            if not endpoint or (urlsplit(endpoint).hostname or "").endswith(".invalid"):
                raise ValueError(f"Configure {name} endpoint in --omni-reward-config before running small eval")
        if services["wer"].get("model") != "Qwen3-ASR-1.7B":
            raise ValueError("Small eval is pinned to served model Qwen3-ASR-1.7B")
        self.client = AudioRewardClient(services, worker_index=worker_index, worker_count=worker_count)
        self._preflight_lock = asyncio.Lock()
        self._ready = False
        self.gt_cache = {}
        self.gt_locks = {}

    async def aclose(self):
        await self.client.aclose()

    async def preflight(self):
        async with self._preflight_lock:
            if self._ready:
                return
            # Validate authentication/config before model generation starts.
            self.client.service("wer")
            sim, judge = await asyncio.gather(
                self.client.service("sim").info(), self.client.service("judge_clap").info()
            )
            if sim.get("model") != "redimnet2" or sim.get("service_task") != "speaker_embedding":
                raise ValueError("Small eval requires ReDimNet2 speaker embeddings")
            if judge.get("model") != "judge-clap" or judge.get("service_task") != "judge_clap":
                raise ValueError("Small eval requires JudgeCLAP embeddings")
            self._ready = True

    async def gt(self, path, queries, request_id):
        # Asset filenames are content hashes; embedding version isolates models.
        identity = await self.client.service("judge_clap").info()
        key = (str(path), tuple(queries), identity["embedding_version"])
        lock = self.gt_locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key not in self.gt_cache:
                result = await self.client.judge_clap(path, queries, request_id, query_version=QUERY_VERSION)
                if result["status"] != "success":
                    raise ValueError("GT JudgeCLAP scoring did not succeed")
                self.gt_cache[key] = result
            return self.gt_cache[key]


states = RewardStateCache(EvalState)


def _token(value):
    return hashlib.sha256(str(value).encode()).hexdigest()[:24]


def directory(args, sample):
    root = getattr(args, "omni_audio_dir", None)
    if not root:
        raise ValueError("Small eval requires --omni-audio-dir on shared storage")
    run = sample.metadata.get("eval_run_id")
    if not run:
        raise ValueError("Small eval requires Slime eval_run_id")
    return Path(root).absolute() / "small-eval" / _token(run)


def audio_path(args, sample, *, evaluation=False):
    if not evaluation or "prism_eval" not in (sample.metadata or {}):
        raise ValueError("Small-eval audio hook is evaluation-only")
    recipe = sample.metadata["prism_eval"]
    stem = _token(recipe["id"] + f":{sample.metadata.get('eval_sample_index', 0)}")
    return directory(args, sample) / "audio" / (stem + ".wav")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def record(args, sample):
    metadata = sample.metadata
    recipe = metadata["prism_eval"]
    result = {
        "schema": "prism-small-eval-result.v1",
        "recipe": recipe,
        "text": metadata["transcript_target"],
        "prompt": metadata["omni_prompt"],
        "generation_status": metadata.get("prism_eval_generation", "missing"),
        "generation_error": metadata.get("prism_eval_generation_error"),
        "audio_path": metadata.get("omni_audio_path"),
        "audio_seconds": metadata.get("omni_audio_seconds"),
        "frames": metadata.get("omni_frames"),
        "truncated": getattr(sample.status, "value", None) == "truncated",
        "sampling_seed": metadata.get("prism_eval_seed"),
        "sampling": metadata.get("prism_eval_sampling"),
        "metrics": metadata.get("prism_eval_metrics", {}),
        "eval_run_id": metadata.get("eval_run_id"),
        "eval_sample_index": metadata.get("eval_sample_index", 0),
        "dataset": metadata.get("eval_dataset_name"),
        "weight_versions": getattr(sample, "weight_versions", []),
        "hf_checkpoint": getattr(args, "hf_checkpoint", None),
    }
    return result


def persist(args, sample):
    result = record(args, sample)
    stem = _token(result["recipe"]["id"] + f":{result['eval_sample_index']}")
    atomic_json(directory(args, sample) / "records" / (stem + ".json"), result)
    return result


def placeholder_reward(args):
    # Slime's eval return schema requires a reward. It is a plumbing sentinel;
    # custom logging below suppresses default reward/accuracy aggregates.
    keys = {getattr(args, "reward_key", None), getattr(args, "eval_reward_key", None)} - {None, ""}
    return dict.fromkeys(keys, 0.0) if keys else 0.0


async def generate(args, sample, sampling_params, *, evaluation=False):
    if not evaluation or "prism_eval" not in (sample.metadata or {}):
        raise ValueError("Small eval generator cannot be used for training")
    if getattr(args, "custom_eval_rollout_log_function_path", None) != "examples.prism.eval.runtime.log_eval":
        raise ValueError(
            "Small eval requires --custom-eval-rollout-log-function-path examples.prism.eval.runtime.log_eval"
        )
    if getattr(args, "group_rm", False):
        raise ValueError("Small eval requires per-sample scoring (group_rm=false)")
    if getattr(args, "use_opd", False) and getattr(args, "opd_eval_teacher_scoring", True):
        raise ValueError("Small eval requires --no-opd-eval-teacher-scoring")
    await states.get(args).preflight()
    from slime.utils.types import Sample
    from slime_plugins.omni_rollout import generate as omni_generate

    eval_args = copy.copy(args)
    eval_args.omni_eval_return_audio = True
    eval_args.omni_audio_path_function_path = "examples.prism.eval.runtime.audio_path"
    params = dict(sampling_params)
    seed = (
        sample.metadata["prism_eval"]["generation_seed"] + sample.metadata.get("eval_sample_index", 0)
    ) & 0x7FFFFFFF
    params["sampling_seed"] = seed
    sample.metadata.update(prism_eval_seed=seed, prism_eval_sampling=params)
    try:
        sample = await omni_generate(eval_args, sample, params, evaluation=True)
        sample.metadata["prism_eval_generation"] = "success"
        # Persist generated audio before remote scoring, so interrupted evaluations
        # can be rescored without regenerating or changing student weights.
        persist(args, sample)
        return sample
    except asyncio.CancelledError:
        raise
    except Exception as error:
        sample.metadata.update(prism_eval_generation="error", prism_eval_generation_error=type(error).__name__)
        sample.status, sample.response, sample.reward = Sample.Status.ABORTED, "", placeholder_reward(args)
        persist(args, sample)
        logger.exception("Small eval generation failed: %s", sample.metadata["prism_eval"]["id"])
        return sample


async def evaluate(state, metadata):
    recipe = metadata["prism_eval"]
    audio = metadata.get("omni_audio_path")
    metrics = {}
    if metadata.get("prism_eval_generation") != "success" or not audio or metadata.get("omni_frames") == 0:
        return {name: {"status": "invalid_audio"} for name in recipe["metrics"]}
    path = Path(audio)
    if not path.is_file():
        return {name: {"status": "missing_audio"} for name in recipe["metrics"]}

    async def component(name):
        start = time.perf_counter()
        request_id = uuid.uuid4().hex
        try:
            if name == "wer":
                response = await state.client.service("wer").transcribe(path, request_id)
                result = wer_record(
                    metadata["transcript_target"],
                    response["result"]["text"],
                    recipe["language"],
                    recipe["normalization"],
                )
                result["identity"] = response["identity"]
            elif name == "sim":
                _, reference = reference_audio(metadata["omni_prompt"])
                response = await state.client.sim(path, reference, request_id)
                result = {"status": response["status"], "value": response["raw"], "identity": response["identity"]}
            else:
                queries = recipe["queries"]
                response = await state.client.judge_clap(path, queries, request_id, query_version=QUERY_VERSION)
                if response["status"] != "success":
                    raise ValueError("Candidate JudgeCLAP scoring did not succeed")
                gt = (
                    await state.gt(recipe["gt_audio"], queries, request_id + "-gt") if recipe.get("gt_audio") else None
                )
                if gt is not None and gt["identity"] != response["identity"]:
                    raise ValueError("GT and candidate JudgeCLAP model identity differ")
                result = judge_record(
                    queries, response["details"]["attribute_scores"], gt["details"]["attribute_scores"] if gt else None
                )
                result["identity"] = response["identity"]
            result["request_id"] = request_id
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # Never reinterpret HTTP errors as low model quality or zero scores.
            logger.exception("Small eval scoring failed: %s %s", recipe["id"], name)
            result = {"status": "error", "error_type": type(error).__name__, "request_id": request_id}
        result["seconds"] = time.perf_counter() - start
        metrics[name] = result

    tasks = [asyncio.create_task(component(name)) for name in recipe["metrics"]]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await _drain_on_cancel(asyncio.gather(*tasks, return_exceptions=True))
    return metrics


async def score(args, sample, **kwargs):
    if "prism_eval" not in (sample.metadata or {}):
        raise ValueError("Small eval scorer requires frozen eval metadata")
    state = states.get(args)
    sample.metadata["prism_eval_metrics"] = await evaluate(state, sample.metadata)
    persist(args, sample)
    return placeholder_reward(args)


score.close = states.close


def finalize(args, sample):
    from slime_plugins.omni_rollout import finalize_sample

    finalize_sample(args, sample)


generate.finalize = finalize


def export_results(output, records, *, rollout_id=None, extra_metrics=None):
    """Write auditable records, scalar summaries and a listening index; no audio in TB."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Never blend embeddings from different scorer versions into one curve.
    for metric in ("wer", "sim", "judge_clap"):
        identities = {
            json.dumps(r["metrics"][metric].get("identity"), sort_keys=True)
            for r in records
            if r["metrics"].get(metric, {}).get("status") == "success"
        }
        if len(identities) > 1:
            raise ValueError(f"Mixed {metric} service identities in one evaluation")
    packages = {r["recipe"].get("package_fingerprint") for r in records}
    if len(packages) > 1:
        raise ValueError("Mixed frozen packages in one evaluation")
    scalars = summarize(records)
    atomic_json(
        output / "summary.json",
        {
            "schema": "prism-small-eval-summary.v1",
            "rollout_id": rollout_id,
            "scalars": scalars,
            "extra_metrics": extra_metrics or {},
        },
    )
    with (output / "samples.jsonl").open("w") as handle:
        for result in records:
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
    with (output / "listening.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "domain",
                "benchmark",
                "language",
                "arm",
                "text",
                "audio_path",
                "reference_path",
                "gt_path",
                "generation_status",
                "truncated",
                "wer",
                "sim",
                "judge_pass",
                "metric_status",
            ],
        )
        writer.writeheader()
        for r in records:
            recipe, metrics = r["recipe"], r["metrics"]
            writer.writerow(
                {
                    **{k: recipe[k] for k in ("id", "domain", "benchmark", "language", "arm")},
                    "text": r["text"],
                    "audio_path": r["audio_path"],
                    "reference_path": r["prompt"].get("references", {}).get("audio1"),
                    "gt_path": recipe.get("gt_audio"),
                    "generation_status": r["generation_status"],
                    "truncated": r["truncated"],
                    "wer": metrics.get("wer", {}).get("value"),
                    "sim": metrics.get("sim", {}).get("value"),
                    "judge_pass": metrics.get("judge_clap", {}).get("value"),
                    "metric_status": json.dumps({k: v["status"] for k, v in metrics.items()}),
                }
            )
    return scalars


def collect_records(args, data):
    """Restore expected denominators if Slime filtered or omitted prompt rows."""
    from types import SimpleNamespace

    records = []
    configs = {config.name: config for config in args.eval_datasets}
    if set(data) - set(configs):
        raise ValueError("Unexpected dataset in small eval results")
    for name, config in configs.items():
        samples = data.get(name, {}).get("samples", [])
        actual = {}
        for sample in samples:
            key = (sample.metadata["prism_eval"]["id"], sample.metadata.get("eval_sample_index", 0))
            if key in actual:
                raise ValueError(f"Duplicate eval result: {key}")
            actual[key] = sample
        run = samples[0].metadata.get("eval_run_id") if samples else uuid.uuid4().hex
        rows = (json.loads(line) for line in Path(config.path).read_text().splitlines() if line.strip())
        for row in rows:
            for repeat in range(config.n_samples_per_eval_prompt):
                key = (row["metadata"]["prism_eval"]["id"], repeat)
                sample = actual.pop(key, None)
                if sample is None:
                    meta = copy.deepcopy(row["metadata"])
                    meta.update(
                        prism_eval_generation="missing",
                        eval_run_id=run,
                        eval_sample_index=repeat,
                        eval_dataset_name=name,
                    )
                    sample = SimpleNamespace(metadata=meta, status=None, weight_versions=[])
                records.append(persist(args, sample))
        if actual:
            raise ValueError(f"Eval results contain IDs absent from frozen dataset: {name}")
    return records


def log_eval(rollout_id, args, data, extra_metrics):
    from slime.observability import logging_utils
    from slime.observability.metric_utils import compute_rollout_step

    records = collect_records(args, data)
    # New UUID even at the same training step/resume; never overwrite old runs.
    output = Path(args.omni_audio_dir) / "small-eval" / "summaries" / f"step-{rollout_id}-{uuid.uuid4().hex}"
    metrics = export_results(output, records, rollout_id=rollout_id, extra_metrics=extra_metrics)
    metrics = {f"eval/prism/{key}": value for key, value in metrics.items()}
    metrics["eval/step"] = compute_rollout_step(args, rollout_id)
    logging_utils.log(args, metrics, step_key="eval/step")
    logger.info("Prism small eval artifacts: %s", output)
    return True
