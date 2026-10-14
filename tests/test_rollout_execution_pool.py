import asyncio
from types import SimpleNamespace

import pytest

from slime.rollout import execution_pool, sample_hooks, sglang_rollout
from slime.utils.types import Sample

NUM_GPUS = 0


@pytest.fixture
def worker_args(monkeypatch):
    sglang_rollout.GenerateState.clear_instances()
    monkeypatch.setattr(sglang_rollout, "load_tokenizer", lambda *a, **kw: None)
    monkeypatch.setattr(sglang_rollout, "load_processor", lambda *a, **kw: None)
    monkeypatch.setattr(execution_pool, "init_http_client", lambda *a, **kw: None)
    yield SimpleNamespace(
        hf_checkpoint="test",
        rollout_temperature=1,
        rollout_top_p=1,
        rollout_top_k=-1,
        rollout_max_response_len=32,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        sglang_dp_size=None,
        use_distributed_post=False,
        sglang_server_concurrency=2,
        rollout_num_gpus=2,
        rollout_num_gpus_per_engine=1,
        n_samples_per_prompt=2,
        sglang_enable_deterministic_inference=False,
        group_rm=False,
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        custom_generate_function_path="generate",
        rollout_sample_hook_path=["teacher"],
    )
    sglang_rollout.GenerateState.clear_instances()


def group(offset=0):
    return [Sample(index=offset + i, group_index=offset // 2, prompt="test", metadata={}) for i in range(2)]


def test_group_execution_preserves_hooks_rewards_and_rollout_boundary(monkeypatch, worker_args):
    seen = []

    async def generate(args, sample, sampling_params, **kwargs):
        sample.tokens = [1, sample.index]
        sample.response_length = 1
        sample.response = "output"
        sample.status = Sample.Status.COMPLETED
        sample.weight_versions = [sampling_params["version"]]
        return sample

    async def teacher(args, sample, *, rollout_id):
        sample.metadata["teacher_rollout"] = rollout_id
        seen.append((rollout_id, sample.index))

    async def reward(args, sample):
        assert "teacher_rollout" in sample.metadata
        return float(sample.index)

    generate.abort_mode = "request"
    monkeypatch.setattr(sglang_rollout, "load_function", lambda path: generate)
    monkeypatch.setattr(sample_hooks, "load_function", lambda path: teacher)
    monkeypatch.setattr(sglang_rollout, "async_rm", reward)

    async def run():
        worker = execution_pool.GroupWorker(worker_args, 2)
        for rollout_id in (7, 8):
            await worker.start(rollout_id)
            output = await worker.generate(rollout_id, group(2 * rollout_id), {"version": str(rollout_id)})
            assert [s.index for s in output] == [2 * rollout_id, 2 * rollout_id + 1]
            assert [s.reward for s in output] == [float(s.index) for s in output]
            assert all(s.weight_versions == [str(rollout_id)] for s in output)
            assert all(s.metadata["teacher_rollout"] == rollout_id for s in output)
            with pytest.raises(RuntimeError, match="Drain"):
                await worker.start(rollout_id + 1)
            stats = await worker.stop()
            assert stats["groups"] == 1
            assert not worker.state.pendings and not worker.state.cancellable_tasks
        with pytest.raises(RuntimeError, match="stale"):
            await worker.generate(7, group(), {})
        assert seen == [(7, 14), (7, 15), (8, 16), (8, 17)]

    asyncio.run(run())


def test_stop_drains_request_cancellation_before_next_round(monkeypatch, worker_args):
    worker_args.rollout_sample_hook_path = None

    async def run():
        entered = asyncio.Event()
        exited = []

        async def generate(args, sample, sampling_params, **kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                exited.append(sample.index)

        generate.abort_mode = "request"
        monkeypatch.setattr(sglang_rollout, "load_function", lambda path: generate)
        worker = execution_pool.GroupWorker(worker_args, 2)
        await worker.start(0)
        pending = asyncio.create_task(worker.generate(0, group(), {}))
        await entered.wait()
        await worker.stop()
        output = await pending
        assert all(s.status == Sample.Status.ABORTED for s in output)
        assert sorted(exited) == [0, 1]
        assert not worker.state.pendings and not worker.state.cancellable_tasks
        await worker.start(1)
        assert not worker.state.aborted
        await worker.stop()

    asyncio.run(run())


def test_failed_reward_is_not_returned_as_a_successful_group(monkeypatch, worker_args):
    worker_args.rollout_sample_hook_path = None

    async def generate(args, sample, sampling_params, **kwargs):
        sample.status = Sample.Status.COMPLETED
        sample.response = "output"
        return sample

    async def reward(args, sample):
        raise ValueError("frozen scoring failed")

    generate.abort_mode = "request"
    monkeypatch.setattr(sglang_rollout, "load_function", lambda path: generate)
    monkeypatch.setattr(sglang_rollout, "async_rm", reward)

    async def run():
        worker = execution_pool.GroupWorker(worker_args, 2)
        await worker.start(0)
        with pytest.raises(ValueError, match="frozen scoring failed"):
            await worker.generate(0, group(), {})
        stats = await worker.stop()
        assert stats["groups"] == 0

    asyncio.run(run())


def test_reward_failure_drains_other_scoring_tasks(monkeypatch, worker_args):
    worker_args.rollout_sample_hook_path = None

    async def run():
        sibling_started = asyncio.Event()
        sibling_finished = []

        async def generate(args, sample, sampling_params, **kwargs):
            sample.status = Sample.Status.COMPLETED
            sample.response = "output"
            return sample

        async def reward(args, sample):
            if sample.index == 0:
                await sibling_started.wait()
                raise ValueError("scoring failed")
            sibling_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                sibling_finished.append(sample.index)

        generate.abort_mode = "request"
        monkeypatch.setattr(sglang_rollout, "load_function", lambda path: generate)
        monkeypatch.setattr(sglang_rollout, "async_rm", reward)
        worker = execution_pool.GroupWorker(worker_args, 2)
        await worker.start(0)
        with pytest.raises(ValueError, match="scoring failed"):
            await worker.generate(0, group(), {})
        assert sibling_finished == [1]
        await worker.stop()

    asyncio.run(run())


def test_stop_prevents_queued_groups_from_starting():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        dispatched = []

        async def execute(rollout_id, samples, sampling_params):
            dispatched.append(samples[0].index)
            entered.set()
            await release.wait()
            return samples

        async def stop():
            release.set()
            return {"cpu_seconds": 0, "groups": 1}

        def remote(function):
            return SimpleNamespace(remote=lambda *args: asyncio.create_task(function(*args)))

        pool = object.__new__(execution_pool.RolloutExecutionPool)
        pool.workers = [SimpleNamespace(generate_serialized=remote(execute), stop=remote(stop))]
        pool.limits = [asyncio.Semaphore(1)]
        pool.pending = set()
        pool.running = True
        pool.stopping = None
        pool.metrics = {}
        pool.rollout_id = 0
        pool.next_worker = 0
        pool.available = asyncio.Queue()
        pool.available.put_nowait(0)
        first = asyncio.create_task(pool.generate(group(), {}))
        await entered.wait()
        queued = asyncio.create_task(pool.generate(group(2), {}))
        await asyncio.sleep(0)
        await pool.stop()
        await first
        assert all(s.status == Sample.Status.ABORTED for s in await queued)
        assert dispatched == [0]
        assert not pool.pending
        await pool.stop()  # A second drain is harmless.

    asyncio.run(run())


def test_native_observer_records_teacher_failure_without_replacing_reward(monkeypatch, worker_args, tmp_path):
    import json
    from pathlib import Path
    from test_omni_rollout_plugin import geometry, output, teacher_binding
    from slime_plugins import omni_rollout
    from slime_plugins.prism import artifacts, teacher

    worker_args.rollout_sample_observer_path = ["observer"]
    worker_args.prism_artifact_dir = str(tmp_path / "attempt")
    worker_args.omni_audio_dir = str(tmp_path / "audio")
    worker_args.omni_return_audio = False
    worker_args.custom_rm_path = "original.reward"
    worker_args.prompt_data = "fixture"
    worker_args.omni_teacher_max_attempts = 1
    worker_args.omni_teacher_config = {"models": {"a": teacher_binding()}, "by_domain": {"chinese": "a"}}

    async def generate(args, sample, sampling_params, **kwargs):
        sample.metadata.update(
            task_type="Full VoiceClone",
            show_language=False,
            domain="chinese",
            references=[],
            transcript_target="hello",
            language="zh",
            reference_targets={},
        )
        return omni_rollout.parse_rollout(output(), sample, geometry())

    async def failed_request(*args, **kwargs):
        raise TimeoutError("injected teacher failure")

    generate.abort_mode = "request"
    monkeypatch.setattr(sglang_rollout, "load_function", lambda path: generate)
    monkeypatch.setattr(
        sample_hooks,
        "load_function",
        lambda path: {"teacher": teacher.score_teacher, "observer": artifacts.observe}[path],
    )
    monkeypatch.setattr(teacher, "_request", failed_request)

    async def run():
        worker = execution_pool.GroupWorker(worker_args, 2)
        await worker.start(0)
        items = group()
        result = await worker.generate_serialized(0, items, {})
        assert result is not None
        await worker.stop()
        for item in items:
            assert item.custom_rm_path is None
            assert item.remove_sample and item.reward is None
            assert item.metadata["rollout_rejection"]["reason"] == "teacher_request_failed"
            assert "policy_teacher_code_logprobs" not in item.multimodal_train_inputs
            row = json.loads(Path(item.metadata["prism_artifact"]["record_path"]).read_text())
            assert row["reward_status"] == "error"
            assert row["failure_stage"] == "sample_hooks"
            assert row["teacher"]["status"] == "error"
            assert row["teacher"]["error"] == "TimeoutError"
        await teacher.close_teacher_clients(worker.args)

    asyncio.run(run())


@pytest.mark.parametrize("transport", ["object-store", "straw"])
@pytest.mark.parametrize("group_size,group_rm", [(1, False), (2, False), (2, True)])
def test_teacher_403_is_archived_and_replaced_before_batch_publication(
    monkeypatch, worker_args, tmp_path, transport, group_size, group_rm
):
    import json

    import httpx
    from test_omni_rollout_plugin import geometry, output, teacher_binding, teacher_output

    from slime.data.transport import load_rollout_samples
    from slime.rollout.filter_hub.base_types import DynamicFilterOutput
    from slime.utils.http_utils import _post
    from slime_plugins import omni_rollout
    from slime_plugins.prism import artifacts, teacher

    args = worker_args
    binding = teacher_binding()
    binding["auth_token_env"] = "TEST_TEACHER_TOKEN"
    monkeypatch.setenv("TEST_TEACHER_TOKEN", "test-secret-never-archive")
    args.n_samples_per_prompt = group_size
    args.group_rm = group_rm
    args.rollout_batch_size = 2048 if group_size == 1 and transport == "object-store" else 4
    args.over_sampling_batch_size = 2
    args.rollout_data_transport = transport
    args.rollout_data_dir = str(tmp_path / "transport")
    args.save = None
    args.rollout_sample_filter_path = None
    args.rollout_all_samples_process_path = None
    args.dynamic_sampling_filter_path = "filter"
    args.rollout_sample_observer_path = ["observer"]
    args.prism_artifact_dir = str(tmp_path / "attempt")
    args.omni_audio_dir = str(tmp_path / "audio")
    args.omni_return_audio = False
    args.omni_compact_replay = True
    args.custom_rm_path = "original.reward"
    args.prompt_data = "fixture"
    args.omni_teacher_batch_size = 1
    args.omni_teacher_config = {"models": {"a": binding}, "by_domain": {"chinese": "a"}}
    candidates, rewarded, filtered = [], [], []

    def source(count):
        start = len(candidates)
        rows = [
            Sample(index=i, group_index=i // group_size, prompt="hello", metadata={})
            for i in range(start, start + count * group_size)
        ]
        candidates.extend(rows)
        return [rows[i : i + group_size] for i in range(0, len(rows), group_size)]

    async def generate(args, sample, sampling_params, **kwargs):
        sample.metadata.update(domain="chinese", task_type="Full VoiceClone", show_language=False, references=[])
        return omni_rollout.parse_rollout(output(), sample, geometry())

    async def reward(args, value):
        rows = value if isinstance(value, list) else [value]
        assert all(row.metadata["omni_teacher_status"] == "success" for row in rows)
        rewarded.extend(row.index for row in rows)
        return [0.0] * len(rows) if isinstance(value, list) else 0.0

    def keep_with_fallback(args, rows):
        # A custom filter must not be able to re-admit missing teacher scores.
        assert not any(row.remove_sample for row in rows)
        filtered.extend(row.index for row in rows)
        return DynamicFilterOutput(keep=True, keep_when_insufficient=True)

    generate.abort_mode = "request"
    generate.finalize = omni_rollout.finalize_sample
    monkeypatch.setattr(
        sglang_rollout, "load_function", lambda path: keep_with_fallback if path == "filter" else generate
    )
    monkeypatch.setattr(
        sample_hooks,
        "load_function",
        lambda path: {
            "teacher": teacher.score_teacher,
            "observer": artifacts.observe,
            "generate": generate,
        }[path],
    )
    monkeypatch.setattr(sglang_rollout, "async_rm", reward)
    monkeypatch.setattr(sglang_rollout, "batched_async_rm", reward)

    async def run():
        failed_id = "0:0:request-1"

        async def handle(request):
            payload = json.loads(request.content)
            if payload["samples"][0]["sample_id"] == failed_id:
                return httpx.Response(
                    403, text="denied Bearer test-secret-never-archive", headers={"x-request-id": "fault-403"}
                )
            return httpx.Response(
                200, json={"version": 1, "results": [teacher_output(row, binding) for row in payload["samples"]]}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:

            async def request(_args, url, payload, **kwargs):
                return await _post(client, url, payload, max_retries=1, **kwargs)

            monkeypatch.setattr(teacher, "_request", request)
            with sample_hooks.rollout_context(0):
                result, _ = await asyncio.wait_for(sglang_rollout.generate_rollout_async(args, 0, source), 45)
            groups = load_rollout_samples(result.samples)
            assert len(groups) == args.rollout_batch_size
            assert all(len(rows) == group_size for rows in groups)
            selected = [item for rows in groups for item in rows]
            assert all(item.group_index != 0 for item in selected)
            assert all(item.metadata["omni_teacher_status"] == "success" for item in selected)
            assert all("policy_teacher_code_logprobs" in item.multimodal_train_inputs for item in selected)
            assert result.metrics["rollout/dynamic_filter/drop_teacher_request_failed"] == 1
            assert 0 not in rewarded and 0 not in filtered
            assert candidates[0].reward is None and candidates[0].teacher_log_probs is None
            metrics = artifacts._summarize(args, 0, selected)
            assert metrics["rollout/prism/selected/sample_count"] == args.rollout_batch_size * group_size
            assert metrics["rollout/prism/teacher_failures/a/http_403"] == 1
            assert metrics["rollout/prism/teacher_failures/a/count"] == 1
            rows = [
                json.loads(line)
                for line in (tmp_path / "attempt/records/train/rollout_0000000.jsonl").read_text().splitlines()
            ]
            failed = next(row for row in rows if row["sample_index"] == 0)
            assert failed["selected_for_training"] is False and failed["remove_sample"]
            assert failed["reward_status"] == "error" and failed["teacher"]["status"] == "error"
            assert failed["teacher"]["failure"]["request_id"] == "fault-403"
            assert "test-secret-never-archive" not in json.dumps(failed)
            await teacher.close_teacher_clients(args)

    asyncio.run(run())


def test_fixed_evaluation_does_not_silently_drop_teacher_failures(monkeypatch, worker_args):
    async def generate(args, sample, sampling_params, **kwargs):
        sample.status = Sample.Status.COMPLETED
        return sample

    async def reject(args, sample, **kwargs):
        raise sample_hooks.SampleRejected("teacher_request_failed", {"status_code": 403})

    async def reward(*args):
        pytest.fail("A rejected evaluation sample must not get a fabricated reward")

    generate.abort_mode = "request"
    monkeypatch.setattr(sglang_rollout, "load_function", lambda path: generate)
    monkeypatch.setattr(sample_hooks, "load_function", lambda path: reject)
    monkeypatch.setattr(sglang_rollout, "async_rm", reward)
    with pytest.raises(sample_hooks.SampleRejected):
        asyncio.run(sglang_rollout.generate_and_rm(worker_args, Sample(), {}, evaluation=True))


def test_available_dispatch_uses_a_free_worker_while_another_is_busy():
    async def run():
        release = asyncio.Event()
        received = []

        async def slow(rollout, items, parameters):
            received.append((0, items[0].index))
            await release.wait()
            return items

        async def fast(rollout, items, parameters):
            received.append((1, items[0].index))
            return items

        pool = execution_pool.RolloutExecutionPool.__new__(execution_pool.RolloutExecutionPool)
        pool.workers = [SimpleNamespace(generate_serialized=SimpleNamespace(remote=fn)) for fn in (slow, fast)]
        pool.available = asyncio.Queue()
        for index in (0, 1):
            pool.available.put_nowait(index)
        pool.pending, pool.running, pool.rollout_id = set(), True, 0
        first = asyncio.create_task(pool.generate(group(0), {}))
        await asyncio.sleep(0)
        await pool.generate(group(2), {})
        await pool.generate(group(4), {})
        assert received == [(0, 0), (1, 2), (1, 4)]
        release.set()
        await first
        assert pool.available.qsize() == 2

    asyncio.run(run())
