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
    worker_args.rollout_sample_hook_path = []
    worker_args.use_opd = True
    worker_args.opd_teacher_config = {
        "score_function": "slime_plugins.prism.teacher.score_teacher",
        "models": {"a": teacher_binding()},
        "by_domain": {"chinese": "a"},
        "client": {"max_attempts": 1},
    }

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
        lambda path: {
            "slime_plugins.prism.teacher.score_teacher": teacher.score_teacher,
            "observer": artifacts.observe,
            "original.reward": lambda *args: None,
        }[path],
    )
    monkeypatch.setattr(teacher, "_request", failed_request)

    async def run():
        worker = execution_pool.GroupWorker(worker_args, 2)
        await worker.start(0)
        items = group()
        with pytest.raises(TimeoutError, match="injected"):
            await worker.generate(0, items, {})
        await worker.stop()
        for item in items:
            assert item.custom_rm_path is None
            row = json.loads(Path(item.metadata["prism_artifact"]["record_path"]).read_text())
            assert row["reward_status"] in {"error", "cancelled"}
            assert row["failure_stage"] == "teacher_scoring"
            assert row["teacher"]["status"] in {"error", "cancelled"}
        await worker.close()

    asyncio.run(run())


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


def test_cancelled_mopd_generation_never_enters_teacher_scoring(monkeypatch, worker_args):
    from slime.rollout import on_policy_distillation as opd

    worker_args.rollout_sample_hook_path = []
    worker_args.use_opd = True
    worker_args.opd_teacher_config = {"models": {}}

    async def run():
        entered = asyncio.Event()

        async def generate(args, item, sampling_params, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        generate.abort_mode = "request"
        monkeypatch.setattr(sglang_rollout, "load_function", lambda path: generate)
        monkeypatch.setattr(
            opd, "load_teacher_config", lambda *a: pytest.fail("cancelled request initialized a teacher")
        )
        worker = execution_pool.GroupWorker(worker_args, 2)
        await worker.start(0)
        pending = asyncio.create_task(worker.generate(0, group(), {}))
        await entered.wait()
        await worker.stop()
        result = await pending
        assert all(item.status == Sample.Status.ABORTED for item in result)
        assert worker.state.teacher_runtime is None

    asyncio.run(run())
