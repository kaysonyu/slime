import asyncio
from types import SimpleNamespace

import pytest

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

NUM_GPUS = 0


def make_state(generate):
    state = object.__new__(GenerateState)
    state.executor = SimpleNamespace(generate=generate)
    state.sampling_params = {}
    state.reset()
    return state


def test_completion_queue_accepts_refills_and_returns_each_group_once():
    async def run():
        gates = [asyncio.Event() for _ in range(4)]

        async def generate(group, params):
            await gates[group[0].index].wait()
            return group

        state = make_state(generate)
        state.submit_generate_tasks([[Sample(index=i)] for i in range(3)])
        gates[1].set()
        done = await state.next_completed()
        assert [task.result()[0].index for task in done] == [1]
        assert len(state.pendings) == 2
        state.submit_generate_tasks([[Sample(index=3)]])
        for gate in gates:
            gate.set()
        indices = [1]
        while state.pendings:
            indices.extend(task.result()[0].index for task in await state.next_completed())
        assert sorted(indices) == [0, 1, 2, 3]
        assert state.completed.empty()
        state.reset()
        state.submit_generate_tasks([[Sample(index=0)]])
        assert [task.result()[0].index for task in await state.next_completed()] == [0]

    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_completion_queue_surfaces_task_failure_and_cancellation():
    async def run():
        gate = asyncio.Event()

        async def generate(group, params):
            await gate.wait()
            raise ValueError("scoring failed")

        state = make_state(generate)
        state.submit_generate_tasks([[Sample(index=0)], [Sample(index=1)]])
        next(iter(state.pendings)).cancel()
        gate.set()
        outcomes = []
        while state.pendings:
            for task in await state.next_completed():
                try:
                    task.result()
                except (asyncio.CancelledError, ValueError) as error:
                    outcomes.append(type(error))
        assert set(outcomes) == {asyncio.CancelledError, ValueError}
        with pytest.raises(ValueError, match="No pending"):
            await state.next_completed()

    asyncio.run(asyncio.wait_for(run(), timeout=5))
