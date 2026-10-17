"""Process workers for finite, synchronous rollout batches.

The manager owns input selection, filtering and the training boundary. Workers
execute complete prompt groups, including sample hooks and reward scoring.
"""

import asyncio
import copy
import math
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from slime.ray.utils import add_default_ray_env_vars
from slime.rollout.base_types import iter_samples
from slime.rollout.gc_utils import configure_rollout_gc
from slime.rollout.group_serialization import SerializedGroup
from slime.rollout.sample_hooks import close_sample_hooks, flush_sample_observers, rollout_context
from slime.utils.http_utils import get_rollout_num_engines, init_http_client
from slime.utils.types import Sample


class GroupWorker:
    def __init__(self, args, concurrency):
        from slime.rollout.sglang_rollout import GenerateState

        configure_rollout_gc(getattr(args, "rollout_gc_threshold", None))
        self.args = copy.copy(args)
        self.args.rollout_execution_workers = 0
        self.args.use_distributed_post = False
        init_http_client(self.args, concurrency=concurrency)
        self.state = GenerateState(self.args, concurrency=concurrency)
        self.rollout_id = None
        self.running = False
        self.groups = 0
        self.started = time.perf_counter()
        self.cpu_started = time.process_time()

    async def start(self, rollout_id):
        if self.running or self.state.pendings:
            raise RuntimeError("Drain the previous rollout before starting another weight version")
        self.state.reset()
        self.rollout_id = rollout_id
        self.running = True
        self.groups = 0
        self.started = time.perf_counter()
        self.cpu_started = time.process_time()

    async def generate(self, rollout_id, group, sampling_params):
        from slime.rollout.sglang_rollout import generate_and_rm_group

        if rollout_id != self.rollout_id:
            raise RuntimeError("A stale rollout request reached an execution worker")
        if not self.running:
            return aborted_group(group)
        with rollout_context(rollout_id):
            task = asyncio.create_task(generate_and_rm_group(self.args, group, sampling_params, evaluation=False))
        self.state.pendings.add(task)
        try:
            result = await task
            self.groups += 1
            return result
        finally:
            self.state.pendings.discard(task)

    async def stop(self):
        from slime.rollout.sglang_rollout import abort

        if self.running:
            self.running = False
            await abort(self.args, self.rollout_id)
        if self.state.teacher_runtime is not None:
            await self.state.teacher_runtime.flush()
        await flush_sample_observers(self.args)
        return {
            "groups": self.groups,
            "wall_seconds": time.perf_counter() - self.started,
            "cpu_seconds": time.process_time() - self.cpu_started,
        }

    async def generate_serialized(self, rollout_id, group, sampling_params):
        return SerializedGroup(await self.generate(rollout_id, group, sampling_params))

    async def close(self):
        if self.running:
            await self.stop()
        if self.state.teacher_runtime is not None:
            await self.state.teacher_runtime.close()
            self.state.teacher_runtime = None
        await close_sample_hooks(self.args)
        from slime.utils import http_utils

        if http_utils._http_client is not None:
            await http_utils._http_client.aclose()


def aborted_group(group):
    for sample in iter_samples(group):
        if sample.status == Sample.Status.PENDING:
            sample.status = Sample.Status.ABORTED
    return group


class RolloutExecutionPool:
    def __init__(self, args):
        self.workers = []
        count = args.rollout_execution_workers
        total = args.sglang_server_concurrency * get_rollout_num_engines(args)
        if not 0 < count <= total:
            raise ValueError("Rollout execution workers must be positive and no greater than generation concurrency")
        nodes = [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) >= 1]
        gpu_nodes = [node for node in nodes if node["Resources"].get("GPU", 0) > 0]
        nodes = sorted(gpu_nodes or nodes, key=lambda node: (node["NodeManagerAddress"], node["NodeID"]))
        if not nodes:
            raise RuntimeError("No CPU resources are available for rollout execution")
        size, extra = divmod(total, count)
        self.available = asyncio.Queue()
        self.capacities = []
        self.pending = set()
        self.running = False
        self.stopping = None
        self.metrics = {}
        try:
            for index in range(count):
                concurrency = size + (index < extra)
                # Bound outstanding complete groups, while allowing scoring to
                # overlap the next generation wave. The worker's sample-level
                # semaphore independently enforces the generation budget.
                capacity = 2 * max(1, math.ceil(concurrency / args.n_samples_per_prompt))
                self.capacities.append(capacity)
                node = nodes[index % len(nodes)]
                self.workers.append(
                    ray.remote(GroupWorker)
                    .options(
                        num_cpus=1,
                        max_concurrency=capacity + 2,
                        max_restarts=0,
                        max_task_retries=0,
                        scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False),
                        runtime_env={
                            "env_vars": add_default_ray_env_vars(
                                {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
                            )
                        },
                    )
                    .remote(args, concurrency)
                )
            for slot in range(max(self.capacities)):
                for index, capacity in enumerate(self.capacities):
                    if slot < capacity:
                        self.available.put_nowait(index)
        except BaseException:
            for worker in self.workers:
                ray.kill(worker, no_restart=True)
            raise

    async def start(self, rollout_id):
        if self.running or self.pending:
            raise RuntimeError("Rollout execution has not drained")
        await asyncio.gather(*(worker.start.remote(rollout_id) for worker in self.workers))
        self.rollout_id = rollout_id
        self.running = True
        self.stopping = None

    async def generate(self, group, sampling_params):
        index = await self.available.get()
        try:
            if not self.running:
                return aborted_group(group)
            ref = self.workers[index].generate_serialized.remote(self.rollout_id, group, sampling_params)
            self.pending.add(ref)
            try:
                return await ref
            finally:
                self.pending.discard(ref)
        finally:
            self.available.put_nowait(index)

    async def stop(self, *, suppress_errors=False):
        if self.stopping is None:
            self.running = False
            self.stopping = asyncio.create_task(self._stop())
        results = await asyncio.shield(self.stopping)
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures and not suppress_errors:
            raise failures[0]

    async def _stop(self):
        results = await asyncio.gather(*(worker.stop.remote() for worker in self.workers), return_exceptions=True)
        completed = [row for row in results if isinstance(row, dict)]
        self.metrics = {
            "rollout/execution_workers": len(self.workers),
            "rollout/execution_worker_cpu_seconds": sum(row["cpu_seconds"] for row in completed),
        }
        return results

    def close(self):
        try:
            ray.get([worker.close.remote() for worker in self.workers], timeout=30)
        finally:
            for worker in self.workers:
                ray.kill(worker, no_restart=True)
            self.workers.clear()
