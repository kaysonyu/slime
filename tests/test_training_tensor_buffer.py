import os

import pytest
import ray
import torch
from test_dp_schedule import make_args, make_tp

from slime.data.batch_builder import BatchBuilder
from slime.utils.data import process_rollout_data

NUM_GPUS = 0


@ray.remote
def consume_partition(boxes, rank, buffered):
    original_load = torch.load

    def disallow_storage_load(*args, **kwargs):
        raise AssertionError("Buffered training tensors must avoid per-storage torch.load")

    try:
        if buffered:
            torch.load = disallow_storage_load
        data = process_rollout_data(boxes, rank, 2)
        assert data["num_microbatches"] == [1]
        assert data["raw_reward"] == list(range(8))
        assert data["local_raw_reward"] == data["partition"]
        for index, tokens, inputs in zip(
            data["partition"], data["tokens"], data["multimodal_train_inputs"], strict=True
        ):
            assert tokens.tolist() == [index, index + 1]
            assert inputs["audio"].dtype == torch.bfloat16
            assert torch.equal(inputs["audio"], torch.full((2, 24), index, dtype=torch.bfloat16))
            assert ray.get(inputs["reference"])["value"] == 7
        data["multimodal_train_inputs"][0]["audio"].fill_(-1)
        again = process_rollout_data(boxes, rank, 2)
        assert again["multimodal_train_inputs"][0]["audio"][0, 0] == data["partition"][0]
        return os.getpid(), data["partition"]
    finally:
        torch.load = original_load


@pytest.mark.parametrize("buffered", [False, True])
def test_real_ray_training_partitions_preserve_schedule_values_and_nested_refs(buffered):
    ray.init(address="local", num_cpus=2, object_store_memory=128 * 1024**2, include_dashboard=False)
    try:
        args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=64)
        args.global_batch_size = 8
        args.rollout_data_transport = "object-store"
        args.rollout_tensor_buffer = buffered
        args.custom_reward_post_process_path = None
        args.custom_convert_samples_to_train_data_path = None
        builder = BatchBuilder(args)
        builder.train_parallel_config = make_tp(dp_size=2)
        reference = ray.put({"value": 7})
        data = {
            "tokens": [[index, index + 1] for index in range(8)],
            "rollout_ids": list(range(8)),
            "raw_reward": list(range(8)),
            "multimodal_train_inputs": [
                {"audio": torch.full((2, 24), index, dtype=torch.bfloat16), "reference": reference}
                for index in range(8)
            ],
        }
        boxes = builder.split_by_dp(data)
        results = ray.get([consume_partition.remote(boxes, rank, buffered) for rank in range(2)])
        assert all(pid != os.getpid() for pid, _ in results)
        assert sorted(index for _, partition in results for index in partition) == list(range(8))
        assert data["multimodal_train_inputs"][0]["audio"][0, 0] == 0
    finally:
        ray.shutdown()
