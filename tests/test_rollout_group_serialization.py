import numpy as np
import pytest
import ray.cloudpickle as pickle
import torch

from slime.rollout.group_serialization import SerializedGroup, deserialize_group, serialize_group
from slime.utils.types import Sample

NUM_GPUS = 0


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.float32, torch.float16, torch.bfloat16, torch.bool])
def test_sample_tensors_round_trip_with_writable_isolated_storage(dtype, monkeypatch):
    def disallow_per_tensor_load(*args, **kwargs):
        pytest.fail("CPU worker tensors must not use a per-storage torch.load")

    tensor = torch.arange(12).reshape(3, 4).to(dtype)
    sample = Sample(index=7, weight_versions=["2"], metadata={"nested": {"tensor": tensor}})
    sample.multimodal_train_inputs = {"input": tensor, "empty": torch.empty((0, 4), dtype=dtype)}
    sample.reward = {"asr": 0.75}
    payload = serialize_group([sample])
    payload.setflags(write=False)  # Ray's object-store arrays are read-only.
    monkeypatch.setattr(torch, "load", disallow_per_tensor_load)
    left = deserialize_group(payload)[0]
    right = deserialize_group(payload)[0]
    restored = left.multimodal_train_inputs["input"]
    assert restored is left.metadata["nested"]["tensor"]
    assert restored.dtype == dtype and torch.equal(restored, tensor)
    assert left.multimodal_train_inputs["empty"].shape == (0, 4)
    assert left.index == 7 and left.weight_versions == ["2"] and left.reward == {"asr": 0.75}
    restored.fill_(0)
    assert torch.equal(right.multimodal_train_inputs["input"], tensor)
    assert torch.count_nonzero(tensor) > 0


def test_tensor_fallback_preserves_strides_gradients_and_custom_attributes():
    transposed = torch.arange(12).reshape(3, 4).T
    gradient = torch.ones(3, requires_grad=True)
    custom = torch.tensor(5)
    custom.description = "retain tensor attributes"
    sparse = torch.sparse_coo_tensor([[0, 1]], [2.0, 3.0], (3,))
    result = deserialize_group(serialize_group([transposed, gradient, custom, sparse]))
    assert result[0].stride() == transposed.stride() and torch.equal(result[0], transposed)
    assert result[1].requires_grad and torch.equal(result[1], gradient)
    assert result[2].description == custom.description and torch.equal(result[2], custom)
    assert torch.equal(result[3].to_dense(), sparse.to_dense())


def test_nested_metadata_aliases_and_scalar_tensors_survive_without_global_reducer_changes():
    shared = {"array": np.arange(5), "scalar": torch.tensor(3.5)}
    shared["self"] = shared
    before = pickle.CloudPickler.dispatch_table.get(torch.Tensor)
    output = deserialize_group(serialize_group([shared, shared]))
    assert output[0] is output[1] and output[0]["self"] is output[0]
    assert output[0]["scalar"].shape == () and output[0]["scalar"].item() == 3.5
    np.testing.assert_array_equal(output[0]["array"], shared["array"])
    assert pickle.CloudPickler.dispatch_table.get(torch.Tensor) is before


def test_worker_wrapper_deserializes_to_the_original_result_type():
    group = [Sample(index=3, multimodal_train_inputs={"x": torch.arange(4)})]
    result = pickle.loads(pickle.dumps(SerializedGroup(group), protocol=5))
    assert isinstance(result, list) and result[0].index == 3
    assert torch.equal(result[0].multimodal_train_inputs["x"], group[0].multimodal_train_inputs["x"])
