"""Transport trusted CPU tensor payloads without per-storage torch.load calls.

Only ordinary contiguous tensors use the buffer reducer. Other values retain
their normal cloudpickle representation; each receiver owns writable memory.
"""

import io
from collections import ChainMap
from typing import Any

import numpy as np
import ray.cloudpickle as pickle
import torch


def _restore_tensor(data, dtype, shape):
    if data.size == 0:
        return torch.empty(shape, dtype=dtype)
    return torch.from_numpy(data.copy()).view(dtype).reshape(shape)


def _reduce_tensor(tensor):
    if (
        tensor.device.type != "cpu"
        or tensor.layout != torch.strided
        or tensor.is_quantized
        or tensor.requires_grad
        or not tensor.is_contiguous()
        or tensor.is_conj()
        or tensor.is_neg()
        or tensor.__dict__
        or any(name is not None for name in getattr(tensor, "names", ()))
    ):
        return tensor.__reduce_ex__(5)
    data = tensor.reshape(-1).view(torch.uint8).numpy()
    return _restore_tensor, (data, tensor.dtype, tensor.shape)


class _TensorPickler(pickle.CloudPickler):
    # Keep Ray's live dispatch table: ObjectRef reducers registered after this
    # module is imported must retain the usual ownership bookkeeping.
    dispatch_table = ChainMap({torch.Tensor: _reduce_tensor}, pickle.CloudPickler.dispatch_table)


def serialize_tensor_payload(value: Any) -> np.ndarray:
    stream = io.BytesIO()
    _TensorPickler(stream, protocol=5).dump(value)
    return np.frombuffer(stream.getvalue(), dtype=np.uint8)


def deserialize_tensor_payload(payload: np.ndarray) -> Any:
    return pickle.loads(memoryview(payload))


class SerializedTensorPayload:
    def __init__(self, value: Any):
        self.value = value

    def __reduce__(self):
        # Ray invokes this inside its serialization context; do not pre-encode
        # payloads outside that context when they can contain ObjectRefs.
        return deserialize_tensor_payload, (serialize_tensor_payload(self.value),)
