"""Serialize worker results without a torch.load call for every CPU tensor.

The payload stays inside the existing trusted Ray cluster. Only ordinary,
contiguous CPU tensors use the buffer representation; other values retain
their normal cloudpickle reducers. Each receiver owns writable tensor memory.
"""

import io
from collections import ChainMap

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


class SerializedGroup:
    def __init__(self, group):
        self.group = group

    def __reduce__(self):
        # Run inside Ray's serialization context so nested ObjectRefs retain
        # the same ownership bookkeeping as an ordinary worker result.
        return deserialize_group, (serialize_group(self.group),)


class _GroupPickler(pickle.CloudPickler):
    # The C pickler reads the class table during construction. Keep the base
    # table live so Ray serializers registered later remain visible as well.
    dispatch_table = ChainMap({torch.Tensor: _reduce_tensor}, pickle.CloudPickler.dispatch_table)


def serialize_group(group):
    stream = io.BytesIO()
    _GroupPickler(stream, protocol=5).dump(group)
    # Ray transports this array as a buffer instead of walking the nested
    # sample graph again. No process-wide tensor serializer is registered.
    return np.frombuffer(stream.getvalue(), dtype=np.uint8)


def deserialize_group(payload):
    return pickle.loads(memoryview(payload))
