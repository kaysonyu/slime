"""Transport trusted CPU tensor payloads without per-storage torch.load calls.

Only ordinary contiguous tensors use the buffer reducer. Other values retain
their normal cloudpickle representation; each receiver owns writable memory.
"""

import io
import pickle as stdlib_pickle
import struct
from collections import ChainMap
from typing import Any

import numpy as np
import ray.cloudpickle as pickle
import torch


def _restore_tensor(data, dtype, shape):
    if data.size == 0:
        return torch.empty(shape, dtype=dtype, device="cpu")
    return torch.from_numpy(data.copy()).view(dtype).reshape(shape)


def _buffer_eligible(tensor):
    return not (
        tensor.device.type != "cpu"
        or tensor.layout != torch.strided
        or tensor.is_quantized
        or tensor.requires_grad
        or not tensor.is_contiguous()
        or tensor.is_conj()
        or tensor.is_neg()
        or tensor.__dict__
        or any(name is not None for name in getattr(tensor, "names", ()))
    )


def _reduce_tensor(tensor):
    if not _buffer_eligible(tensor):
        return tensor.__reduce_ex__(5)
    data = tensor.reshape(-1).view(torch.uint8).numpy()
    return _restore_tensor, (data, tensor.dtype, tensor.shape)


class _TensorPickler(pickle.CloudPickler):
    # Keep Ray's live dispatch table: ObjectRef reducers registered after this
    # module is imported must retain the usual ownership bookkeeping.
    dispatch_table = ChainMap({torch.Tensor: _reduce_tensor}, pickle.CloudPickler.dispatch_table)


_PACKED_MAGIC = b"SLIMETB2"
_PACKED_HEADER = struct.Struct("<8sQ")


class _PackedTensorPickler(pickle.CloudPickler):
    """Put tensor bytes in one arena; pickle contains only tensor identities."""

    def __init__(self, stream):
        super().__init__(stream, protocol=5)
        self.tensors = {}
        self.tensor_owners = []
        self.buffers = []
        self.nbytes = 0

    def persistent_id(self, value):
        if type(value) is not torch.Tensor or not _buffer_eligible(value):
            return None
        identity = id(value)
        if identity not in self.tensors:
            data = value.reshape(-1).view(torch.uint8).numpy()
            self.tensor_owners.append(value)
            self.tensors[identity] = (len(self.tensors), value.dtype, tuple(value.shape), self.nbytes, data.nbytes)
            self.buffers.append(data)
            self.nbytes += data.nbytes
        return self.tensors[identity]


class _PackedTensorUnpickler(stdlib_pickle.Unpickler):
    def __init__(self, stream, arena):
        super().__init__(stream)
        self.arena = arena
        self.tensors = {}

    def persistent_load(self, descriptor):
        index, dtype, shape, offset, nbytes = descriptor
        if index not in self.tensors:
            if offset < 0 or nbytes < 0 or offset + nbytes > len(self.arena):
                raise ValueError("Packed tensor slice is outside its payload")
            if nbytes:
                self.tensors[index] = torch.frombuffer(
                    self.arena, dtype=dtype, count=nbytes // dtype.itemsize, offset=offset
                ).reshape(shape)
            else:
                self.tensors[index] = torch.empty(shape, dtype=dtype, device="cpu")
        return self.tensors[index]


def serialize_tensor_payload(value: Any, *, packed=False) -> np.ndarray:
    stream = io.BytesIO()
    if packed:
        pickler = _PackedTensorPickler(stream)
        pickler.dump(value)
        metadata = stream.getvalue()
        header = _PACKED_HEADER.pack(_PACKED_MAGIC, len(metadata))
        result = np.empty(len(header) + len(metadata) + pickler.nbytes, dtype=np.uint8)
        result[: len(header)] = np.frombuffer(header, dtype=np.uint8)
        position = len(header)
        result[position : position + len(metadata)] = np.frombuffer(metadata, dtype=np.uint8)
        position += len(metadata)
        for data in pickler.buffers:
            result[position : position + data.nbytes] = data
            position += data.nbytes
        return result
    _TensorPickler(stream, protocol=5).dump(value)
    return np.frombuffer(stream.getvalue(), dtype=np.uint8)


def deserialize_tensor_payload(payload: np.ndarray) -> Any:
    if memoryview(payload)[:8] == _PACKED_MAGIC:
        if payload.nbytes < _PACKED_HEADER.size:
            raise ValueError("Truncated packed tensor header")
        _, size = _PACKED_HEADER.unpack_from(payload)
        end = _PACKED_HEADER.size + size
        if end > payload.nbytes:
            raise ValueError("Truncated packed tensor metadata")
        # Exactly one writable copy for this receiver, not a NumPy reconstruction
        # plus another copy for every tensor. Views occupy disjoint arena slices.
        arena = bytearray(memoryview(payload)[end:])
        return _PackedTensorUnpickler(io.BytesIO(memoryview(payload)[_PACKED_HEADER.size : end]), arena).load()
    return pickle.loads(memoryview(payload))


class SerializedTensorPayload:
    def __init__(self, value: Any, *, packed=False):
        self.value = value
        self.packed = packed

    def __reduce__(self):
        # Ray invokes this inside its serialization context; do not pre-encode
        # payloads outside that context when they can contain ObjectRefs.
        return deserialize_tensor_payload, (serialize_tensor_payload(self.value, packed=self.packed),)
