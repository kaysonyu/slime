"""Serialize worker groups with isolated, writable CPU tensor buffers."""

from slime.utils import tensor_serialization
from slime.utils.tensor_serialization import deserialize_tensor_payload, serialize_tensor_payload

# Existing in-flight worker buffers refer to this reducer's original module.
_restore_tensor = tensor_serialization._restore_tensor


class SerializedGroup:
    def __init__(self, group, *, packed=False):
        self.group = group
        self.packed = packed

    def __reduce__(self):
        # Run inside Ray's serialization context so nested ObjectRefs retain
        # the same ownership bookkeeping as an ordinary worker result.
        return deserialize_group, (serialize_group(self.group, packed=self.packed),)


def serialize_group(group, *, packed=False):
    return serialize_tensor_payload(group, packed=packed)


def deserialize_group(payload):
    return deserialize_tensor_payload(payload)
