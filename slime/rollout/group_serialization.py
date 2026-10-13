"""Serialize worker groups with isolated, writable CPU tensor buffers."""

from slime.utils.tensor_serialization import deserialize_tensor_payload, serialize_tensor_payload


class SerializedGroup:
    def __init__(self, group):
        self.group = group

    def __reduce__(self):
        # Run inside Ray's serialization context so nested ObjectRefs retain
        # the same ownership bookkeeping as an ordinary worker result.
        return deserialize_group, (serialize_group(self.group),)


def serialize_group(group):
    return serialize_tensor_payload(group)


def deserialize_group(payload):
    return deserialize_tensor_payload(payload)
