"""Versioned weight-update handles, refreshed at the existing recovery barrier.

Sending all E engine actor handles to each of R trainer ranks costs O(E * R)
handle serialization per publication. Stable topology needs only a revision and
the live reconnect count. This does not cache health checks or weight versions.
"""

from copy import deepcopy
from uuid import uuid4


def _actor_id(handle):
    return None if handle is None else handle._actor_id


class WeightTopologyPublisher:
    def __init__(self):
        self._key = None
        self._revision = None
        self._payload = None
        self._num_new = 0
        self._ready = False

    def invalidate(self):
        # Keep the previous identity for comparison, but never serve a snapshot
        # between rollout/reattachment and a successful recovery boundary.
        self._ready = False

    def refresh(self, snapshot):
        engines, lock, num_new, counts, offsets, configs = snapshot
        key = (
            tuple(_actor_id(engine) for engine in engines),
            _actor_id(lock),
            tuple(counts),
            tuple(offsets),
            deepcopy(configs),
        )
        if self._revision is None or key != self._key:
            self._revision = uuid4().hex
            self._key = key
            self._payload = (list(engines), lock, list(counts), list(offsets), deepcopy(configs))
        self._num_new = num_new
        self._ready = True

    def clear_new(self):
        self._num_new = 0

    def get(self, known_revision=None):
        if not self._ready:
            raise RuntimeError("Weight topology must be refreshed after engine recovery")
        return self._revision, self._num_new, None if known_revision == self._revision else self._payload


class WeightTopologyClient:
    def __init__(self):
        self.revision = None
        self._payload = None

    def resolve(self, response):
        revision, num_new, payload = response
        if payload is not None:
            self.revision, self._payload = revision, payload
        elif revision != self.revision or self._payload is None:
            raise RuntimeError("Missing handles for a new weight topology revision")
        engines, lock, counts, offsets, configs = self._payload
        return engines, lock, num_new, counts, offsets, configs
