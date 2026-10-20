"""CPU checks for an opt-in topology protocol; not GPU performance acceptance."""

import ast
import copy
import logging
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import ray

from slime.ray.weight_topology import WeightTopologyClient, WeightTopologyPublisher

NUM_GPUS = 0


def actor(identity):
    return SimpleNamespace(_actor_id=identity)


def snapshot():
    return ([actor("a"), actor("b")], actor("lock"), 2, [1, 1], [0, 1], [{"tp_size": 1}] * 2)


def test_unchanged_handles_are_omitted_but_reconnect_count_is_fresh():
    publisher, client = WeightTopologyPublisher(), WeightTopologyClient()
    value = snapshot()
    publisher.refresh(value)
    initial = client.resolve(publisher.get())
    assert initial == value
    revision = client.revision
    publisher.clear_new()
    response = publisher.get(revision)
    assert response == (revision, 0, None)
    assert client.resolve(response)[2] == 0
    publisher.invalidate()
    with pytest.raises(RuntimeError, match="refreshed"):
        publisher.get(revision)
    # Re-created Python handle wrappers with the same Ray IDs aren't a change.
    publisher.refresh(snapshot())
    assert publisher.get(revision) == (revision, 2, None)
    assert client.resolve(publisher.get(revision))[2] == 2


@pytest.mark.parametrize("change", ["engine", "order", "empty", "lock", "counts", "offsets", "parallel"])
def test_every_connection_identity_change_requires_new_payload(change):
    publisher, client = WeightTopologyPublisher(), WeightTopologyClient()
    value = snapshot()
    publisher.refresh(value)
    client.resolve(publisher.get())
    old_revision = client.revision
    engines, lock, num_new, counts, offsets, configs = copy.deepcopy(value)
    if change == "engine":
        engines[0] = actor("replacement")
    elif change == "order":
        engines.reverse()
    elif change == "empty":
        engines, counts, offsets, configs = [], [], [], []
    elif change == "lock":
        lock = actor("new-lock")
    elif change == "counts":
        counts[0] = 2
    elif change == "offsets":
        offsets.reverse()
    elif change == "parallel":
        configs[0]["tp_size"] = 2
    changed = (engines, lock, num_new, counts, offsets, configs)
    publisher.refresh(changed)
    response = publisher.get(old_revision)
    assert response[0] != old_revision and response[2] is not None
    assert client.resolve(response) == changed


def test_replacement_trainer_and_manager_cannot_reuse_an_unknown_revision():
    old, replacement = WeightTopologyPublisher(), WeightTopologyPublisher()
    old.refresh(snapshot())
    replacement.refresh(snapshot())
    revision = old.get()[0]
    assert replacement.get(revision)[2] is not None
    assert WeightTopologyClient().resolve(replacement.get())[2] == 2
    with pytest.raises(RuntimeError, match="Missing handles"):
        WeightTopologyClient().resolve((revision, 0, None))


def manager(monkeypatch, *, enabled=True):
    from slime.ray.rollout import RolloutManager

    owner = object.__new__(RolloutManager.__ray_metadata__.modified_class)
    owner.args = SimpleNamespace(cache_weight_topology=enabled)
    owner._weight_topology = WeightTopologyPublisher()
    events = []
    current = {"value": snapshot(), "failure": None}

    def recover():
        events.append("recover")
        if current["failure"]:
            raise RuntimeError(current["failure"])
        return {}

    def fetch():
        events.append("fetch")
        return current["value"]

    def clear():
        events.append("clear")
        return "cleared"

    owner.serving = SimpleNamespace(
        recover_updatable_engines=SimpleNamespace(remote=recover),
        get_updatable_engines_and_lock=SimpleNamespace(remote=fetch),
        clear_updatable_num_new_engines=SimpleNamespace(remote=clear),
        health_monitoring_resume=SimpleNamespace(remote=lambda: {}),
    )
    monkeypatch.setattr(ray, "get", lambda x: x)
    return owner, events, current


def test_production_manager_fetches_once_per_boundary_and_invalidates_on_failure(monkeypatch):
    owner, events, current = manager(monkeypatch)
    owner.recover_updatable_engines()
    revision = owner.get_cached_updatable_engines_and_lock()[0]
    for _ in range(128):
        assert owner.get_cached_updatable_engines_and_lock(revision) == (revision, 2, None)
    assert events == ["recover", "fetch"]
    assert owner.clear_updatable_num_new_engines() == "cleared"
    assert owner.get_cached_updatable_engines_and_lock(revision) == (revision, 0, None)
    owner.health_monitoring_resume()
    with pytest.raises(RuntimeError, match="refreshed"):
        owner.get_cached_updatable_engines_and_lock(revision)
    current["failure"] = "engine recovery failed"
    with pytest.raises(RuntimeError, match="engine recovery failed"):
        owner.recover_updatable_engines()
    with pytest.raises(RuntimeError, match="refreshed"):
        owner.get_cached_updatable_engines_and_lock(revision)
    current["failure"] = None
    current["value"][0][0] = actor("replacement")
    owner.recover_updatable_engines()
    assert owner.get_cached_updatable_engines_and_lock(revision)[2] is not None


def test_flag_off_preserves_original_recovery_without_an_extra_owner_fetch(monkeypatch):
    owner, events, _ = manager(monkeypatch, enabled=False)
    owner.recover_updatable_engines()
    assert events == ["recover"]


@pytest.mark.parametrize(
    "enabled,external,tolerant",
    [(True, False, False), (False, False, False), (True, True, False), (True, True, True)],
)
def test_actual_megatron_update_method_keeps_recovery_and_connection_order(enabled, external, tolerant):
    # Execute the actual method without importing CUDA/Megatron in a CPU test.
    path = Path(__file__).parents[1] / "slime/backends/megatron_utils/actor.py"
    tree = ast.parse(path.read_text())
    method = next(
        method
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "MegatronTrainRayActor"
        for method in cls.body
        if isinstance(method, ast.FunctionDef) and method.name == "update_weights"
    )
    method.decorator_list = []
    events, revisions = [], []
    publisher = WeightTopologyPublisher()
    current = list(snapshot())

    def recover():
        events.append("recover")
        publisher.refresh(current)

    def cached(revision):
        events.append("cached")
        revisions.append(revision)
        return publisher.get(revision)

    def clear():
        events.append("clear")
        current[2] = 0
        publisher.clear_new()

    env = {
        "ray": SimpleNamespace(get=lambda x: x),
        "dist": SimpleNamespace(get_rank=lambda: 0, barrier=lambda **kw: events.append("barrier")),
        "get_gloo_group": lambda: "gloo",
        "nullcontext": nullcontext,
        "print_memory": lambda text: None,
        "logger": logging.getLogger(__name__),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), env)
    instance = SimpleNamespace(
        args=SimpleNamespace(
            debug_train_only=False,
            debug_rollout_only=False,
            rollout_external=external,
            use_fault_tolerance=tolerant,
            cache_weight_topology=enabled,
            offload_train=False,
            use_critic=False,
            colocate=True,
        ),
        rollout_manager=SimpleNamespace(
            recover_updatable_engines=SimpleNamespace(remote=recover),
            get_cached_updatable_engines_and_lock=SimpleNamespace(remote=cached),
            get_updatable_engines_and_lock=SimpleNamespace(remote=lambda: (events.append("legacy"), current)[1]),
            clear_updatable_num_new_engines=SimpleNamespace(remote=clear),
        ),
        weight_updater=SimpleNamespace(
            connect_rollout_engines=lambda *a, **kw: events.append("connect"),
            update_weights=lambda: events.append("publish"),
            weight_version=1,
        ),
    )
    for _ in range(2):
        env["update_weights"](instance)
    boundary = ["recover", "barrier"] if not external or tolerant else []
    fetch = ["cached" if enabled and (not external or tolerant) else "legacy"]
    assert events == boundary + fetch + ["connect", "barrier", "clear", "publish"] + boundary + fetch + ["publish"]
    if revisions:
        assert revisions[0] is None and revisions[1] is not None


@ray.remote(num_cpus=0)
class _Engine:
    def ping(self):
        return "alive"


@ray.remote(num_cpus=0)
class _Owner:
    def __init__(self, engine, lock):
        from slime.ray.rollout import RolloutManager

        self.manager = object.__new__(RolloutManager.__ray_metadata__.modified_class)
        self.manager.args = SimpleNamespace(cache_weight_topology=True)
        self.manager._weight_topology = WeightTopologyPublisher()
        self.manager.serving = None
        self.manager.rollout_id = -1
        self.manager.rollout_engine_lock = lock
        self.manager.servers = {
            "default": SimpleNamespace(
                update_weights=True,
                engines=[engine],
                num_new_engines=1,
                engine_gpu_counts=[1],
                engine_gpu_offsets=[0],
                engine_parallel_configs=[{"tp_size": 1}],
            )
        }

    def recover(self):
        self.manager.recover_updatable_engines()

    def fetch(self, revision=None):
        return self.manager.get_cached_updatable_engines_and_lock(revision)

    def replace(self, engine):
        self.manager.servers["default"].engines = [engine]
        self.manager.servers["default"].num_new_engines = 1
        self.manager.recover_updatable_engines()

    def clear(self):
        self.manager.clear_updatable_num_new_engines()


def test_real_ray_handles_survive_compact_updates_and_engine_replacement():
    ray.init(address="local", num_cpus=2, num_gpus=0, object_store_memory=128 * 1024**2, include_dashboard=False)
    try:
        engine, lock = _Engine.remote(), _Engine.remote()
        owner = _Owner.remote(engine, lock)
        ray.get(owner.recover.remote())
        clients = [WeightTopologyClient(), WeightTopologyClient()]
        for client in clients:
            handles = client.resolve(ray.get(owner.fetch.remote(client.revision)))
            assert handles[0][0]._actor_id == engine._actor_id
            assert ray.get(handles[0][0].ping.remote()) == "alive"
        ray.get(owner.clear.remote())
        ray.get(owner.recover.remote())
        for client in clients:
            response = ray.get(owner.fetch.remote(client.revision))
            assert response[1:] == (0, None)
            assert client.resolve(response)[2] == 0
        replacement = _Engine.remote()
        ray.get(owner.replace.remote(replacement))
        for client in clients:
            response = ray.get(owner.fetch.remote(client.revision))
            assert response[2] is not None
            handles = client.resolve(response)
            assert handles[0][0]._actor_id == replacement._actor_id and handles[2] == 1
            assert ray.get(handles[0][0].ping.remote()) == "alive"
    finally:
        ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
