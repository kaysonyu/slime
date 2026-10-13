"""Exercise real train/save orchestration with CPU stand-ins for GPU operations."""

import ast
import runpy
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from slime.ray.actor_group import RayTrainGroup

NUM_GPUS = 0
pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def actor_class(monkeypatch):
    source = ROOT / "slime/backends/megatron_utils/actor.py"
    actor = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef))
    actor.bases = []
    actor.body = [
        node
        for node in actor.body
        if isinstance(node, ast.FunctionDef) and node.name in {"train", "save_model", "sleep", "wake_up"}
    ]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), actor],
        type_ignores=[],
    )
    events = []
    namespace = {
        "timer": lambda value: value if callable(value) else nullcontext(),
        "clear_memory": lambda **kwargs: None,
        "print_memory": lambda message: None,
        "destroy_process_groups": lambda: events.append("destroy_groups"),
        "reload_process_groups": lambda: events.append("reload_groups"),
        "reset_cuda_stack_size": lambda: None,
        "torch_memory_saver": SimpleNamespace(
            pause=lambda: events.append("pause"), resume=lambda: events.append("resume")
        ),
        "mpu": SimpleNamespace(get_pipeline_model_parallel_world_size=lambda: 1),
        "Path": Path,
    }
    async_utils = ModuleType("megatron.training.async_utils")
    async_utils.maybe_finalize_async_save = lambda **kwargs: events.append(("finalize", kwargs))
    monkeypatch.setitem(sys.modules, async_utils.__name__, async_utils)
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)

    def save(rollout_id, model, optimizer, scheduler):
        assert not model._is_offloaded
        events.append(("save", model.role, rollout_id))
        if model.fail_save:
            raise OSError("checkpoint write failed")

    namespace["save"] = save
    return namespace["MegatronTrainRayActor"], events


def make_actor(actor_class, role="actor", *, offload=True, async_save=False):
    actor_type, events = actor_class
    actor = actor_type()
    actor.args = SimpleNamespace(
        debug_rollout_only=False,
        offload_train=offload,
        use_critic=False,
        colocate=True,
        async_save=async_save,
        save_hf=None,
    )
    actor.role = role
    actor._is_offloaded = offload
    actor.fail_save = False
    actor.model = actor
    actor.optimizer = actor.opt_param_scheduler = None
    actor._switch_model = lambda name: None
    actor._get_rollout_data = lambda value: value

    def train_actor(rollout_id, rollout_data, external_data=None):
        assert not actor._is_offloaded
        events.append(("train", role, rollout_id))

    def train_critic(rollout_id, rollout_data):
        train_actor(rollout_id, rollout_data)
        return {"values": [rollout_id]}

    actor.train_actor, actor.train_critic = train_actor, train_critic
    return actor


@pytest.mark.parametrize("role", ["actor", "critic"])
@pytest.mark.parametrize("async_save", [False, True])
def test_train_then_save_has_one_memory_roundtrip(actor_class, role, async_save):
    actor = make_actor(actor_class, role, async_save=async_save)
    _, events = actor_class
    result = actor.train(7, {}, offload_after_train=False)
    assert not actor._is_offloaded
    assert result == ({"values": [7]} if role == "critic" else None)
    actor.save_model(7, force_sync=True)
    operations = [event for event in events if event in ("pause", "resume") or isinstance(event, tuple)]
    expected = ["resume", ("train", role, 7)]
    if async_save:
        expected.append(("finalize", {"blocking": True}))
    expected.append(("save", role, 7))
    if async_save:
        expected.append(("finalize", {"blocking": True}))
    assert operations == expected + ["pause"]
    assert actor._is_offloaded


def test_training_without_pending_checkpoint_still_offloads(actor_class):
    actor = make_actor(actor_class)
    actor.train(1, {})
    assert actor._is_offloaded
    assert actor_class[1].count("resume") == actor_class[1].count("pause") == 1
    actor.save_model(1)
    assert actor_class[1].count("resume") == actor_class[1].count("pause") == 2


def test_repeated_memory_controls_do_not_repeat_gpu_operations(actor_class):
    actor = make_actor(actor_class)
    actor.wake_up()
    actor.wake_up()
    actor.sleep()
    actor.sleep()
    events = actor_class[1]
    assert events.count("pause") == events.count("resume") == 1
    assert events.count("destroy_groups") == events.count("reload_groups") == 1


def test_resident_save_failure_propagates_and_allows_explicit_cleanup(actor_class):
    actor = make_actor(actor_class)
    actor.train(2, {}, offload_after_train=False)
    actor.fail_save = True
    with pytest.raises(OSError, match="checkpoint write failed"):
        actor.save_model(2, force_sync=True)
    actor.sleep()
    assert actor._is_offloaded
    assert actor_class[1].count("pause") == 1


@pytest.mark.parametrize("per_rank", [False, True])
def test_pending_checkpoint_residency_is_forwarded_to_every_rank(monkeypatch, per_rank):
    received = []
    group = RayTrainGroup(SimpleNamespace(), 1, 2, pg=None)
    group._actor_handlers = [
        SimpleNamespace(train=SimpleNamespace(remote=lambda *args, **kwargs: received.append((args, kwargs))))
        for _ in range(2)
    ]
    external = [{"values": [0]}, {"values": [1]}] if per_rank else {"values": [2]}
    group.async_train(7, "batch", external_data=external, offload_after_train=False)
    assert len(received) == 2
    for index, (args, kwargs) in enumerate(received):
        assert args == (7, "batch")
        assert kwargs == {
            "external_data": external[index] if per_rank else external,
            "offload_after_train": False,
        }


@pytest.mark.parametrize("use_critic,critic_only", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("offload", [False, True])
def test_driver_keeps_only_last_trained_role_for_checkpoint(
    actor_class, monkeypatch, use_critic, critic_only, offload
):
    monkeypatch.setitem(sys.modules, "slime.utils.arguments", SimpleNamespace(parse_args=None))
    driver = runpy.run_path(str(ROOT / "train.py"))["train"]
    _, events = actor_class
    args = SimpleNamespace(
        release_train=False,
        offload_rollout=offload,
        offload_train=offload,
        check_weight_update_equal=False,
        eval_interval=None,
        start_rollout_id=0,
        num_rollout=2,
        num_critic_only_steps=2 if critic_only else 0,
        use_critic=use_critic,
        save_interval=2,
        rollout_data_transport="object-store",
        debug_train_only=False,
        debug_rollout_only=False,
        use_fault_tolerance=False,
        update_weight_mode="full",
        update_weight_transport="nccl",
        update_weight_start_version=0,
    )
    workers = {role: make_actor(actor_class, role, offload=offload) for role in ("actor", "critic")}

    def rpc(function):
        return SimpleNamespace(remote=function)

    groups = {}
    for role, worker in workers.items():
        group = RayTrainGroup(args, 1, 1, pg=None, role=role)
        group._actor_handlers = [
            SimpleNamespace(
                train=rpc(worker.train),
                save_model=rpc(worker.save_model),
                update_weights=rpc(lambda: events.append("publish")),
                clear_memory=rpc(lambda: None),
            )
        ]
        groups[role] = group
    manager = SimpleNamespace(
        **{
            name: rpc(function)
            for name, function in {
                "onload_weights": lambda: None,
                "onload_kv": lambda: None,
                "offload": lambda: None,
                "training_ready": lambda: None,
                "generate": lambda rollout: {},
                "training_completed": lambda rollout: None,
                "save": lambda rollout: events.append(("rollout_save", rollout)),
                "pause_rollout_admission": lambda: True,
                "resume_rollout_admission": lambda value: None,
            }.items()
        }
    )
    monkeypatch.setattr("ray.get", lambda value: value)
    monkeypatch.setitem(
        driver.__globals__, "create_training_models", lambda *args: (groups["actor"], groups["critic"])
    )
    driver(args, {}, manager, None, None)
    last_role = "critic" if critic_only else "actor"
    trained = events.index(("train", last_role, 1))
    saved = events.index(("save", last_role, 1))
    assert "pause" not in events[trained + 1 : saved]
    assert "resume" not in events[trained + 1 : saved]
    if offload:
        assert workers[last_role]._is_offloaded
        if use_critic and not critic_only:
            critic_done = events.index(("train", "critic", 1))
            assert "pause" in events[critic_done + 1 : trained]
    else:
        assert "pause" not in events and "resume" not in events
    assert events.index(("rollout_save", 1)) < len(events) - 1
    assert events[-1] == "publish"


def test_joint_checkpoint_rejects_incomplete_or_corrupted_dataset_restore(tmp_path, monkeypatch):
    import torch
    from slime.data.checkpoint import resolve_checkpoint, save_checkpoint
    from slime.data.data_source import RolloutDataSource

    args = SimpleNamespace(
        save=str(tmp_path),
        load=None,
        rollout_data_transport="object-store",
        checkpoint_joint_commit=True,
        debug_train_only=False,
        debug_rollout_only=False,
        release_train=False,
        num_rollout=100,
        use_critic=False,
        no_save_rng=False,
        no_save_optim=False,
        ckpt_step=None,
        start_rollout_id=None,
        finetune=False,
        rollout_shuffle=False,
    )
    source = RolloutDataSource.__new__(RolloutDataSource)
    source.args, source.dataset = args, None
    source.consumers, source._restored_consumers, source.metadata = {}, {}, {}
    source.epoch_id, source.sample_offset = 1, 17
    source.sample_index, source.sample_group_index = 99, 99

    def save_model(step, *, force_sync):
        assert force_sync
        directory = tmp_path / f"iter_{step:07d}"
        directory.mkdir()
        torch.save({"weight": torch.ones(2), "step": step}, directory / "model.pt")
        (tmp_path / "latest_checkpointed_iteration.txt").write_text(str(step))

    actor = SimpleNamespace(args=args, save_model=save_model)
    manager = SimpleNamespace(
        save=SimpleNamespace(remote=source.save), get_weight_version=SimpleNamespace(remote=lambda: "4")
    )
    monkeypatch.setattr("ray.get", lambda value: value)
    save_checkpoint(args, 3, actor, None, manager, actor_trains=True)
    args.load = str(tmp_path)
    restored, _ = resolve_checkpoint(args)
    assert (restored.ckpt_step, restored.start_rollout_id, restored.update_weight_start_version) == (3, 4, 4)
    source.sample_offset = 0
    source.load(3)
    assert source.sample_offset == 17

    def failed_save(step):
        raise OSError("injected data checkpoint failure")

    manager.save.remote = failed_save
    with pytest.raises(OSError, match="injected"):
        save_checkpoint(args, 4, actor, None, manager, actor_trains=True)
    restored, _ = resolve_checkpoint(args)
    assert restored.ckpt_step == 3
    args.ckpt_step = 4
    with pytest.raises(ValueError, match="complete"):
        resolve_checkpoint(args)
    args.ckpt_step = None
    data_file = tmp_path / "rollout/global_dataset_state_dict_3.pt"
    content = bytearray(data_file.read_bytes())
    content[-1] ^= 1
    data_file.write_bytes(content)
    with pytest.raises(ValueError, match="changed"):
        resolve_checkpoint(args)
    with pytest.raises(FileNotFoundError, match="exact resume"):
        source.load(4)
    args.finetune = True
    source.load(4)


def test_dataset_resume_rejects_changed_input_identity():
    from slime.data.data_source import RolloutDataSource

    source = RolloutDataSource.__new__(RolloutDataSource)
    source.dataset_identity = {"sha256": "new"}
    with pytest.raises(ValueError, match="Dataset contents"):
        source.load_state_dict({"dataset_identity": {"sha256": "old"}, "sample_offset": 5})


@pytest.mark.parametrize("direct_iteration", [False, True])
def test_legacy_checkpoint_upgrade_keeps_original_recovery_directory(tmp_path, direct_iteration):
    from slime.data.checkpoint import resolve_checkpoint

    original = tmp_path / "legacy"
    iteration = original / "iter_0000003"
    iteration.mkdir(parents=True)
    tracker = original / "latest_checkpointed_iteration.txt"
    tracker.write_text("3")
    args = SimpleNamespace(
        rollout_data_transport="object-store",
        checkpoint_joint_commit=True,
        load=str(iteration if direct_iteration else original),
        save=str(original),
        finetune=False,
    )
    with pytest.raises(ValueError, match="new --save directory"):
        resolve_checkpoint(args)
    args.save = str(tmp_path / "continued")
    restored, _ = resolve_checkpoint(args)
    assert restored.load == args.load
    assert tracker.read_text() == "3"
    assert not (original / "rollout/joint_checkpoint.json").exists()
    args.save, args.checkpoint_joint_commit = str(original), False
    resolve_checkpoint(args)
