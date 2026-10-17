"""Slime deployment options stay separate from native Omni configuration."""

from argparse import Namespace
from pathlib import Path

import pytest

from slime.backends.sglang_utils import omni_config
from slime.backends.sglang_utils.omni_config import resolve_omni_deployment
from slime.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, SglangConfig

NUM_GPUS = 0
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def args(tmp_path):
    source = tmp_path / "pipeline.yaml"
    source.write_text("config_cls: MossTTSPrismPipelineConfig\nmodel_path: checkpoint\nenable_rl: true\n")
    return Namespace(
        rollout_external=False,
        sglang_router_ip=None,
        omni_config=str(source),
        omni_update_stage="decoder",
        omni_worker_capabilities=["chat"],
        num_gpus_per_node=8,
        rollout_num_gpus_per_engine=1,
        hf_checkpoint="student",
    )


def test_deployment_preserves_native_configs_and_explicit_options(args, monkeypatch):
    source = Path(args.omni_config)
    original = source.read_bytes()
    explicit = {
        "decoder.engine.max_running_requests": 7,
        "custom_decoder.engine.enable_weights_cpu_backup": False,
        "enable_memory_saver": False,
        "omni_update_stage": "custom_decoder",
        "worker_capabilities": ["audio_input"],
    }
    models = [
        ModelConfig("student", update_weights=True, server_groups=[ServerGroupConfig("regular", 1)]),
        ModelConfig("teacher", update_weights=False, server_groups=[ServerGroupConfig("regular", 1)]),
        ModelConfig(
            "custom", update_weights=False, server_groups=[ServerGroupConfig("regular", 1, overrides=explicit)]
        ),
    ]
    monkeypatch.setattr(omni_config, "resolve_sglang_config", lambda args: SglangConfig(models))

    student, teacher, custom = resolve_omni_deployment(args)

    student_options = student.server_groups[0].overrides
    teacher_options = teacher.server_groups[0].overrides
    custom_options = custom.server_groups[0].overrides
    assert "decoder.engine.enable_weights_cpu_backup" not in student_options
    assert teacher_options["decoder.engine.enable_weights_cpu_backup"] is True
    assert custom_options["custom_decoder.engine.enable_weights_cpu_backup"] is False
    assert custom_options["decoder.engine.max_running_requests"] == 7
    assert custom_options["enable_memory_saver"] is False
    assert student_options["omni_update_stage"] == teacher_options["omni_update_stage"] == "decoder"
    assert custom_options["omni_update_stage"] == "custom_decoder"
    assert student_options["worker_capabilities"] == teacher_options["worker_capabilities"] == ["chat"]
    assert custom_options["worker_capabilities"] == ["audio_input"]
    assert source.read_bytes() == original
    assert all(model.server_groups[0].overrides["omni_config"] == str(source) for model in (student, teacher, custom))
    assert "worker_capabilities" not in models[0].server_groups[0].overrides


def test_group_options_can_supply_pipeline_stage_and_capabilities_without_global_defaults(args, monkeypatch):
    config = args.omni_config
    args.omni_config = args.omni_update_stage = args.omni_worker_capabilities = None
    group = ServerGroupConfig(
        "regular",
        1,
        overrides={"omni_config": config, "omni_update_stage": "decoder", "worker_capabilities": ["audio_input"]},
    )
    model = ModelConfig("teacher", update_weights=False, server_groups=[group])
    monkeypatch.setattr(omni_config, "resolve_sglang_config", lambda args: SglangConfig([model]))

    teacher = resolve_omni_deployment(args)[0]

    assert teacher.server_groups[0].overrides["worker_capabilities"] == ["audio_input"]
    assert teacher.server_groups[0].overrides["decoder.engine.enable_weights_cpu_backup"] is True


def test_each_regular_group_requires_a_pipeline(args, monkeypatch):
    args.omni_config = None
    model = ModelConfig("student", server_groups=[ServerGroupConfig("regular", 1)])
    monkeypatch.setattr(omni_config, "resolve_sglang_config", lambda args: SglangConfig([model]))
    with pytest.raises(ValueError, match="omni_config override"):
        resolve_omni_deployment(args)


@pytest.mark.parametrize("capabilities", [None, "chat", [], ["chat", "chat"], ["two names"]])
def test_invalid_worker_capabilities_fail_before_launch(args, monkeypatch, capabilities):
    args.omni_worker_capabilities = capabilities
    model = ModelConfig("student", server_groups=[ServerGroupConfig("regular", 1)])
    monkeypatch.setattr(omni_config, "resolve_sglang_config", lambda args: SglangConfig([model]))
    with pytest.raises(ValueError, match="worker-capabilities"):
        resolve_omni_deployment(args)


@pytest.mark.parametrize(
    "models,message", [([], "at least one model"), ([ModelConfig("same"), ModelConfig("same")], "unique")]
)
def test_invalid_model_sets_fail_before_pipeline_loading(monkeypatch, models, message):
    monkeypatch.setattr(omni_config, "resolve_sglang_config", lambda args: SglangConfig(models))
    with pytest.raises(ValueError, match=message):
        resolve_omni_deployment(Namespace(rollout_external=False, sglang_router_ip=None))


@pytest.mark.parametrize("filename", ["student.yaml", "teacher-score.yaml"])
def test_shipped_pipelines_load_directly_with_native_omni(filename):
    manager = pytest.importorskip("sglang_omni.config.manager")
    path = REPO / "examples/prism/configs" / filename
    configuration = manager.ConfigManager.from_file(str(path))
    resolved = configuration.merge_config([("enable_memory_saver", "true")])
    engine = next(stage.engine for stage in resolved.stages if stage.name == "tts_engine")
    assert engine.overrides()["cuda_graph_backend_prefill"] == "disabled"
    if filename == "student.yaml":
        assert resolved.enable_rl is True
        assert resolved.stage_factory_kwargs("vocoder")["vocoder_cuda_graph"] is False
    else:
        assert engine.overrides()["enable_weights_cpu_backup"] is True


@pytest.mark.parametrize(
    "suffix,message",
    [
        ("model_path: second-checkpoint\n", "duplicate mapping key"),
        ("stages:\n  unknown_stage:\n    gpu: 0\n", "no stage named"),
        ("slime:\n  worker_capabilities: [chat]\n", "slime"),
    ],
)
def test_native_config_errors_are_not_hidden_by_slime(args, monkeypatch, suffix, message):
    manager = pytest.importorskip("sglang_omni.config.manager")
    path = Path(args.omni_config)
    path.write_text(path.read_text() + suffix)
    original = path.read_bytes()
    model = ModelConfig("student", server_groups=[ServerGroupConfig("regular", 1)])
    monkeypatch.setattr(omni_config, "resolve_sglang_config", lambda args: SglangConfig([model]))

    options = resolve_omni_deployment(args)[0].server_groups[0].overrides

    assert Path(options["omni_config"]).read_bytes() == original
    with pytest.raises(ValueError, match=message):
        manager.ConfigManager.from_file(options["omni_config"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
