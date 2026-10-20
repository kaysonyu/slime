"""Deployment validation runs before Ray or GPU processes are created."""

import argparse
import ast
import json
import math
import os
import sys
from pathlib import Path

import pytest

NUM_GPUS = 0
pytestmark = pytest.mark.unit


def _load_arguments():
    # Exercise the real CLI definitions without importing the SGLang server.
    source = Path(__file__).resolve().parents[1] / "slime/utils/arguments.py"
    names = {"reset_arg", "get_slime_extra_args_provider", "validate_omni_args"}
    functions = [
        node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"argparse": argparse, "json": json, "math": math, "os": os, "sys": sys}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["get_slime_extra_args_provider"], namespace["validate_omni_args"]


get_slime_extra_args_provider, validate_omni_args = _load_arguments()


def _args(tmp_path, **overrides):
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text("config_cls: MossTTSPrismPipelineConfig\nmodel_path: checkpoint\nenable_rl: true\n")
    values = dict(
        rollout_backend="sglang-omni",
        omni_config=str(pipeline),
        omni_python=None,
        colocate=False,
        offload_rollout=False,
        rollout_external=False,
        update_weight_mode="full",
        update_weight_transport="nccl",
        check_weight_update_equal=False,
        debug_train_only=False,
        omni_request_timeout_s=600.0,
        omni_cancel_drain_timeout_s=600.0,
    )
    return argparse.Namespace(**(values | overrides))


def test_cli_defaults_keep_original_backend():
    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(["--rollout-batch-size", "1"])
    assert args.rollout_backend == "sglang"
    assert args.omni_update_stage is None
    assert args.omni_worker_capabilities is None
    assert args.custom_megatron_log_prob_function_path is None
    assert not args.use_rollout_logprobs
    assert args.policy_loss_weights is None
    assert args.omni_router_processes == 1
    assert not args.cache_weight_topology
    assert not args.rollout_packed_tensor_buffer


def test_cli_accepts_opt_in_scaling_without_changing_reward_configuration():
    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--rollout-batch-size",
            "1024",
            "--omni-router-processes",
            "16",
            "--cache-weight-topology",
            "--rollout-packed-tensor-buffer",
            "--omni-reward-config",
            "rewards.yaml",
        ]
    )
    assert args.omni_router_processes == 16
    assert args.cache_weight_topology and args.rollout_packed_tensor_buffer
    assert args.omni_reward_config == "rewards.yaml"


def test_cli_accepts_an_explicit_weight_update_stage_and_router_capabilities():
    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--rollout-batch-size",
            "1",
            "--omni-update-stage",
            "decoder",
            "--omni-worker-capabilities",
            "chat",
            "audio_input",
            "audio_output",
        ]
    )
    assert args.omni_update_stage == "decoder"
    assert args.omni_worker_capabilities == ["chat", "audio_input", "audio_output"]


def test_disaggregated_omni_defaults(tmp_path):
    args = _args(tmp_path)
    validate_omni_args(args)
    assert args.omni_python == sys.executable
    assert args.omni_config == str(tmp_path / "pipeline.yaml")


@pytest.mark.parametrize("processes", [0, -1, 1.5, True])
def test_invalid_router_process_count(tmp_path, processes):
    with pytest.raises(ValueError, match="positive integer"):
        validate_omni_args(_args(tmp_path, omni_router_processes=processes))


def test_multiprocess_router_rejects_local_only_least_request(tmp_path):
    with pytest.raises(ValueError, match="least_request"):
        validate_omni_args(_args(tmp_path, omni_router_processes=4, omni_router_policy="least_request"))
    validate_omni_args(_args(tmp_path, omni_router_processes=4, omni_router_policy="round_robin"))


def test_cli_defers_gpu_topology_to_resolved_deployment(tmp_path):
    args = _args(tmp_path, rollout_num_gpus_per_engine=1)
    pipeline = Path(args.omni_config)
    pipeline.write_text(pipeline.read_text() + "stages:\n  tts_engine:\n    gpu: 0\n  vocoder:\n    gpu: 1\n")
    args.sglang_config = str(tmp_path / "deployment.yaml")
    Path(args.sglang_config).write_text(
        "sglang:\n  - name: student\n    num_gpus_per_engine: 2\n"
        "    server_groups:\n      - worker_type: regular\n        num_gpus: 2\n"
    )

    validate_omni_args(args)


@pytest.mark.parametrize("flag", ["rollout_external", "check_weight_update_equal"])
def test_unsupported_capabilities_fail_before_launch(tmp_path, flag):
    with pytest.raises(ValueError):
        validate_omni_args(_args(tmp_path, **{flag: True}))


def test_omni_rejects_an_explicit_external_router(tmp_path):
    with pytest.raises(ValueError, match="sglang-router-ip"):
        validate_omni_args(_args(tmp_path, sglang_router_ip="192.0.2.7"))


@pytest.mark.parametrize("overrides", [{"update_weight_mode": "delta"}, {"update_weight_transport": "ipc"}])
def test_requires_full_nccl_publication(tmp_path, overrides):
    with pytest.raises(ValueError, match="full NCCL"):
        validate_omni_args(_args(tmp_path, **overrides))


@pytest.mark.parametrize("name", ["omni_request_timeout_s", "omni_cancel_drain_timeout_s"])
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_timeouts_fail_before_launch(tmp_path, name, value):
    with pytest.raises(ValueError, match="positive and finite"):
        validate_omni_args(_args(tmp_path, **{name: value}))


def test_missing_pipeline_and_train_only(tmp_path):
    with pytest.raises(ValueError, match="existing --omni-config"):
        validate_omni_args(_args(tmp_path, omni_config=None))
    validate_omni_args(_args(tmp_path, omni_config=None, debug_train_only=True))


def test_per_group_pipelines_do_not_require_a_global_pipeline(tmp_path):
    args = _args(tmp_path, omni_config=None, sglang_config=str(tmp_path / "deployment.yaml"))
    validate_omni_args(args)
    assert args.omni_config is None


def test_original_backend_bypasses_omni_validation(tmp_path):
    args = _args(tmp_path, rollout_backend="sglang", colocate=True, omni_config=None)
    validate_omni_args(args)
    assert args.omni_python is None


@pytest.mark.parametrize("flag", ["colocate", "offload_rollout"])
def test_omni_supports_shared_gpu_memory_lifecycle(tmp_path, flag):
    validate_omni_args(_args(tmp_path, **{flag: True}))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
