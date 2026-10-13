"""CPU checks for sample-weighted Prism diagnostics and sparse DP groups."""

import json
from argparse import Namespace
from pathlib import Path

import _cp_dist_helpers
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from slime.backends.megatron_utils.policy_actions import grouped_opd_metrics
from slime.observability.train_metric_utils import gather_and_reduce_log_dict, reduce_train_step_metrics

NUM_GPUS = 0


def metric_rollout():
    inputs, masks = [], []
    for frames, code_delta, stop_delta, enabled in (
        (1, 1.0, 2.0, True),
        (3, 3.0, 4.0, True),
        (0, 9.0, 5.0, True),
        (2, 99.0, 99.0, False),
    ):
        code_mask = torch.zeros(frames + 1, 2, dtype=torch.bool)
        code_mask[:frames] = True
        codes = torch.full((frames + 1, 2), -2.0)
        stops = torch.full((frames + 1,), -1.0)
        inputs.append(
            dict(
                policy_code_mask=code_mask,
                policy_stop_mask=torch.ones(frames + 1, dtype=torch.bool),
                policy_group_advantage=torch.zeros(1),
                policy_sample_mask=torch.tensor([enabled]),
                policy_old_code_logprobs=codes,
                policy_old_stop_logprobs=stops,
                policy_teacher_code_logprobs=codes - code_delta,
                policy_teacher_stop_logprobs=stops - stop_delta,
            )
        )
        masks.append(torch.ones(frames))
    return dict(
        multimodal_train_inputs=inputs,
        loss_masks=masks,
        policy_domains=["zh", "zh", "en", "removed"],
        policy_teachers=["a", "b", "a", "c"],
    )


def test_grouped_opd_uses_sample_means_and_keeps_zero_frame_stop():
    metrics = grouped_opd_metrics(Namespace(use_opd=True, use_rollout_logprobs=False), metric_rollout())
    assert metrics["prism/domain/zh/rvq/opd_logratio_mean"] == (4.0, 2.0)
    assert metrics["prism/teacher/a/stop/opd_logratio_mean"] == (7.0, 2.0)
    assert metrics["prism/teacher/a/rvq/opd_logratio_mean"] == (1.0, 1.0)
    assert not any("removed" in key or "/teacher/c/" in key or "/en/rvq/" in key for key in metrics)


def test_split_metric_reporting_uses_active_counts_but_loss_keeps_batch_denominator(monkeypatch):
    monkeypatch.setattr(dist, "all_reduce", lambda *a, **kw: None)
    keys = [
        "pg_loss",
        "rvq/pg_loss",
        "stop/pg_loss",
        "rvq/active_ratio",
        "stop/active_ratio",
        "rvq/opd_logratio_mean",
        "stop/opd_logratio_mean",
    ]
    # Three trajectories: two with RVQ, all three with stop. Loss numerators
    # are 6 + 0.5 * 12 = 12; diagnostics have a different active denominator.
    values = torch.tensor([0.0, 12.0, 6.0, 12.0, 2.0, 3.0, 4.0, 15.0])
    actual = reduce_train_step_metrics(
        [dict(keys=keys, values=values)],
        calculate_per_token_loss=False,
        step_global_batch_size=3,
        cp_size=1,
        dp_with_cp_group=None,
    )
    assert actual["pg_loss"] == actual["rvq/pg_loss"] + 0.5 * actual["stop/pg_loss"] == 4
    assert actual["rvq/opd_logratio_mean"] == 2
    assert actual["stop/opd_logratio_mean"] == 5
    assert actual["rvq/active_samples"] == 2
    values[4] = values[6] = 0
    actual = reduce_train_step_metrics(
        [dict(keys=keys, values=values)],
        calculate_per_token_loss=False,
        step_global_batch_size=3,
        cp_size=1,
        dp_with_cp_group=None,
    )
    assert "rvq/opd_logratio_mean" not in actual


def _distributed_groups(rank, port, result_path):
    group = _cp_dist_helpers.init_worker_process_group(rank, 2, port)
    try:
        data = metric_rollout()
        indices = (0, 1) if rank == 0 else (2, 3)
        shard = {key: [value[i] for i in indices] for key, value in data.items()}
        local = grouped_opd_metrics(Namespace(use_opd=True, use_rollout_logprobs=False), shard)
        result = gather_and_reduce_log_dict(local, dp_size=2, dp_src_rank=0, dp_group=group)
        if rank == 0:
            Path(result_path).write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


def test_sparse_teacher_and_domain_metrics_across_real_dp_ranks(tmp_path):
    result_path = tmp_path / "metrics.json"
    mp.spawn(_distributed_groups, args=(_cp_dist_helpers.free_port(), str(result_path)), nprocs=2, join=True)
    result = json.loads(result_path.read_text())
    expected = grouped_opd_metrics(Namespace(use_opd=True, use_rollout_logprobs=False), metric_rollout())
    assert result == {key: total / count for key, (total, count) in expected.items()}


def test_prism_performance_keeps_sequence_row_units_without_generic_flops(monkeypatch):
    from megatron.core import mpu
    from slime.observability import train_metric_utils as metrics
    from slime.observability.timer import Timer

    for name in ("get_tensor_model_parallel_rank", "get_pipeline_model_parallel_rank", "get_data_parallel_rank"):
        monkeypatch.setattr(mpu, name, lambda **kw: 0, raising=False)
    monkeypatch.setattr(mpu, "is_pipeline_last_stage", lambda: True, raising=False)
    monkeypatch.setattr(Timer(), "seq_lens", [3, 7], raising=False)
    monkeypatch.setattr(Timer(), "log_dict", lambda: {"actor_train": 2.0, "log_probs": 1.0})
    monkeypatch.setattr(Timer(), "reset", lambda: None)

    def unexpected(*args, **kwargs):
        raise AssertionError("Prism must not use the generic LM FLOPs estimate")

    monkeypatch.setattr(metrics, "calculate_fwd_flops", unexpected)
    logged = []
    monkeypatch.setattr(metrics.logging_utils, "log", lambda args, data, **kw: logged.append(data))
    metrics.log_perf_data(
        0, Namespace(custom_megatron_log_prob_function_path="fixture", wandb_always_use_train_step=False)
    )
    assert logged[0]["perf/train_sequence_rows_per_sec"] == 5
    assert not any("tflops" in key for key in logged[0])


@pytest.mark.parametrize("transport", ["object-store", "nixl"])
def test_policy_labels_follow_conversion_and_dp_partition(monkeypatch, transport):
    from slime.data import batch_builder
    from slime.utils.types import Sample

    args = Namespace(
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        advantage_estimator="grpo",
        rewards_normalization=False,
        reward_key=None,
        rollout_top_p=1.0,
        rollout_data_transport=transport,
        global_batch_size=2,
    )
    builder = batch_builder.BatchBuilder(args)
    builder.train_parallel_config = {"dp_size": 2}
    samples = [
        Sample(
            index=i,
            tokens=[1, 2],
            response_length=1,
            reward=0.0,
            loss_mask=[1],
            metadata={"domain": domain, "omni_teacher_key": teacher},
            multimodal_train_inputs={"policy_code_mask": torch.tensor([[False], [True]])},
        )
        for i, (domain, teacher) in enumerate((("chinese", "zh-teacher"), ("english", "en-teacher")))
    ]
    converted = builder.convert(samples)
    monkeypatch.setattr(batch_builder, "build_dp_schedule", lambda *a, **kw: ([[1], [0]], [[[0]], [[0]]], [1], [2]))
    monkeypatch.setattr(batch_builder.ray, "put", lambda value, **kw: value)
    refs = builder.split_by_dp(converted)
    assert refs[0].inner["policy_domains"] == ["english"]
    assert refs[0].inner["policy_teachers"] == ["en-teacher"]
    assert refs[1].inner["policy_domains"] == ["chinese"]
    assert refs[1].inner["policy_teachers"] == ["zh-teacher"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
