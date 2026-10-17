"""Native GRPO/OPD integration for chronological multi-codebook trajectories."""

from argparse import Namespace

import _cp_dist_helpers  # noqa: F401
import pytest
import torch

from slime.backends.megatron_utils import loss as loss_module
from slime.backends.megatron_utils.policy_actions import store_old_action_scores, validate_action_policy_args

NUM_GPUS = 0


def arguments(**overrides):
    defaults = dict(
        custom_megatron_log_prob_function_path="test.extractor",
        policy_loss_weights={},
        advantage_estimator="grpo",
        loss_type="policy_loss",
        pg_loss_type=None,
        use_rollout_logprobs=False,
        use_opd=False,
        opd_type="sglang",
        opd_kl_coef=0.4,
        kl_coef=0.0,
        custom_advantage_function_path=None,
        normalize_advantages=False,
        use_score_centering=False,
        use_opsm=False,
        get_mismatch_metrics=False,
        use_tis=False,
        eps_clip=0.2,
        eps_clip_high=0.2,
        eps_clip_c=None,
        entropy_coef=0.0,
        use_kl_loss=False,
        kl_loss_coef=0.0,
        kl_loss_type="k1",
        use_unbiased_kl=False,
        custom_pg_loss_reducer_function_path=None,
        calculate_per_token_loss=False,
        allgather_cp=False,
        recompute_loss_function=False,
    )
    return Namespace(**(defaults | overrides))


def sample(rows=5, frames=2, terminal=True, advantage=1.0, codebooks=2):
    start = rows - frames - 1
    code_mask = torch.zeros(rows, codebooks, dtype=torch.bool)
    code_mask[start : start + frames] = True
    stop_mask = torch.zeros(rows, dtype=torch.bool)
    stop_mask[start : start + frames + int(terminal)] = True
    stops = torch.zeros(rows, dtype=torch.long)
    if terminal:
        stops[rows - 1] = 1
    # Different scores per action detect accidental broadcasting and summing.
    old_codes = -torch.arange(1, rows * codebooks + 1).reshape(rows, codebooks).float() / 10 - 1
    old_stop = -torch.arange(1, rows + 1).float() / 10 - 1
    streams = {
        "rvq": {
            "actions": torch.zeros_like(code_mask, dtype=torch.long),
            "valid_mask": code_mask,
            "loss_mask": (code_mask).clone(),
            "old_log_probs": old_codes,
            "rollout_log_probs": old_codes - 0.03,
        },
        "stop": {
            "actions": stops,
            "valid_mask": stop_mask,
            "loss_mask": (stop_mask).clone(),
            "old_log_probs": old_stop,
            "rollout_log_probs": old_stop - 0.07,
        },
    }
    delta = torch.linspace(-0.3, 0.35, rows * (codebooks + 1)).reshape(rows, codebooks + 1)
    output = (torch.cat((old_codes, old_stop[:, None]), dim=-1) + delta).requires_grad_()
    return {
        "streams": streams,
        "advantage": advantage,
        "enabled": True,
        "output": output,
        "rows": rows,
        "frames": frames,
    }


def batch_for(samples, padding=0, args=None):
    trajectories = [s["streams"] for s in samples]
    rollout = {"policy_actions": trajectories}
    loss_module.prepare_action_advantages(args or arguments(), rollout, [s["advantage"] for s in samples])
    streams = {
        name: {key: torch.cat([t[name][key] for t in trajectories]) for key in trajectories[0][name]}
        for name in trajectories[0]
    }
    outputs = torch.cat([s["output"] for s in samples])
    if padding:
        outputs = torch.cat((outputs, torch.full((padding, outputs.shape[-1]), 99.0)))
    batch = {
        "policy_sample_masks": torch.tensor([s["enabled"] for s in samples]),
        "policy_actions": streams,
        "total_lengths": [s["rows"] for s in samples],
        "response_lengths": [s["frames"] for s in samples],
        "loss_masks": [torch.ones(s["frames"]) for s in samples],
        "rollout_mask_sums": torch.tensor([float(s["frames"]) for s in samples]),
        "advantages": [torch.full((s["frames"],), s["advantage"]) for s in samples],
        "log_probs": [torch.zeros(s["frames"]) for s in samples],
    }
    return batch, outputs[None]


def extract(args, batch, output):
    codes, stops, summaries = [], [], []
    offset = 0
    for rows, frames in zip(batch["total_lengths"], batch["response_lengths"], strict=True):
        part = output[0, offset : offset + rows]
        codes.append(part[:, :-1])
        stops.append(part[:, -1])
        start = rows - frames - 1
        summaries.append(part[start : start + frames, :-1].sum(-1))
        offset += rows
    return {"log_probs": summaries, "policy/rvq/log_probs": codes, "policy/stop/log_probs": stops}


@pytest.fixture(autouse=True)
def native_eager(monkeypatch):
    from slime.utils import misc

    monkeypatch.setattr(misc, "load_function", lambda path: extract)
    # Exercise the production operator body without CPU CI compilation cost.
    native = loss_module.compute_policy_loss
    monkeypatch.setattr(loss_module, "compute_policy_loss", getattr(native, "__wrapped__", native))
    native_kl = loss_module.compute_approx_kl
    monkeypatch.setattr(loss_module, "compute_approx_kl", getattr(native_kl, "__wrapped__", native_kl))
    monkeypatch.setattr(loss_module.mpu, "is_pipeline_last_stage", lambda: True, raising=False)
    monkeypatch.setattr(loss_module.mpu, "get_data_parallel_world_size", lambda **kwargs: 1, raising=False)


def run_policy(args, samples, padding=0):
    batch, output = batch_for(samples, padding, args)
    value, metrics = loss_module.policy_loss_function(args, batch, output, lambda x: x.sum())
    return value, metrics


def independent_reference(args, samples):
    result = samples[0]["output"].new_zeros(())
    for s in samples:
        for column, name in ((slice(None, -1), "rvq"), (-1, "stop")):
            stream = s["streams"][name]
            current = s["output"][:, column][stream["loss_mask"]]
            if current.numel() == 0:
                continue
            source = "rollout" if args.use_rollout_logprobs else "old"
            old = stream[f"{source}_log_probs"][stream["loss_mask"]].detach()
            adv = torch.full_like(current, s["advantage"])
            if args.use_opd:
                teacher = stream["teacher_log_probs"][stream["loss_mask"]]
                adv = adv - args.opd_kl_coef * (old - teacher)
            ratio = (current - old).exp()
            clipped = ratio.clamp(1 - args.eps_clip, 1 + args.eps_clip_high)
            objective = torch.maximum(-adv * ratio, -adv * clipped).mean()
            if args.use_kl_loss:
                reference = stream["ref_log_probs"][stream["loss_mask"]].detach()
                log_ratio = current - reference
                if args.kl_loss_type == "k1":
                    kl = log_ratio
                elif args.kl_loss_type == "k2":
                    kl = log_ratio.square() / 2
                else:
                    kl = (reference - current).exp() - 1 + log_ratio
                if args.use_unbiased_kl:
                    kl = kl * ratio
                if args.kl_loss_type == "low_var_kl":
                    kl = kl.clamp(-10, 10)
                objective = objective + args.kl_loss_coef * kl.mean()
            result = result + objective * args.policy_loss_weights.get(name, 1) * s["enabled"]
    return result


@pytest.mark.parametrize("frames,terminal", [(2, True), (2, False), (0, True)])
def test_native_policy_actions_and_gradients(frames, terminal):
    args = arguments(policy_loss_weights={"stop": 0.7})
    s = sample(frames=frames, terminal=terminal, advantage=-1.2)
    value, metrics = run_policy(args, [s], padding=7)
    ref = independent_reference(args, [s])
    torch.testing.assert_close(value, ref)
    torch.testing.assert_close(torch.autograd.grad(value, s["output"])[0], torch.autograd.grad(ref, s["output"])[0])
    assert "entropy_loss" not in metrics  # Selected logp is not distribution entropy.
    assert s["frames"] == frames  # No fabricated RVQ or expanded response_length.


def test_native_opd_is_applied_to_selected_stop_and_code(monkeypatch):
    args = arguments(use_opd=True)
    samples = [sample(advantage=0.0), sample(rows=3, frames=0, advantage=0.0)]
    for s in samples:
        for stream in s["streams"].values():
            old = stream["old_log_probs"]
            stream["teacher_log_probs"] = old + torch.linspace(-0.8, 0.9, old.numel()).reshape_as(old)
    calls = []
    native = loss_module.apply_opd_kl_to_advantages

    def spy(*args, **kwargs):
        calls.append(True)
        return native(*args, **kwargs)

    monkeypatch.setattr(loss_module, "apply_opd_kl_to_advantages", spy)
    value, metrics = run_policy(args, samples)
    torch.testing.assert_close(value, independent_reference(args, samples))
    assert calls == [True]
    assert "opd_reverse_kl" in metrics


def test_packed_and_microbatch_gradients_are_equal():
    args = arguments(policy_loss_weights={"stop": 1.4})
    samples = [
        sample(advantage=0.4),
        sample(rows=4, frames=2, terminal=False, advantage=-1.2),
        sample(rows=3, frames=0, advantage=0.9),
    ]
    packed, _ = run_policy(args, samples, padding=4)
    separate = sum(run_policy(args, [s])[0] for s in samples)
    torch.testing.assert_close(packed, separate)
    targets = [s["output"] for s in samples]
    for pgrad, sgrad in zip(torch.autograd.grad(packed, targets), torch.autograd.grad(separate, targets), strict=True):
        torch.testing.assert_close(pgrad, sgrad)


@pytest.mark.parametrize("use_opd", [False, True])
def test_split_metrics_preserve_objective_gradients_and_component_denominators(use_opd):
    args = arguments(use_opd=use_opd, policy_loss_weights={"stop": 0.3})
    samples = [sample(frames=2), sample(rows=3, frames=0), sample(rows=6, frames=3)]
    for s in samples:
        for stream in s["streams"].values():
            stream["teacher_log_probs"] = stream["old_log_probs"] - 0.2
    value, metrics = run_policy(args, samples)
    reference = independent_reference(args, samples)
    torch.testing.assert_close(value, reference)
    targets = [s["output"] for s in samples]
    for actual, expected in zip(
        torch.autograd.grad(value, targets), torch.autograd.grad(reference, targets), strict=True
    ):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(metrics["pg_loss"], metrics["rvq/pg_loss"] + 0.3 * metrics["stop/pg_loss"])
    assert metrics["rvq/active_ratio"] == 2 and metrics["stop/active_ratio"] == 3
    for key, metric in metrics.items():
        assert not metric.requires_grad, key
        separate = sum(run_policy(args, [s])[1][key] for s in samples)
        torch.testing.assert_close(metric, separate)


def test_prism_extractor_integrates_with_native_policy_loss(monkeypatch):
    from slime.utils import misc
    from slime_plugins.models.prism_policy import score_policy

    monkeypatch.setattr(misc, "load_function", lambda path: score_policy)
    args = arguments()
    samples = [sample(), sample(rows=3, frames=0, advantage=-0.5)]
    value, _ = run_policy(args, samples, padding=4)
    torch.testing.assert_close(value, independent_reference(args, samples))


@pytest.mark.parametrize("use_kl_loss", [False, True])
def test_existing_megatron_wrapper_preserves_global_sample_mean(monkeypatch, use_kl_loss):
    args = arguments(use_kl_loss=use_kl_loss, kl_loss_coef=0.3, kl_loss_type="k3")
    samples = [sample(advantage=0.4), sample(rows=3, frames=0, advantage=-0.8)]
    for s in samples:
        for stream in s["streams"].values():
            stream["ref_log_probs"] = stream["old_log_probs"] - 0.2
    packed_batch, packed_output = batch_for(samples)
    packed, _, _ = loss_module.loss_function(args, packed_batch, 1, 2, packed_output)
    micros = []
    for s in samples:
        batch, output = batch_for([s])
        micros.append(loss_module.loss_function(args, batch, 2, 2, output)[0])
    # Megatron averages the microbatch losses at the end of its schedule.
    torch.testing.assert_close(packed, sum(micros) / 2)
    monkeypatch.setattr(loss_module.mpu, "get_data_parallel_world_size", lambda **kwargs: 2)
    shards = []
    for s in samples:
        batch, output = batch_for([s])
        shards.append(loss_module.loss_function(args, batch, 1, 2, output)[0])
    # DDP averages rank gradients; both shards still contribute one sample.
    torch.testing.assert_close(packed, sum(shards) / 2)


@pytest.mark.parametrize("prefix", ["old", "ref", "teacher"])
def test_scores_are_detached_and_frozen_across_updates(prefix):
    s = sample()
    batch, output = batch_for([s])
    scores = extract(arguments(), batch, output)
    rollout = {"policy_actions": [s["streams"]]}
    store_old_action_scores(rollout, scores, prefix)
    key = f"{prefix}_log_probs"
    old = rollout["policy_actions"][0]["rvq"][key].clone()
    assert not rollout["policy_actions"][0]["rvq"][key].requires_grad
    with torch.no_grad():
        s["output"].add_(0.2)
    torch.testing.assert_close(rollout["policy_actions"][0]["rvq"][key], old)


def test_zero_frame_gets_native_grpo_scalar_and_skips_frame_opd():
    args = arguments(use_opd=False)
    samples = [sample(advantage=0.0), sample(rows=3, frames=0, advantage=0.0)]
    rollout = {
        "policy_actions": [s["streams"] for s in samples],
        "log_probs": [torch.zeros(s["frames"]) for s in samples],
        "loss_masks": [torch.ones(s["frames"]) for s in samples],
        "rewards": [1.3, -0.4],
        "response_lengths": [s["frames"] for s in samples],
        "total_lengths": [s["rows"] for s in samples],
    }
    # No frame teacher_log_probs: action teacher scores live on a distinct axis.
    loss_module.compute_advantages_and_returns(args, rollout)
    torch.testing.assert_close(rollout["policy_actions"][1]["stop"]["advantages"][-1], torch.tensor(-0.4))
    assert "advantages" not in rollout and "returns" not in rollout


def test_rollout_source_is_configurable():
    args = arguments(use_rollout_logprobs=True)
    s = sample()
    value, _ = run_policy(args, [s])
    torch.testing.assert_close(value, independent_reference(args, [s]))


def test_missing_teacher_or_frozen_old_fails():
    s = sample()
    with pytest.raises(ValueError, match="Missing selected teacher"):
        run_policy(arguments(use_opd=True), [s])
    del s["streams"]["rvq"]["old_log_probs"]
    with pytest.raises(ValueError, match="Missing selected old"):
        run_policy(arguments(), [s])


@pytest.mark.parametrize("kl_loss_type", ["k1", "k2", "k3", "low_var_kl"])
@pytest.mark.parametrize("use_unbiased_kl", [False, True])
@pytest.mark.parametrize("use_rollout_logprobs", [False, True])
def test_reference_kl_preserves_stream_weights_masks_and_frozen_scores(
    kl_loss_type, use_unbiased_kl, use_rollout_logprobs
):
    args = arguments(
        use_kl_loss=True,
        kl_loss_coef=0.37,
        kl_loss_type=kl_loss_type,
        use_unbiased_kl=use_unbiased_kl,
        use_rollout_logprobs=use_rollout_logprobs,
        policy_loss_weights={"rvq": 0.6, "stop": 1.4},
    )
    samples = [
        sample(rows=6, frames=3, codebooks=3, advantage=0.0),
        sample(rows=3, frames=0, codebooks=3, advantage=0.0),
        sample(rows=4, frames=1, codebooks=3, advantage=0.0),
    ]
    samples[0]["streams"]["rvq"]["loss_mask"][2, 1] = False
    samples[-1]["enabled"] = False
    frozen = []
    for s in samples:
        for stream in s["streams"].values():
            old = stream["old_log_probs"]
            stream["ref_log_probs"] = old + torch.linspace(-0.4, 0.5, old.numel()).reshape_as(old)
            for key in ("old_log_probs", "rollout_log_probs", "ref_log_probs"):
                frozen.append(stream[key].requires_grad_())
    # Exercise the low_var_kl clamp, as distinct from the unclamped k3 estimator.
    with torch.no_grad():
        samples[0]["streams"]["rvq"]["ref_log_probs"][2, 0] -= 20
    actual, metrics = run_policy(args, samples, padding=4)
    expected = independent_reference(args, samples)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, args.kl_loss_coef * metrics["kl_loss"])
    torch.testing.assert_close(metrics["kl_loss"], 0.6 * metrics["rvq/kl_mean"] + 1.4 * metrics["stop/kl_mean"])
    targets = [s["output"] for s in samples]
    gradients = torch.autograd.grad(actual, targets + frozen, allow_unused=True)
    expected_gradients = torch.autograd.grad(expected, targets)
    for actual_grad, expected_grad in zip(gradients[: len(targets)], expected_gradients, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)
    assert gradients[0][2, 1] == 0
    assert not gradients[2].any()
    assert all(grad is None for grad in gradients[len(targets) :])
    assert all(not metric.requires_grad for metric in metrics.values())


def test_reference_kl_requires_reference_scores_only_when_enabled():
    s = sample()
    args = arguments(kl_loss_coef=0.3)
    actual, metrics = run_policy(args, [s])
    torch.testing.assert_close(actual, independent_reference(args, [s]))
    assert "kl_loss" not in metrics
    args.use_kl_loss = True
    with pytest.raises(ValueError, match="Missing selected reference action scores for rvq"):
        run_policy(args, [s])


@pytest.mark.parametrize(
    "config",
    [
        dict(use_tis=True),
        dict(advantage_estimator="gspo"),
        dict(normalize_advantages=True),
        dict(kl_coef=0.1),
        dict(calculate_per_token_loss=True),
        dict(compute_advantages_and_returns=False),
        dict(custom_advantage_function_path="fixture.advantages"),
        dict(context_parallel_size=2),
    ],
)
def test_unadapted_token_features_fail_explicitly(config):
    with pytest.raises(ValueError):
        validate_action_policy_args(arguments(**config))


def test_entropy_requires_actual_distribution_scores():
    with pytest.raises(ValueError, match="entropy"):
        run_policy(arguments(entropy_coef=0.01), [sample()])


@pytest.mark.parametrize("stop_weight", [0.0, 0.7, 1.0, 3.0])
def test_clip_fraction_counts_real_actions_independently_of_stop_weight(stop_weight):
    s = sample(advantage=1.0)
    streams = s["streams"]
    with torch.no_grad():
        s["output"].copy_(torch.cat((streams["rvq"]["old_log_probs"], streams["stop"]["old_log_probs"][:, None]), -1))
        s["output"].add_(torch.tensor(1.5).log())
    _, metrics = run_policy(arguments(policy_loss_weights={"stop": stop_weight}), [s])
    torch.testing.assert_close(metrics["pg_clipfrac"], torch.tensor(1.0))
    torch.testing.assert_close(metrics["ppo_kl"], -torch.tensor(1.5).log())


@pytest.mark.parametrize("frames", [0, 2])
def test_removed_trajectory_has_no_code_or_terminal_stop_gradient(frames):
    s = sample(frames=frames)
    s["enabled"] = False
    value, metrics = run_policy(arguments(), [s])
    torch.testing.assert_close(value, torch.tensor(0.0))
    torch.testing.assert_close(torch.autograd.grad(value, s["output"])[0], torch.zeros_like(s["output"]))
    torch.testing.assert_close(metrics["pg_clipfrac"], torch.tensor(0.0))


def test_action_masks_control_loss_independently_of_frame_summaries():
    s = sample()
    batch, output = batch_for([s])
    batch["loss_masks"][0].zero_()
    value, _ = loss_module.policy_loss_function(arguments(), batch, output, lambda x: x.sum())
    ref = independent_reference(arguments(), [s])
    torch.testing.assert_close(value, ref)
    torch.testing.assert_close(torch.autograd.grad(value, s["output"])[0], torch.autograd.grad(ref, s["output"])[0])


@pytest.mark.parametrize("stop_weight", [0.0, 3.0])
def test_sample_max_logprob_error_averages_trajectory_maxima_with_stop_and_removal(stop_weight):
    samples = [sample(), sample(rows=3, frames=0), sample(rows=4, frames=1)]
    for s in samples:
        s["enabled"] = True
        for stream in s["streams"].values():
            stream["rollout_log_probs"] = stream["old_log_probs"].clone()
    samples[0]["streams"]["rvq"]["rollout_log_probs"][2, 0] -= 0.8
    samples[0]["streams"]["stop"]["rollout_log_probs"][-1] -= 0.2
    samples[1]["streams"]["stop"]["rollout_log_probs"][-1] -= 0.3
    samples[2]["streams"]["rvq"]["rollout_log_probs"] -= 8.0
    samples[2]["streams"]["stop"]["rollout_log_probs"] -= 9.0
    samples[2]["enabled"] = False
    args = arguments(policy_loss_weights={"stop": stop_weight})
    _, packed = run_policy(args, samples)
    metric = "mean_sample_train_rollout_logprob_max_abs_diff"
    # The loss reports a numerator. The existing Megatron reporting path
    # divides by the global sample count, after summing microbatches/DP.
    torch.testing.assert_close(packed[metric], torch.tensor(0.8 + 0.3))
    assert packed[metric] > packed["train_rollout_logprob_abs_diff"]
    micro_sum = sum(run_policy(args, [s])[1][metric] for s in samples)
    torch.testing.assert_close(packed[metric] / len(samples), micro_sum / len(samples))
    torch.testing.assert_close(run_policy(args, [samples[1]])[1][metric], torch.tensor(0.3))
    torch.testing.assert_close(run_policy(args, [samples[2]])[1][metric], torch.tensor(0.0))


def test_partial_head_loss_mask_preserves_the_other_action_gradients():
    item = sample()
    selected = item["streams"]["rvq"]["valid_mask"].nonzero()[0]
    row, head = selected.tolist()
    item["streams"]["rvq"]["loss_mask"][row, head] = False
    actual, _ = run_policy(arguments(), [item])
    expected = independent_reference(arguments(), [item])
    torch.testing.assert_close(actual, expected)
    grad = torch.autograd.grad(actual, item["output"])[0]
    assert grad[row, head] == 0
    assert item["streams"]["rvq"]["valid_mask"][row, head]
    assert grad[:, -1].abs().sum() > 0


def test_ppo_keeps_each_streams_time_and_head_axes(monkeypatch):
    native = loss_module.compute_policy_loss
    shapes = []

    def record(ppo_kl, advantages, *args, **kwargs):
        assert ppo_kl.shape == advantages.shape
        shapes.append(tuple(ppo_kl.shape))
        return native(ppo_kl, advantages, *args, **kwargs)

    monkeypatch.setattr(loss_module, "compute_policy_loss", record)
    samples = [sample(codebooks=3), sample(rows=3, frames=0, codebooks=3)]
    samples[0]["streams"]["rvq"]["loss_mask"][2, 1] = False
    args = arguments(policy_loss_weights={"stop": 0.3})
    actual, _ = run_policy(args, samples)
    expected = independent_reference(args, samples)
    assert shapes == [(5, 3), (5,), (3, 3), (3,)]
    torch.testing.assert_close(actual, expected)
    targets = [s["output"] for s in samples]
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual, targets), torch.autograd.grad(expected, targets), strict=True
    ):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_action_objective_applies_dual_clip():
    s = sample(advantage=-2.0)
    with torch.no_grad():
        for column, name in ((slice(None, -1), "rvq"), (-1, "stop")):
            s["output"][:, column].copy_(s["streams"][name]["old_log_probs"] + torch.tensor(2.0).log())
    value, metrics = run_policy(arguments(eps_clip_c=1.5, policy_loss_weights={"stop": 0.3}), [s])
    torch.testing.assert_close(value, torch.tensor(3.0 * 1.3))
    torch.testing.assert_close(metrics["rvq/pg_loss"], torch.tensor(3.0))
    torch.testing.assert_close(torch.autograd.grad(value, s["output"])[0], torch.zeros_like(s["output"]))


@pytest.mark.parametrize("use_opd", [False, True])
@pytest.mark.parametrize("use_kl_loss", [False, True])
def test_entropy_keeps_stream_weights_and_gradients(monkeypatch, use_opd, use_kl_loss):
    from slime.utils import misc

    def with_entropy(args, batch, output):
        scores = extract(args, batch, output)
        for name in ("rvq", "stop"):
            # A binary distribution with the supplied selected probability.
            scores[f"policy/{name}/entropy"] = [
                -(value.exp() * value + (1 - value.exp()) * torch.log1p(-value.exp()))
                for value in scores[f"policy/{name}/log_probs"]
            ]
        return scores

    monkeypatch.setattr(misc, "load_function", lambda path: with_entropy)
    args = arguments(
        entropy_coef=0.07,
        use_opd=use_opd,
        policy_loss_weights={"stop": 0.3},
        use_kl_loss=use_kl_loss,
        kl_loss_coef=0.25,
        kl_loss_type="k3",
    )
    samples = [sample(), sample(rows=3, frames=0), sample(rows=4, frames=1)]
    samples[-1]["enabled"] = False
    samples[0]["streams"]["rvq"]["loss_mask"][2, 0] = False
    expected_entropy = samples[0]["output"].new_zeros(())
    for s in samples:
        for column, name in ((slice(None, -1), "rvq"), (-1, "stop")):
            stream = s["streams"][name]
            stream["teacher_log_probs"] = stream["old_log_probs"] + 0.2
            stream["ref_log_probs"] = stream["old_log_probs"] - 0.3
            values = s["output"][:, column][stream["loss_mask"]]
            if values.numel():
                entropy = torch.distributions.Bernoulli(probs=values.exp()).entropy().mean()
                expected_entropy = expected_entropy + args.policy_loss_weights.get(name, 1) * entropy * s["enabled"]
    actual, metrics = run_policy(args, samples, padding=3)
    expected = independent_reference(args, samples) - args.entropy_coef * expected_entropy
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(metrics["entropy_loss"], expected_entropy)
    torch.testing.assert_close(
        metrics["entropy_loss"], metrics["rvq/entropy_mean"] + 0.3 * metrics["stop/entropy_mean"]
    )
    targets = [s["output"] for s in samples]
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual, targets), torch.autograd.grad(expected, targets), strict=True
    ):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("use_opd", [False, True])
@pytest.mark.parametrize("use_kl_loss", [False, True])
def test_nonfinite_inactive_rows_do_not_poison_loss_metrics_or_gradients(empty, use_opd, use_kl_loss):
    s = sample()
    args = arguments(
        use_opd=use_opd, use_kl_loss=use_kl_loss, kl_loss_coef=0.2, kl_loss_type="k3", use_unbiased_kl=True
    )
    for stream in s["streams"].values():
        stream["teacher_log_probs"] = stream["old_log_probs"] + 0.2
        stream["ref_log_probs"] = stream["old_log_probs"] - 0.3
        if empty:
            stream["valid_mask"].zero_()
            stream["loss_mask"].zero_()
    batch, output = batch_for([s], padding=3, args=args)
    with torch.no_grad():
        for column, name in ((slice(None, -1), "rvq"), (-1, "stop")):
            stream = batch["policy_actions"][name]
            invalid = ~stream["valid_mask"]
            output[0, : s["rows"], column].masked_fill_(invalid, float("nan"))
            for field in (
                "old_log_probs",
                "rollout_log_probs",
                "teacher_log_probs",
                "ref_log_probs",
                "advantages",
                "opd_reverse_kl",
            ):
                if field in stream:
                    stream[field].masked_fill_(invalid, float("nan"))
        output[:, s["rows"] :].fill_(float("inf"))
    actual, metrics = loss_module.policy_loss_function(args, batch, output, lambda values: values.sum())
    assert torch.isfinite(actual)
    assert all(torch.isfinite(value) for value in metrics.values())
    actual_grad = torch.autograd.grad(actual, output)[0]
    assert torch.isfinite(actual_grad).all()
    assert not actual_grad[:, s["rows"] :].any()
    if empty:
        assert actual.item() == 0
        assert not actual_grad.any()
        assert all(value == 0 for value in metrics.values())
    else:
        expected = independent_reference(args, [s])
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_grad[0, : s["rows"]], torch.autograd.grad(expected, s["output"])[0])


def test_empty_detached_scores_still_allow_backward(monkeypatch):
    from slime.utils import misc

    def empty_scores(args, batch, output):
        return {
            key: [torch.full_like(value, float("nan")) for value in values]
            for key, values in extract(args, batch, output).items()
            if key.startswith("policy/")
        }

    monkeypatch.setattr(misc, "load_function", lambda path: empty_scores)
    s = sample()
    for stream in s["streams"].values():
        stream["valid_mask"].zero_()
        stream["loss_mask"].zero_()
    value, metrics = run_policy(arguments(), [s])
    value.backward()
    assert value.item() == 0
    assert not s["output"].grad.any()
    assert all(value == 0 for value in metrics.values())


@pytest.mark.parametrize("use_kl_loss", [False, True])
def test_single_step_score_reuse_preserves_policy_gradients(use_kl_loss):
    args = arguments(use_kl_loss=use_kl_loss, kl_loss_coef=0.3, kl_loss_type="k3", use_unbiased_kl=True)
    s = sample()
    for column, name in ((slice(None, -1), "rvq"), (-1, "stop")):
        s["streams"][name]["old_log_probs"] = s["output"][:, column].detach().clone()
        s["streams"][name]["ref_log_probs"] = s["streams"][name]["old_log_probs"] - 0.2
    expected = independent_reference(args, [s])
    batch, output = batch_for([s])
    for stream in batch["policy_actions"].values():
        del stream["old_log_probs"]
    batch["reuse_current_policy_scores"] = True
    actual, _ = loss_module.policy_loss_function(args, batch, output, lambda values: values.sum())
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, s["output"])[0], torch.autograd.grad(expected, s["output"])[0]
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
