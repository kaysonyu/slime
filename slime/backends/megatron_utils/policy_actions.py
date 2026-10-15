"""Map model-specific selected actions onto the existing policy-loss operators.

Chronological token rows remain the packing axis. This view only selects and
weights real actions for the objective; it never changes response_lengths.
"""

import math
from argparse import Namespace
from dataclasses import dataclass
from functools import cached_property
from urllib.parse import quote

import torch


def uses_custom_policy_scores(args: Namespace) -> bool:
    return bool(getattr(args, "custom_megatron_log_prob_function_path", None))


def validate_action_policy_args(args: Namespace) -> None:
    """Reject combinations whose token assumptions have not been adapted."""
    unsupported = {
        "calculate_per_token_loss": False,
        "normalize_advantages": False,
        "use_critic": False,
        "use_opsm": False,
        "use_tis": False,
        "get_mismatch_metrics": False,
        "use_score_centering": False,
        "use_kl_loss": False,
        "kl_coef": 0,
        "custom_pg_loss_reducer_function_path": None,
        "custom_advantage_function_path": None,
        "compute_advantages_and_returns": True,
        "opd_full_vocab": False,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "sequence_parallel": False,
    }
    for key, expected in unsupported.items():
        value = getattr(args, key, expected)
        if value != expected:
            raise ValueError(f"Selected-action policy scoring requires {key}={expected!r}; got {value!r}")
    if args.advantage_estimator != "grpo" or args.loss_type != "policy_loss":
        raise ValueError("Selected-action scoring currently supports native GRPO/OPD with policy_loss")
    if getattr(args, "pg_loss_type", None) not in (None, "ppo"):
        raise ValueError("Selected-action scoring currently supports PPO clipping")
    stop_weight = getattr(args, "policy_stop_loss_weight", 1.0)
    if not math.isfinite(stop_weight) or stop_weight < 0:
        raise ValueError("policy_stop_loss_weight must be finite and nonnegative")


def get_custom_policy_scores(args: Namespace, batch: dict, output: torch.Tensor) -> dict:
    """The same extractor is used by frozen-old scoring and training forwards.

    The callback ``(args, batch, output)`` returns lists of chronological
    ``policy_code_logprobs[S,Q]`` / ``policy_stop_logprobs[S]`` and the existing
    frame-summary ``log_probs[F]``. Optional entropy uses the analogous names.
    """
    from slime.utils.misc import load_function

    validate_action_policy_args(args)
    scores = load_function(args.custom_megatron_log_prob_function_path)(args, batch, output)
    if not isinstance(scores, dict):
        raise TypeError("Custom policy score extractor must return a dict of per-sample tensor lists")
    for key in ("log_probs", "policy_code_logprobs", "policy_stop_logprobs"):
        values = scores.get(key)
        if not isinstance(values, list) or len(values) != len(batch["total_lengths"]):
            raise ValueError(f"Custom policy scores {key} must contain one tensor per sample")
    return scores


@dataclass
class PolicyActionView:
    current: list[torch.Tensor]
    old: list[torch.Tensor]
    advantages: list[torch.Tensor]
    rollout: list[torch.Tensor] | None
    teacher: list[torch.Tensor] | None
    entropy: list[torch.Tensor] | None
    weights: list[torch.Tensor]
    sample_masks: list[torch.Tensor]
    code_counts: list[int]

    @cached_property
    def metric_segment_lengths(self) -> torch.Tensor:
        return torch.tensor(
            [
                length
                for count, weight in zip(self.code_counts, self.weights, strict=True)
                for length in (count, weight.numel() - count)
            ],
            dtype=torch.int64,
            device=self.weights[0].device,
        )

    @cached_property
    def metric_sample_masks(self) -> torch.Tensor:
        return torch.stack(self.sample_masks)

    def split_metrics(self, values: torch.Tensor, name: str) -> dict[str, torch.Tensor]:
        """Detached sums of per-sample RVQ/stop means, before DP reduction.

        The reporter divides diagnostics by active sample counts; pg_loss keeps
        the global-batch denominator so its two components reconstruct the loss.
        """
        lengths = self.metric_segment_lengths
        # Segment lengths come from this view; unsafe avoids a device validation sync.
        means = torch.segment_reduce(values.detach(), "mean", lengths=lengths, unsafe=True)
        means = torch.where(lengths > 0, means, 0).reshape(-1, 2)
        totals = (means * self.metric_sample_masks[:, None]).sum(dim=0)
        return {f"{kind}/{name}": total for kind, total in zip(("rvq", "stop"), totals, strict=True)}

    def reduce(self, values: torch.Tensor) -> torch.Tensor:
        """Sum sample means: mean(RVQ) + stop_weight * mean(stop)."""
        if values.ndim != 1 or values.numel() != sum(w.numel() for w in self.weights):
            raise ValueError("Action reducer received a tensor outside its selected-action view")
        chunks = values.split([w.numel() for w in self.weights])
        return sum((chunk * weight).sum() for chunk, weight in zip(chunks, self.weights, strict=True))

    def reduce_metrics(self, values: torch.Tensor) -> torch.Tensor:
        """Diagnostics average real actions without objective coefficients."""
        if values.ndim != 1 or values.numel() != sum(w.numel() for w in self.weights):
            raise ValueError("Action metric received a tensor outside its selected-action view")
        chunks = values.split([w.numel() for w in self.weights])
        return sum(chunk.mean() * enabled for chunk, enabled in zip(chunks, self.sample_masks, strict=True))

    def reduce_sample_max(self, values: torch.Tensor) -> torch.Tensor:
        """Sum trajectory maxima for the native sample/microbatch/DP average."""
        if values.ndim != 1 or values.numel() != sum(w.numel() for w in self.weights):
            raise ValueError("Action maximum metric received a tensor outside its selected-action view")
        chunks = values.split([w.numel() for w in self.weights])
        return sum(chunk.max() * enabled for chunk, enabled in zip(chunks, self.sample_masks, strict=True))


def _select_frozen_scores(mm, prefix, section, code_mask, stop_mask):
    code_key, stop_key = f"policy_{prefix}_code_logprobs", f"policy_{prefix}_stop_logprobs"
    if code_key not in mm or stop_key not in mm:
        raise ValueError(f"Missing selected {prefix} action scores; old scores must be frozen before training")
    code, stop = mm[code_key][section], mm[stop_key][section]
    if code.shape != code_mask.shape or stop.shape != stop_mask.shape:
        raise ValueError(f"{prefix} action scores are not aligned to the chronological carrier")
    return torch.cat((code[code_mask], stop[stop_mask])).detach()


def build_policy_action_view(args: Namespace, batch: dict, scores: dict) -> PolicyActionView:
    """Select identical action positions in old/current/teacher distributions."""
    mm = batch["multimodal_train_inputs"]
    if not isinstance(mm, dict):
        raise ValueError("Selected-action scoring requires multimodal_train_inputs")
    count = len(batch["total_lengths"])
    scalars = mm.get("policy_group_advantage")
    if not isinstance(scalars, torch.Tensor) or scalars.shape != (count,):
        raise ValueError("policy_group_advantage must contain one scalar per sample, including zero-frame samples")
    source = "rollout" if args.use_rollout_logprobs else "old"
    has_rollout = "policy_rollout_code_logprobs" in mm and "policy_rollout_stop_logprobs" in mm
    has_entropy = "policy_code_entropy" in scores and "policy_stop_entropy" in scores
    if getattr(args, "entropy_coef", 0) and not has_entropy:
        raise ValueError("Nonzero entropy_coef requires actual per-action entropy from the model")
    sample_masks = mm.get("policy_sample_mask", scalars.new_ones(count))
    if sample_masks.shape != (count,) or bool(((sample_masks != 0) & (sample_masks != 1)).any()):
        raise ValueError("policy_sample_mask must contain one binary value per trajectory")
    view = PolicyActionView(
        [], [], [], [] if has_rollout else None, [] if args.use_opd else None, [] if has_entropy else None, [], [], []
    )
    offset = 0
    for i, rows in enumerate(batch["total_lengths"]):
        code = scores["policy_code_logprobs"][i]
        stop = scores["policy_stop_logprobs"][i]
        if code.ndim != 2 or code.shape[0] != rows or stop.shape != (rows,):
            raise ValueError("Extracted action scores must retain each sample's chronological rows")
        section = slice(offset, offset + rows)
        code_mask = mm["policy_code_mask"][section]
        stop_mask = mm["policy_stop_mask"][section]
        if code_mask.shape != code.shape or stop_mask.shape != stop.shape:
            raise ValueError("Selected-action masks and chronological scores have different shapes")
        if code_mask.dtype != torch.bool or stop_mask.dtype != torch.bool:
            raise ValueError("Selected-action masks must be boolean")
        selected_code = code[code_mask]
        selected_stop = stop[stop_mask]
        if selected_stop.numel() == 0:
            raise ValueError("Every policy trajectory must contain a real sampled stop/continue action")
        selected = torch.cat((selected_code, selected_stop))
        frame_mask = batch["loss_masks"][i]
        if frame_mask.shape != (batch["response_lengths"][i],):
            raise ValueError("Frame loss mask must retain the original response length")
        enabled = sample_masks[i].detach().to(dtype=selected.dtype)
        if getattr(args, "opd_mask_truncated", False):
            if not args.use_opd:
                raise ValueError("opd_mask_truncated requires OPD")
            enabled = enabled * (1 - batch["truncated"][i])
        if frame_mask.numel():
            if bool((frame_mask == 0).all()):
                enabled = enabled * 0
            elif not bool((frame_mask == 1).all()):
                raise ValueError("Partial frame loss masks require an explicit action-mask policy")

        view.current.append(selected)
        view.old.append(_select_frozen_scores(mm, source, section, code_mask, stop_mask))
        view.advantages.append(scalars[i].detach().expand_as(selected))
        if view.rollout is not None:
            view.rollout.append(_select_frozen_scores(mm, "rollout", section, code_mask, stop_mask))
        if view.teacher is not None:
            view.teacher.append(_select_frozen_scores(mm, "teacher", section, code_mask, stop_mask))
        if view.entropy is not None:
            view.entropy.append(
                torch.cat((scores["policy_code_entropy"][i][code_mask], scores["policy_stop_entropy"][i][stop_mask]))
            )
        code_weight = selected_code.new_full(selected_code.shape, 1 / max(selected_code.numel(), 1))
        stop_weight = selected_stop.new_full(
            selected_stop.shape, getattr(args, "policy_stop_loss_weight", 1.0) / selected_stop.numel()
        )
        view.weights.append(torch.cat((code_weight, stop_weight)) * enabled)
        view.sample_masks.append(enabled)
        view.code_counts.append(selected_code.numel())
        offset += rows
    return view


def grouped_opd_metrics(args: Namespace, rollout_data: dict) -> dict[str, tuple[float, float]]:
    """Group the actual frozen training/teacher scores without another forward.

    Each value is (sum of sample means, active sample count), so missing domains
    on a DP rank never become zero-valued samples. No per-sample device sync.
    """
    if not args.use_opd:
        return {}
    source = "rollout" if args.use_rollout_logprobs else "old"
    inputs = rollout_data["multimodal_train_inputs"]
    domains = rollout_data.get("policy_domains", [None] * len(inputs))
    teachers = rollout_data.get("policy_teachers", [None] * len(inputs))
    totals = {}
    counts = {}
    for mm, domain, teacher, frame_mask in zip(inputs, domains, teachers, rollout_data["loss_masks"], strict=True):
        if domain is None and teacher is None:
            continue
        reference = mm[f"policy_{source}_stop_logprobs"]
        enabled = mm.get("policy_sample_mask", reference.new_ones(1)).reshape(()).to(reference)
        if frame_mask.numel():
            enabled = enabled * frame_mask.any()
        for name, kind in (("rvq", "code"), ("stop", "stop")):
            mask = mm[f"policy_{kind}_mask"]
            old = mm[f"policy_{source}_{kind}_logprobs"][mask].detach()
            if old.numel() == 0:
                continue
            teacher_scores = mm[f"policy_teacher_{kind}_logprobs"][mask].detach()
            value = (old - teacher_scores).mean() * enabled
            for field, label in (("domain", domain), ("teacher", teacher)):
                if label is None:
                    continue
                key = f"prism/{field}/{quote(str(label), safe='')}/{name}/opd_logratio_mean"
                totals[key] = totals.get(key, 0) + value
                counts[key] = counts.get(key, 0) + enabled
    if not totals:
        return {}
    keys = sorted(totals)
    values = (
        torch.stack([torch.stack((totals[key], counts[key].to(totals[key].dtype))) for key in keys]).cpu().tolist()
    )
    return {key: (total, count) for key, (total, count) in zip(keys, values, strict=True) if count > 0}


def store_old_action_scores(rollout_data: dict, scores: dict, prefix: str = "old") -> None:
    """Persist frozen per-row scores in the existing multimodal payload."""
    for mm, code, stop in zip(
        rollout_data["multimodal_train_inputs"],
        scores["policy_code_logprobs"],
        scores["policy_stop_logprobs"],
        strict=True,
    ):
        mm[f"policy_{prefix}_code_logprobs"] = code.detach().clone()
        mm[f"policy_{prefix}_stop_logprobs"] = stop.detach().clone()
