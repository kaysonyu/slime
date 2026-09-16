"""Exact student actions and their causal prediction positions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import MossLocalConfig


def validate_model_identity(config, identity):
    config.validate_rollout_identity(identity)


def generation_inputs(sample, temperature):
    parameters = dict(sample.metadata.get("tts_params") or {})
    for key in ("ref_audio", "ref_text", "instructions"):
        if key in sample.metadata:
            parameters[key] = sample.metadata[key]
    parameters.update(
        text_temperature=temperature,
        audio_temperature=temperature,
        text_top_p=1,
        audio_top_p=1,
        text_top_k=-1,
        audio_top_k=-1,
        audio_repetition_penalty=1,
    )
    return sample.prompt, parameters


@dataclass
class MossLocalTrajectory:
    prompt_rows: torch.Tensor
    decisions: torch.Tensor
    codes: torch.Tensor
    decision_logprobs: torch.Tensor
    code_logprobs: torch.Tensor
    finish_reason: str
    weight_version: str
    request_id: str
    sampling: dict
    model_identity: dict

    @classmethod
    def from_omni(cls, trace, meta, config: MossLocalConfig):
        if trace.get("version") != 2 or trace.get("model_family") != "moss_tts_local":
            raise ValueError("Expected MOSS Local structured rollout schema v2")
        if trace.get("logprob_semantics") != "temperature_scaled_full_vocab_v1":
            raise ValueError("Unsupported rollout probability distribution")
        streams = {s["name"]: s for s in trace["action_streams"]}
        if set(streams) != {"decision", "codes"}:
            raise ValueError("MOSS trace must contain exactly decision and codes streams")
        for stream in streams.values():
            mask = torch.as_tensor(stream["action_mask"], dtype=torch.bool)
            if not mask.all():
                raise ValueError("MOSS trace contains missing or forced actions")
        d, c = streams["decision"], streams["codes"]
        version = str(trace["admission_weight_version"])
        if meta.get("weight_version") is None or version != str(meta["weight_version"]):
            raise ValueError("Generation crossed a weight update or omitted its version")
        value = cls(
            prompt_rows=torch.as_tensor(trace["replay_inputs"]["prompt_rows"], dtype=torch.long),
            decisions=torch.as_tensor(d["actions"], dtype=torch.long),
            codes=torch.as_tensor(c["actions"], dtype=torch.long).reshape(-1, config.n_vq),
            decision_logprobs=torch.as_tensor(d["logprobs"], dtype=torch.float32),
            code_logprobs=torch.as_tensor(c["logprobs"], dtype=torch.float32).reshape(-1, config.n_vq),
            finish_reason=trace["finish_reason"],
            weight_version=version,
            request_id=trace["request_id"],
            sampling=dict(trace["sampling"]),
            model_identity=dict(trace["model_identity"]),
        )
        value.validate(config)
        return value

    def validate(self, config: MossLocalConfig):
        config.validate_rollout_identity(self.model_identity)
        if self.prompt_rows.ndim != 2 or self.prompt_rows.shape[1] != config.channels or not len(self.prompt_rows):
            raise ValueError("Invalid prompt rows")
        if self.codes.ndim != 2 or self.codes.shape[1] != config.n_vq:
            raise ValueError("Invalid RVQ code grid")
        if self.decisions.ndim != 1 or self.decisions.numel() == 0:
            raise ValueError("A trajectory must contain at least one decision")
        frames = len(self.codes)
        if self.finish_reason == "stop":
            if len(self.decisions) != frames + 1 or self.decisions[-1] != 1:
                raise ValueError("Natural termination requires T continue actions followed by a stop")
        elif self.finish_reason == "length":
            if len(self.decisions) != frames:
                raise ValueError("Length truncation cannot invent a terminal action")
        else:
            raise ValueError(f"Unsupported finish reason: {self.finish_reason}")
        if self.decisions[:frames].ne(0).any():
            raise ValueError("Emitted audio frames require continue decisions")
        if self.codes.numel() and (self.codes.min() < 0 or self.codes.max() >= config.audio_vocab_size):
            raise ValueError("Audio code outside the model vocabulary")
        if self.prompt_rows[:, 0].min() < 0 or self.prompt_rows[:, 0].max() >= config.language["vocab_size"]:
            raise ValueError("Prompt text token outside the model vocabulary")
        if self.prompt_rows[:, 1:].min() < 0 or self.prompt_rows[:, 1:].max() > config.audio_pad_id:
            raise ValueError("Prompt audio code outside the model vocabulary")
        for actions, logprobs in [(self.decisions, self.decision_logprobs), (self.codes, self.code_logprobs)]:
            if actions.shape != logprobs.shape or not torch.isfinite(logprobs).all() or (logprobs > 1e-5).any():
                raise ValueError("Selected logprobs must be finite, non-positive and action-aligned")
        for name, expected in {
            "text_top_p": 1.0,
            "audio_top_p": 1.0,
            "text_top_k": -1,
            "audio_top_k": -1,
            "audio_repetition_penalty": 1.0,
        }.items():
            if self.sampling.get(name) != expected:
                raise ValueError(f"Behavior distribution requires {name}={expected}")
        temps = [self.sampling.get("text_temperature", 0), self.sampling.get("audio_temperature", 0)]
        if any(not 0 < float(t) < float("inf") for t in temps):
            raise ValueError("Behavior temperatures must be finite and positive")
        if not self.weight_version or not self.request_id:
            raise ValueError("Trajectory requires a weight version and request identity")

    @property
    def num_actions(self):
        return self.decisions.numel() + self.codes.numel()

    @property
    def num_frames(self):
        return len(self.codes)

    @property
    def sequence_length(self):
        return len(self.prompt_rows) + len(self.codes)

    def training_tensors(self, config: MossLocalConfig):
        """Padding exists only in a physical batch; the trace has no invented audio actions."""
        self.validate(config)
        d, t = len(self.decisions), len(self.codes)
        rows = torch.cat(
            (
                self.prompt_rows,
                torch.cat(
                    (
                        self.codes.new_full((t, 1), config.audio_slot_id),
                        self.codes,
                    ),
                    dim=1,
                ),
            )
        )
        targets = self.codes.new_zeros((d, config.channels))
        targets[:, 0] = self.decisions
        targets[:t, 1:] = self.codes
        mask = torch.zeros_like(targets, dtype=torch.bool)
        mask[:, 0] = True
        mask[:t, 1:] = True
        old = torch.zeros_like(targets, dtype=torch.float32)
        old[:, 0] = self.decision_logprobs
        old[:t, 1:] = self.code_logprobs
        positions = torch.arange(len(self.prompt_rows) - 1, len(self.prompt_rows) - 1 + d)
        return rows, positions, targets, mask, old

    def score_request(self, sample_id, temperature):
        if any(self.sampling[key] != temperature for key in ("text_temperature", "audio_temperature")):
            raise ValueError("Local teacher scoring currently requires one matching decision/audio temperature")
        return dict(
            version=1,
            sample_id=str(sample_id),
            prompt_rows=self.prompt_rows.tolist(),
            decisions=self.decisions.tolist(),
            codes=self.codes.tolist(),
            temperature=temperature,
        )


def batch_inputs(samples, config):
    return (
        [sample.trajectory.training_tensors(config) for sample in samples],
        [
            [sample.trajectory.sampling["text_temperature"]]
            + [sample.trajectory.sampling["audio_temperature"]] * config.n_vq
            for sample in samples
        ],
        torch.tensor([config.text_pad_id] + [config.audio_pad_id] * config.n_vq),
        ("decision", *(f"codebook_{index}" for index in range(config.n_vq))),
    )


def teacher_scores(trajectory, response, config):
    config.validate_rollout_identity(response["model_identity"])
    _, _, targets, mask, _ = trajectory.training_tensors(config)
    decisions = torch.as_tensor(response["decision_logprobs"], dtype=torch.float32)
    codes = torch.as_tensor(response["code_logprobs"], dtype=torch.float32).reshape(-1, config.n_vq)
    if decisions.shape != trajectory.decisions.shape or codes.shape != trajectory.codes.shape:
        raise ValueError("Teacher scores do not cover the student's original actions")
    scores = torch.zeros_like(targets, dtype=torch.float32)
    scores[:, 0] = decisions
    scores[: trajectory.num_frames, 1:] = codes
    if not torch.isfinite(scores[mask]).all() or (scores[mask] > 1e-5).any():
        raise ValueError("Teacher returned invalid selected-action probabilities")
    return scores
