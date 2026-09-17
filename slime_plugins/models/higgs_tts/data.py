"""Higgs delayed-code trajectories using the common source-position CP contract."""

import hashlib
import json
from dataclasses import dataclass

import torch


def validate_model_identity(config, identity):
    expected = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if identity.get("config_sha256") != expected:
        raise ValueError("Higgs rollout does not match the checkpoint configuration")


def generation_inputs(sample, temperature):
    if sample.metadata.get("instructions"):
        raise ValueError("Higgs conditioning uses text/reference audio; instructions are not supported")
    inputs = {"text": sample.prompt}
    for source, target in (("ref_audio", "reference_audio"), ("ref_text", "reference_text")):
        if source in sample.metadata:
            inputs[target] = sample.metadata[source]
    return inputs, {}


@dataclass
class HiggsTrajectory:
    prompt_ids: torch.Tensor
    reference_codes: torch.Tensor
    codes: torch.Tensor
    logprobs: torch.Tensor
    sampled_mask: torch.Tensor
    audio_mask: torch.Tensor
    sampling: dict
    weight_version: str
    finish_reason: str
    request_id: str
    model_identity: dict

    @classmethod
    def from_omni(cls, trace, meta, config):
        if trace.get("model_family") != "higgs_tts" or trace.get("version") != 2:
            raise ValueError("Higgs requires its versioned exact-input trajectory")
        validate_model_identity(config, trace["model_identity"])
        if trace.get("logprob_semantics") != "temperature_scaled_full_vocab_v1":
            raise ValueError("Unknown Higgs probability semantics")
        (stream,) = trace["action_streams"]
        if "sampled_action_mask" not in stream:
            raise ValueError("Higgs RL requires actual sampled actions, including sampled termination")
        n = config["audio_encoder_config"]["num_codebooks"]
        inputs = trace["replay_inputs"]
        value = cls(
            prompt_ids=torch.as_tensor(inputs["prompt_token_ids"], dtype=torch.long),
            reference_codes=torch.as_tensor(inputs.get("reference_codes_delayed") or [], dtype=torch.long).reshape(
                -1, n
            ),
            codes=torch.as_tensor(stream["actions"], dtype=torch.long).reshape(-1, n),
            logprobs=torch.as_tensor(stream["logprobs"], dtype=torch.float32).reshape(-1, n),
            sampled_mask=torch.as_tensor(stream["sampled_action_mask"], dtype=torch.bool).reshape(-1, n),
            audio_mask=torch.as_tensor(stream["action_mask"], dtype=torch.bool).reshape(-1, n),
            sampling=trace["sampling"],
            weight_version=str(trace["admission_weight_version"]),
            finish_reason=trace["finish_reason"],
            request_id=trace["request_id"],
            model_identity=trace["model_identity"],
        )
        if value.weight_version != str(meta.get("weight_version")):
            raise ValueError("Higgs trajectory crossed a weight publication")
        value.validate(config)
        return value

    def validate(self, config):
        vocab = config["audio_encoder_config"]["vocab_size"]
        if self.codes.ndim != 2 or not len(self.codes) or not len(self.prompt_ids):
            raise ValueError("Empty or malformed Higgs trajectory")
        if not (self.codes.shape == self.logprobs.shape == self.sampled_mask.shape == self.audio_mask.shape):
            raise ValueError("Higgs code, score and mask shapes differ")
        expected = torch.arange(len(self.codes))[:, None] >= torch.arange(self.codes.shape[1])[None]
        if not torch.equal(self.sampled_mask, expected):
            raise ValueError("Sampled mask does not follow Higgs delay initialization")
        if self.codes.min() < 0 or self.codes.max() >= vocab:
            raise ValueError("Higgs code is outside its vocabulary")
        if (
            not torch.isfinite(self.logprobs[self.sampled_mask]).all()
            or (self.logprobs[self.sampled_mask] > 1e-5).any()
        ):
            raise ValueError("Higgs has invalid behavior probabilities")
        if len(self.reference_codes) != int(self.prompt_ids.eq(-100).sum()):
            raise ValueError("Reference codes do not match the prompt audio slots")
        if self.finish_reason not in {"stop", "length"}:
            raise ValueError("Unsupported Higgs termination")
        if self.sampling.get("top_k") != -1 or self.sampling.get("top_p") != 1:
            raise ValueError("Higgs RL requires unfiltered behavior sampling")
        if not 0 < self.sampling.get("temperature", 0) < float("inf"):
            raise ValueError("Higgs RL requires a finite positive temperature")

    @property
    def sequence_length(self):
        return len(self.prompt_ids) + len(self.codes) - 1

    @property
    def num_frames(self):
        return int(self.audio_mask[:, 0].sum())

    @property
    def num_actions(self):
        return int(self.sampled_mask.sum())

    def training_tensors(self, config):
        self.validate(config)
        rows = torch.full((self.sequence_length, self.codes.shape[1] + 1), -1, dtype=torch.long)
        rows[: len(self.prompt_ids), 0] = self.prompt_ids
        references = self.prompt_ids.eq(-100).nonzero().flatten()
        rows[references, 1:] = self.reference_codes
        rows[len(self.prompt_ids) :, 0] = -100
        rows[len(self.prompt_ids) :, 1:] = self.codes[:-1]
        positions = torch.arange(len(self.prompt_ids) - 1, self.sequence_length)
        return rows, positions, self.codes, self.sampled_mask, self.logprobs

    def score_request(self, sample_id, temperature):
        if temperature != self.sampling["temperature"]:
            raise ValueError("Teacher scoring must use the student's probability temperature")
        return dict(
            version=1,
            sample_id=str(sample_id),
            prompt_token_ids=self.prompt_ids.tolist(),
            reference_codes_delayed=self.reference_codes.tolist(),
            codes=self.codes.tolist(),
            temperature=temperature,
        )


def batch_inputs(samples, config):
    n = config["audio_encoder_config"]["num_codebooks"]
    pad = config["text_config"].get("pad_token_id") or 0
    return (
        [s.trajectory.training_tensors(config) for s in samples],
        [[s.trajectory.sampling["temperature"]] * n for s in samples],
        torch.tensor([pad] + [-1] * n),
        tuple(f"codebook_{index}" for index in range(n)),
    )


def teacher_scores(trajectory, response, config):
    if response["model_identity"] != trajectory.model_identity:
        raise ValueError("Higgs teacher uses a different action/model configuration")
    scores = torch.as_tensor(response["code_logprobs"], dtype=torch.float32)
    if scores.shape != trajectory.codes.shape:
        raise ValueError("Higgs teacher scores are not aligned with the original actions")
    values = scores[trajectory.sampled_mask]
    if not torch.isfinite(values).all() or (values > 1e-5).any():
        raise ValueError("Higgs teacher returned invalid selected-action probabilities")
    return scores
