"""MOSS diagnostics use the same equal-utterance averaging as training."""


def rollout_metrics(samples):
    count = len(samples)
    values = {"model/decision/nll": sum(float(-s.trajectory.decision_logprobs.mean()) for s in samples) / count}
    channels = samples[0].trajectory.codes.shape[1]
    for depth in range(channels):
        values[f"model/codebook_{depth}/nll"] = (
            sum(
                float(-s.trajectory.code_logprobs[:, depth].mean()) if s.trajectory.num_frames else 0.0
                for s in samples
            )
            / count
        )
    return values
