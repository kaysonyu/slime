"""Higgs sampled-head diagnostics with equal utterance weights."""


def rollout_metrics(samples):
    values = {}
    for head in range(samples[0].trajectory.codes.shape[1]):
        total = 0.0
        for sample in samples:
            trace = sample.trajectory
            active = trace.sampled_mask[:, head]
            if active.any():
                total += float(-trace.logprobs[:, head][active].mean())
        values[f"model/codebook_{head}/nll"] = total / len(samples)
    return values
