# Examples

These examples provide concrete examples to leverage slime in your own RL workflow. Some examples are just demonstrative, but most of them are verifiable with a concrete performance score.

## Directory Structure

- **[eval_multi_task](./eval_multi_task)**: Example for supporting evaluation multiple tasks with different configs.
- **[fully_async](./fully_async)**: Demonstrates fully asynchronous rollout generation for higher efficiency.
- **[low_precision](../scripts/low_precision/)**: Launch recipes for FP8/INT4 training and inference.
- **[on_policy_distillation](./on_policy_distillation)**: Example implementation for on-policy distillation, extending the reinforcement learning pipeline to support teacher–student distillation directly within on-policy training.
- **[delta_weight_sync](./delta_weight_sync)**: Non-colocated weight sync that ships only the changed bytes over a shared filesystem (training/inference disaggregation), reloading via the vanilla `update_weights_from_disk` path.
- **[reproducibility](../docs/en/advanced/reproducibility.md)**: Guide to bitwise experiment reproduction using deterministic modes.
- **[train_infer_mismatch_helper](./train_infer_mismatch_helper)**: Algorithmic methods for rollout correction (e.g., TIS, MIS).
