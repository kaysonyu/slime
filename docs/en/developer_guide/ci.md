# Continuous Integration

This TTS cleanup baseline preserves the v0.3.2 training core. Model-specific
plugins, launch recipes and their dedicated end-to-end tests have been removed.
TTS integration tests will be added with the corresponding feature PRs.

## Retained checks

| Trigger | Job | Coverage |
| --- | --- | --- |
| PR / push to `main` or `tts-rl`, manual dispatch | `cpu-unittest` | Argument validation, DP/CP, loss algebra, metrics, rewards, samples, rollout/configuration, checkpoint utilities and plugin contracts |
| `run-ci-sglang-config` | `e2e-test-sglang-config` | Two CPU SGLang argument/configuration tests in the self-hosted container |
| `run-ci-megatron` | `e2e-test-megatron` | Generic CUDA log-probability/entropy parity test, 2 GPUs |
| `run-ci-image` | `e2e-test-image` | The same numeric test in `slimerl/slime-test:latest` |
| `run-ci-changed` | `e2e-test-changed` | Added/modified tests, using each file's `NUM_GPUS` |

CPU jobs run on GitHub-hosted runners and do not acquire GPUs. GPU jobs remain
label-gated or manually dispatched. The `run-ci-precision`, `run-ci-ckpt`, agent
and model-specific Conda smoke jobs were removed with their associated recipes.
A passing numeric test does not establish end-to-end training correctness.

## Running and adding tests

```bash
PYTHONPATH=. python tests/test_docs_consistency.py
PYTHONPATH=. python tests/test_dp_schedule.py
PYTHONPATH=. python tests/plugin_contracts/test_plugin_generate_contracts.py
```

CPU test files should declare `NUM_GPUS = 0` and run directly:

```python
if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
```

GPU tests declare their actual GPU requirement and use
`tests/ci/gpu_lock_exec.py` through CI. The removed model-specific
`slime.utils.external_utils.command_utils` launcher is no longer available.
Use the existing generic numeric test as a reference for standalone CUDA tests;
new TTS end-to-end tests should accompany their implementation.

## Workflow generation

Edit `.github/workflows/pr-test.yml.j2`, then regenerate and commit both files:

```bash
python .github/workflows/generate_github_workflows.py
```

The changed-test job currently compares against `origin/main` and defaults to
8 GPUs when a test omits `NUM_GPUS`. Use explicit matrix registration for checks
that should run on every PR. Historical model E2E recipes remain available in
[upstream v0.3.2](https://github.com/THUDM/slime/tree/v0.3.2/tests).
