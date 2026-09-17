# 持续集成

当前 TTS 精简基线保留 v0.3.2 训练核心，移除了模型专用插件、启动示例及对应端到端测试。
TTS 集成测试将在功能 PR 中一起加入。

## 保留的检查

| 触发方式 | Job | 覆盖范围 |
| --- | --- | --- |
| 面向 `main` / `tts-rl` 的 PR、分支 push、手工触发 | `cpu-unittest` | 参数、DP/CP、loss 数值、指标、奖励、Sample、rollout/config、checkpoint 工具与 plugin contracts |
| `run-ci-sglang-config` | `e2e-test-sglang-config` | 在自托管容器内运行两个 CPU SGLang 参数/config 测试 |
| `run-ci-megatron` | `e2e-test-megatron` | 通用 CUDA log-probability/entropy 数值测试，2 张 GPU |
| `run-ci-image` | `e2e-test-image` | 在 `slimerl/slime-test:latest` 中运行同一数值测试 |
| `run-ci-changed` | `e2e-test-changed` | 新增/修改的测试，资源取自文件的 `NUM_GPUS` |

CPU jobs 使用 GitHub-hosted runner，不申请 GPU。GPU jobs 仍由 label 或手工触发。
`run-ci-precision`、`run-ci-ckpt`、agent 和模型专用 Conda smoke jobs 已随相关示例移除。
通用数值测试通过不代表完整训练闭环已验证。

## 本地运行与添加测试

```bash
PYTHONPATH=. python tests/test_docs_consistency.py
PYTHONPATH=. python tests/test_dp_schedule.py
PYTHONPATH=. python tests/plugin_contracts/test_plugin_generate_contracts.py
```

CPU 测试应声明 `NUM_GPUS = 0`，并支持直接运行：

```python
if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
```

GPU 测试声明真实卡数，由 CI 通过 `tests/ci/gpu_lock_exec.py` 运行。
原模型示例使用的 `slime.utils.external_utils.command_utils` 已删除。
独立 CUDA 测试可参考保留的通用数值测试，TTS 端到端测试与后续实现一并加入。

## 生成工作流

修改 `.github/workflows/pr-test.yml.j2` 后运行生成器，并同时提交模板和生成文件：

```bash
python .github/workflows/generate_github_workflows.py
```

Changed-test job 目前仍相对 `origin/main` 检查，未声明 `NUM_GPUS` 时默认申请 8 卡。
每个 PR 都应运行的检查需要显式加入固定 matrix。
历史模型端到端示例可在 [上游 v0.3.2](https://github.com/THUDM/slime/tree/v0.3.2/tests) 查看。
