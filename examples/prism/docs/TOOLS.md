# Prism GPU 与数值诊断工具

训练使用 [run-grpo.sh](../run-grpo.sh) 和 [run-mopd.sh](../run-mopd.sh)。诊断实现统一位于仓库 `scripts/`，从仓库根目录运行；它们不参与默认训练入口。

| 工具 | 调用方式 | 用途 |
| --- | --- | --- |
| [test_prism_gpu.py](../../../scripts/test_prism_gpu.py) | `python -m scripts.test_prism_gpu --help` | 单 learner 的 native 加载、动作打分、反向与 Adam 检查 |
| [test_prism_omni.py](../../../scripts/test_prism_omni.py) | `python -m scripts.test_prism_omni --help` | 真实 Omni 协议检查；static 或 publication 模式 |
| [prism_publication_smoke.py](../../../scripts/prism_publication_smoke.py) | `scripts.prism_publication_smoke:publication` / `:verify_generated` | Omni smoke 的初始发布、更新和再次发布 hooks |
| [prism_numerical_diagnostic.py](../../../scripts/prism_numerical_diagnostic.py) | `scripts.prism_numerical_diagnostic.diagnose` | GPU smoke 的 native / HF / projection 数值对照 hooks |

完整 publication 命令、输入要求和容差说明见 [GUIDE.md](GUIDE.md#恢复与检查)。例如查看入口参数：

```bash
python -m scripts.test_prism_gpu --help
python -m scripts.test_prism_omni --help
```

需要 Megatron/Omni/CUDA 的实际检查应在相应 GPU 环境中执行。独立工具的产物使用其显式 `--artifact-dir`，不自动加入训练的 `ROUND_PATH`。它们的工程闭环不替代真实 prompt group 的 GRPO/MoPD 训练和语音质量验收。

## 配置与旧工具

学生 pipeline、教师服务绑定和奖励服务直接编辑 [configs/](../configs/) 下的四份 YAML。模型架构参数仍复用 `python -m slime_plugins.models.prism_configuration --hf-checkpoint /assets/student-hf`，无需生成另一套配置。

旧 `examples.prism.tools` 目录已合并整理：四个诊断实现移入上述 `scripts` 模块；一次性 prompt 迁移、YAML 生成和集群探测工具不再作为示例分发。统一 request 数据格式见 [GUIDE.md](GUIDE.md#jsonl-输入)，Ray 环境与启停见 [RAY.md](RAY.md)。

历史实验继续以其固定源码版本和 [VALIDATION.md](VALIDATION.md) 的记录为准。
