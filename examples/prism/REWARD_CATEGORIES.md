---
orphan: true
---

# Prism 样本奖励类别

`examples.prism.rewards.reward` 是这套示例自己的奖励入口。每个类别在
`examples/prism/rewards/` 中有一个文件，组合公式直接写在类别代码里。
`__init__.py` 的 `CATEGORY_REWARDS` 根据样本选择文件；`common.py` 复用
`slime_plugins.tts_reward.RewardScorer` 的 WER、SIM、JudgeCLAP 基础评分，
管理共享缓存、批量调用和结果记录。domain、Prism task type 与奖励类别分别保留。

| `metadata.reward_category` | 文件 | 当前组合公式 |
| --- | --- | --- |
| `base` | `rewards/base.py` | 加权 WER/CER reward |
| `voiceclone` | `rewards/voiceclone.py` | WER/CER reward 与 SIM 加权和 |
| `instruction` | `rewards/instruction.py` | WER/CER reward 与 JudgeCLAP 加权和 |
| `voiceclone_gated` | `rewards/voiceclone_gated.py` | 原始 WER/CER ≤ 0.15 时取 SIM，否则 0 |

统一入口要求每条 prompt 显式填写类别，未知或缺失类别会报错。每个 GRPO prompt group
的候选继承同一类别；批量调用也检查这一约束。
类别不会根据 `domain` 或 `task_type` 自动推断。

## 配置与启动

沿用现有 `REWARD_CONFIG` 的 `services` 和 `weights` 格式。前三个类别保持原来的
加权和语义与正系数检查。也可以只提供 `services`；此时这三个类别的系数默认为 1。
非线性类别自行决定如何组合分量，`voiceclone_gated` 不使用配置里的线性系数。
每个类别只调用自己需要的服务；不同类别共享服务的 fingerprint 缓存、并发上限和
SIM 参考 embedding 缓存。

例如，用现有配置工具将三个系数显式设为 `1.0`：

```bash
"$TRAIN_PYTHON" -m examples.prism.configure \
  --student "$HF_CHECKPOINT" --codec "$CODEC_MODEL" --output "$RUN_CONFIG" \
  --reward wer http://asr-host:8101 1.0 \
  --reward sim http://sim-host:8102 1.0 \
  --reward judge_clap http://judge-host:8103 1.0

export OMNI_CONFIG="$RUN_CONFIG/student.yaml"
export REWARD_CONFIG="$RUN_CONFIG/rewards.yaml"
export PROMPT_DATA=/data/prism-category-prompts.jsonl
bash examples/prism/run-grpo.sh \
  --custom-rm-path examples.prism.rewards.reward
```

此时三类分别得到 `WER_reward`、`WER_reward + SIM_reward`、
`WER_reward + JudgeCLAP_reward`。WER/CER 分量沿用 `1 / (1 + error_rate)`；SIM
沿用非负 cosine similarity；JudgeCLAP 沿用全局属性分数平均。修改 `weights` 可以
调整这三个类别的分量系数，但其要求的分量不能通过零权重关闭。reward 总和不再额外归一化，
后续组优势仍由原 GRPO 流程计算。

将样本的 `metadata.reward_category` 改为 `voiceclone_gated` 就会选择门控示例。
`0.15` 是文件中的示例实验阈值；可以直接修改 `MAX_ERROR_RATE`，或另建一个类别文件。
门控、乘积、最小值等组合都在评分之后、GRPO 组内归一化之前执行。同一 prompt group
的所有候选应使用同一类别。全组都未通过门控时，组优势为零。

单独使用一个类别文件时，可以直接指定
`--custom-rm-path examples.prism.rewards.voiceclone_gated.reward`。
原 `examples.prism.rewards.reward_base`、`reward_voiceclone`、`reward_instruction` 入口保留。

## 添加或修改类别

复制一个类别文件，在 `reward()` 中声明需要的基础分量，在 `aggregate()` 中写公式。
例如，下面的类别使用 WER reward 和 SIM 的几何平均：

```python
from .common import score


def aggregate(sample, components):
    wer = components["wer"]["transformed"]
    sim = components["sim"]["transformed"]
    return (wer * sim) ** 0.5


async def reward(args, sample, **kwargs):
    return await score(
        args, sample, category="voiceclone_product",
        components=("wer", "sim"), aggregate=aggregate,
    )
```

保存为 `rewards/voiceclone_product.py`，在 `__init__.py` 导入它的 `reward` 并加入
`CATEGORY_REWARDS["voiceclone_product"]`；也可以直接通过其完整函数路径使用。
增加类别不需要更改基础评分客户端或 Megatron loss。

`aggregate(sample, components)` 是同步函数，返回有限标量。`components` 的每项
包含 `raw`、`transformed`、`status`、`identity` 和 `details`，所以可以用原始 WER
作阈值，或读取 `judge_clap.details.attribute_scores` 改变属性间的归约。
当前变换为：WER/CER `1 / (1 + error_rate)`、SIM 非负 cosine、JudgeCLAP 属性均分。
技术故障由评分客户端抛出；`invalid_audio` 是需要类别处理的明确评分结果。

参见 `category-prompts.example.jsonl`，填写实际参考音频路径后作为数据模板使用。
每行填写 `metadata.reward_category`，其他 Prism 输入沿用已有协议：

- `base`：需要 `transcript_target` 和 `language`，不需要 SIM 参考或 JudgeCLAP 属性。
- `voiceclone`：还需要有效的音色参考。示例显式填写 Full VoiceClone 的完整参考属性映射。
- `instruction`：还需要可被 JudgeCLAP 评分的全局属性指令；该类别不会调用 SIM。

这些输入会严格验证，技术故障继续抛错。合法零帧 stop 的所选奖励分量均记录为明确的
`zero_frame` 结果，reward 为零。分类路由同时支持单样本和混合类别的样本列表；
列表返回值保持输入顺序，并等待已启动的评分完成后传播故障。

## 记录与检查

评分结果继续写入 `metadata.omni_reward`，包含选中分量的 raw/transformed 值、服务
身份和 total，并补充 `category`；加权类别还记录实际系数，门控示例记录阈值与通过状态。
样本自身的 domain 和 task type 保持其原有语义。组合公式由运行时的源码版本固定。

控制类别的训练占比时，使用采样比例或明确的训练权重。开启 GRPO 标准差归一化后，
将某类别的整个 reward 乘以正常数，其整体尺度通常会被组内归一化抵消。

```bash
"$TRAIN_PYTHON" -m pytest \
  tests/test_prism_reward_categories.py tests/test_prism_reward_composition.py \
  tests/test_tts_reward_plugin.py tests/test_grouped_rewards.py -q
```

测试使用冻结评分协议的 CPU 替身，覆盖分量选择、混合批次、共享并发上限、配置与
输入验证、零帧 stop 和故障传播，并验证门控能改变候选排序而后续原 GRPO 分组保持一致。
实际混合任务的 GPU 训练与语音质量收益需另行验收。
