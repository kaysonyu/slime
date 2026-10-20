---
orphan: true
---

# Prism 样本奖励类别

完整流程与文件入口见[奖励使用指南](../rewards/README.md)。
本页专门说明组合公式、类别路由和样本输入；组件算法示例见
[`scorers.py`](../rewards/scorers.py)，训练过滤示例见 [`filters.py`](../rewards/filters.py)。

`examples.prism.rewards.reward` 是 MixRL 的奖励入口。各类别的公式分别放在
[rewards/](../rewards/) 的独立文件中，`__init__.py` 的 `CATEGORY_REWARDS` 按 `metadata.reward_category` 选择函数。
`common.py` 复用 `slime_plugins.prism.reward_adapter.PrismRewardAdapter` 的 WER、SIM、口音复用、韵律及 JudgeCLAP 基础评分，
复用共享客户端状态，处理批量调用和结果记录。domain、Prism request 与奖励类别分别保留。

| `metadata.reward_category` | 文件 | 当前组合公式 |
| --- | --- | --- |
| `base` | [`base.py`](../rewards/base.py) | 加权 WER/CER reward |
| `voiceclone` | [`voiceclone.py`](../rewards/voiceclone.py) | WER/CER、SIM 与非 none 的 accent / prosody 加权和；emotion 暂不计入 |
| `instruction` | [`instruction.py`](../rewards/instruction.py) | WER/CER reward 与 JudgeCLAP 加权和 |
| `voiceclone_gated` | [`voiceclone_gated.py`](../rewards/voiceclone_gated.py) | 原始 WER/CER ≤ 0.15 时取 SIM，否则 0 |

统一入口要求每条 prompt 显式填写类别，未知或缺失类别会报错。每个 GRPO prompt group
的候选继承同一类别；批量调用也检查这一约束。
类别不会根据 `domain` 或模型 request 自动推断。

## 配置与启动

沿用 `--omni-reward-config` 指向文件的 `services` 和 `weights` 格式。前三个类别保持原来的
加权和语义与正系数检查。也可以只提供 `services`；此时这三个类别的系数默认为 1。
非线性类别自行决定如何组合分量，`voiceclone_gated` 不使用配置里的线性系数。
每个类别只调用自己需要的服务；不同类别共享服务的身份查询缓存、并发上限和
SIM 参考 embedding 缓存。

同一配置文件还支持可选的顶层 `scorers`，按组件指定评分函数路径和 `kwargs`。
它负责把 ASR 文本或 embedding 转成该组件的 `raw` / `transformed`；未配置的组件
继续使用默认算法。函数接口和完整示例见
[自定义组件 scorer](../rewards/README.md#组件评分)。
服务调用、限流、身份校验和缓存由客户端继续管理；类别函数负责组合这些组件结果。

直接填写 [rewards.yaml](../configs/rewards.yaml) 中的服务地址和认证环境；模板五个系数均为 `1.0`，其中 accent / prosody 仅在参考属性启用时生效。
填写模型配置与样例参考音频后，运行：

```bash
OUTPUT_ROOT=/shared/experiments RUN_NAME=prism-mixrl JOB_NAME=categories \
bash examples/prism/run-grpo.sh \
  --hf-checkpoint /assets/student-hf \
  --pretrained-checkpoint /assets/student-native/iter_0100000 \
  --omni-config examples/prism/configs/student.yaml \
  --omni-reward-config examples/prism/configs/rewards.yaml \
  --prompt-data examples/prism/data/mixrl.example.jsonl \
  --custom-rm-path examples.prism.rewards.reward
```

仅绑定音色参考时，三类分别得到 `WER_reward`、`WER_reward + SIM_reward`、
`WER_reward + JudgeCLAP_reward`。绑定口音、韵律时，voiceclone 还会加入对应分量；
公式、参考槽位规则、依赖安装及旧配置迁移见[参考声音属性](../rewards/README.md#参考声音属性)。WER/CER 分量沿用 `1 / (1 + error_rate)`；SIM
沿用非负 cosine similarity；JudgeCLAP 沿用全局属性分数平均。修改 `weights` 可以
调整这三个类别的分量系数，但其要求的分量不能通过零权重关闭。reward 总和不再额外归一化，
后续组优势仍由原 GRPO 流程计算。

将样本的 `metadata.reward_category` 改为 `voiceclone_gated` 就会选择门控示例。
`0.15` 是示例实验阈值；可以直接修改 `MAX_ERROR_RATE`，或增加自己的类别函数。
门控、乘积、最小值等组合都在评分之后、GRPO 组内归一化之前执行。同一 prompt group
的所有候选应使用同一类别。全组都未通过门控时，组优势为零。
门控返回零分的候选仍保留。需要拒绝整个组或屏蔽单条候选的 loss 时，使用
[原生过滤 hook](../rewards/README.md#过滤)；这两种操作不属于组合公式。

单独使用一种奖励时，直接指定对应函数，例如
`--custom-rm-path examples.prism.rewards.voiceclone_gated.reward`。固定函数不要求填写 `reward_category`。包级 `reward_base`、`reward_voiceclone`、
`reward_instruction`、`reward_voiceclone_gated` 也继续导出同一个函数。

## 添加或修改类别

复制一个类别文件，在 `reward()` 中声明需要的基础分量，在同步 `aggregate()` 中写组合公式。
例如，下面的类别使用 WER reward 和 SIM 的几何平均：

```python
from .common import reward_states, score


def aggregate(sample, components):
    wer = components["wer"]["transformed"]
    sim = components["sim"]["transformed"]
    return (wer * sim) ** 0.5


async def reward(args, sample, **kwargs):
    return await score(
        args, sample, category="voiceclone_product",
        components=("wer", "sim"), aggregate=aggregate,
    )


reward.close = reward_states.close
```

保存为 `rewards/voiceclone_product.py`，即可使用
`--custom-rm-path examples.prism.rewards.voiceclone_product.reward`。
需要通过 MixRL 按样本分发时，在 `__init__.py` 导入它并加入 `CATEGORY_REWARDS`。
`reward.close` 复用共享客户端的清理，各类别不需要自行维护一套服务缓存。
增加类别不需要更改基础评分客户端或 Megatron loss。

`aggregate(sample, components)` 是同步函数，返回有限标量。`components` 的每项
包含 `raw`、`transformed`、`status`、`identity` 和 `details`，所以可以用原始 WER
作阈值，或读取 `judge_clap.details.attribute_scores` 改变属性间的归约。
默认变换为：WER/CER `1 / (1 + error_rate)`、SIM 非负 cosine、JudgeCLAP 属性均分。
自定义 scorer 可以改变这些公式；若类别使用原始 WER 阈值或 `attribute_scores`，
替换 scorer 时应保持对应字段的语义，或同时调整类别公式。
技术故障由评分客户端抛出；`invalid_audio` 是需要类别处理的明确评分结果。

参见 [mixrl.example.jsonl](../data/mixrl.example.jsonl)，填写实际参考音频路径后作为数据模板使用。
每行填写 `metadata.reward_category`，其他 Prism 输入沿用已有协议：

`metadata.task_type` 仅记录任务类型，不推断或切换奖励类别；例如 Full VoiceClone 输入仍可选择 base 奖励。

- `base`：需要 `metadata.transcript_target` 和 `metadata.language`，不需要 SIM 参考或 JudgeCLAP 属性。
- `voiceclone`：还需要有效的音色参考。示例在 `metadata.omni_prompt` 填写
  `references.audio1` 与 `reference_attributes.timbre="audio1"`。非 none 的口音复用音色分数，
  非 none 的韵律按自己的参考槽位比较；emotion 不参与计算。
- `instruction`：还需要可被 JudgeCLAP 评分的全局属性对象；`Language` 不参与评分，
  当前服务未支持的音乐属性（如 `Genre`）会报错。该类别不会调用 SIM。

新数据使用 Chinese、English 等规范全名，并用 `metadata.show_language` 控制是否向
模型注入 Language。隐藏语言时，`metadata.language` 仍用于 WER/CER；切换显示状态不改变
评分语言或奖励类别。JudgeCLAP 读取 rollout 保存的有效全局属性对象并跳过 Language。
完整语言名单和兼容规则见 [语言元数据与显示开关](GUIDE.md#语言元数据与显示开关)。

这些输入会严格验证，技术故障继续抛错。合法零帧 stop 的所选奖励分量均记录为明确的
`zero_frame` 结果，reward 为零。分类路由同时支持单样本和混合类别的样本列表；
列表返回值保持输入顺序，并等待已启动的评分完成后传播故障。

## 记录与检查

评分结果继续写入 `metadata.omni_reward`，包含选中分量的 raw/transformed 值、服务
身份和 total，并补充 `category`；加权类别还记录实际系数，门控示例记录阈值与通过状态。
样本自身的 domain 和模型 request 保持其原有语义。组合公式由运行时的源码版本固定。

控制类别的训练占比时，使用采样比例或明确的训练权重。开启 GRPO 标准差归一化后，
将某类别的整个 reward 乘以正常数，其整体尺度通常会被组内归一化抵消。

```bash
"$TRAIN_PYTHON" -m pytest \
  tests/test_prism_reward_categories.py tests/test_prism_reward_composition.py \
  tests/test_audio_reward_clients.py tests/test_grouped_rewards.py -q
```

测试使用冻结评分协议的 CPU 替身，覆盖分量选择、混合批次、共享并发上限、配置与
输入验证、零帧 stop 和故障传播，并验证门控能改变候选排序而后续原 GRPO 分组保持一致。
实际混合任务的 GPU 训练与语音质量收益需另行验收。
