# Prism 的三种训练用法

算法入口仍是 GRPO 和 MOPD。固定奖励 GRPO 与 MixRL 使用同一个原生 GRPO
训练流程，区别在于奖励函数的选择。下表对应当前 Prism 示例与实验归档入口。

| 用法 | 启动入口 | 决定评分方式的位置 | 所需配置 |
| --- | --- | --- | --- |
| 固定奖励 GRPO | `run-grpo.sh --custom-rm-path <函数路径>` | 整个运行固定一个函数 | `REWARD_CONFIG`，包含函数实际需要的服务 |
| MixRL GRPO | `run-grpo.sh --custom-rm-path examples.prism.rewards.reward` | 每行 `metadata.reward_category` | 所有类别所需服务的并集 |
| MOPD | `run-mopd.sh` | `metadata.domain` → `TEACHER_CONFIG.by_domain` | 固定的教师绑定，不要求 `REWARD_CONFIG` |

只在数据里填写 `reward_category` 不会自动打开 MixRL；必须选择上述分发入口。
默认 `run-grpo.sh` 的 `slime_plugins.tts_reward.reward` 会使用配置中的统一奖励组合。
MixRL 支持注册在 `CATEGORY_REWARDS` 中的类别名，不从每行数据动态导入任意 Python
函数，也不执行 JSON 中的公式字符串。

示例默认的 grpo.yaml/mopd.yaml 不覆盖奖励入口。若额外的自定义 YAML 设置了
`custom_rm_path`，它按框架原有规则覆盖 CLI；本轮 `config/experiment.yaml` 中的
`prism_reward_function_path` 记录了实际委托的函数，可以用它核对最终选择。

## 数据格式与可复制模板

每个 JSONL 文件一行一个 prompt。脚本使用 `--input-key script --metadata-key metadata`。
GRPO 的候选组由数据源复制同一行产生，不需要在数据集里预先写四条或八条候选。
同组候选继承相同的奖励类别；不同 prompt group 可以使用不同类别。

| 文件 | 用途 | 每行额外字段 |
| --- | --- | --- |
| [grpo-base.example.jsonl](data/grpo-base.example.jsonl) | 固定 WER/CER 奖励 | `transcript_target`、`language` |
| [grpo-voiceclone.example.jsonl](data/grpo-voiceclone.example.jsonl) | 固定 WER/CER + SIM | `references.audio1`，音色参考由任务推导 |
| [grpo-instruction.example.jsonl](data/grpo-instruction.example.jsonl) | 固定 WER/CER + JudgeCLAP | 支持的全局属性指令 |
| [mixrl.example.jsonl](data/mixrl.example.jsonl) | 混合 base、voiceclone、instruction、voiceclone_gated | 每行显式 `reward_category` |
| [mopd.example.jsonl](data/mopd.example.jsonl) | 按 chinese/english 路由到两个教师 | 每行显式 `domain` |

模板中的 `/assets/reference-*.wav` 是待替换路径。基础模型、codec、评分服务和教师
必须使用实际资产；模板本身不启动这些服务。

参考音频沿用 revision 6 的任务位置，直接用槽位名填写，无需逐音频 `id`：

```json
"references": {
  "audio1": "/assets/reference-speaker.wav",
  "audio2": "/assets/reference-style.wav"
}
```

上例适用于 `Attribute Clone Triplet`：音色/口音取 `audio1`，韵律/情绪取 `audio2`。
`Full VoiceClone` 只需 `audio1`，四项属性都由它提供。无参考的任务省略
`references` 或写 `{}`。只接受连续的 `audio1`、`audio2`，解析顺序不取决于 JSON
字段顺序；标准任务省略 `reference_targets`，SIM 自动使用任务指定的音色参考。
旧的 `[{"id": "speaker", "uri": "/assets/reference-speaker.wav"}]` 仍然兼容，
保留原 ID 和列表位置；若写了 `reference_targets`，继续校验它与任务映射一致。

字段的职责分别是：

- `script`：实际交给 Prism 生成的文本。
- `metadata.id`：建议填写唯一、稳定的数据 ID，用于音频目录及跨实验定位。
- `metadata.omni_prompt.task_type`：Prism 生成任务，例如 `TTS`、`Full VoiceClone`、`Instruction`。
- `metadata.omni_prompt.references`：可选的 `audio1`/`audio2` 到参考音频路径/URI 的映射。
- `metadata.omni_prompt.show_language`：必填布尔值；语言用于奖励计算时，即便这里是 false，也填写 `language`。
- `metadata.omni_prompt.transcript_target`：期望的 ASR 转写文本，模板中与 script 相同。
- `metadata.reward_category`：MixRL 的奖励选择；固定函数模式不要求填写。
- `metadata.domain`：MOPD 的教师选择；GRPO 中可作为统计标签。domain 也可以是业务领域，不限于语言。

`task_type`、`reward_category`、`domain` 三者独立。例如，Full VoiceClone 输入可以
只用 WER 奖励，也可以用 voiceclone 或 voiceclone_gated；设置哪个函数/类别由实验决定。
同一 category 的整个 reward 乘以正常数通常会被 GRPO 组内标准差归一化抵消；
需要改变类别占比时，应调整数据采样或明确的训练权重。

## 1. 固定奖励 GRPO

先按 [README](README.md) 设置模型、native 初始化来源、解释器、Ray 和 `OMNI_CONFIG`。
以下命令假设 `REWARD_CONFIG` 已绑定 WER 和 SIM 服务：

```bash
RUN_NAME=prism-grpo-voiceclone JOB_TAG=trial-001 RUNNING_ROUND=0 \
PROMPT_DATA=examples/prism/data/grpo-voiceclone.example.jsonl \
bash examples/prism/run-grpo.sh \
  --custom-rm-path examples.prism.rewards.voiceclone.reward
```

这个模板故意不填写 `reward_category`：入口已经固定，所有样本调用 voiceclone 函数。
其他固定入口为：

| 函数路径 | 对应模板 | 实际奖励 |
| --- | --- | --- |
| `examples.prism.rewards.base.reward` | grpo-base | WER/CER reward |
| `examples.prism.rewards.voiceclone.reward` | grpo-voiceclone | WER/CER reward 与 SIM 加权和 |
| `examples.prism.rewards.instruction.reward` | grpo-instruction | WER/CER reward 与 JudgeCLAP 加权和 |
| `examples.prism.rewards.voiceclone_gated.reward` | grpo-voiceclone | 原始 WER/CER ≤ 0.15 时取 SIM，否则 0 |

也可以指定自己的异步奖励函数，沿用 `(args, sample, **kwargs)` 接口并返回标量或
对应顺序的批量标量。当前 GRPO recipe 仍要求 `REWARD_CONFIG` 文件存在；完全不使用
这些评分服务的自定义函数可提供内容为 `{}` 的配置文件。

归档包装保留原函数路径。内置类别函数返回的 `omni_reward.category` 会记录为实际
`reward_category`；数据原先填写的值另存为 `requested_reward_category`。仅返回标量
的自定义函数仍保存总 reward 和函数路径，不会从函数名猜测类别或凭空生成分项。

## 2. MixRL GRPO

使用三个评分服务的绑定配置，将数据换成 MixRL 模板：

```bash
RUN_NAME=prism-grpo-mixrl JOB_TAG=trial-001 RUNNING_ROUND=0 \
PROMPT_DATA=examples/prism/data/mixrl.example.jsonl \
bash examples/prism/run-grpo.sh \
  --custom-rm-path examples.prism.rewards.reward
```

每行的 `metadata.reward_category` 选择一个类别文件。当前类别为 `base`、`voiceclone`、
`instruction`、`voiceclone_gated`；缺失或未知类别会报错。各类只调用自己需要的评分服务。
新增公式时，在 `examples/prism/rewards/` 新建类别函数并注册到 `CATEGORY_REWARDS`，
再在数据中填写该类别名。配置和组合规则见 [REWARD_CATEGORIES.md](REWARD_CATEGORIES.md)。

MixRL 的类别共用一个学生、optimizer 和 checkpoint。音频仍按 prompt group 组织，
每条 JSON/JSONL 明确记录类别，TensorBoard 按实际类别分别统计。

## 3. MOPD

模板中的 `chinese` 与 `english` 必须与教师配置的 domain 映射一致。可以用已运行的
冻结教师生成配置，工具会核验教师身份与已加载权重 checksum：

```bash
export RUN_CONFIG=/runs/prism-mopd-config-trial-001
"$TRAIN_PYTHON" -m examples.prism.configure \
  --student "$HF_CHECKPOINT" --codec "$CODEC_MODEL" --output "$RUN_CONFIG" \
  --teacher zh_teacher http://teacher-zh-host:8201 "$TEACHER_ZH" \
  --teacher en_teacher http://teacher-en-host:8202 "$TEACHER_EN" \
  --domain chinese zh_teacher --domain english en_teacher

export OMNI_CONFIG="$RUN_CONFIG/student.yaml"
export TEACHER_CONFIG="$RUN_CONFIG/teachers.yaml"

RUN_NAME=prism-mopd JOB_TAG=trial-001 RUNNING_ROUND=0 \
PROMPT_DATA=examples/prism/data/mopd.example.jsonl \
bash examples/prism/run-mopd.sh
```

学生生成后，hook 根据 domain 选择教师并取得所选 RVQ/stop 动作的 logprob；原生
OPD 使用这些分数形成训练信号。这里 task reward 固定为 0 是预期行为，不能用
GRPO 的总 reward 曲线判断 MOPD 是否在学习。显式 domain 没有映射时会报错。

MOPD 默认每个 prompt 一条轨迹、`omni_return_audio: false`，因此保留 prompt、候选
JSON、JSONL、教师 key/身份/checksum 和 checkpoint，但不产生生成 WAV。完整动作和
logprob 张量继续由原 debug dump 参数控制。需要合成波形用于听音时，可复制 mopd.yaml，
将 `omni_return_audio` 改为 true，再用 `--custom-config-path` 指定该文件；这会增加
vocoder 与音频保存开销，不会自动启用 GRPO 奖励。

## 独立评测（GRPO 与 MOPD 共用质量指标）

默认 recipe 不开启 eval。准备与训练集独立的 revision 6 JSONL，复制并修改
[eval.example.yaml](eval.example.yaml) 中各数据集路径，再在启动命令后追加
`--eval-interval 10 --eval-config /path/to/eval.yaml`。每个数据集明确指定自己的
`custom_rm_path`，因此质量评分不必与训练奖励相同。只启用需要的评测集，
`REWARD_CONFIG` 绑定这些评测集需要的评分服务；固定函数评测不要求 `reward_category`。

MOPD 使用 [mopd-eval.yaml](mopd-eval.yaml) 在 eval 阶段请求 WAV，训练仍使用无波形的
教师动作蒸馏和零 task reward：

```bash
REWARD_CONFIG=/path/to/frozen-rewards.yaml \
  bash examples/prism/run-mopd.sh \
  --custom-config-path examples/prism/mopd-eval.yaml \
  --eval-config /path/to/eval.yaml --eval-interval 10
```

该配置设置 `omni_eval_return_audio: true` 和 `omni_eval_teacher_scoring: false`。
评测直接调用各数据集指定的质量 reward，跳过教师评分；教师仍照常参与训练。
需要同时检查教师时可将后一项设为 true，并为评测 domain 提供有效教师路由。
这两个字段也接受 CLI：`--omni-eval-return-audio`、`--no-omni-eval-teacher-scoring`。
未配置时，eval 的音频开关继承训练，教师 hook 保持原行为。

评测使用同步后的学生权重，从第 0 轮开始训练时默认先评一次，后续按 rollout
间隔或 epoch 边界触发。它不参与梯度更新，也不自动选择最佳 checkpoint。
结果进入本轮 `rollout_audio/eval/`、`records/eval/` 和 `eval/<dataset>/prism/` 指标。
MOPD 的音频解码和质量评分开销仅在配置的 eval 中发生。

## 保存目录与恢复

建议始终显式填写 RUN_NAME 区分三种实验。GRPO 的两种用法都走 grpo 入口，默认
RUN_NAME 都是 `prism-grpo`，不会根据奖励函数自动改名。

```text
OUTPUT_ROOT/
├── prism-grpo-voiceclone/trial-001/running_round_0/
├── prism-grpo-mixrl/trial-001/running_round_0/
└── prism-mopd/trial-001/running_round_0/
    ├── config/
    ├── logs/
    ├── checkpoints/
    ├── tensorboard/
    ├── wandb/
    ├── rollout_audio/{train,eval}/...
    └── records/{train,eval}/...
```

三种模式使用相同的目录合同。默认音频在本轮的 `rollout_audio/`；显式 `AUDIO_DIR`
指向轮次目录之外时，归档会增加对应运行目录的末尾层级与唯一摘要。例如标准布局下：

```text
AUDIO_DIR/prism-grpo-mixrl/trial-001/running_round_0-<run-path-hash>/train/...
```

这样多次运行可以共用同一个外置音频存储根，不覆盖其他实验的 prompt/reference 文件，
也不把其他轮次的候选算入本轮统计。每个候选记录里都有绝对 `audio_path` 和 `record_path`。
MixRL 的子目录按 group 组织，不为每个 reward category 建一套独立 checkpoint。

`OUTPUT_DIR` 仍是 checkpoint 路径的显式覆盖项，使用时应为本次运行选择独立写入目录。
续训通过 `RL_LOAD` 指向选定的旧 checkpoint，使用新的 RUNNING_ROUND；新目录继续记录
恢复后的训练 step。原生 checkpoint 的 optimizer、scheduler、RNG 与数据源状态保存
沿用框架行为，配置快照同时记录原始奖励入口和加载来源。

CPU 验证使用上述 JSONL 的真实数据读取/分组、原 Omni 协议解析、实际奖励类别代码、
教师 hook 与归档函数；生成结果和外部评分服务使用协议替身，没有运行 GPU 训练：

```bash
python -m pytest tests/test_prism_experiment_artifacts.py \
  tests/test_prism_reward_categories.py tests/test_prism_reward_composition.py \
  tests/test_tts_reward_plugin.py tests/test_omni_rollout_plugin.py -q
```
