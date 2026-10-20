# Prism 的三种训练用法

三种用法共用原生 `train.py`，由 [run-grpo.sh](../run-grpo.sh) / [run-mopd.sh](../run-mopd.sh) 提供默认参数：

| 用法 | 脚本与奖励选择 | 所需配置 |
| --- | --- | --- |
| 固定奖励 GRPO | `run-grpo.sh --custom-rm-path <函数>` | `--omni-reward-config` 绑定实际需要的服务 |
| MixRL GRPO | `run-grpo.sh --custom-rm-path examples.prism.rewards.reward` | 数据 `metadata.reward_category` 选择类别；绑定所需服务并集 |
| MoPD | `run-mopd.sh` | `--opd-teacher-config` 按 `metadata.domain` 路由冻结教师 |

只填写 `reward_category` 不会自动打开 MixRL。默认 GRPO 使用
`examples.prism.rewards.base.reward` 的 WER/CER 奖励；纯 MoPD 使用零 task reward。
`--custom-config-path` 中的同名参数按框架规则覆盖 CLI。最终参数用原生
`--resolved-config-path` 保存，不再生成 `experiment.yaml`。

两个入口管理 Ray 启停，默认使用平台分配的所有 GPU；未提供节点环境时使用单机单 GPU。
准备公共原生参数和输出根目录：

```bash
export OUTPUT_ROOT=/shared/experiments
COMMON_ARGS=(
  --hf-checkpoint /assets/prism-student-hf
  --pretrained-checkpoint /assets/prism-native-dcp
  --omni-config /shared/configs/student.yaml
)
```

每次运行通过 `JOB_NAME` 命名任务，产物由脚本放入本轮 `ROUND_PATH`；多机所有节点使用相同命名。
完整目录与平台环境见 [RAY.md](RAY.md)。

## 数据格式与可复制模板

每个 JSONL 文件一行一个 prompt。脚本使用 `--input-key script --metadata-key metadata`。
GRPO 的候选组由数据源复制同一行产生，不需要在数据集里预先写四条或八条候选。
同组候选继承相同的奖励类别；不同 prompt group 可以使用不同类别。

| 文件 | 用途 | 每行额外字段 |
| --- | --- | --- |
| [grpo-base.example.jsonl](../data/grpo-base.example.jsonl) | 固定 WER/CER 奖励 | `transcript_target`、`language` |
| [mixrl.example.jsonl](../data/mixrl.example.jsonl) | 混合 base、voiceclone、instruction、voiceclone_gated | 每行显式 `reward_category` |
| [mopd.example.jsonl](../data/mopd.example.jsonl) | 按 chinese/english 配置教师路由 | 每行显式 `domain` |
| [language-control.example.jsonl](../data/language-control.example.jsonl) | 显示/隐藏模型语言，包含顶层局部指令 | 规范语言全名与显式 `show_language` |

模板中的 `/assets/reference-*.wav` 是待替换路径。基础模型、codec、评分服务和教师
必须使用实际资产；模板本身不启动这些服务。

参考音频使用统一 request 的槽位及显式属性映射，无需逐音频 `id`：

```json
"references": {
  "audio1": "/assets/reference-speaker.wav",
  "audio2": "/assets/reference-style.wav"
},
"reference_attributes": {
  "timbre": "audio1", "accent": "audio1",
  "prosody": "audio2", "emotion": "audio2"
}
```

上例对应原 `Attribute Clone Triplet`：音色/口音取 `audio1`，韵律/情绪取 `audio2`。
完整克隆可只提供 `audio1`，并将四项属性显式映射到它。无参考时省略
`references` 或写 `{}`。槽位允许跳号及两个以上参考；SIM 使用
`reference_attributes.timbre`。旧任务标签和参考列表在数据准备阶段转换。
预训练的语言开关放到 `metadata.show_language`，解析后的语言全名放到 `metadata.language`；
参见 [语言元数据与显示开关](GUIDE.md#语言元数据与显示开关)。

字段的职责分别是：

- `script`：实际交给 Prism 生成的文本或局部指令片段数组。
- `metadata.id`：建议填写唯一、稳定的数据 ID，用于音频目录及跨实验定位。
- `metadata.task_type`：非空任务类型标签，例如 TTS、Full VoiceClone、Local Instruction；原样记录，不影响模型条件或教师/奖励路由。旧数据可省略。
- `metadata.omni_prompt.modality`：`speech` 或 `sing`，默认 `speech`。
- `metadata.omni_prompt.script`：可选的正文覆盖；新样本只在顶层保存一份 script。
- `metadata.omni_prompt.references` 与 `reference_attributes`：参考音频及模型可见的属性来源。
- `metadata.language`：目标语言的规范全名，如 Chinese、English；用于 WER/CER，并作为显示语言时的来源。
- `metadata.show_language`：布尔开关；true 注入规范 Language，false 去掉语言行。开关不进入模型 request。
- `metadata.omni_prompt.global_instruction`：全局属性对象。新数据不重复填写 Language；未填写开关的旧数据仍保留显式 Language 行为。
- `metadata.transcript_target`：期望的 ASR 转写文本，模板中与 script 相同。
- `metadata.reward_category`：MixRL 的奖励选择；固定函数模式不要求填写。
- `metadata.domain`：MOPD 的教师选择；GRPO 中可作为统计标签。domain 也可以是业务领域，不限于语言。

任务标签 `task_type`、模型 request、`reward_category`、`domain` 相互独立。例如，完整语音克隆输入可以
只用 WER 奖励，也可以用 voiceclone 或 voiceclone_gated；设置哪个函数/类别由实验决定。
同一 category 的整个 reward 乘以正常数通常会被 GRPO 组内标准差归一化抵消；
需要改变类别占比时，应调整数据采样或明确的训练权重。

## 1. 固定奖励 GRPO

```bash
JOB_NAME=wer-baseline bash examples/prism/run-grpo.sh "${COMMON_ARGS[@]}" \
  --prompt-data examples/prism/data/grpo-base.example.jsonl \
  --omni-reward-config /shared/configs/rewards.yaml \
  --custom-rm-path examples.prism.rewards.base.reward
```

| 函数路径 | 实际奖励 |
| --- | --- |
| `examples.prism.rewards.base.reward` | WER/CER reward |
| `examples.prism.rewards.voiceclone.reward` | WER/CER 与 SIM 加权和 |
| `examples.prism.rewards.instruction.reward` | WER/CER 与 JudgeCLAP 加权和 |
| `examples.prism.rewards.voiceclone_gated.reward` | 原始 WER/CER ≤ 0.15 时取 SIM，否则 0 |

固定函数不要求数据填写 reward_category。自定义异步函数继续使用
`(args, sample, **kwargs)`，返回标量或按输入顺序返回批量标量；不使用内置评分服务的
函数可以省略奖励配置。启用归档后记录实际函数路径和奖励分项，不从函数名猜测类别。

## 2. MixRL GRPO

```bash
JOB_NAME=mixed-baseline RUN_NAME=prism-mixrl \
bash examples/prism/run-grpo.sh "${COMMON_ARGS[@]}" \
  --prompt-data examples/prism/data/mixrl.example.jsonl \
  --omni-reward-config /shared/configs/rewards.yaml \
  --custom-rm-path examples.prism.rewards.reward
```

每行 metadata.reward_category 选择 `base`、`voiceclone`、`instruction` 或
`voiceclone_gated`；缺失或未知类别报错。同组候选使用同一类别，共用学生、optimizer
和 checkpoint。新增公式放在 `rewards/<category>.py`，需要 MixRL 分发时注册到 `__init__.py` 的 `CATEGORY_REWARDS`，见
[REWARD_CATEGORIES.md](REWARD_CATEGORIES.md)。

## 3. MoPD

在 [teacher-endpoints.yaml](../configs/teacher-endpoints.yaml) 中直接填写教师绑定；例如两位教师：

```yaml
score_function: slime_plugins.prism.teacher.score_teacher
models:
  zh_teacher:
    endpoint: http://teacher-zh-host:8201
  en_teacher:
    endpoint: http://teacher-en-host:8202
by_domain:
  chinese: zh_teacher
  english: en_teacher
```

教师通过 [teacher-score.yaml](../configs/teacher-score.yaml) 单独启动。训练命令：

```bash
JOB_NAME=teacher-baseline bash examples/prism/run-mopd.sh "${COMMON_ARGS[@]}" \
  --prompt-data examples/prism/data/mopd.example.jsonl \
  --opd-teacher-config /shared/configs/teacher-endpoints.yaml \
  --opd-kl-coef 1 --policy-loss-weights '{"stop": 1}'
```

数据的 domain 必须在配置中有映射。学生生成后，公共 OPD 路由选择教师，配置中的评分函数取得同一轨迹所选
RVQ/stop 动作的 logprob，原生 OPD 将它用于训练。零 task reward 是预期行为。
教师可以比学生大，但动作空间、codec、特殊 token、tokenizer 和重放语义必须兼容。

MoPD 默认每个 prompt 一条轨迹，脚本使用 `--no-omni-return-audio`。如需训练试听，
在脚本调用末尾追加 `--omni-return-audio`；
这会增加 vocoder 开销，不会自动启用 GRPO 奖励。

## 独立评测（GRPO 与 MOPD 共用质量指标）

默认不启用评测。创建自己的 eval YAML，填写独立数据路径及奖励函数，例如：

```yaml
eval:
  defaults:
    input_key: script
    metadata_key: metadata
    apply_chat_template: false
    n_samples_per_eval_prompt: 1
    temperature: 1.0
    top_p: 1.0
    top_k: -1
    max_response_len: 512
  datasets:
    - name: content
      path: /data/eval/content.jsonl
      custom_rm_path: examples.prism.rewards.base.reward
    - name: voiceclone
      path: /data/eval/voiceclone.jsonl
      custom_rm_path: examples.prism.rewards.voiceclone.reward
    - name: instruction
      path: /data/eval/instruction.jsonl
      custom_rm_path: examples.prism.rewards.instruction.reward
```

追加 `--eval-config /path/eval.yaml --eval-interval 10`。评测集可以使用与训练不同的奖励，
奖励绑定需包含这些评测实际需要的服务。

MoPD 质量评测示例：

```bash
JOB_NAME=teacher-with-eval bash examples/prism/run-mopd.sh "${COMMON_ARGS[@]}" \
  --prompt-data /data/mopd-train.jsonl \
  --opd-teacher-config /shared/configs/teacher-endpoints.yaml \
  --omni-reward-config /shared/configs/rewards.yaml \
  --omni-eval-return-audio --no-opd-eval-teacher-scoring \
  --eval-config /shared/configs/eval.yaml --eval-interval 10
```

这两个 CLI 开关使评测请求 WAV 并跳过教师，训练保持无波形蒸馏。
需要同时检查教师时使用 `--opd-eval-teacher-scoring`，并为评测数据提供有效 domain 路由。
采样温度、top-p/top-k 仍受 Prism 协议限制。

评测用同步后的学生权重，默认从第 0 轮开始前先评一次，后续按 rollout 间隔或
数据 epoch 边界触发。不参与梯度更新，不自动选择最佳 checkpoint。

## 保存目录与恢复

`--save`、`--tensorboard-dir`、`--resolved-config-path` 等默认从本轮 `ROUND_PATH` 派生，
也支持原生 CLI 覆盖。需要完整候选归档时使用 [GUIDE 中的 YAML](GUIDE.md#显式归档与监控)，
填写本轮的 `prism_artifact_dir` 与 `omni_audio_dir`，通过 `--custom-config-path` 使用。
归档配置只包含目录和钩子，沿用脚本的训练/评测音频开关，MoPD 无需另做修改。
`--custom-config-path` 仍是单个原生自定义 YAML；其中同名字段覆盖 CLI，不会自动合并多份文件。

归档保留 reward 函数、类别、教师身份、generation/training rollout ID 和 policy version；
纯 MoPD 的无波形候选 JSON 按 worker 分片，音频训练/评测按 prompt group 保存 WAV。
外置音频根按归档目录摘要隔离。MixRL 按 group 组织，不为每个类别建立 checkpoint。

恢复将公共参数中的 `--pretrained-checkpoint` 替换为 `--load <旧 checkpoint>`，
保持任务名称，增加 `RUNNING_ROUND`，将新产物写入下一轮目录。显式归档路径也应更新。原生 optimizer、scheduler、RNG 与数据源状态
按框架规则恢复；延长总轮数时用 `--use-checkpoint-opt-param-scheduler`。

CPU 协议替身验证不替代 GPU 训练验收：

```bash
python -m pytest tests/test_prism_experiment_artifacts.py \
  tests/test_prism_reward_categories.py tests/test_prism_reward_composition.py \
  tests/test_audio_reward_clients.py tests/test_omni_rollout_plugin.py -q
```
