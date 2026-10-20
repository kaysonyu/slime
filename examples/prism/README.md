# Prism 训练示例

保留两个训练入口，覆盖三种用法。脚本使用原生 `train.py` 和 Slime 参数，并在每个节点管理 Ray 的启动与退出。

| 用法 | 入口 | 数据 | 奖励 / 教师 |
| --- | --- | --- | --- |
| GRPO | [run-grpo.sh](run-grpo.sh) | [grpo-base.example.jsonl](data/grpo-base.example.jsonl) | WER/CER，`examples.prism.rewards.base.reward` |
| MixRL | 同一个 `run-grpo.sh`，覆盖数据和奖励函数 | [mixrl.example.jsonl](data/mixrl.example.jsonl) | `examples.prism.rewards.reward` 按样本类别选择公式 |
| MoPD | [run-mopd.sh](run-mopd.sh) | [mopd.example.jsonl](data/mopd.example.jsonl) | 冻结教师，默认不生成波形 |

三个 JSONL 分别有 2、4、2 条记录，用于说明输入格式。训练时替换为自己的数据；MixRL 的参考音频路径也需要填写。
`metadata.task_type` 记录任务类型，`domain` 选择 MoPD 教师，`reward_category` 选择 MixRL 奖励；三者独立。
样例的 `metadata.language` 使用 Chinese、English 等规范全名，`metadata.show_language`
显式决定是否把语言加入模型全局指令；正文只在顶层 script 保存。
[language-control.example.jsonl](data/language-control.example.jsonl) 另给出显示、隐藏与局部指令示例，
完整 25 种语言及中文说明见 [语言元数据与显示开关](docs/GUIDE.md#语言元数据与显示开关)。
奖励使用从 [rewards/README.md](rewards/README.md) 开始：服务调用、组件评分、组合、
过滤和诊断均有对应入口。类别公式在 `rewards/`；自定义评分示例在 `scorers.py`，
过滤示例在 `filters.py`，MixRL 分发在 `__init__.py`。`common.py` 连接共享状态和聚合流程；
HTTP 客户端与向量缓存由 `slime_plugins.audio_rewards.client.AudioRewardClient` 管理。
[ray.sh](ray.sh) 是两个入口共享的生命周期实现，不是独立的启动命令。

## 准备和运行

1. 准备兼容的 HF、native checkpoint 和 codec，安装训练及 Omni 依赖。
2. 修改 [student.yaml](configs/student.yaml) 的 codec 路径；Slime 用 `--hf-checkpoint` 指定学生模型。
3. GRPO/MixRL 填写 [rewards.yaml](configs/rewards.yaml) 中实际需要的服务。MoPD 填写 [teacher-endpoints.yaml](configs/teacher-endpoints.yaml)；独立教师使用 [teacher-score.yaml](configs/teacher-score.yaml)。
4. 在所有节点启动 Ray 前准备相同的源码、共享路径、依赖和服务认证环境。

pipeline YAML 是可以直接交给 Omni 的原生配置，不含 `slime:` 段。两个入口通过
`--omni-update-stage tts_engine` 选择权重接收阶段，通过
`--omni-worker-capabilities chat audio_input audio_output` 配置 router 能力。多模型部署可在
`server_groups.overrides` 中分别设置 `omni_update_stage`、`worker_capabilities` 和原生推理参数。
显存释放和恢复默认操作全部阶段，无需配置阶段名单；托管教师直接复用 `teacher-score.yaml`。

单机示例：

```bash
OUTPUT_ROOT=/shared/experiments \
RUN_NAME=prism-grpo JOB_NAME=wer-debug-001 \
bash examples/prism/run-grpo.sh \
  --hf-checkpoint /assets/student-hf \
  --pretrained-checkpoint /assets/student-native/iter_0100000 \
  --num-rollout 1
```

Inspire 多机任务在每个节点执行同一份 run 脚本，直接使用平台注入的 `PET_*`、`MASTER_*`、`TRAIN_JOB_ID` 和 `RUNNING_ROUND`。默认使用全部分配 GPU 共卡并 offload；未提供分布式环境时按单机单 GPU 运行。只在 rank 0 执行训练。

输出统一放在：

```text
${OUTPUT_ROOT}/${RUN_NAME}/${JOB_NAME}_${TRAIN_JOB_ID}/running_round_${RUNNING_ROUND}/
├── checkpoint/
├── rollout_audio/              # 请求波形时产生
├── tensorboard/
├── config/resolved-args.json
├── logs/rank<N>.log
├── logs/phases.jsonl
└── .ray/                      # 本轮启动与结束状态
```

本地默认任务 ID 为 `local`、round 为 `0`。下一次独立实验更换 `JOB_NAME`；同一任务重新启动时使用新的 `RUNNING_ROUND`。恢复仍通过原生 `--load` 指向已有 checkpoint。目录、环境变量和 Ray 生命周期详见 [RAY.md](docs/RAY.md)。

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [EXAMPLE_RUNS.md](docs/EXAMPLE_RUNS.md) | GRPO、MixRL、MoPD 的最小运行命令和多节点示例 |
| [RAY.md](docs/RAY.md) | 平台环境、目录规则、Ray 启停与恢复 |
| [TRAINING_MODES.md](docs/TRAINING_MODES.md) | 数据字段、训练方式和可选评测 |
| [奖励使用入口](rewards/README.md) | 启动服务 → client → scorer → reward 组合 → 过滤 → 诊断，含完整命令 |
| [REWARD_CATEGORIES.md](docs/REWARD_CATEGORIES.md) | 奖励组合、类别路由和自定义公式 |
| [reward_services/README.md](reward_services/README.md) | 四个服务的部署脚本、运行环境、协议及验证工具 |
| [REWARD_SERVICES.md](docs/REWARD_SERVICES.md) | 评分耗时、客户端排队、缓存和联合评分诊断 |
| [GUIDE.md](docs/GUIDE.md) | 模型协议、动作流、性能、归档与诊断 |
| [TOOLS.md](docs/TOOLS.md) | 位于 `scripts/` 的 GPU 与数值诊断工具 |
| [VALIDATION.md](docs/VALIDATION.md) | 历史验证记录与验收范围 |
