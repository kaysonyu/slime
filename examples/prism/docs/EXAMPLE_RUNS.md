# Prism 最小运行示例

两个 run 脚本覆盖 GRPO、MixRL 和 MoPD。样例数据用于说明格式；模型路径、参考音频和在线服务需要替换为实际资产。默认资源从节点环境读取，单机默认为一张 GPU，使用 learner/rollout 共卡 offload。

## 公共准备

在仓库根目录执行。填写 [student.yaml](../configs/student.yaml) 的 codec 路径，准备 HF 和 native 模型；Slime 用 `--hf-checkpoint` 作为学生的最终模型路径。

```bash
export TRAIN_PYTHON=python3
export OUTPUT_ROOT=/shared/experiments
COMMON_ARGS=(
  --hf-checkpoint /assets/student-hf
  --pretrained-checkpoint /assets/student-native/iter_0100000
  --omni-config "$PWD/examples/prism/configs/student.yaml"
  --num-rollout 10
)
```

需要给不同实验使用不同的 `JOB_NAME`。输出路径由运行脚本计算，详见 [RAY.md](RAY.md)。下列命令是三种独立运行选项。

## GRPO：WER/CER

填写 [rewards.yaml](../configs/rewards.yaml) 的 `wer` endpoint 和认证变量。默认 `reward_base` 只调用 WER/CER，其他服务无需为了这条示例配置。

```bash
RUN_NAME=prism-grpo JOB_NAME=wer-baseline \
bash examples/prism/run-grpo.sh "${COMMON_ARGS[@]}" \
  --omni-reward-config "$PWD/examples/prism/configs/rewards.yaml"
```

默认数据为 2 条中文/英文 [grpo-base.example.jsonl](../data/grpo-base.example.jsonl)。总 GPU 数为 G 时，每轮 `2G` 个 prompt，每个 prompt 4 个候选，global batch 为 `8G`。这些只是脚本默认值；较大实验可用原生参数一起覆盖，例如：

```bash
RUN_NAME=prism-grpo JOB_NAME=wer-batch2048 \
bash examples/prism/run-grpo.sh "${COMMON_ARGS[@]}" \
  --prompt-data /data/grpo-train.jsonl \
  --rollout-batch-size 256 --n-samples-per-prompt 8 --global-batch-size 2048
```

## MixRL：按样本选择奖励

填写 WER、SIM、JudgeCLAP 中实际需要的服务。将 [mixrl.example.jsonl](../data/mixrl.example.jsonl) 的两条参考音频改为共享路径；四条记录分别演示 base、voiceclone、instruction 和 voiceclone_gated。

```bash
RUN_NAME=prism-mixrl JOB_NAME=mixed-baseline \
bash examples/prism/run-grpo.sh "${COMMON_ARGS[@]}" \
  --prompt-data "$PWD/examples/prism/data/mixrl.example.jsonl" \
  --custom-rm-path examples.prism.rewards.reward \
  --omni-reward-config "$PWD/examples/prism/configs/rewards.yaml"
```

奖励调用、组件算法、组合与过滤的完整入口见 [rewards/README.md](../rewards/README.md)。
上面的命令默认不启用过滤；[完整流程示例](../rewards/README.md#完整训练示例) 展示两个原生过滤 hook。
各类别共享学生、optimizer 和 checkpoint。此入口仍使用 GRPO 的候选分组。

## MoPD：冻结教师动作评分

独立教师使用 [teacher-score.yaml](../configs/teacher-score.yaml)，需要填写教师模型路径。可在独立 GPU 服务中启动：

```bash
python -m sglang_omni.cli serve \
  --config examples/prism/configs/teacher-score.yaml \
  --host 0.0.0.0 --port 8000
```

将服务地址和认证环境变量填入 [teacher-endpoints.yaml](../configs/teacher-endpoints.yaml)，再运行：

```bash
RUN_NAME=prism-mopd JOB_NAME=teacher-baseline \
bash examples/prism/run-mopd.sh "${COMMON_ARGS[@]}" \
  --opd-teacher-config "$PWD/examples/prism/configs/teacher-endpoints.yaml"
```

默认使用 2 条 [mopd.example.jsonl](../data/mopd.example.jsonl)，`chinese` 和 `english` 都指向配置中的 `prism_v04`；可以修改映射选择不同教师。每轮 prompt 数和 global batch 都为 G，每个 prompt 一条轨迹。纯 MoPD 默认零任务奖励、无波形；训练试听可显式添加 `--omni-return-audio`。

## Inspire 和手动多机

Inspire 在每个节点执行相同命令。使用平台注入的 `PET_NODE_RANK`、`PET_NNODES`、`PET_NPROC_PER_NODE`、`MASTER_ADDR`、`MASTER_PORT`、`TRAIN_JOB_ID`、`RUNNING_ROUND`，不再额外运行 `inspire/run.sh`。

手动模拟两节点、每节点 8 GPU 时，两边都设置：

```bash
export MASTER_ADDR=10.0.0.1 MASTER_PORT=6379
export PET_NNODES=2 PET_NPROC_PER_NODE=8
export OUTPUT_ROOT=/shared/experiments RUN_NAME=prism-grpo JOB_NAME=two-node
export TRAIN_JOB_ID=manual-001 RUNNING_ROUND=0
# head 上设置 0，另一台设置 1。
export PET_NODE_RANK=0
bash examples/prism/run-grpo.sh "${COMMON_ARGS[@]}"
```

`COMMON_ARGS`、依赖、源码、服务认证和通信环境在两个节点分别准备。脚本只在 head 执行训练，各节点在结束时清理本地 Ray。checkpoint、TensorBoard、音频和节点日志均位于本轮目录。

可选 eval 和候选归档分别见 [TRAINING_MODES.md](TRAINING_MODES.md#独立评测grpo-与-mopd-共用质量指标) 与 [GUIDE.md](GUIDE.md#显式归档与监控)。CPU 测试覆盖启动行为和协议接线；实际 GPU 训练、通信和服务质量需在所用资产上验证。
