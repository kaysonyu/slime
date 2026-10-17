# Prism 的运行目录与 Ray 生命周期

[run-grpo.sh](../run-grpo.sh) 和 [run-mopd.sh](../run-mopd.sh) 在每个节点启动 Ray，只有 head 执行 `train.py`。两个入口共享 [ray.sh](../ray.sh)，无需单独运行平台引导脚本，也无需套 `torchrun`。

## 输出目录

| 配置 | 默认值 | 用途 |
| --- | --- | --- |
| `OUTPUT_ROOT` | `${REPO_ROOT}/outputs` | 绝对路径；多机时必须是各节点可见的同一共享目录 |
| `RUN_NAME` | `prism-grpo` / `prism-mopd` | 相关实验的分组名称 |
| `JOB_NAME` | `manual` | 当前任务的可读名称，由用户提供 |
| `TRAIN_JOB_ID` | `local` | Inspire 注入的任务 ID |
| `RUNNING_ROUND` | `0` | Inspire 注入的本次运行轮次；本地可手动设置 |

脚本派生 `JOB_TAG=${JOB_NAME}_${TRAIN_JOB_ID}`、`RUN_ROOT=${OUTPUT_ROOT}/${RUN_NAME}/${JOB_TAG}` 和 `ROUND_PATH=${RUN_ROOT}/running_round_${RUNNING_ROUND}`。无需设置 `RUN_DIR` 或 `RAY_RUN_DIR`。

```text
/shared/experiments/
└── prism-grpo/
    └── wer-lr1e6_<train-job-id>/
        ├── running_round_0/
        │   ├── checkpoint/
        │   ├── rollout_audio/
        │   ├── tensorboard/
        │   ├── config/resolved-args.json
        │   ├── logs/
        │   │   ├── rank0.log
        │   │   ├── rank1.log
        │   │   └── phases.jsonl
        │   └── .ray/
        │       ├── head-ready
        │       ├── node-0.ready
        │       ├── node-1.ready
        │       └── exit-code
        └── running_round_1/
            └── ...
```

每个 round 的产物独立保存。脚本首先创建日志与协调目录，checkpoint、音频和 TensorBoard 等由训练组件按需写入。MoPD 默认不请求波形。`.ray` 只负责启动同步和完成通知，Ray 自身日志仍位于各节点的 `/tmp/ray/session_*`。

head 拒绝复用已经存在的 `.ray` 目录，worker 拒绝加入已经完成的 round。新实验更换 `JOB_NAME`；重启同一任务增加 `RUNNING_ROUND`，旧日志和完成标记保持不变。所有节点使用相同名称，不在每个节点各自生成时间戳。

原生 `--save`、`--omni-audio-dir`、`--tensorboard-dir`、`--resolved-config-path`、`--phase-trace-path` 仍可显式覆盖；它们只覆盖对应产物，节点日志和协调状态仍归属 `ROUND_PATH`。

## 平台环境

| 含义 | 读取顺序 | 单机默认 |
| --- | --- | --- |
| head 地址 | `MASTER_ADDR` → `PET_MASTER_ADDR` | `127.0.0.1`；多机必须指定 |
| Ray GCS 端口 | `MASTER_PORT` → `PET_MASTER_PORT` | `6379` |
| 节点编号 | `PET_NODE_RANK` → `RANK` | `0` |
| 节点数 | `PET_NNODES` → `WORLD_SIZE` | `1` |
| 每节点 GPU 数 | `PET_NPROC_PER_NODE` | `1` |

这里 `PET_NPROC_PER_NODE` 按每 GPU 一个进程的数值配置使用。`RANK/WORLD_SIZE` 的兼容输入必须表示节点编号和节点数。Slime 为 learner 分配自己的训练进程 rank、world size 和 rendezvous 端口；本入口直接启动 Ray。

平台注入的 `NCCL_*` 和 `GLOO_SOCKET_IFNAME` 原样继承，包括 NIC、HCA、GID、超时和重试设置。脚本仅为未设置的 `CUDA_DEVICE_MAX_CONNECTIONS` 和 `NCCL_CUMEM_ENABLE` 提供原有默认值。认证变量、`PYTHONPATH`、通信环境必须在每个节点启动 Ray 前准备好。

`TRAIN_PYTHON` 指定训练及 Ray 使用的解释器，默认 `python3`。`SLIME_ROOT` 可指定仓库路径，方便复制 run 脚本到仓库外使用。模型、数据、服务和训练参数继续使用 Slime 的原生 CLI/YAML。

## 启停顺序

1. 每个节点计算同一个 `ROUND_PATH`，输出写入自己的 `logs/rank<N>.log`。
2. head 启动 Ray；worker 根据已知的 `MASTER_ADDR:MASTER_PORT` 加入。共享标记用于等待启动完成，不承担地址发现。
3. head 等待所有节点加入，再直接启动 `train.py`；不经过 Ray Jobs/dashboard。
4. 训练正常结束或失败时，head 写入退出码。worker 读取相同退出码后退出。
5. 各节点在 EXIT trap 中执行本机 `ray stop --force`。收到 INT/TERM 时，head 同时终止自己启动的训练进程组。

脚本用于本任务独占的节点或容器；`ray stop` 会停止该节点上的 Ray 进程。外部奖励/教师服务应独立管理，脚本不执行 `pkill python`。启动/节点等待由 `RAY_START_TIMEOUT` 限制，默认 900 秒；worker 等待任务完成由 `RAY_JOB_TIMEOUT` 限制，默认 172800 秒。

## 单机开发和恢复

```bash
export OUTPUT_ROOT=/shared/experiments
export RUN_NAME=prism-grpo
export JOB_NAME=wer-debug-001
export PET_NPROC_PER_NODE=1

bash examples/prism/run-grpo.sh \
  --hf-checkpoint /assets/student-hf \
  --pretrained-checkpoint /assets/student-native/iter_0100000 \
  --num-rollout 10
```

目录为 `/shared/experiments/prism-grpo/wer-debug-001_local/running_round_0/`。恢复时读取上一轮 checkpoint，并把新的产物保存在下一轮：

```bash
RUNNING_ROUND=1 bash examples/prism/run-grpo.sh \
  --hf-checkpoint /assets/student-hf \
  --load /shared/experiments/prism-grpo/wer-debug-001_local/running_round_0/checkpoint \
  --num-rollout 20 --use-checkpoint-opt-param-scheduler
```

`RUNNING_ROUND` 只选择输出目录，不自动选择恢复 checkpoint。`--load`、`--ckpt-step`、optimizer 和数据游标恢复沿用 Slime 原生行为。多机手动启动时，在所有节点设置相同的输出配置、head 地址和节点数，仅 `PET_NODE_RANK` 不同。
