# slime TTS RL

基于 **slime v0.3.2** 的 TTS RL 分支。推理使用单独维护的 **SGLang-Omni**，训练使用 **Megatron-LM**。
当前接入 MOSS TTS Local 与 Higgs TTS 的离散多码本策略，提供 WER GRPO 和同模型多教师 MOPD。

这里保留 slime 的 `train.py → Ray → rollout / Megatron` 组织方式。模型自身的时序、码本、停止规则与参数映射放在
`slime_plugins/models/`；公共框架负责 batch、DP 调度、CP、优化器、checkpoint、权重同步和日志。
训练样本不再用一维 `tokens` / `response_length` 表达语音轨迹。

## 数据边界

| 对象 | 表达内容 | 所属位置 |
| --- | --- | --- |
| `Sample` | prompt、领域、奖励、音频路径、原生轨迹、教师分数 | `slime/utils/types.py` |
| MOSS Local trajectory | `[P,C+1]` 提示行、`[T,C]` 音频码、独立 continue/stop 决策 | `slime_plugins/models/moss_tts_local/data.py` |
| Higgs trajectory | 原始提示及参考码、延迟码本、实际采样 mask 与音频有效 mask | `slime_plugins/models/higgs_tts/data.py` |
| `PolicyBatch` | 输入行、预测来源位置、所选动作、mask、行为/教师 logprob、完整样本分母 | `slime/backends/megatron_utils/policy_batch.py` |

CP 按**产生动作的 hidden state 所在位置**分配完整动作行。码本维不参与时间切分；
loss 的每条样本分母在 CP 切分前计算，梯度通过 Megatron/TransformerEngine 归约。
模型适配器只描述输入与预测位置，不实现另一套 CP 通信。

```text
train.py
slime/
  ray/                         生命周期、GPU placement、同步训练迭代
  rollout/                     Omni 生成、领域教师路由、WER
  backends/
    megatron_utils/             公共 batch / CP / loss / optimizer / checkpoint
    sglang_omni_utils/          分阶段 HTTP 控制与完整权重发布
  observability/               公共指标输出、JSONL、TensorBoard、W&B
  utils/                       参数、样本、DP 调度与数值工具
slime_plugins/models/
  moss_tts_local/               config / data / model / Local decoder / weights / diagnostics
  higgs_tts/                    data / model / weights / diagnostics
scripts/run-moss-tts-local.sh
```

删除了原始 SGLang engine/router、agent、critic、异步/部分 rollout、量化及 delta/disk 同步分支、
无关模型插件、示例和 dependency patch 安装流程。保留 v0.3.2 的 DP/CP 与指标归约测试。
SGLang 仍是 **Omni 内部**的依赖，版本为 **0.5.18**。

## 环境与依赖源码

已验证的 GPU 环境是 Python 3.12、CUDA 13.0、Torch 2.13.0+cu130、Transformers 5.12.1、
SGLang 0.5.18、Ray 2.56.0、TransformerEngine 2.17.0。完整锚点见
[docker/tts-runtime-versions.json](docker/tts-runtime-versions.json)。

| 仓库 | 锚定版本 |
| --- | --- |
| slime v0.3.2 | `3778dbf6d1a533ab478ecf5ddaa11449a47752b2` |
| SGLang-Omni 上游共同基点 | `1ddf9c17bb8b211b439e9e91c22f1c722fb3461f` |
| SGLang-Omni 检查时本地 HEAD | `e8c998291c9ae2ec1e48d18f1102952d9f0412a3` |
| Megatron-LM | `3f49de20de81c1ad5dfc10efa0f7d0c8098f1ce4` |
| mossLite 模型参考 | `43892a9b2797591fd6be35ae95066524ac984b1f` |

不生成或应用外部依赖 patch。使用上述单独维护的源码；Omni 使用表中已提交的 revision。
在已准备好 GPU 依赖的环境中安装本项目：

```bash
python -m pip install -e . --no-deps
export PYTHONPATH="$MEGATRON_DIR:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_CUMEM_ENABLE=0
export NCCL_NVLS_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

NCCL 的两个设置要同时用于训练进程和 Omni 服务。Torch/CUDA/TE 由运行环境提供；
`docker/Dockerfile` 接受 `BASE_IMAGE`，不会自动克隆或修改 Megatron、Omni。

## 权重目录与一次性转换

运行时固定使用两种目录：**Omni 读取 Local 布局 HF，Megatron 读取 torch_dist**。
原始 NeoX HF 只作为离线转换输入；服务启动、训练与权重回传中不再做 NeoX 名称或 QKV/RoPE 转换。

本次已生成的持久资产：

```text
/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/MOSS-TTS-Local-0020000-omni
/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/MOSS-TTS-Local-0020000-torch_dist
```

第一步在 CPU 上转换原始 HF，保留独立音频 heads，核验 tied text head 后去重，重排 Local QKV 并补齐 processor shim：

```bash
python tools/convert_moss_tts_2_0_for_sglang_omni.py "$SOURCE_HF" "$MODEL_DIR"
```

第二步在 Megatron 环境生成初始化 checkpoint（新目录，不含优化器状态）：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node 1 tools/convert_hf_to_torch_dist.py \
  --hf-checkpoint "$MODEL_DIR" --save "$TRAIN_CHECKPOINT" \
  --no-gradient-accumulation-fusion --no-masked-softmax-fusion \
  --no-rope-fusion --no-persist-layer-norm --attention-backend flash
```

`MODEL_DIR` 必须是第一步的转换产物。首次训练从 `TRAIN_CHECKPOINT` 的 iteration 0 权重开始，
重新创建优化器；继续训练时把 `TRAIN_CHECKPOINT` 指向训练运行保存的 checkpoint 根目录，恢复优化器和采样状态。
本次生成的 Omni HF 目录包含可直接使用的 `sglang_omni.yaml` 与 `sglang_omni_teacher.yaml`。

模型/codec 实现参考用户提供的 `sglang-omni-tmp` 快照（标注版本
`92a53c268a167e153cbe7f36b1d916b8a72c0b4f`），在其 Local 实现之上增加 RL 返回与 teacher scoring。
[本轮迁移与验收](docs/local_layout_migration.md) 记录具体边界和产物。

## MOSS Local：WER GRPO

训练数据是 JSONL。`text` 同时用于合成与 WER 参考；参考音频和领域为可选字段。

```json
{"text":"Please place the blue notebook beside the wooden box.","domain":"general"}
{"text":"The experiment begins tomorrow.","domain":"science","ref_audio":"/shared/voice.wav","ref_text":"Reference speech."}
```

先启动单独的学生推理服务。模型目录是转换后的 Local HF artifact，包含 tokenizer/config 和 safetensors。
MOSS Local 的 codec 独立指定：

```bash
cd "$OMNI_DIR"
CUDA_VISIBLE_DEVICES=0 python -m sglang_omni.cli serve \
  --config examples/configs/moss_tts_local.yaml \
  --model-path "$MODEL_DIR" --host 0.0.0.0 --port 18410 \
  --preprocessing.factory.codec_model_path "$CODEC_DIR" \
  --vocoder.factory.codec_model_path "$CODEC_DIR"
```

WER 接受提供 `POST /v1/audio/transcriptions` 的 ASR 服务。验证使用本地 Qwen3-ASR-1.7B：

```bash
CUDA_VISIBLE_DEVICES=1 python -m sglang_omni.cli serve \
  --config examples/configs/qwen3_asr_rtx4090.yaml \
  --model-path "$ASR_MODEL_DIR" --host 0.0.0.0 --port 18411
```

在独立训练 GPU 上运行：

```bash
cd "$SLIME_DIR"
CUDA_VISIBLE_DEVICES=2,3 TRAIN_GPUS=2 \
  OMNI_ENDPOINT=http://127.0.0.1:18410 \
  ASR_ENDPOINT=http://127.0.0.1:18411/v1/audio/transcriptions \
  PROMPT_DATA=/shared/train.jsonl OUTPUT_DIR=/shared/runs/moss-grpo \
  bash scripts/run-moss-tts-local.sh
```

`MODEL_DIR`、`TRAIN_CHECKPOINT`、`MEGATRON_DIR` 须预先导出。可附加 `--context-parallel-size 2` 或
`--tensor-model-parallel-size 2`；相应减少 DP，并保证 global batch 能分配给所有 DP rank。
默认训练完整策略；`--train-scope audio` 冻结全局文本 embedding/backbone。

GRPO 使用同 prompt 组内标准化奖励与逐动作 PPO 裁剪，比值来自真实行为 logprob。
WER 奖励为 `1 - WER`，不裁掉大于 1 的 WER；ASR 服务失败会中止当前迭代。
英文按词，`--wer-language zh` 明确使用中文字符级/CER 口径。
默认采样使用正温度、`top_p=1`、`top_k=-1`、无音频重复惩罚，保证训练与采样概率的定义一致。

独立验证集通过 `--eval-data /shared/eval.jsonl --eval-interval 5` 配置。
验证遍历该文件；MOPD 的验证也使用 WER，因此需要 ASR endpoint。

## 同模型多教师 MOPD

每个领域教师都是同一 MOSS Local 基座的独立权重。学生先生成完整轨迹，再按样本 `domain`
选一个冻结教师做 teacher forcing。教师不另生成答案，必须覆盖学生原始输入、动作、温度和停止规则。
服务返回的输入摘要、权重摘要及版本均会检查，教师身份也写入 sampler checkpoint。

教师服务不启动 codec/vocoder：

```bash
cd "$OMNI_DIR"
CUDA_VISIBLE_DEVICES=1 python -m sglang_omni.cli serve \
  --config examples/configs/moss_tts_local_score.yaml \
  --model-path /shared/teachers/science-hf --host 0.0.0.0 --port 18412
CUDA_VISIBLE_DEVICES=2 python -m sglang_omni.cli serve \
  --config examples/configs/moss_tts_local_score.yaml \
  --model-path /shared/teachers/general-hf --host 0.0.0.0 --port 18413
```

两个服务应分别启动。学生服务仍使用自己的 HF 权重和端口。训练示例：

```bash
cd "$SLIME_DIR"
CUDA_VISIBLE_DEVICES=3,4 TRAIN_GPUS=2 OBJECTIVE=mopd \
  OMNI_ENDPOINT=http://127.0.0.1:18410 PROMPT_DATA=/shared/mixed-domains.jsonl \
  OUTPUT_DIR=/shared/runs/moss-mopd bash scripts/run-moss-tts-local.sh \
  --mopd-teachers science=http://127.0.0.1:18412 general=http://127.0.0.1:18413
```

当前 MOPD 使用所选动作上的 `teacher_logprob - student_behavior_logprob` 作为 detach 后的蒸馏优势，
默认幅度限制为 5，然后优化学生的 logprob。它是 sampled on-policy distillation，
没有请求或传输整个词表的 teacher logits，也没有宣称实现任意教师架构间的蒸馏。

## Checkpoint 与教师导出

`--load /shared/run/checkpoints` 恢复 Megatron 权重、优化器、scheduler 及对应的 rollout sampler 状态。
恢复时仍须传转换后的 Local `--hf-checkpoint` 以确定模型结构；数据内容、采样种子与 shuffle 设置必须一致。

训练后的教师可导出为同一 Local 布局的 HF artifact，直接供 Omni 加载。目标目录必须不存在；导出完成前写入 `.incomplete` 目录：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node 1 tools/export_tts_to_hf.py \
  --hf-checkpoint "$MODEL_DIR" --load /shared/teacher-run/checkpoints \
  --debug-train-only --num-rollout 0 --export-dir /shared/teachers/science-hf \
  --no-gradient-accumulation-fusion --no-masked-softmax-fusion \
  --no-rope-fusion --no-persist-layer-norm --attention-backend flash
```

导出使用和权重同步相同的模型映射，并保留原 HF artifact 的 tokenizer、模型代码和冻结 codec 参数。
它不会把训练前的参数填回缺失的可训练策略参数；策略参数不完整会直接报错。

## 日志与验证边界

公共层写 `metrics.jsonl`，可选 `--use-tensorboard --tb-log-dir ...` 和
`--use-wandb --wandb-mode offline`。音频文件按 rollout/sample 编号保存，完整调试样本可用
`--save-debug-rollout-data '/shared/run/rollout_{rollout_id}.pt'` 留存；它包含 ASR 转写和教师身份。

公共指标包含 WER、corpus WER、组内奖励标准差、零方差组比例、帧/动作数、截断率、loss、
概率比值、裁剪比例、logprob 差、梯度范数和 LR。MOPD 另记录蒸馏优势与教师 logprob。
模型专有的 decision/codebook NLL 由各自 `diagnostics.py` 或 action group 名称提供，公共日志代码负责输出。

当前支持 dense BF16、DP/TP/CP；PP、SP、量化、critic 和异步策略版本混用会被拒绝或不提供入口。
MOSS 与 Higgs 的 BF16 推理/训练 logprob **并非逐位一致**；训练保留真实行为分数并检查差异。
独立 HF 回放也观察到同量级差异，详见 [数值与协议说明](docs/tts_rl.md)。
Higgs 冒烟验证显式使用 `--logprob-parity-tolerance 1.5 --logprob-parity-mean-tolerance 0.15`；
这是验证条件，不是精度对齐或训练效果的结论。

[INSPIRE.md](INSPIRE.md) 记录指定 Notebook、模型路径及占卡恢复要求。
`tools/tts_wer_e2e.py` 执行实际生成、WER/MOPD、训练、回传及可选 checkpoint 恢复，
`tools/tts_gpu_guard.py` 负责暂停和恢复用户原有的 `gpu-occupy`。
真实验证结果见 [验证记录](docs/tts_validation.md)。CPU 测试由 `.github/workflows/pr-test.yml.j2` 注册；
私有 HF 资产与 H200 验证通过 Inspire 运行，不混入公开 CI。
