# Prism GRPO 与多 domain MoPD

Prism 使用原 `train.py`、Ray placement、rollout 分组、Megatron optimizer 和 checkpoint。
[run-grpo.sh](../run-grpo.sh) / [run-mopd.sh](../run-mopd.sh) 分别提供完整的原生参数示例；模型 provider、checkpoint initializer、
Omni 生成、教师评分和归档均使用 Slime 的扩展接口。两个 run 脚本管理各节点 Ray，
目录规则、Inspire 环境和退出清理见 [RAY.md](RAY.md)。

## 训练入口与可调参数

在所有分配节点执行同一份 run 脚本。`TRAIN_PYTHON` 选择训练及 Ray 解释器；
Omni 默认使用相同解释器，也可以显式传 `--omni-python`。单机示例：

```bash
export TRAIN_PYTHON=python3
export OUTPUT_ROOT=/shared/experiments
export RUN_NAME=prism-grpo JOB_NAME=wer-baseline
HF_CHECKPOINT=/assets/prism-student-hf
NATIVE_CHECKPOINT=/assets/prism-native-dcp
OMNI_CONFIG=/shared/configs/student.yaml

bash examples/prism/run-grpo.sh \
  --hf-checkpoint "$HF_CHECKPOINT" \
  --pretrained-checkpoint "$NATIVE_CHECKPOINT" \
  --prompt-data /data/prism-train.jsonl \
  --omni-config "$OMNI_CONFIG" \
  --omni-reward-config /shared/configs/rewards.yaml \
  --rollout-batch-size 2 --n-samples-per-prompt 4 --global-batch-size 8 \
  --num-rollout 100 --lr 1e-6
```

输出默认位于 `${OUTPUT_ROOT}/${RUN_NAME}/${JOB_NAME}_${TRAIN_JOB_ID}/running_round_${RUNNING_ROUND}`。
本地任务 ID 默认 `local`、round 默认 `0`。head 先调用
`slime_plugins.models.prism_configuration` 读取有效的 `--hf-checkpoint` 并生成架构参数，
等待所有 Ray 节点就绪后直接执行 `train.py`，保留 shell 处理信号和清理。
重复的单值 CLI 参数以后一个为准；自定义 YAML 按 Slime 原规则覆盖同名参数，
所以建议把学习率、batch 等放在 CLI，将插件字段放在 `--custom-config-path`。

下表 G 表示分配的总 GPU 数，即 `PET_NNODES × PET_NPROC_PER_NODE`；单机默认 G=1。

| 设置 | 原生参数 | 示例脚本默认 |
| --- | --- | --- |
| 学习率、调度 | `--lr`、`--lr-decay-style` | `1e-6`、`constant` |
| Weight decay / Adam | `--weight-decay`、`--adam-beta1`、`--adam-beta2` | `0.01`、`0.9`、`0.98` |
| PPO clip | `--eps-clip`、`--eps-clip-high` | `0.2`、`0.2` |
| 动作流 loss 权重 | `--policy-loss-weights` | `{"stop": 1}`；其他流默认 `1` |
| MoPD 教师权重 | `--opd-kl-coef` | `1` |
| rollout prompt 数 | `--rollout-batch-size` | GRPO `2G`，MoPD `G` |
| 每 prompt 候选数 | `--n-samples-per-prompt` | GRPO `4`，MoPD `1` |
| 训练 global batch | `--global-batch-size` | GRPO `8G`，MoPD `G` |
| 训练总轮数 | `--num-rollout` | GRPO `100`，MoPD `2000` |
| 生成音频帧上限 | `--rollout-max-response-len` | `512` |
| 每副本生成并发 | `--sglang-server-concurrency` | GRPO `2`，MoPD `16` |
| 序列与动态 packing | `--seq-length`、`--use-dynamic-batch-size`、`--max-tokens-per-gpu` | 两者均为 `4096`，启用动态 batching |

`rollout_batch_size × n_samples_per_prompt` 是候选数，`global_batch_size` 是训练 batch；
它们由原生训练配置校验，不在平台层自动绑定。序列长度计入文本、参考和生成时间行，
不要再乘 RVQ 通道数。FLOPs packing 的预算不是最终 pack 长度的硬上限。

## 模型、协议与训练边界

调用方选择结构、执行计划、tokenizer/codec 和精度匹配的 HF 制品与 native DCP。
HF 提供 Omni 权重、processor、协议和模型配置；native initializer 按参数名称、
形状和 checkpoint 格式加载模型权重，optimizer 从新状态开始。
`--ckpt-step` 可指定 native iteration；导入和恢复不比较额外的模型身份或 forward-semantics 清单。

当前策略训练要求 TP=PP=CP=1、关闭 sequence parallel；支持 learner DP 和多个完整
Omni rollout 副本。已验证的示例使用 BF16，模型精度必须匹配制品。Prism 的全部 RVQ
和 sampled stop 参与训练，自然终止和零帧 stop 都保留。采样固定 temperature=1、
top_p=1、top_k=-1、repetition_penalty=1，并采样 stop；评测也遵守该协议。
支持下节的可选 reference KL loss；不支持 `kl_coef` reward shaping、TIS、critic、GSPO 或 full-vocab OPD。激活重计算启用时仅支持
`full / uniform / 1`。

Omni 部署会在创建进程前校验所有模型与 server group，再并行初始化 student/teacher
副本和各模型的 router。router 的 `/live` 仅表示 HTTP 进程存活；完成 engine 绑定后，
还会等待 `/ready`，确认已有可路由的 worker。`placeholder` 只预留 GPU，
不受完整副本的 GPU 整除约束。Omni 自行管理 router，不接受 `--sglang-router-ip`；
可以通过 `--sglang-router-port` 指定第一个模型的本机 router 端口。
router 运行入口是独立脚本，由 `--omni-python` 执行，该解释器无需安装 Slime。

MoPD 每轮先重放轨迹并冻结旧学生 logprob，默认用它作为 PPO 分母。
普通文本在不使用 OPD 的单次更新中保留一次前向的优化；Prism 继续独立打分，
保持现有跨引擎数值诊断的口径。
Omni 分数用于跨引擎对齐诊断。`mean_sample_train_rollout_logprob_max_abs_diff`
是每条轨迹最大误差的平均，不是全 DP 的 global max。工程 GPU 闭环通过不意味着
满足任意更紧的数值容差。Attention 使用原生 TE layer spec 和 auto 后端，可用
原生 `--attention-backend` 指定支持的后端。

MoPD 使用所选 RVQ/stop 动作 logprob，目标归约为
`rvq_weight × mean(RVQ losses) + stop_weight × mean(stop losses)`，权重由 `--policy-loss-weights` 设置。
`--opd-kl-coef` 控制教师相对旧学生的动作蒸馏项；零 task reward 是预期行为。

## 可选 reference KL loss

通过原生参数启用动作流的 reference KL。例如在填写学生 pipeline 和奖励服务后：

```bash
JOB_NAME=wer-with-kl bash examples/prism/run-grpo.sh \
  --hf-checkpoint /assets/student-hf \
  --pretrained-checkpoint /assets/student-native/iter_0100000 \
  --ref-load /assets/reference/checkpoint \
  --use-kl-loss --kl-loss-coef 0.01 --kl-loss-type k3
```

`--ref-load` 使用现有 Megatron reference checkpoint 加载流程；需要指定 iteration 时用
`--ref-ckpt-step`。learner 在训练更新前对同一批动作评分，冻结各流的 `ref_log_probs`。
reference 与 OPD teacher 是两个独立来源，分别使用 `kl_loss_coef` 和 `opd_kl_coef`。

各流在 `valid_mask & loss_mask` 且样本未被移除的位置上计算
`compute_approx_kl(current_log_probs, ref_log_probs)`，随后取每样本的流均值，乘以对应
`policy_loss_weights`，再使用原有 global batch 分母。reference 和 old 分数不参与反向；
只有当前策略保留梯度。零帧轨迹仍计算真实 stop 的 KL，空 RVQ 流贡献为零。

`--kl-loss-type` 支持原有 `k1`、`k2`、`k3`、`low_var_kl`；`--use-unbiased-kl` 使用
`exp(current_log_probs - old_log_probs)` 修正，其中 old 与 PPO 选择的分数来源一致。
总 loss 加上 `kl_loss_coef × kl_loss`，可与 entropy 和 OPD 一起使用。
训练指标 `kl_loss` 是加权 KL 项，尚未乘 `kl_loss_coef`；`rvq/kl_mean`、`stop/kl_mean`
报告各流有训练动作的样本上的 KL 均值。原有 `ppo_kl` 仍表示 old/current log-ratio。

保持 `--kl-coef 0`。优势白化、关闭优势计算、自定义 advantage hook 和 per-token loss
的动作流支持不在这项功能中；默认 GRPO/MoPD 配方仍关闭 reference KL。

## 显式归档与监控

脚本用 `--omni-return-audio` / `--no-omni-return-audio` 控制训练波形。需要完整候选 JSON、JSONL、试听音频和
细分指标时，创建自己的归档 YAML，通过 `--custom-config-path` 使用。例如：

```yaml
prism_artifact_dir: /shared/experiments/prism-grpo/wer_local/running_round_0/artifacts
omni_audio_dir: /shared/experiments/prism-grpo/wer_local/running_round_0/rollout_audio
rollout_sample_observer_path: [slime_plugins.prism.artifacts.observe]
omni_audio_path_function_path: slime_plugins.prism.artifacts.audio_path
custom_rollout_log_function_path: slime_plugins.prism.artifacts.log_rollout_data
custom_eval_rollout_log_function_path: slime_plugins.prism.artifacts.log_eval_rollout_data
```

填写本轮的绝对路径；YAML 不展开 shell 环境变量。归档配置不改变训练/评测音频开关，
纯 MoPD 可以只归档轨迹记录。归档依赖以下显式扩展点：

- `rollout_sample_observer_path: [slime_plugins.prism.artifacts.observe]`
- `omni_audio_path_function_path: slime_plugins.prism.artifacts.audio_path`
- `custom_rollout_log_function_path: slime_plugins.prism.artifacts.log_rollout_data`
- `custom_eval_rollout_log_function_path: slime_plugins.prism.artifacts.log_eval_rollout_data`

自定义奖励和 sample hook 保持原样，不再由启动器包装或重写。教师评分由
`--opd-teacher-config` 独立配置。已有日志回调可以通过
`prism_rollout_log_function_path` / `prism_eval_rollout_log_function_path` 显式组合。
YAML 中的 `rollout_sample_hook_path` 会覆盖 CLI 上的列表。OPD 教师评分独立于这份列表执行。

归档保存所有生成候选，包括未进入训练的候选。请求波形时按 prompt group 保存
公共 prompt、参考音频、候选 WAV 与 JSON；纯 MoPD 的无波形记录按 worker 分片。
生成后先写 pending，评分后写 success/error，保留取消、失败和未选中状态。
每轮汇总 JSONL 与 metrics；不存储重复 replay tensor，完整张量另用
`--save-debug-rollout-data` / `--save-debug-train-data`。

默认训练产物统一归属 `ROUND_PATH`。新任务更换 `JOB_NAME`，重启增加 `RUNNING_ROUND`，
已存在的 Ray 协调目录不会复用。显式归档路径也应跟随本轮目录更新；外置音频根仍按
`prism_artifact_dir` 生成路径摘要，隔离不同运行。

`--resolved-config-path` 保存最终脱敏参数。配置文件直接作为训练输入，不会自动复制
或合成另一份；需要固定源码、服务绑定和输入配置时，将原文件连同实验保存。
TensorBoard / W&B 使用原生 `--use-tensorboard` / `--use-wandb` 及其目录参数。
普通日志默认 Slime INFO、Omni WARNING；诊断时显式设置 `SLIME_LOG_LEVEL=DEBUG`、
`NCCL_DEBUG=INFO`，以及 `--sglang-log-level debug --sglang-log-level-http info
--router-log-level debug`。已有 Ray workers 的环境需要在启动 Ray 时设置，或通过
原生 `--train-env-vars` 传给 learner。
Slime 按日志级别输出，不根据 checkpoint 提示的具体文案过滤消息。Ray 自身的
重复日志合并由 `RAY_DEDUP_LOGS` 控制。

## 指标口径

新增 `rollout/prism/` 与 `eval/<dataset>/prism/` 指标包含：

- reward 及 WER/CER、SIM、JudgeCLAP 分项的原始值/变换后值统计、参与比例；
- 完整 prompt group 的奖励均值、组内标准差、p50/p90、零方差比例；
- 零帧、截断、评分失败、待完成与取消比例；
- 按 reward category、语言和 domain 分组的统计。

训练统计的 `all`、`selected`、`not_selected` 分别表示已归档候选、选入训练的候选
和其余候选。`not_selected` 包括过滤与过采样等原因，不等同于动态过滤失败；
原有 `rollout/dynamic_filter/*` 原因计数继续保留。组内统计只计入完整、评分成功的
group；未启用的奖励分项不当作零分。纯 MOPD 保持原有不生成波形的配置，记录
实际轨迹与教师信息。

Prism 指标的单位与分母如下：

| 指标 | 口径 |
| --- | --- |
| `rollout/audio_frames/*` | 实际生成的目标音频帧数；`response_len/*` 使用 Sample 的输出长度口径，参与训练的动作数见 `policy/*/train_actions` |
| `perf/rollout_audio_frames_per_gpu_sec` | 本次 rollout 收集到的目标帧数 / 耗时 / 学生 rollout GPU 数 |
| `perf/rollout_samples_per_sec` | 本次 rollout 收集到的候选数 / 耗时 |
| `perf/rollout_audio_seconds_per_sec` | 仅当全部候选有真实 WAV 时长（零帧为 0）时报告；不从未知 codec 帧率推算 |
| `perf/train_sequence_rows_per_sec` | 整个训练批的序列行数 / actor 训练耗时，包含 prompt 和音频行，未除 GPU 数 |
| `rollout/policy/{rvq,stop}/{old_log_probs,rollout_log_probs,teacher_log_probs,advantages,opd_reverse_kl}` | 对参与训练的动作求样本内均值，再按有效样本平均；直接反映真实动作分数与优势 |
| `rollout/policy/{rvq,stop}/{valid_actions,train_actions}` | 每条启用轨迹的实际动作数与参与训练的动作数 |
| `train/{rvq,stop}/pg_loss` | 未乘流权重的损失分项，保留全训练 batch 分母；按 `policy_loss_weights` 组合后等于策略损失 |
| `train/{rvq,stop}/{clipfrac,ppo_kl_estimate,opd_logratio_mean,train_rollout_logprob_abs_diff}` | 先按样本内该类动作平均，再按有效样本平均；`active_samples`/`active_ratio` 给出分母和覆盖率 |
| `rollout/{domain,teacher}/<label>/{rvq,stop}/opd_logratio_mean` | 实际训练使用的冻结学生与教师分数差，按有效样本加权聚合；分组标签经过 URL 编码 |

OPD log-ratio 是已采样动作上的统计，单批可以为负，不是完整词表的精确 KL。
零帧样本仍参与 stop 训练；没有 RVQ 动作时不把 RVQ 诊断伪造为零。
新增统计不改变 loss、梯度或采样。分组均值汇总分子与有效计数，不对各 GPU
的均值再平均。熵分项只在原配置已要求计算熵时输出，不额外启动模型计算。

`rollout/prism/` 与 `eval/<dataset>/prism/` 下的质量汇总还包括：

- `frames/*`、`audio_seconds/*` 的均值、p50、p95，以及时长覆盖率；
- `component/<name>/coverage_ratio`（有效评分数 / 当前样本范围）、`valid_ratio`
  （有效评分数 / 该服务已有返回或已知失败数）、`error_ratio` 和无效音频比例；
- `component/wer/language/<language>/corpus_error_rate`：总编辑错误数 / 总参考
  字符或词数。保留原每条错误率均值，语言分母不混合，缺失评分不补零；
- `component/judge_clap/attribute/<attribute>/mean` 与对应覆盖率；
- reward category、语言、domain、teacher 分组。单候选的 MOPD/eval 不再报告
  没有区分意义的组内 reward 零方差指标。

异步预取时，上述 rollout 吞吐是收集窗口的交付速度，不是推理引擎的纯 decode
吞吐；外部教师和评分服务的 GPU 不在学生 GPU 分母内。等待占比不是 GPU 空闲率。
Prism 路径不再报告空文本重复率、未填充的缓存命中率、`reward==1` 的文本 pass@k
和通用 LM 公式估算的 TFLOPS。没有新增 Prism TFLOPS 估算或声学复读检测。

MOPD 可使用独立的音频质量 eval，配置见 [训练与评测说明](TRAINING_MODES.md#独立评测grpo-与-mopd-共用质量指标)。

## JSONL 输入

脚本使用 `--input-key script --metadata-key metadata`。外层 `script` 保留完整文本，
`metadata.omni_prompt` 使用 HF processor 的统一 request；实际渲染发生在 Omni。
模型字段只从该对象读取；程序化调用也可直接把完整 request 放在 `Sample.prompt`。
WER 的 `language`、`transcript_target` 只从顶层 metadata 读取。运行时不接受旧任务标签、
JSON 字符串 instruction 或列表 references，也不在两种存储位置之间回退。
参见 [mixrl.example.jsonl](../data/mixrl.example.jsonl)：

```json
{
  "script": "你好，欢迎来到今天的节目。",
  "metadata": {
    "domain": "chinese",
    "language": "zh",
    "transcript_target": "你好，欢迎来到今天的节目。",
    "omni_prompt": {
      "modality": "speech",
      "global_instruction": {"Gender": "female"},
      "references": {"audio1": "/assets/reference-zh.wav"},
      "reference_attributes": {
        "timbre": "audio1", "accent": "audio1",
        "prosody": "audio1", "emotion": "audio1"
      }
    }
  }
}
```

`modality` 为 `speech` 或 `sing`，默认 `speech`。`omni_prompt.script` 可以覆盖
外层文本，接受字符串或 `[{"text":"你好。","local_instruction":{"Emotion":"happy"}}]`
这样的分段数组。`global_instruction` 和分段 `local_instruction` 都使用对象；
属性名、规范化与字段排序由 HF 制品中的 renderer 校验。

`references` 把 `audio1`、`audio2` 等槽位映射到非空路径/URI，允许槽位跳号及
两个以上参考。`reference_attributes` 显式指定 `timbre`、`accent`、`prosody`、
`emotion` 的来源，值为实际槽位或 `none`；省略的属性不引入参考条件。
SIM 使用显式 `reference_attributes.timbre`，必须指向实际存在的参考音频。
`global_tokens` 为正整数或 null，表示目标音频帧数；生成上限仍由 rollout 参数控制。

模型语言仅由 `global_instruction.Language` 决定是否展示。上例的
`metadata.language="zh"` 只用于 WER/CER 归一化，不会自动插入 `Language: Chinese`。
`metadata.transcript_target` 和 `metadata.language` 不传给 ASR 解码器。
revision 7 支持歌唱音乐属性并按字典序渲染；显式属性输入不依赖自然语言编译服务。

输入数据应直接使用上述统一 request 格式。旧任务标签、列表 references 和隐藏语言设置
需要在数据准备阶段按目标 HF processor 的协议转换；运行入口不执行数据迁移。

学生的 `--hf-checkpoint`、Omni `model_path` 与教师制品一起切换，教师绑定填写服务地址。
教师模型路径由服务端配置，客户端不校验模型身份哈希或权重 checksum。
renderer 代码的哈希属于 replay 身份，旧数据队列
与新的制品配置应使用各自的实验目录。调用方负责选择匹配的模型、tokenizer 与权重制品。

JudgeCLAP 读取实际全局 instruction 对象，忽略 `Language` 这一生成控制字段，
并保留现有 frozen scorer query 协议。它只评分当前服务支持的属性；如 `Genre` 等
未支持的音乐类别会明确报错，不会静默丢弃或当作已评分。局部片段指令也不会被当作
全局评分目标。样本级 `metadata.id` 仍用于归档定位，与参考槽位名无关。

MOPD 使用 `metadata.domain` 选择教师。domain 的采样比例由数据记录比例及数据源决定，不由教师响应速度决定。显式 domain 没有教师映射时会报错。

## Ray 与吞吐配置

两个脚本默认使用平台分配的全部 GPU；无平台变量时使用单机单 GPU。
Ray head/worker 和训练资源都由 [RAY.md](RAY.md#平台环境) 中的节点环境派生，
无需提前启动 Ray。所有节点使用兼容的 Python/依赖，并能访问源码、权重、数据和音频。

一个 rollout GPU 对应完整 preprocessing、AR、vocoder pipeline。共卡通过
`--colocate` 配置。原生共卡默认 offload；
驻留共卡使用 `--no-offload-train --no-offload-rollout`。学生 KV 容量参考
[student.yaml](../configs/student.yaml)。开启 rollout offload
时 adapter 自动启用 memory saver，先恢复权重、完成发布，再恢复 KV/graph。

例如分配 4×8 GPU 后，MoPD 默认 DP32、32 个完整 rollout 副本，
`rollout-batch-size=global-batch-size=32`，每 prompt 一条轨迹。较大 batch 可同时覆盖
这两个参数，例如均设为 2048。`--use-dynamic-batch-size` 配合 `--max-tokens-per-gpu` 和 `--balance-by-flops`
控制 packing；实际是否只有一个 microbatch 要查看运行指标。

`--rollout-execution-workers 16` 分配完整 prompt group 的 CPU 执行。总生成并发是
每副本并发乘副本数；奖励服务 `max_inflight` 是每个 worker 的限额，需一起规划。
`--rollout-gc-threshold 70000 --rollout-tensor-buffer` 是可选性能配置，后者仅用于
object-store。`--resident-cleanup-interval` 默认 1，增大前应测量显存和保存/恢复。
历史性能对照与验收范围见 [VALIDATION.md](VALIDATION.md)。

## 服务绑定、训练方式和评测

`--omni-config` 选择学生 pipeline；`--omni-reward-config` 绑定冻结奖励服务；
`--opd-teacher-config` 按 `metadata.domain` 选择冻结教师。
三种用法共用 [student.yaml](../configs/student.yaml)。GRPO 与 MixRL 共用
[rewards.yaml](../configs/rewards.yaml)，奖励函数只初始化实际选择的服务组件；
MoPD 使用 [teacher-endpoints.yaml](../configs/teacher-endpoints.yaml)。
外部教师直接使用 [teacher-score.yaml](../configs/teacher-score.yaml) 启动 Omni 服务；
Slime 托管教师的资源配置见下方[可选示例](#slime-托管冻结教师可选)。
直接填写这四份 YAML；外部教师和奖励服务需另行启动。
`audio_transport: path` 要求相同共享路径；不共享文件系统时选择 upload。
鉴权变量必须在训练/rollout worker 环境中可读。服务故障抛错，不替换成数字零奖励。
合法零帧 sampled stop 的零奖励仍是有效训练样本。

固定奖励、MixRL 类别和 MoPD 的完整命令见 [TRAINING_MODES.md](TRAINING_MODES.md)。
评测使用 `--eval-config ... --eval-interval ...`，各数据集可以选择独立奖励函数。
MoPD 质量评测还需 `--omni-eval-return-audio` 和
`--no-opd-eval-teacher-scoring`，训练仍保持无波形蒸馏。

## 恢复与检查

将 `--pretrained-checkpoint` 换为 `--load <RL checkpoint>`。原生 loader 会优先执行恢复，
忽略首次初始化 hook；optimizer、scheduler 和训练进度按框架恢复。
`--num-rollout` 是总上限；延长时使用 `--use-checkpoint-opt-param-scheduler`，有意改变
schedule 时用 `--override-opt-param-scheduler`。

启动脚本不再提供 `--dry-run`。可以检查 shell 语法和模型结构参数：

```bash
for script in examples/prism/run-grpo.sh examples/prism/run-mopd.sh examples/prism/ray.sh; do
  bash -n "$script"
done
python -m slime_plugins.models.prism_configuration --hf-checkpoint /assets/student-hf
```

第二条只打印模型架构参数，不启动训练。完整参数的合法性、资产和服务检查由原生
框架/插件处理。首轮 GPU 检查用 `--num-rollout 1` 和短生成预算，覆盖加载、重放、
反向、完整权重发布和保存/恢复。CPU 测试不替代 GPU 验收。

仓库提供独立 GPU 工程验收入口。它先加载 native 学生并发布版本 1，再通过 Omni router 生成真实轨迹，核对更新前 logprob，执行一次 native policy objective/Megatron Adam 更新，发布版本 2，并核对更新后生成动作的 logprob。下面使用 learner GPU 0 和 Omni GPU 1；把 `--gpus` 改为 `1,2` 可在三张以上 GPU 上验证两个完整副本：

```bash
cat > /runs/prism-smoke-options.json <<'JSON'
{"alignment_atol": 0.05, "alignment_rtol": 0, "learner_max_frames": 8, "learning_rate": 0.001}
JSON

"$TRAIN_PYTHON" -m scripts.test_prism_omni \
  --hf-checkpoint "$HF_CHECKPOINT" --config "$OMNI_CONFIG" \
  --python "${OMNI_PYTHON:-$TRAIN_PYTHON}" --gpus 1 \
  --artifact-dir /runs/prism-publication-smoke \
  --native-checkpoint "$NATIVE_CHECKPOINT" \
  --mode publication \
  --publication-hook scripts.prism_publication_smoke:publication \
  --post-generation-hook scripts.prism_publication_smoke:verify_generated \
  --publication-args-json /runs/prism-smoke-options.json \
  --expected-version 2 --max-frames 8
```

`0.05` 是这里显式选定的 BF16 数值容差示例；结果会分别报告 RVQ 与 stop 的最大和平均误差。该工程检查使用一个明确的非零 advantage 来验证反向和权重更新，不调用实际 reward model，也不替代完整 prompt group 的 GRPO 或 domain 教师训练验收。独立脚本直接使用 engine 控制方法，没有执行 Ray placement；实际训练的资源生命周期仍由 `train.py` 管理。

`--mode static` 可单独检查 HF artifact、pipeline、router 和真实 rollout 协议，省去两个 hook 及 native checkpoint。这个模式不发布 learner 权重，结果会明确记录这一点。原始 `/generate` JSON、typed replay、更新前后分数和清理状态均写入结果目录。


## 教师、发布与 checkpoint 协议

The frozen teacher client loads each configuration once per process. The teacher
YAML's `client` mapping controls `batch_size` (default 8), `batch_wait_ms` (2 ms),
`max_inflight` (16 per execution worker), `max_attempts` (2), and
`request_timeout_s` (120 s). These settings bound request aggregation and transient retries.
Only fixed-action scoring retries transport errors and HTTP 429/502/503/504;
input-hash mismatches remain fatal. Prism teacher requests and
results use version 2. Each result contains one `selected_stop_logprobs` value
per student stop/continue action, computed directly from FP32 logits. The teacher
infers these actions from the supplied codes and finish reason: one continue
per frame, plus a final stop only for a natural termination. No probability
fallback or two-column response is accepted. The HTTP batch envelope remains
version 1. Upgrade the Slime client and Omni teacher together.

The native example script enables `--omni-compact-replay`: training tensors retain the exact
replay inputs and actions, while completed samples discard duplicate wire lists.
The teacher request can still be reconstructed from those tensors. Pass
`--no-omni-compact-replay` when a custom postprocessing hook needs the original
wire trace. The native archive does not require that duplicate trace.

Colocated Omni tensor updates follow Slime's ordinary bucket loop: pause
generation, send every bucket and wait for its result, then synchronize training
ranks before resuming generation. Slime checks each update's success, cache flush,
and paused state, and verifies the installed weight version before resuming.
The sender owns completeness; requests do not carry a publication manifest or
require manifest acknowledgements. A failed update requires a fresh pause and a
complete retry because the tensors are loaded in place.

Prism examples use Slime's ordinary checkpoint save and resume flow. Model and
optimizer state are saved by Megatron, and the rollout manager saves the dataset
cursor separately. Use `--load` to resume from the checkpoint root, with
`--ckpt-step` when selecting a specific saved iteration. After an interrupted
save, select an iteration with both complete model files and its dataset state.
Dataset state is required for ordinary resume; use Megatron's explicit
`--finetune` mode to load a model and start a new dataset run. Cursor snapshots
retain dataset progress and epoch state. The existing straw commit format is unchanged.


## 示例入口整理

| 先前的用途 | 当前入口 |
| --- | --- |
| GRPO / MixRL 训练 | `run-grpo.sh`；MixRL 覆盖数据和 reward 函数 |
| MoPD 训练 | `run-mopd.sh` |
| 单独的 Inspire Ray 引导 | 已整合进 run 脚本，直接读取平台环境 |
| GPU / Omni / publication / 数值诊断 | `scripts/` 中四个 Prism 工具，见 [TOOLS.md](TOOLS.md) |
| 服务 YAML 准备 | 直接编辑 `configs/` 中的学生、奖励和教师配置 |
| 可选 eval / 归档 | 使用文档中的原生 YAML 片段 |

训练参数使用 `--lr`、`--global-batch-size`、`--load` 等原生接口。目录由
`OUTPUT_ROOT`、`RUN_NAME`、`JOB_NAME` 和平台任务/轮次派生；GPU 数量由平台环境提供。
组件 YAML 由原生接口读取，没有额外的作业 schema 或生成的 `experiment.yaml`。
复制配方到仓库外时设置 `SLIME_ROOT`，其余参数在副本中修改。历史验收报告和归档保留。


## 公共动作流与教师路由

模型重放输入保留在 `tokens` 和 `multimodal_train_inputs`。策略动作、有效位置、训练 mask、生成分数、
教师分数和训练优势独立存放在 `Sample.policy_actions`，不再混入模型输入。
每个动作流的 tensor 第一维与真实序列行对齐，后续维度保留同一时间行的多个动作头。
Prism 显式声明 `rvq` 和 `stop` 两个动作流。普通 LLM 保留原有 token 训练路径，
不会自动转换成 `text` 动作流；只有显式提供 `Sample.policy_actions` 的样本才进入动作流路径，
并且必须提供 `--custom-megatron-log-prob-function-path` 指定的 extractor。
参考音频只参与条件重放，`valid_mask` 不选择它；零帧轨迹仍保留真实 stop。
`loss_mask` 可以屏蔽任意有效动作或 RVQ 头，不改变教师评分和模型重放。
RVQ `actions` 是监督目标，必须在有效位置与 successor 条件输入一致；原始 trace 只作诊断。

Prism 模型通过 `accepts_policy_actions = True` 显式声明 forward 接收动作目标。
`--custom-megatron-log-prob-function-path` 仅选择输出打分函数，不决定模型 forward 参数。
打分函数仍使用 `(args, batch, output)` 签名和 `policy/<stream>/log_probs`、可选 `entropy`
的扁平返回格式。更新前冻结分数仍由 learner 负责。

共同支持的 GRPO / selected-action OPD / PPO clipping 路径使用相同的动作选择与归约。
旧学生和教师分数冻结后统一计算 OPD 优势，loss 中不会再次添加该项。
`--policy-loss-weights '{"rvq": 1, "stop": 0.5}'` 设置各动作流权重，未指定的流默认权重为 1。
动作分项的 `pg_loss` 保留 global batch 分母；诊断按有效样本归约，空动作流不伪造零分。

新的 MoPD 配方使用 `--opd-teacher-config`。教师配置由公共 OPD 模块选择 domain 绑定，
模型评分函数只负责具体协议：

```yaml
score_function: slime_plugins.prism.teacher.score_teacher
models:
  zh_teacher:
    endpoint: http://teacher-host:8201
by_domain:
  chinese: zh_teacher
```

奖励服务 fingerprint、教师模型身份哈希与权重 checksum 的校验和配置项均已移除。
教师配置只绑定 endpoint（或命名 deployment）及认证；请求对应关系、动作形状和分数有效性仍按协议检查。

教师绑定 YAML 显式填写 `score_function`，仓库模板已包含该字段。教师评分在公共 OPD 阶段执行：
未配置任务奖励时，公共 OPD 使用零任务奖励；配置了 reward 函数则正常执行。
使用 `--no-opd-eval-teacher-scoring` 或自定义 YAML 的 `opd_eval_teacher_scoring: false`
可独立关闭评测教师打分。rollout 执行状态持有 `TeacherRuntime`，统一管理评分客户端的组批、刷新和关闭。

Prism 的轨迹、打分器和启动配置统一使用本节接口；不自动转换先前的 Prism 数据字段、
打分返回格式或启动参数。原生文本 OPD 的 `custom_rm + custom_reward_post_process` 方式保持可用。
OPD 分组指标使用公共 domain / teacher 前缀；任务质量归档使用 `prism/` 前缀。

新增 CPU 回归包括 `tests/test_policy_streams.py`、`tests/test_opd_teacher_routes.py`，
以及样本转换、packing、教师协议和分布式指标测试。CPU 回归不替代真实 GPU 数值对齐、
权重发布和 checkpoint 恢复验收。

教师配置通过 `client` 设置组批、并发、重试和超时：

```yaml
client:
  batch_size: 8
  batch_wait_ms: 2
  max_inflight: 16
  max_attempts: 2
  request_timeout_s: 120
```

默认值由公共 `TeacherClientOptions` 定义。公共 `BatchedTeacherClient` 负责调度与重试，Prism 客户端负责请求编码和响应校验；没有插件全局客户端缓存。


## Omni pipeline 与部署配置

[student.yaml](../configs/student.yaml) 和 [teacher-score.yaml](../configs/teacher-score.yaml)
都是原生 Omni 配置，可以直接传给 Omni CLI，无需导出或转换：

```bash
python -m sglang_omni.cli serve --config examples/prism/configs/student.yaml
```

模型、codec 和引擎参数在 pipeline YAML 中配置。Slime 直接传递原文件，Omni 负责解析、
重复 key 检查和管线配置校验。Slime 按部署覆盖模型路径和 memory saver，启动后通过实际的
模型信息检查更新目标；当前权重发布仍要求单个更新 stage、TP=1。

Slime 的训练入口设置：

```bash
--omni-update-stage tts_engine
--omni-worker-capabilities chat audio_input audio_output
```

前者选择权重通信组、权重更新及配套暂停、恢复和版本查询的目标；后者声明向 router 注册的
能力。多模型部署通过现有的 `server_groups.overrides` 覆盖 `omni_update_stage` 和
`worker_capabilities`。显存释放和恢复省略 `stages`，由 Omni 操作全部阶段；CPU stage
返回空的显存状态，未启用 memory saver 的 GPU stage 报错。Slime 校验逐 stage/rank 结果，
失败或超时保持入口关闭。

`overrides` 中的 `omni_config`、`omni_update_stage`、`worker_capabilities`、
`model_path` 和 `enable_memory_saver` 由 Slime 使用；其余标量键值按原生 Omni CLI 选项传递，
例如 `tts_engine.engine.max_running_requests: 64`。列表和字典放在原生 pipeline YAML 中，避免
原生 CLI 的自由字段将结构化文本保留为字符串。这些显式选项优先于 pipeline YAML，
host、port 等由 Slime 分配的启动参数不能通过此处替换。
如果每个实际服务组都提供了 `omni_config`，可以只传 `--sglang-config`，省略全局
`--omni-config`；更新目标和 router 能力也可以全部由各组 overrides 提供。

示例 pipeline 显式将 `tts_engine.engine.cuda_graph_backend_prefill` 设为 `disabled`，
匹配默认共卡训练。无 offload 的部署需要 prefill graph 时，可通过原生配置或部署 override
改为 `breakable`。vocoder 的 graph 开关由 Omni 在 memory saver 模式下处理。
冻结模型默认通过原生 `<update_stage>.engine.enable_weights_cpu_backup` 选项保留权重，
不依赖后续 learner 发布；独立 scoring 配置也显式开启 CPU 权重备份。

`update_weights` 仍由模型部署层保管，训练端只向可更新模型发布权重。Omni 启动后使用
已加载的 checkpoint 提供服务；训练流程完成权重同步后开始 rollout。独立工具需要在首次
发布前暂停服务时，显式调用 `pause_generation()`。显存恢复遵循后端保存的运行/暂停状态；
通信组重置不改变接单状态，完整发布确认前不会恢复接单。

迁移旧配置时，移除整个 `slime:` 块，将 router 能力放到 Slime 参数或部署 overrides，
将推理设置移到原生 pipeline 字段。原来的 `update_stages`、`memory_stages`、
`offload_overrides`、`frozen_offload_overrides` 和 YAML 导出入口已移除。

## Slime 托管冻结教师（可选）

默认训练入口使用外部 teacher，只绑定 URL。需要 Slime 管理 teacher GPU 时，可直接引用
[teacher-score.yaml](../configs/teacher-score.yaml)，无需复制另一份 managed pipeline。

下面的 `/shared/configs/deployment.yaml` 配置一个 student 和两个冻结 teacher，
共占 3 个 rollout GPU；learner 资源另行配置。模型部署的 `model_path` 覆盖 pipeline
中的模型路径，因此两个 teacher 可以共用同一份 scoring 配置。

```yaml
sglang:
  - name: student
    update_weights: true
    num_gpus_per_engine: 1
    server_groups:
      - worker_type: regular
        num_gpus: 1
        overrides:
          omni_config: /shared/configs/student.yaml
          omni_update_stage: tts_engine
          worker_capabilities: [chat, audio_input, audio_output]
  - name: teacher_zh
    model_path: /assets/prism-teacher-zh
    update_weights: false
    num_gpus_per_engine: 1
    server_groups:
      - worker_type: regular
        num_gpus: 1
        overrides:
          omni_config: /shared/configs/teacher-score.yaml
          omni_update_stage: tts_engine
          worker_capabilities: [audio_input]
  - name: teacher_en
    model_path: /assets/prism-teacher-en
    update_weights: false
    num_gpus_per_engine: 1
    server_groups:
      - worker_type: regular
        num_gpus: 1
        overrides:
          omni_config: /shared/configs/teacher-score.yaml
          omni_update_stage: tts_engine
          worker_capabilities: [audio_input]
```

复制上述两个 pipeline 到对应路径并填写资产。训练侧的
`/shared/configs/teacher-endpoints.yaml` 按 deployment 名称绑定：

```yaml
models:
  zh_teacher:
    deployment: teacher_zh
  en_teacher:
    deployment: teacher_en
by_domain:
  chinese: zh_teacher
  english: en_teacher
score_function: slime_plugins.prism.teacher.score_teacher
```

运行时传入 `--sglang-config /shared/configs/deployment.yaml` 和
`--opd-teacher-config /shared/configs/teacher-endpoints.yaml`，并按分配设置 learner、
rollout 和共卡参数。`update_weights: false` 阻止 learner 向 teacher 发布权重；
`omni_update_stage` 仍用于定位模型信息等管理操作，不会把冻结 teacher 变成可训练模型。
