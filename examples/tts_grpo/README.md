# Local 的公共语音 RL 能力

从 Delay 适配项目 `0a1f6024` 迁入奖励、数据、评估、产物和部署能力。
Local 继续使用自身的 Omni 原生轨迹、Megatron 模型、CP 切分和权重映射。
本目录是正式训练入口，替代此前 `outputs/` 下的四卡草稿。

## WER 定义

当前默认 WER 与旧项目一致：

- NFKC、转小写、归一化空白，删除 Unicode 标点和符号。
- 中文与粤语先做繁简归一化；中文、粤语、日语、韩语和泰语按 Unicode grapheme 分词，其余支持语言按词。
- 每条样本可指定 `language`；缺省使用 `--wer-language`，保留 `en`、`zh` 别名。
- 奖励为 **`1 - min(raw_wer, 1)`**。原始 WER 可以超过 1，仍单独记录。
- 归一化后为空的参考文本对应 raw WER `+inf`、奖励 0，保持旧定义。
- 真实空生成和不足 0.1 秒的音频得 0 分；网络故障不会被转换成模型低分。

这改变了此前 Local 的未截断 `1 - WER` 目标。比较历史曲线时应标明定义版本。
Local 的组内标准差仍使用 `unbiased=False`，动作 loss 与 CP 归约不变。

## 奖励配置与服务

只使用 WER 时继续传 `--asr-endpoint`。组合奖励使用 `--reward-config`：

```yaml
reward:
  components:
    - {name: wer, weight: 0.4}
    - {name: reference_similarity, weight: 0.6}
services:
  wer:
    endpoint: ${TTS_ASR_URL}
    protocol: qwen3_asr_chat_path
    model: qwen3-asr-1.7b
    auth_token_env: INSPIRE_API_KEY
    runtime: {timeout_seconds: 120, concurrency: 32, max_retries: 2}
  timbre_sim:
    endpoint: ${TTS_SIM_URL}
    auth_token_env: INSPIRE_API_KEY
    runtime: {timeout_seconds: 120, concurrency: 32, max_retries: 2}
```

提供 `reward_wer_sim.yaml`、`reward_wer_sim_judge.yaml`、`reward_noop.yaml`。
权重显式相加，不自动归一化；零权重组件不会访问服务。示例权重不是 Local 最优参数的实验结论。

每次 rollout 共享每个服务的连接池、并发限制和统计。429、5xx、连接失败和超时有限重试；
鉴权、参数和协议错误直接失败。取消时关闭连接池。生成许可在奖励调用前释放，避免 ASR 等待占住生成并发。
指标包括 requests、attempts、failures、queue_wait_seconds、request_seconds，另保留各奖励组件的 reward/raw。

WER 同时支持 multipart `/v1/audio/transcriptions` 和旧版 `qwen3_asr_chat_path` 的共享 WAV URI 协议。
Judge 对每条 rubric 请求 targeted yes/no logprob，并平均归一化后的 yes 概率。
SIM 按参考音频分组，每次最多 16 对路径，按响应 ID 恢复原始顺序。

参考相似度目前真正实现的是 **timbre**。旧实现的 accent/prosody/emotion 仍是明确标记为
`implemented=false` 的零分占位，若指定仍参与平均。使用 `uses: [timbre]` 可得到完整实现的音色目标。
SIM 的预处理取第一声道、重采样到 16 kHz、最多评估前 30 秒。
独立服务位于 `slime/serving/tts_sim`，包含动态 batching、参考 embedding 缓存和请求准入限制。

## 数据、校验和恢复身份

兼容已有简单 JSONL，并允许独立的评分文本：

```json
{"id":"example-1","text":"Read this sentence.","target_text":"Read this sentence.","language":"english"}
```

可选字段为 `instructions`、`ref_audio`、`ref_text`、`reference_audios`、`rubric`、`domain`、`metadata`。
`target_text` 缺省为 `text`；ID 必须唯一，未提供时按源行号生成。
也支持旧项目的 canonical 数据语义：

```json
{"id":"example-2","script":"Please read this calmly.","target_text":"Please read this calmly.","global_instruction":"A calm voice.","language":"english","reference_audios":[{"id":"voice","path":"/inspire/REPLACE_ME/reference.wav","uses":["timbre"]}],"rubric":[{"dimension":"delivery","statement":"The speaker sounds calm."}]}
```

`script` 可以是字符串或文本条目列表。Local 目前只有全局 instruction，逐条 `local_instruction`
会明确报错。有多个评分参考音频时，必须用 `ref_audio` 显式选择生成时的参考音频。
MossFlux/ChatML/codec 渲染继续由 Local 的 Omni processor 负责。

启动前检查全部 JSONL、重复 ID、语言、奖励必需字段和可解码的参考 WAV，未知顶层字段报错。
恢复指纹覆盖 JSONL、参考音频真实路径/文件状态/音频元数据、模型和 processor 配置、奖励定义及 metadata overrides。
sampler 状态原子写入，恢复时检查 seed、shuffle、fanout 和计数器。
旧 sampler 无法验证参考音频新指纹，仅无参考音频的数据可按旧 JSONL/seed 检查恢复，并输出警告。

生成音频需通过完整 PCM WAV 检查并匹配响应声明的采样率，再原子写入。
公共层支持 Local 单声道等合法布局，不沿用 Delay 固定双声道的约束。
产物身份包含数据集、rollout、样本和 attempt，重试与不同评估集不会相互覆盖。

## 完整组重试

`--rollout-group-max-retries` 默认 2，`--max-recoverable-rollout-failures` 默认每轮 32 个失败样本。
可恢复故障导致整个 prompt group 重新生成，使用确定性的 attempt seed。
失败尝试的正常兄弟样本也不进入训练。达到任一预算就结束本轮，绝不凑出不完整训练批次。
所有尝试和所有组必须来自同一个已发布策略版本；轨迹错误、评分协议错误和权重版本漂移直接失败。

## 独立评估

三种互斥入口：

```bash
--eval-data /inspire/project/eval.jsonl
--eval-prompt-data english /inspire/project/en.jsonl chinese /inspire/project/zh.jsonl
--eval-config /inspire/project/eval.yaml
```

配置见 `eval.example.yaml`。优先级为数据集覆盖 `eval.defaults`、再覆盖 CLI 缺省值。
每个数据集可配置采样次数、温度、长度、语言、奖励配置和 metadata overrides。
默认每条评估文本采样一次。评估新建 sampler，不推进训练 sampler；指标为 `eval/<name>/...`。
Local 原生动作协议仍要求训练和评估都使用 `top_p=1, top_k=-1`，不支持的值在预检时拒绝。

## Inspire 单节点训练

默认配置为 4 张 H200：Local 学生和 ASR 各一张，Megatron 使用剩余两张。
`plan` 运行真实 Inspire dry-run，只有 `submit` 创建任务。

```bash
export PROMPT_DATA=/inspire/your-project/train.jsonl
export EVAL_DATA=/inspire/your-project/eval.jsonl
export WER_LANGUAGE=zh
export JOB_NAME=moss-local-grpo-$(date +%Y%m%d-%H%M%S)
export NUM_ROLLOUTS=3 MAX_RESPONSE_LEN=128 SAVE_INTERVAL=1 EVAL_INTERVAL=1
bash examples/tts_grpo/submit_inspire.sh --dry-run
bash examples/tts_grpo/submit_inspire.sh --submit
```

| 变量 | 默认值或用途 |
| --- | --- |
| `INSPIRE_WORKSPACE` / `INSPIRE_PROJECT` / `INSPIRE_GROUP` | 当前仓库 CQ 资产范围 |
| `INSPIRE_QUOTA` / `INSPIRE_NODES` | `4,60,800` / `1`，quota 按每节点计算 |
| `INSPIRE_IMAGE` / `INSPIRE_SHM_SIZE` / `INSPIRE_PRIORITY` | 现有训练镜像 / 128 GiB / 10；改资源时以 Live 目录为准 |
| `MODEL_DIR` / `TRAIN_CHECKPOINT` / `CODEC_DIR` / `ASR_MODEL_DIR` | `INSPIRE.md` 中的规范模型资产 |
| `OMNI_DIR` / `MEGATRON_DIR` | 独立维护的 editable checkout |
| `TRAIN_GPUS` / `TP_SIZE` / `CP_SIZE` | 扣除本地服务后的卡数 / 1 / 1 |
| `ROLLOUT_BATCH_SIZE` / `N_SAMPLES_PER_PROMPT` / `MICRO_BATCH_SIZE` | 2 起、按 DP 向上对齐 / 4 / 1；global batch 自动取前两项乘积 |
| `LR` / `NUM_ROLLOUTS` / `MAX_RESPONSE_LEN` | 3e-6 / 100 / 512 |
| `SAVE_INTERVAL` / `EVAL_INTERVAL` / `EVAL_SAMPLES` | 10 / 10 / 1 |
| `EVAL_CONFIG` / `EVAL_TEMPERATURE` / `EVAL_MAX_RESPONSE_LEN` | 独立评估控制 |
| `REWARD_CONFIG` / `ENDPOINT_BUNDLE` / `REWARD_SECRET_ENV` | 组合奖励、非可执行 endpoint 文件、仅 owner 可读的凭据文件 |
| `ASR_ENDPOINT` / `ASR_PROTOCOL` / `ASR_MODEL` / `ASR_AUTH_TOKEN_ENV` | 外部 ASR；外部奖励配置会关闭本地 ASR |
| `OMNI_ENDPOINTS` | 空格分隔的外部学生地址，设置后关闭本地学生启动 |
| `REF_AUDIO_ROOT` | 传给 Omni 的本地参考音频允许目录 |
| `REWARD_CONCURRENCY` / `REWARD_TIMEOUT` / `REWARD_MAX_RETRIES` | 8 / 120 秒 / 2 |
| `GROUP_MAX_RETRIES` / `FAILURE_BUDGET` | 2 / 32 |
| `OUTPUT_DIR` | qb-ilm2 运行根目录；产物在 `running_round_N` 下 |
| `AUTO_FAULT_TOLERANCE` / `MAX_JOB_RETRIES` | 0 / 3，显式开启平台重启 |
| `MAX_HOURS` / `STARTUP_TIMEOUT` | 默认不设平台时限 / 启动等待 1200 秒 |
| `EXTRA_PYTHONPATH` | 旧镜像使用的可选共享依赖目录 |

训练环境需安装更新后的 requirements。当前 `20260908-cu130-v1` Notebook 缺少 `zhconv`；
本次验证准备了共享纯 Python 依赖目录，代码已在 requirements 中声明依赖。
可在联网 CPU 空间准备依赖后固化镜像，或使用共享 `EXTRA_PYTHONPATH`，保持训练镜像的 Torch/CUDA/Megatron 栈。
该变量会传给训练 Ray 进程。SIM 环境另需 `tts-sim` extra，以及与 Torch 匹配的 torchaudio。

## 独立服务、多节点和自动恢复

服务生命周期由 `serving/manage.py` 管理，参见其 README。填好 TOML 中的资产与资源后，逐服务显式部署。
然后接入 bundle 与 owner-only 凭据文件：

```bash
export ENDPOINT_BUNDLE=/inspire/your-project/endpoints.env
export REWARD_SECRET_ENV=/inspire/your-project/reward-secret.env
export REWARD_CONFIG="$PWD/examples/tts_grpo/reward_wer_sim.yaml"
```

凭据文件使用严格的 `KEY=VALUE` 格式，不作为 shell 执行，不放入 Ray runtime-env JSON。
Endpoint 文件仅包含 URL。服务模型注册名称、镜像和 quota 都需按当前平台资产填写。

多节点必须使用每节点 8 卡配额和外部奖励服务。每个节点启动一个 Local 学生，或使用显式外部学生地址。
启动器根据 PET rank 启动 Ray head/worker，等待学生健康检查和全部节点就绪，再传入 `--ray-address` 和学生列表。
例如 2 个节点各 1 张学生卡、7 张训练卡，TP=CP=1 时 DP=14，7 个 prompt × 4 次采样得到 global batch 28。

平台重启时，启动器扫描此前 `RUNNING_ROUND`，寻找完整 torch_dist checkpoint 和同轮 sampler 状态，
包括 rollout 0；找不到完整恢复点会报错，不静默回到初始化权重。
手工恢复设置 `TRAIN_CHECKPOINT=<旧输出>/checkpoints` 并使用新的 `OUTPUT_DIR`。
`NUM_ROLLOUTS` 是最终总轮数，不是新增轮数。

本地服务失败会传递到 Job；worker 观察 head 的成功/失败状态；启动等待有时限；退出清理自己启动的服务进程组。
多节点 Ray 的清理限定在专用 Job 容器中，`run` 入口应在分配好的 Job 内使用。

## 验证

CPU 回归包含真实本地 HTTP 连接池、重试和取消；多语言 WER；Judge/SIM 协议；数据漂移；
完整组重试和策略版本检查；WAV 原子写入；评估隔离；Inspire 生命周期与 endpoint bundle。
新增测试已注册到 `.github/workflows/pr-test.yml.j2`，可作为独立 CI 入口运行。

```bash
python -m pytest tests -q
PYTHONPATH=. python tests/test_tts_rollout_services.py
PYTHONPATH=. python tests/test_tts_inspire_serving.py
```

GPU 验证可用 `tools/tts_wer_e2e.py --eval-config ...`。
本次完整 CPU suite 通过；真实 GPU 验证在 guard 的资源检查阶段退出：指定 Notebook 中没有原占卡 supervisor，
GPU 已被其他进程使用，因此没有停止进程或开始训练。多节点 NCCL、新组合奖励的模型质量和全新镜像启动仍需实际 GPU 验证。
