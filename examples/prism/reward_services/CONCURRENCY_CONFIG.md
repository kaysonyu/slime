# Prism 音频奖励：简化后的并发与服务配置

本页描述当前实现的总并发预算、服务默认值和迁移要求。
历史实验使用各自冻结源码，旧的配置字段需要按本页迁移。

## 1. 服务总并发与默认部署参数

训练侧的 [rewards.yaml](../configs/rewards.yaml) 使用每个服务的 `concurrency`：

| 服务 | YAML 与代码默认总预算 |
| --- | ---: |
| WER / ASR | 128 |
| SIM / ReDimNet2 | 64 |
| JudgeCLAP | 64 |

预算属于当前奖励配置在本次训练执行池中的服务请求，不是每个 worker 的值，
也不是跨多个训练作业的集群级限额。同一配置的不同奖励类别共享客户端状态。
当前 `max_inflight` 字段已移除，出现时会明确报错，避免把旧的每 worker 数值
静默解释为总预算。

假设总预算为 WER/SIM/JudgeCLAP **256/128/256**：

| 实际执行 worker 数 | 每 worker 分到的预算 | 全部 worker 合计 |
| --- | --- | --- |
| 8 | 32 / 16 / 32 | 256 / 128 / 256 |
| 16 | 16 / 8 / 16 | 256 / 128 / 256 |
| 单进程或本地评测 | 256 / 128 / 256 | 256 / 128 / 256 |

除不尽时按整数商和余数分配，所有份额之和严格等于总预算，不额外向上取整。
总预算小于实际 worker 数时会报错；应提高预算或减少 worker。静态分配不会
动态借用空闲 worker 的份额，因此总预算是上限，不保证所有名额始终被使用。
同步 execution worker 和分布式异步 reader 都传递实际 worker 序号/数量。
脱离 Slime 自行启动多个客户端进程时，需要显式传入 `worker_index` 和
`worker_count`；不传时每个客户端都按单进程预算运行，不会自动发现其他进程。

模型部署侧，两个 embedding 启动脚本直接使用已测的单卡 H200 参数：

```bash
PYTHON=/shared/reward-env/bin/python \
  bash /shared/slime/examples/prism/reward_services/start_redimnet2.sh

PYTHON=/shared/reward-env/bin/python \
  bash /shared/slime/examples/prism/reward_services/start_judge_clap.sh
```

上述路径需要替换为实际仓库和运行环境。副本数仍在 Inspire 模型部署中设置。
ASR 继续使用原生 vLLM 启动脚本及其默认运行参数。

## 2. 客户端只保留一处请求准入

每个服务客户端按分配到的份额维护一个 semaphore。它覆盖编码/转写请求的
音频上传准备、HTTP 调用和重试；等名额的时间计入 `client_queue_s`。参考或文本缓存命中不会额外
发送编码请求；未命中的请求与候选请求共享同一服务预算。

JudgeCLAP 按模型版本和文本共享正在进行的缓存填充，包括成功结果和失败异常。
缓存命中和无关文本不会等待其他填充；取消一个样本不会取消其他样本共用的查询。
失败后会移除正在进行的填充记录，后续调用可以重新尝试。

取消或超时会停止请求；已经开始的后台音频读取/编码会先完成，再释放客户端名额并
向调用方抛出取消或超时异常。重复取消不会跳过这一步，也不会在取消后继续发送 HTTP。
因此 `timeout_s` 到期后，调用返回前可能还有文件读取的清理时间。

奖励客户端拥有自己的 HTTP 连接，不再依赖 Slime 的生成端 `_http_client`。
连接池不另设活跃连接数上限，空闲 keep-alive 连接数由本地请求份额决定。
请求量由前述唯一准入控制，因此取消独立连接池阈值不表示取消客户端限流。

由此移除了旧的隐含关系：

```text
旧：奖励 HTTP 连接预算 = sglang_server_concurrency × engine 数 / worker 数
新：奖励请求预算 = services.<name>.concurrency，按实际 worker 分配
```

修改生成并发不再直接改变奖励连接数。HTTP 耗时仍包含网络、服务端处理和返回等
工作，但不再混入由生成预算造成的额外连接池等待。

通用客户端可用 `async with AudioRewardClient(...)`，Prism 适配器提供
`await adapter.aclose()`。关闭会拒绝新请求，取消完整评分操作、排队和在途请求及
共享参考/文本缓存任务。上传准备、参考音频哈希和 SQLite 读取结束、HTTP 连接关闭后，
清理才算完成；并发或重复关闭共享同一次清理。
训练的现有 hook 清理会等待异步关闭，不需用户初始化 Slime HTTP 全局变量。
客户端不导入训练包或服务端模型依赖。

来源：[concurrency.py](../../../slime_plugins/audio_rewards/concurrency.py)、
[client.py](../../../slime_plugins/audio_rewards/client.py)、
[transport.py](../../../slime_plugins/audio_rewards/transport.py)、
[execution_pool.py](../../../slime/rollout/execution_pool.py)。

## 3. embedding 服务只保留一处积压保护

HTTP 入口对音频、文本共享一份每进程输入项预算，覆盖 CPU 准备、排队和前向。
默认值自动取 `max(64, 8 × max_batch)`。此预算按输入项计数，客户端则按 HTTP
请求计数，两者不是同一个单位。只有这一处进行容量拒绝；内部 batcher
不再重复限制队列长度。已接纳的 CPU/GPU 工作在完成前持续占用名额，包括请求取消后。
超出预算返回 503，技术失败不会转换为数值奖励。

| 模型 | 模型进程 / 单卡副本 | CPU 准备线程 / 进程 | 音频 batch 上限 | batch 成本预算 | 凑批窗口 | 每进程接纳上限 | 编译 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ReDimNet2 | 1 | 4 | 32 | 16384 | 5 ms | 256 | 开启 |
| JudgeCLAP | 4 | 4 | 16 | 16384 | 10 ms | 128 | 关闭 |

默认部署中，Judge 的四个进程共享一张 H200。
ReDim 使用 masked batching、SDPA、FP16。Judge 保留 per-audio CPU 特征、
BF16 基座、未合并 FP32 LoRA。音频策略与分数公式见 [服务接口说明](README.md)。

`/info` 返回实际运行参数及限制；`/metrics` 提供 `active`、`active_limit`、
`peak_active`、`rejected_inputs`、batch 分布和计算时间。`worker_id` 同时包含
容器和进程身份，避免不同副本的 PID 相同而被误认为同一个进程。

### 保留的高级覆盖

必要时可在启动脚本末尾传入 `--max-batch`、`--max-cost`、`--batch-window-ms`、
`--decode-workers`、`--processes`、`--max-pending`、`--precision` 等 CLI 参数，
它们覆盖服务默认值。`--no-compile` 可关闭 ReDimNet2 编译；
较小显存的 GPU 可用 `--processes 1` 将 JudgeCLAP 改为单模型进程。

`PROFILE` 环境变量和 `--profile` 参数已移除，启动时无需选择档位。
旧的 `MAX_BATCH`、`MAX_BATCH_COST`、`BATCH_WINDOW_MS`、`MODEL_PROCESSES`、
`DECODE_WORKERS` 等各脚本性能环境变量已移除，显式设置时脚本会报错并提示迁移。
`--max-queue` 已移除；唯一的接纳上限高级覆盖为 `--max-pending`。
`--gpu-workers` 已移除，模型进程内固定一个 GPU worker；Judge 音频和文本前向
仍保留同进程互斥。标准 `OMP_NUM_THREADS` 和 CPU 算术线程配置不代表第二份请求额度。

batch 条数和成本预算仍需同时保留：长音频可能先触及成本预算。ReDim 的成本
是特征帧，Judge 是音频 token 估计，两者的 16384 单位不同。
Judge 文本仍使用独立的 64 条、65536 成本组批配置，并共享入口接纳预算。

来源：[server.py](../../../slime_plugins/audio_rewards/serving/server.py)、
[batching.py](../../../slime_plugins/audio_rewards/serving/batching.py)。

## 4. 哪些控制保留、哪些省掉

| 原有控制 | 处理 | 原因 |
| --- | --- | --- |
| 每 worker `max_inflight` | 改为服务总 `concurrency` | worker 数改变时无需手工重算总请求量 |
| 奖励 HTTP 活跃连接预算 | 移除独立阈值，与生成连接池解耦 | 已有客户端准入，第二次等待只会藏入 HTTP 时间 |
| 服务端入口与 batcher 两份队列上限 | 合并成入口一份自动接纳预算 | 同一批输入无须重复配置容量限制 |
| 只能取 1 的 `gpu_workers` 参数 | 移除，固定为 1 | 模型线程约束不是有效的用户调参范围 |
| 多个常用部署环境变量 | 改为服务端默认值 | 直接使用已测的单卡 H200 参数；调优仍可用显式 CLI |
| 服务端 batch 条数与成本 | 保留 | 控制实际计算量和显存，单位与请求数不同 |
| CPU 线程池与 GPU 前向互斥 | 保留内部执行约束 | 控制执行资源并保持模型调用正确性 |
| rollout group 投放、生成并发、数据 I/O 限制 | 保留 | 约束生成与训练数据的在途资源，职责不同于奖励服务流量 |
| 超时、重试 | 保留 | 结束故障调用并处理可重试错误 |
| 音频文件、base64 与 HTTP 请求体字节上限 | 移除 | 内部音频链路不再配置传输字节阈值 |

日常配置聚焦服务总并发，服务端的 batch 成本、音频时长和 token 长度按模型执行需求约束。

## 5. 其它仍生效的配置

- 当前 [run-grpo.sh](../run-grpo.sh) 的 `sglang_server_concurrency` 默认仍为 2，
  `rollout_execution_workers` 默认仍为 0（本地执行）；它们负责生成及执行拓扑。
- 生成总并发由 `sglang_server_concurrency × engine 数` 决定；每 worker 的
  group 投放量由框架计算，与奖励服务预算分别配置。
- TTS 的 [student.yaml](../configs/student.yaml) 仍设置每 engine 32 个运行请求、
  65536 token 容量和最大 32 的 CUDA 图 batch。
- ASR 脚本继续默认 `MAX_NUM_SEQS=128`、`API_SERVER_COUNT=2`、显存预算 0.80；
  Qwen 的 batch token 预算与模型长度为 16384，输出上限为 1024。
  Whisper 的模型长度为 448，脚本不单独指定 batch token 预算。
- 奖励 YAML 的 `timeout_s=300`、`max_retries=2` 保持不变；字段缺失的代码默认
  仍分别为 120 秒、0 次额外重试。`concurrency` 的 YAML 与代码默认值已经统一。
- embedding 服务仍限制每 HTTP 请求 64 个输入项、输入音频 120 秒。
  Judge 文本限制为 4096 字符且完整模板不超过 1024 token。
- ReDim 保留完整音频；Judge 的非空短音频补到 2 秒，长音频取前 30 秒。

修改奖励配置需要重新创建训练客户端；修改服务进程参数需要重新部署。
副本数可用平台扩缩容调整。当前没有这些客户端/进程参数的在线热更新接口。

## 6. 监控和迁移

`client_queue_s` 测量唯一客户端准入等待；`http_s` 测量实际 HTTP 调用；
`total_s` 与 `last_completed` 等指标继续用于判断整体耗时和拖尾。
阶段时间可能重叠，分位数不能相加。

每个组件的原始 `timing` 还记录 `concurrency`、`worker_concurrency`、
`worker_index`、`worker_count`，便于核对实际分配而不是凭配置猜测。
模型 embedding identity 仍用于验证缓存和服务版本。

服务入口接纳超限返回 503；平台代理也可能返回 429。应结合响应状态、响应头和
服务指标判断限制来源，不应假定网关会自动设置适合模型推理的并发上限。

从每 worker 预算迁移到总预算时（以 8 个 worker 为例）：

1. 把每 worker 的 `max_inflight: 32/16/32` 改为总 `concurrency: 256/128/256`。
2. embedding 启动命令删除 `PROFILE` 及旧的进程、线程、组批环境变量，直接使用服务默认值；
   需要调整时传入显式 CLI 参数。
3. 保留 endpoint、认证、模型路径和既定评分公式。
4. 检查服务 `/info` 与实际 worker 分配，使用相同负载比较完整步时和组件计时。

原始实验记录仍使用其冻结源码下的旧字段，作为历史证据保留。
相关文档：[服务接口](README.md)、
[联合评分诊断](../docs/REWARD_SERVICES.md#联合评分变慢时怎么定位)。
