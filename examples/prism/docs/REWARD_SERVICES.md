# Prism frozen reward services

Start with the [reward workflow](../rewards/README.md) for service calls,
component scoring, composition and training filters. This page covers diagnostics;
deployment commands and environments belong to the [service guide](../reward_services/README.md).

The installed client and server implementations live in
[`slime_plugins/audio_rewards`](../../../slime_plugins/audio_rewards/).
Prism sample adaptation lives in
[`reward_adapter.py`](../../../slime_plugins/prism/reward_adapter.py).
Container launch scripts, runtime preparation and benchmark recipes are documented
in [reward_services/README.md](../reward_services/README.md).

- Qwen3-ASR and Whisper use native vLLM OpenAI audio transcription endpoints.
- ReDimNet2 returns unit speaker embeddings for local cosine rewards.
- JudgeCLAP returns unnormalized audio/text embeddings for the checkpoint's
  calibrated margin reward, with 27 instruction categories and no Loudness.
- The [shared configuration](../configs/rewards.yaml) selects native ASR and
  embedding APIs. Configure `api` explicitly; only these two protocols are supported.

Default GRPO selects WER only; MixRL and explicit component selection determine
whether SIM/JudgeCLAP are called. Choose service replicas from the scoring workload
and measured endpoint capacity. Deployment and validation tools are documented in
[reward_services/README.md](../reward_services/README.md).

## 联合评分变慢时怎么定位

[当前参数清单](../reward_services/CONCURRENCY_CONFIG.md) 解释服务总并发预算与部署参数。
每服务一个 `concurrency`，自动分给执行 worker；奖励 HTTP 连接与生成端独立。

每个非零帧样本都会记录组件耗时，随 `omni_reward` 一起进入现有 Prism
归档；使用 [GUIDE 中的归档 hook](GUIDE.md) 即可，无需额外的 profiling 开关。
下一次启动训练时加载新代码才会生效，正在运行的旧源码快照无法补出历史组件计时。

训练日志搜索 `Prism reward timing`，每步会输出 WER、SIM、JudgeCLAP 的
样本数、完整组件耗时 p50/p95/max、客户端排队 p95、HTTP p95，最后列出
`last_completed` 组件、样本编号、请求 ID、音频时长。
该行统计训练选中的样本；全部候选、未选中样本及按类别/语言拆分的数据仍在
TensorBoard/W&B 和 `records/train/rollout_*.metrics.json` 中。

在 TensorBoard/W&B 搜索 `rollout/prism/selected/`，优先看：

| 指标后缀 | 含义与用途 |
| --- | --- |
| `reward_timing/total_s/p95` | 单条样本完整联合评分耗时，不是整步耗时。 |
| `component/<name>/timing/total_s/p95`、`max` | 对比 `wer`、`sim`、`judge_clap`，找整体慢和长尾慢的组件。 |
| `component/<name>/timing/client_queue_s/p95` | 请求等待该 worker 分到的 `concurrency` 名额的时间，请求尚未发出。 |
| `component/<name>/timing/http_s/p95` | 发出 HTTP 调用到返回的时间，包括连接池/网络、服务端排队、推理、传输及重试等待；不是纯 GPU 计算时间。 |
| `component/<name>/timing/cache_wait_s/p95` | 等待共享参考 embedding 或文本缓存锁；冷缓存时尤其有用。 |
| `component/<name>/timing/last_ratio` | 在同时使用多个组件的样本中，该组件最后完成的比例。 |
| `component/<name>/timing/last_completed` | 该组件是否包含本批选中样本中最后完成的评分请求，取值 0/1。 |
| `component/<name>/timing/finish_gap_s` | 该组件最后一次完成，比本批最后一次评分完成早多少秒。 |

`<name>` 是 `wer`、`sim` 或 `judge_clap`；WER 的实际模型由服务配置决定。
其他字段包括 `prepare_s`（文件读取/上传编码）、`identity_wait_s`、
`identity_http_s`（首次身份查询）、`cache_lookup_s`（文件摘要/SQLite 查询）、
缓存命中/缺失数、错误与取消比例。失败或超时也保留耗时，零帧跳过和没有计时的旧记录
不会被填成零延迟拉低组件统计。`http_requests` 统计逻辑 HTTP 调用，不拆分底层重试次数。

同一组件内部的候选/参考音频、音频/文本可并发，阶段耗时是各操作耗时之和，
不能相加当作组件墙钟时间；应以 `total_s` 为准。三个组件也并发，联合评分
通常等待最晚完成者。`sample_tail_s` 只表示同一样本最后一个组件比倒数第二个
多等了多久，不等于删除该组件后的训练提速。跨 worker 的最后完成顺序使用系统
时间戳，依赖节点时钟同步；时长本身使用单调时钟。

诊断顺序：先对照整步的 rollout、train、weight sync 时间；若 rollout 慢，再看
每个组件的完整耗时和最后完成情况。`client_queue_s` 高且服务吞吐已饱和，优先增加
瓶颈服务容量；服务未饱和时再检查总并发限制。`http_s` 高则进一步看对应服务的
网关延迟、队列和计算指标。`services.<name>.concurrency` 是本次训练的服务总并发预算，由实际执行 worker 自动分配。
奖励 HTTP 连接不再受生成端连接池预算限制。不要仅凭提高总并发后队列位置改变就判断提速。

历史联合评分诊断、扩容和并发对照保存在
[独立实验归档](https://github.com/VoiceAgentGroup/slime/blob/e57ed66c1ed6bb35015db134e5a0dc1a508caff3/examples/prism/reward_services/experiments/README.md)，其中各次实验使用各自冻结源码与统计窗口。
