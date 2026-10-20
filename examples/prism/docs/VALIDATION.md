# Prism 训练验收与性能记录

本文记录固定源码、配置和资产对应的验收结果；各轮结论只适用于其记录的范围。
当前训练入口见 [EXAMPLE_RUNS.md](EXAMPLE_RUNS.md)，输出目录与 Ray 生命周期见 [RAY.md](RAY.md)。

## Prism GRPO 64/128 H200 扩展与优化（2026-10-20 UTC，按用户要求结束）

最终优化版在 64 卡 / B4096 达到 **14.532 秒/步**，128 卡 / B8192 两次
独立 25 步短跑为 **18.685 / 18.673 秒/步**。同代码吞吐扩展为 **1.556×**，
弱扩展效率约 **77.8%**，不是线性 2×。用户要求记录当前优化结果并关闭实验，
因此不再启动准备中的 128 卡 100 步与 checkpoint 恢复，也不再复核 64 卡。
本节是实际短跑结果与用户终止范围的记录，不是原始全量验收通过声明。

机器可读结果、逐项配置、统计及证据索引见
[`20261020-final.json`](../benchmarks/scaling/results/20261020-final.json)。
完整产物位于
`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/grpo128-20261020`。
所有失败、排除、主动取消和未启动的记录均保留，不混入有效吞吐。

### 固定版本与测量方法

最终训练源码为 Slime `cd50e5a934681a54a4cec098dcff7f44512ef085`，
不可变快照为 `source/postprocess-cd50e5a9`；优化前的正确 router 拓扑基线是
`f08b40550219da661e6f2732a949b238ea2d6ffa`，router 启动适配器固定为
`40487d9223189f24e45b872cd9ed1b8c0d9013f2`。Omni 固定为
`b3f9b9bdd9a2a40334c159b5fbe4beda2408dffb`；Megatron 为 `3f49de20`。
Git 分支为 `perf/prism-grpo-128gpu-20261020`，上述 runtime pin 不随记录提交改变。

沿用 Prism 1.7B v0.4、iteration 100000 / renderer 7、8 万条
fullclone_chinese、每 prompt 8 候选和真实 Qwen3-ASR WER 奖励；
模型、训练算法和奖励语义没有为测速简化。训推共卡最多 128 张 H200，
ASR 独立最多 32 张；数据及 ASR 身份哈希记录在 JSON 中。
最终 64 卡配置为 B4096 / ASR16 / workers32 / router8 / ASR inflight4；
128 卡为 B8192 / ASR32 / workers64 / router16 / ASR inflight8。
两者 concurrency16、round_robin，并启用 topology cache 和 packed tensor buffer。

每个短跑完成 25 次更新，用相邻 generate 开始时间统计 rollout 3–23 的
21 个完整窗口，排除前三步预热及末尾未闭合窗口，不删除慢步。整步包括生成、
真实评分、归档、转换与分片、训练、清理和权重发布；不包括初始化及末尾 checkpoint。
CPU profile 仅在末尾未计时的 rollout 24 采样，不污染表中窗口。

| 配置 / 实验 | 秒/步 | P95 秒 | samples/s | generate 秒 | async_train 秒 | 权重发布秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 优化前 128 / B8192，正确 router 拓扑 | 22.980 | 24.109 | 356.48 | 17.882 | 1.747 | 3.305 |
| 最终 64 / B4096 | 14.532 | 15.056 | 281.86 | 11.775 | 1.438 | 1.276 |
| 最终 128 / B8192，首次 | 18.685 | 19.278 | 438.42 | 15.390 | 1.640 | 1.610 |
| 最终 128 / B8192，重复 | 18.673 | 20.024 | 438.71 | 15.395 | 1.705 | 1.528 |
| 同版 128，仅关闭 topology cache | 20.340 | 20.894 | 402.75 | 15.364 | 1.678 | 3.253 |

128 卡相对正确拓扑优化前版本减少 **4.308 秒/步（18.74%）**，吞吐增加
**23.07%**。两次最终 128 均值仅差 **0.066%**，但重复组 P95 高 **0.746 秒**，
不能据此宣称尾延迟完全一致或统计显著。最终 128 平均生成长度为 92.75 / 92.80
帧，64 为 92.25 帧；同时保留 audio frames/s 供核对，未用截短生成换取吞吐。

弱扩展效率定义为 `T64 / T128`，吞吐倍率为 `2 × T64 / T128`。
最终比较使用同一 `cd50e5a9` 的现有 64 卡对照；用户免除了末尾 64 复核，
因此没有优化版前后双对照漂移区间。早先以旧 64 基线与新 128 混算的 83.5%
不是同代码扩展效率，最终结论不采用该数字。

### 已保留的优化与消融

1. **Router 多进程与独立连接池。** 最终使用真实 8 / 16 个 router 数据面进程，
   为后端副本建立 HTTP 池，并核对实际拓扑。上一轮 64 卡、ASR8 的 router/HTTP
   联合改动从 37.023 降至 17.827 秒；这是组合优化，不是纯进程数或硬件扩展收益。
2. **ASR 容量和在途并发匹配。** 64 卡采用 16 个 ASR，128 卡采用 32 个，
   128 的 inflight 从 4 调为 8；同时测过 workers64/128 和 concurrency16/8。
   最终保留 W64/I8，没有因 worker 更多就默认更快。
3. **HTTP1 连接所有权保护。** Router 和 rollout 客户端保护已分配连接，避免
   pool eviction 关闭活动流；不通过重试生成 POST 掩盖错误。中间的乐观分配
   `a42aa468` 候选已撤回，最终 HTTP 实现与 `f08b4055` 对照一致。
4. **带版本的权重拓扑缓存（`7ce376a9`）。** 缓存 actor handle 拓扑，
   在权威恢复边界刷新，engine 替换时失效；每步权重仍完整发布，不缓存旧权重。
   同源码开关消融节省 1.655 秒整步，其中权重发布节省 1.643 秒，
   其余阶段合计只差约 0.013 秒；该消融吞吐提高 8.86%。
5. **Packed tensor 传输（`cd50e5a9`）。** 将符合条件的普通连续 CPU tensor
   合入一个 buffer，携带 dtype/shape/offset，接收端保持可写副本；不符合条件
   时走原路径，保留嵌套 Ray ObjectRef 所有权。CPU 64-sample 微测 encode
   13.485→10.262 ms、decode 8.438→4.867 ms，不直接当作整步 GPU 收益。
6. **保持精确结果的 KK 分片优化。** 复用空分片 sentinel、跳过空合并并转移
   所有权，246 个 golden case 分片完全一致。CPU B4096/DP64 为
   0.497→0.192 秒，B8192/DP128 为 1.613→0.584 秒。
7. **CPU tensor 快路径。** 对 dtype、连续性、设备、梯度及附加属性均满足
   条件的普通 tensor 直接复用，避免重复转换；其余情况保留原语义。

实际拓扑校验、失败后精确进程清理、污染测量排除、native object-store
checkpoint 校验和计时窗外 profile 是验收可靠性修复，不另算吞吐收益。
部署前 CPU、精确分片、集成与真实 Ray 测试证据保存在 JSON 索引中；
本次关闭阶段没有重新运行 GPU 训练。

### 剩余瓶颈与验证边界

64→128 的整步增加约 4.141 秒，其中 generate 增加 3.620 秒，
async_train 增加 0.267 秒，权重发布增加 0.252 秒。实际 actor_train
仅 0.802→0.834 秒，主要拖慢量不在 learner 的前后向计算。
collection 为 8.828→9.209 秒，而 collection 结束到 driver 返回为
2.946→6.186 秒，后者是剩余扩展损失的主要区间。

128 重复组末轮 GIL 采样中，deserialize_group、split_by_dp/Ray put 和
Prism artifact/log 的 inclusive 占比分别约 22.2%、20.3%、19.0%。
这是 522 次采样、88 次非阻塞采样错误的定位证据，**不是互斥 wall-time 分解**，
不能相加或把整个后处理区间归因到单一函数。代码中仍存在各 DP payload
重复遍历全局 raw_reward/total_lengths、逐 tensor materialization、同步归档与
指标计算等工作；共享元数据、进一步异步化及 serving snapshot 缓存没有继续实施。
最终 128 GPU 平均 busy 约 30–31%，ASR 约 19%，峰值显存约 50.3 GiB；
这些设备采样是辅助证据，不单独证明瓶颈。

最终 64 短跑的 102,400 条样本，以及两次 128 各 204,800 条样本，逐条核验
无失败奖励、零帧、截断、错组或陈旧策略样本，梯度有限且非零。128 每组的
3200 份 rank/step microbatch 记录均为 `[1]`，版本 1–26 每次发布至 128 副本。
这些证明短跑业务完整性，不证明长期收敛或语音主观质量。

优化前的 `f08b4055` **64 卡**已完成 100 更新及从 checkpoint 实际恢复后
追加 2 更新：optimizer `[100,100]→[102,102]`、scheduler/data
409600→417792，模型及 Adam 状态完整。该结果不能替代最终 `cd50e5a9`
**128 卡**长跑与恢复验收；后者按用户关闭指示没有启动。

关闭记录为 `campaign-finished.json` 与 `platform/user-closure.json`：
训练 Job `prism-grpo128-matrix-1020` 于 10:52:30 UTC 取消，ASR
`prism-grpo64-asr-1019` 停止，两条 CPU 监控退出，同线程待触发队列为零。
64 卡末尾复核在初始化阶段主动取消、0 更新、16 节点完成清理；
不将其视为训练失败或有效测量。所有日志、源码快照、样本与已有 checkpoint
保留；没有为关闭实验删除证据或释放后重新申请 GPU。

## 当前 ttsrl 的 32 卡配对测速（2026-10-16 UTC）

当前组为 Slime `43683a89` 加动作流校验工作区补丁、Omni `ttsrl@0cc51ec4e`；
基线为 Slime `e99476bd` 与 Omni `c2f8c453`。为固定教师服务，基线只回移当前
`slime_plugins/prism/teacher.py` 的接口兼容：移除旧身份/checksum 必填校验，保留
请求 hash、动作形状及有限分数检查；请求构造函数保持一致，learner、loss、optimizer
和学生推理代码保持在基线版本。补丁及兼容检查保存在验收目录。

两组使用同一份 Prism v0.4.0 iteration 100000 renderer 7 HF/native 权重，以及
同序的 8 万条 fullclone_chinese 数据。旧、新请求格式分别适配各自客户端；全部
记录的语言展示与参考属性条件一致，256 条 prompt 的渲染 token、参考 codes 和
打包函数检查通过。实际训练的 prompt 顺序与文本摘要也逐步一致。

四组均在同一次 4×8 H200 分配上运行，32 个 GPU UUID 一致。DP32、TP=PP=CP=1、
BF16、B2048、每卡 20480 行预算、单 packed microbatch；learner/rollout 共卡常驻，
关闭 offload，启用 distributed optimizer 与 full/uniform/1 重计算。生成上限
1024 帧、种子 6211；MoPD 为 2048×1，GRPO 为 256×8。学生没有额外的独立 Serving。
两组共用当前 ttsrl 的 2 副本冻结教师和同一个 4 副本 ASR；教师、ASR 均为独立
Inspire Serving，整个配对期间保持在线。有效参数的 76 个字段及 pipeline 核对一致。

每组从相同初始权重执行 25 次更新，共 100 次更新、204,800 条训练样本。
计时使用相邻 generate 开始时间，排除前 3 步预热与最终 checkpoint 步，保留
rollout 3–23 的全部 21 个窗口；整步包含生成、评分、归档、learner 和权重发布，
没有删除慢步。以下为每种算法一次配对短跑的结果。

| 算法 | 基线秒/步 | 当前秒/步 | 整步耗时变化 | samples/s 基线 → 当前 | audio frames/s 变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MoPD | 13.836 | 13.808 | -0.20% | 148.02 → 148.31 | +0.26% |
| GRPO | 16.864 | 17.485 | +3.68% | 121.44 → 117.13 | -3.92% |

MoPD 基本持平；GRPO 本轮有小幅下降，两项都未超过测试前采用的 5% 整步耗时阈值。
GRPO 的 generate 阶段（含生成、评分和归档）为 14.979 → 15.564 秒，贡献约 0.585 秒
增量；async_train 为 1.282 → 1.298 秒，权重发布为 0.567 → 0.588 秒。
该分阶段结果不能进一步单独归因于推理、ASR 或网络，且一次配对不等于统计显著性结论。

四组均完成版本 1–26 向 32 个副本的发布，全部 3200 份 rank/step 组批记录核验通过，
loss/梯度有限且梯度非零。四份 checkpoint 均包含 302 个 BF16 模型 tensor、完整
FP32 master 参数与 Adam 状态；iteration 为 24，optimizer 计数为 `[25,25]`，
scheduler 样本计数为 51200。102,400 条 MoPD 训练样本逐条确认使用同一教师 endpoint。
源码 pin/补丁检查通过，584 个 editable 运行文件与被测当前源码快照一致。

完整产物位于
`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/ttsrl-32-ab-20261016-183137/attempt-002`。
`analysis/report.json`、`analysis/steps.csv`、`analysis/verification.json` 为统计入口，
`analysis/argument-comparison.json` 与 `analysis/baseline-teacher-compatibility.json`
记录可比性检查；模型、音频、逐候选记录和配置保留在各 run 目录。
Job `slime-prism-ttsrl-32-ab-1016-a2` 已 SUCCEEDED。该目录的上一层保留首次配对
尝试：基线成功，当前组因测试启动器把 hostname 的回环解析值传入 `SLIME_HOST_IP`
而在 worker 注册阶段失败。重试改用 `hostname -I` 中的非回环 IPv4，并逐节点记录；
未修改训练实现。首次失败组的数字不混入本表。

## ttsrl 与当前 Slime 的 DP2 功能验收（2026-10-16 UTC）

Slime 使用 `43683a89` 加当前动作流校验工作区补丁；Omni 使用
`kaysonyu/sglang-omni` 的 `ttsrl@0cc51ec4e`。完整源码、补丁和内容指纹已归档。
实际 import 指向共享 editable 目录，镜像为 `slime-ttsrl:editable-v1`，
模型为 Prism v0.4.0 iteration 100000 的 renderer 7 HF/native 制品。

DP2、TP=PP=CP=1、B32、生成上限 512 帧、learner/rollout 共卡 offload。
GRPO 每 prompt 4 个候选，使用真实 Qwen3-ASR；MoPD 每 prompt 一条轨迹，
训练不生成波形、task reward 为零，通过独立 Inspire Serving 调用冻结教师。
两种模式均执行 5 次更新，再从完整 checkpoint 恢复并追加一次更新。

| 模式 | fresh / resume 更新数 | 训练样本总数 | fresh 梯度范数范围 | resume 梯度范数 |
| --- | ---: | ---: | ---: | ---: |
| GRPO | 5 / 1 | 192 | 0.783395–1.266445 | 0.693883 |
| MoPD | 5 / 1 | 192 | 0.071301–0.077754 | 0.073940 |

全部 loss、动作分数和优势有限，梯度非零；候选分组、两个 rank 的样本数及
RVQ/stop 动作记录通过核验。两种模式均发布版本 1–6，恢复后发布 6–7。
四份 checkpoint 各含 302 个 BF16 模型 tensor、1,979,878,400 个参数，
以及完整 FP32 master 参数和 Adam 一二阶状态。optimizer 计数由 `[5,5]`
恢复到 `[6,6]`，scheduler 样本计数由 160 递增到 192。
每种模式训练前后均完成 8 条独立音频评测，MoPD 教师发布版本保持一致。
CPU 回归为 346 passed、2 skipped；运行前后源码内容指纹一致。

复现资产为
`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/ttsrl-grpo-mopd-20261016-180409`。
汇总入口是 `analysis/verification.json`；GRPO 成功记录位于 `attempt-002`，
独立教师的 MoPD 成功记录位于 `attempt-003`。失败记录也保留：首次验证脚本的
Ray GCS 端口落入 worker 范围；随后同 Job 教师继承 `MASTER_PORT`，造成 NCCL
与 GCS 冲突。分别通过调整验证端口和独立部署教师解决，训练算法没有修改。
本节验证功能闭环和恢复正确性，小样本评测不作为语音质量收益结论。

## 示例精简与启动脚本 CPU 验证（2026-10-16 UTC）

Python 3.10.20 下，启动脚本 29 项、奖励组合与分类/评分服务/归档/rollout 接线/数值诊断
155 项，共 184 项通过。启动测试用进程替身覆盖单机、两节点、PET 环境优先级、通信环境继承、
本轮目录、恢复输入与新输出分离、训练失败、启动失败、缺失 worker 和 TERM 清理。
`test_prism_launch.py` 与 `test_prism_numerical_diagnostic.py` 的 CI 直接执行入口也通过；
数值诊断已注册到 CPU CI。Bash 语法、文档命令语法、Python 3.10 解析、Black、isort、Ruff
及文档本地链接检查通过。诊断工具的两个 CLI `--help` 在新模块路径下可用。

这轮没有启动真实 Ray/GPU 集群，没有新增 GPU 吞吐、数值对齐或服务质量结论。
下列历史记录继续对应各自固定的源码和资产。

## Policy streams 与 Omni deployment 重构配对验收（2026-10-15 UTC）

使用干净提交 `b73f0fd4bceddd66320de82569c7ca08dcd84d56` 作为两次重构前的
基线，验证 `e99476bd8779980fc0fd1f960ae774511da0982c`，后者包含 policy stream
轴保留与 Omni deployment/router 生命周期重构。两份代码均来自固定 worktree，
没有混入当时工作区的未提交修改；Omni 固定为 `c2f8c4532fc889bef06f2e31d5161bdd591b6c58`，
Megatron 固定为 `3f49de20de81c1ad5dfc10efa0f7d0c8098f1ce4`。

在同一组 4×8 H200 上，每种算法按基线、最新、最新、基线的顺序运行，每次从
Prism v0.4 native iteration 100000 重新初始化模型与 optimizer，各 25 次更新，
共 200 次。DP32、TP=PP=CP=1、B2048、每卡 20480 行预算、单 packed microbatch；
learner/Omni 共卡常驻，无 offload，distributed optimizer 与 `full/uniform/1`
重计算开启，生成上限 1024 帧，种子 6211。MoPD 为 2048×1、两个真实冻结教师
副本；GRPO 为 256×8、四个真实 ASR WER 副本。数据为同一份 fullclone_chinese，
并发、归档、每 16 步清理、256 MiB 权重 bucket 等一致。每轮重启 Ray 和模型。

实际仅配置 `--omni-config`，`sglang_config=None`。同算法四轮的 effective options
在规范化产物路径与自动分配的 router 地址/端口后完全一致；GPU UUID 和逐步
prompt 指纹也逐项一致。源码、ASR Python 实现和数据的前后内容指纹核验通过。

计时使用相邻 generate 开始时间，排除 rollout 0–2 预热和最终 rollout 24 的
checkpoint，保留 rollout 3–23 全部 21 个窗口，每个提交/算法合计 42 个窗口。
整步包含生成、评分、归档、训练、清理与权重发布，没有删除慢步；启动与 checkpoint
分别记录在原始报告。汇总吞吐用总样本/帧数除以总时间。

| 算法 | 提交 | 两次整步秒/步 | 汇总秒/步 | samples/s | audio frames/s |
| --- | --- | ---: | ---: | ---: | ---: |
| MoPD | `b73f0fd4` | 13.659 / 13.604 | 13.631 | 150.24 | 13697.93 |
| MoPD | `e99476bd` | 13.802 / 13.753 | 13.777 | 148.65 | 13550.06 |
| GRPO | `b73f0fd4` | 16.020 / 15.859 | 15.939 | 128.49 | 11796.42 |
| GRPO | `e99476bd` | 16.051 / 16.125 | 16.088 | 127.30 | 11704.56 |

最新相对基线：MoPD 整步耗时 +1.07%，样本吞吐 -1.06%；GRPO 整步耗时 +0.93%，样本吞吐 -0.92%。两项均未触发测试前设定的“整步耗时增加 5%”回退线。
原生 actor train 计时（含前后向及 optimizer）为：MoPD 0.728 → 0.754 秒；GRPO 0.712 → 0.753 秒。
这些结果覆盖本次固定负载下的重复短跑，保留了实际测得的小幅差异。

八轮各自完成 25 次 optimizer 更新，loss/梯度有限且梯度非零；6400 份 rank/step
组批记录均为单 pack，版本 1–26 均发布到全部 32 个 rollout 副本。全部选中样本
的策略版本、教师权重身份或 ASR fingerprint、候选组大小核验通过。每份 checkpoint
均包含 302 个 BF16 模型 tensor、1,979,878,400 个参数，以及对应完整 FP32 master
参数和 Adam 一二阶状态；两组 optimizer step 均为 `[25,25]`，scheduler sample
steps 为 51200，iteration 为 24。

复现与核验资产位于 `/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/latest-refactors-ab-20261015-190912`。
`analysis/report.json` 保存逐轮、分阶段与配对统计，`analysis/steps.csv` 保存逐步
窗口，`analysis/verification.json` 保存最终核验收据，`manifest.json`、`source/`、
`configs/`、`scripts/`、`runs/` 保存源码、参数、程序、评分记录及 checkpoint。
Job `slime-prism-latest-ab-1015-1909` 成功完成，两个专用评分 Serving 均已停止；
平台记录保留用于复核，GPU 资源已释放。

## Distributed optimizer 与完整重计算消融（2026-10-15 UTC）

在同一台 8×H200 上顺序完成 MoPD、GRPO 各四种配置，每组 20 次 optimizer
更新，共 160 次。两项都关闭可以完成训练、权重发布和 checkpoint 保存，但本轮
没有带来加速：MoPD 整步增加 14.09%，GRPO 增加 11.77%。目前保留 distributed
optimizer 与 `full/uniform/1` 重计算更合适，不能把显存占用更高当作吞吐改善。

源码复用下节 DP32 测速的固定快照：Slime `47b00de9` 加补丁
`325a1185027af313e7d6b937f7ca6f397fea8ef8263ba145cb9bcf8aef6bdfca`，
Omni `c2f8c453`、Megatron `3f49de20`。没有修改训练实现，各配置使用相同的测量
hook；每组重启 Ray 和模型，从同一个 Prism v0.4.0 native iteration 100000
初始化新的 optimizer。模型实际为 1,979,878,400 个参数、24 RVQ、BF16。

本轮可调度的整节点只有一台，因此使用 DP8、TP=PP=CP=1、B512，保持此前 DP32
测试的每卡 64 条样本。每卡仍为 20480 时间行预算、单 packed microbatch，实际
平均约 17785–17880 行，平均生成约 91–93 帧，上限 1024 帧；训练与 Omni
共卡驻留、无 offload。MoPD 为 512×1、不生成 WAV、两个真实冻结教师副本；
GRPO 为 64×8、生成 WAV、四个真实 ASR WER 副本。采样、归档、并发、每 16 步
resident cleanup 和 256 MiB 权重 bucket 均保持一致。DP8 的网络表现不能直接
外推到 DP32 跨节点训练，也不能直接把本表秒/步与 B2048 的秒/步比较。

计时排除 rollout 0–2 热身和最终 rollout 19 的 checkpoint，保留 rollout 3–18
全部 16 个完整窗口。整步包含生成、评分、归档、收集、训练、清理与权重发布；
actor train 为原生训练计时，包括前后向和 optimizer，并非纯 GPU kernel 时间。
下表整卡峰值取整个 case 的 5 Hz nvidia-smi 采样，包含 learner 和 Omni；
learner allocated 峰值为全部 20 步各 rank 的 PyTorch allocator 高水位，
不包含其他进程，也不包含 reserved 中尚未使用的缓存。

| 模式 | 配置 | 整卡采样峰值 GiB | Learner allocated 峰值 GiB | Actor train 秒/步 | 整步秒/步 | 整步变化 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MoPD | 两项开启 | 54.24 | 22.06 | 0.755 | 6.494 | 基线 |
| MoPD | 只关闭 distributed optimizer | 73.37 | 41.42 | 0.787 | 6.547 | +0.82% |
| MoPD | 只关闭完整重计算 | 96.79 | 65.06 | 0.961 | 7.398 | +13.93% |
| MoPD | 两项关闭 | 116.68 | 83.91 | 1.003 | 7.408 | +14.09% |
| GRPO | 两项开启 | 55.79 | 22.39 | 0.739 | 8.341 | 基线 |
| GRPO | 只关闭 distributed optimizer | 74.72 | 41.76 | 0.776 | 8.558 | +2.60% |
| GRPO | 只关闭完整重计算 | 100.72 | 66.72 | 0.931 | 9.310 | +11.62% |
| GRPO | 两项关闭 | 120.23 | 86.08 | 0.996 | 9.323 | +11.77% |

关闭 distributed optimizer 后，运行时类型确实从 `DistributedOptimizer`
变为 `Float16OptimizerWithFloat16Params`，后者名称不表示切换到 FP16：实际模型
仍为 BF16，master 参数与 Adam 一二阶状态仍为 FP32。DP8 下取消状态分片的
理论增量为 `(18 - (6 + 12/8)) * 1979878400 / 2**30 = 19.36 GiB/卡`，
与 learner 峰值差一致。此开关保留数据并行与梯度同步，每卡承担完整 optimizer
状态和更新；本轮 actor train 反而增加约 4%–5%。整步的 0.82%–2.60% 差异
来自各一轮短跑，尚不能判定为稳定回退，但也没有支持关闭后提速的证据。

关闭完整重计算后，激活显存明显增大，训练与发布都变慢。四种配置的生成、评分
与收集阶段接近：MoPD 4.68–4.73 秒，GRPO 6.66–6.73 秒；差异主要不在采样。
只关闭重计算时 actor train 增加约 26%–27%，权重发布由 MoPD 0.421 / GRPO
0.401 秒增至 1.111 / 1.111 秒。代码检查发现
`slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py`
在每个权重 bucket 完成后以及发布末尾都调用 `ipc_collect()` 和 `empty_cache()`；
这条路径每次发布都会执行，与每 16 步的 resident cleanup 是两回事。

测量也显示缓存反复丢失：MoPD 基线各 rank 的平均 reserved 从 optimizer
结束时 23.38 GiB 降至下一次训练前 15.92 GiB，关闭重计算为 65.83 → 16.23
GiB；GRPO 对应为 23.39 → 15.91、65.96 → 16.27 GiB。结合发布耗时和源码，
大量缓存释放、重新分配及其同步开销是重要嫌疑，足以说明“少做重计算就一定更快”
在当前共卡流程中不成立。本轮没有做 CUDA allocator/Nsight trace，不能把全部
差值定量归因于某个 CUDA API；也没有直接移除管理 IPC 生命周期的缓存清理。

配置通过现有 `--custom-config-path` 覆盖生效。当前默认初始化会强制设置
`use_distributed_optimizer=True`，仅省略命令行 flag 不会关闭它。两项都关闭
需要把下列字段合入原有 recipe，保留 MoPD 的 `omni_return_audio: false`
等其他配置；三个 recompute 字段都置空，不能只改 granularity。

```yaml
use_distributed_optimizer: false
recompute_granularity: null
recompute_method: null
recompute_num_layers: null
```

八组配置均检查了实际 optimizer 类型与模型 `_full_activation_recompute`。
全部 loss/梯度有限、梯度非零，1280 份 rank/step 调度记录均为单 pack；每组
10240 条选中记录的真实评分状态与策略版本检查通过，版本 1–21 发布到全部
八个副本。八份 checkpoint 均包含 302 个模型 tensor，optimizer 两组 step
为 `[20,20]`，scheduler sample steps 为 10240，三类 FP32 optimizer 状态
元素数均与完整参数量一致。分片 optimizer 保存为 `dp_reshardable`，关闭后
保存复制状态。本轮没有测试 checkpoint 恢复、已有分片 optimizer checkpoint
切换为复制 optimizer 的续训、固定轨迹逐梯度等价或语音质量提升。

两项关闭后，全程采样峰值已达到约 117–120 GiB/卡；该结论只覆盖本次短轨迹、
每卡 64 条样本负载，不能保证更长轨迹或更大 batch 也能容纳。每个配置只有一次
20 步短跑，稳定收益仍需重复实验。本轮仅增加配置、测量产物和文档，没有改变
生产训练默认值。

固定复现产物：

```text
/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/memory-ablation-20261015-094424
```

`analysis/report.json`、`steps.csv`、`comparison.png` / `comparison.pdf`
保存完整统计、逐步结果和图表，`configs/` 保存八组完整 recipe，`scripts/`
保存运行、测量和核验程序。每个 `runs/<case>/` 保留原始日志、5 Hz 显存采样、
各 rank allocator 记录、checkpoint 和两份 verification JSON。Job
`slime-prism-memory-dp8-1015-0944` 成功结束，专用 teacher 与 ASR 服务均已停止，
平台回执保留在 `platform/`。

## 公共动作流的 25 步 DP32 测速（2026-10-15 UTC）

`refactor/multichannel-policy-streams` 使用同一组 32 张 H200 顺序完成 MoPD、GRPO
各 25 次 optimizer 更新。Slime 快照基于 `47b00de962b85c6354862facf26b931d51ccadbb`，
包含当时工作区的未提交修改；补丁 SHA256 为
`325a1185027af313e7d6b937f7ca6f397fea8ef8263ba145cb9bcf8aef6bdfca`。
完整 tracked-content 指纹、原始补丁与独立 checkout 保存在产物根目录的 `manifest.json`
和 `source/`，不能把此次结果解释为仅验收裸 `47b00de9`。Omni 固定为
`c2f8c4532fc889bef06f2e31d5161bdd591b6c58`，Megatron 固定为 `3f49de20`。

模型为 Prism 1.7B v0.4.0 native iteration 100000，24 RVQ、BF16；两种模式分别
从相同预训练 checkpoint 初始化。DP32、TP=PP=CP=1、B2048、驻留共卡，
每卡 20480 时间行预算、16 rollout workers、每 16 步清理、256 MiB 权重 bucket。
采样上限 1024 帧。MoPD 为 2048×1、不生成 WAV，冻结 v0.4.0 教师使用 2 个 GPU
副本；GRPO 为 256×8，生成 WAV，真实 Qwen3-ASR WER 奖励使用 4 个 GPU 副本。
两者使用同一份 8 万条 fullclone_chinese 数据；此次不是六领域异构教师或 SIM/JudgeCLAP
联合奖励验收。

计时使用相邻 generate 起点，包含生成、评分、归档、数据收集、训练、清理和权重发布。
排除 rollout 0–2 热身与末步 rollout 24 的 checkpoint 窗口，保留 rollout 3–23
全部 21 个稳态窗口，没有剔除慢步骤。

| 指标 | MoPD | GRPO |
| --- | ---: | ---: |
| 平均秒/步 | 13.946 | 16.024 |
| 中位秒/步 | 13.915 | 15.968 |
| p90 秒/步 | 14.326 | 16.383 |
| 样本/秒 | 146.86 | 127.81 |
| 音频帧/秒 | 13380.36 | 11744.05 |
| 生成、评分与收集秒/步 | 11.947 | 14.118 |
| learner 阶段秒/步 | 1.382 | 1.291 |
| 其中 actor train 秒/步 | 0.769 | 0.750 |
| 其中旧策略重放秒/步 | 0.178 | 0.178 |
| 权重发布秒/步 | 0.571 | 0.571 |
| 32 张学生卡平均 GPU busy | 38.48% | 52.77% |
| 采样显存峰值 GiB/卡 | 50.07 | 51.20 |

GPU busy 与显存来自各节点每秒一次的 nvidia-smi 采样，外部评分 GPU 不计入学生
GPU 分母。driver 启动至第一次 rollout 分别为 331.66 / 247.85 秒，最终 checkpoint
阶段分别为 4.06 / 3.69 秒，均未混入上表稳态均值。本轮没有在同一分配上重跑旧版
基线，因此不据此宣称重构零回退或稳定加速。

全部 50 次更新的 loss 与梯度有限且梯度非零；每个 case 的 800 份 rank/step
调度记录均为单 packed microbatch。每个 case 核验 51200 条选中样本的评分完成
状态与策略版本，以及版本 1–26 向全部 32 个副本的发布记录。两份联合 checkpoint
都核验了文件大小、带 hash 的元数据、302 个模型 tensor、三类 optimizer 状态元素
数；两组 optimizer step 均为 `[25,25]`，scheduler sample steps 均为 51200。
本轮没有执行从这些新 checkpoint 恢复后的继续训练，也不声称语音质量提升。

严格概率对齐仍未得到保证。首末步按全部 rank 的实际动作检查，MoPD 的 RVQ
平均绝对误差为 0.02724 / 0.02730、全局最大误差为 0.90164 / 0.81325；GRPO
为 0.02709 / 0.02783、0.85320 / 2.56921。stop 平均误差约 0.00012，四次检查
中的全局最大值为 0.19060。这里的全局最大值不同于历史日志的“轨迹最大误差的
平均”，不能直接按两个不同口径判断精度回退。训练仍使用更新前 learner replay
作为 PPO 分母；没有为本轮测速修改采样、loss 或误差容差。

固定复现产物：

```text
/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/policy-streams-25step-20261015-085228
```

`analysis/report.json`、`steps.csv` 与 `speed.png` / `speed.pdf` 保存汇总、逐步数据和
曲线；`runs/<mode>/verification.json` 与 `checkpoint-verification.json` 保存检查结果。
Job `slime-prism-streams-32-25step-1015-a1` 成功完成，训练资源释放，专用教师和 ASR
Serving 均已停止；平台条目与回执保留用于追溯。

## 兼容层清理与 2 卡回归（2026-10-14 UTC）

删除三个 examples.prism 运行时转发模块及旧 reward 归档包装器，归档统一使用
sample observer。调用方直接导入配置、checkpoint、policy 和 teacher 的定义模块；
模型模块的公开导出从 30 项缩为 10 项，rollout 模块从 19 项缩为 4 项。
Teacher 在 GPU 评分分块内选取学生实际执行的 stop/continue 分数，响应仅返回
一维 selected_stop_logprobs，删除额外概率及客户端的概率回退和列选择。
Prism 请求/结果升级到 version 2，HTTP batch envelope 保持 version 1。
生成端的停止策略及 rollout trace 协议、checkpoint 恢复、显存卸载和权重发布逻辑
均保持原实现。本轮要求 Slime 和 Omni teacher 一起升级。

对照源码为 Slime bb9c1e84 / Omni 547f00e3；重构源码为 Slime f01ed5e1 /
Omni c2f8c453。四个 case 使用干净、固定的独立 worktree；后续仅增加验收文档。
当时没有空闲整节点，本轮使用同一组两张 H200，DP2、TP=PP=CP=1、每步 128
条样本、每卡 20480 时间位置预算、一条 packed microbatch、resident cleanup 16。
MoPD 为 128×1，GRPO 为 16×8；teacher 和 ASR 通过独立 Serving 部署，各使用
一个 GPU 副本。没有把本轮小规模结果当作 32 卡性能复验。

每个 case 完成 20 次更新。下表取同样的 rollout 3–18，共 16 个完整窗口，包含
生成、评分、归档、训练、清理和 tensor 权重发布；排除前三步热身和最终 checkpoint。
所有稳态窗口均保留。

| 模式 | 基线秒/步 | 重构秒/步 | 步耗时变化 | 音频帧吞吐变化 |
| --- | ---: | ---: | ---: | ---: |
| MOPD | 4.827 | 4.819 | -0.17% | +0.25% |
| GRPO | 6.501 | 6.538 | +0.57% | -0.69% |

本轮表现接近：MoPD 约快 8 毫秒/步，GRPO 约慢 37 毫秒/步。GRPO 的变化主要在
生成与评分阶段（5.111 → 5.155 秒）；训练为 1.036 → 1.032 秒，发布为
0.314 → 0.311 秒。短跑只有一组配对，不能据此保证亚百分比级别零回退。
四组共完成 80 次参数更新；每步检查所有 rank 的单 pack 条件，全部梯度和 loss
有限，末步联合 checkpoint、数据游标、完整候选归档和评分服务身份检查均通过。

真实 v0.4.0 模型为 24 RVQ 通道。Teacher 采用相同单请求输入进行新旧数值对照，
覆盖自然停止、长度截断、零帧、128/512 帧及多条输入，code 和 selected stop
logprob 的最大差值均为 0。128/512 帧测试的响应体缩小 8.16%–8.29%；HTTP
中位耗时因形状而异，batch8/512 帧为 390 → 304 ms，但尾延迟存在明显波动，
不能将其解释为稳定的整步加速。Teacher 探针在首次训练 rollout 前结束。

首次跨服务混合 batch 的逐点比较失败。进一步同版本对照发现，旧版本自身在
服务端组批变化时也会产生分数差异；固定单请求和匹配组批时新旧分数一致。
重复 token 的合成输入中，旧版同一 stop 分数出现约 0.875 的变化。真实短轨迹的
单请求新旧分数完全一致；跨批次的 RVQ 分数也存在变化，stop 差异约 4e-7。
本轮没有修改 attention 或组批算法，也没有定位该既有批次敏感性的根因。
原始失败日志和进一步的对照均保留，不声称任意 HTTP batch 都逐点等价。

CPU 回归：Slime 344 passed / 1 skipped，Omni 62 passed / 6 skipped。
Omni 的全仓 pre-commit 和 Slime 的改动文件检查通过。GPU Job
slime-prism-cleanup-dp2-1014-a1 成功，全部 Job / Serving 条目保留。

固定产物目录：

```text
/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/compat-cleanup-20261014-024941
```

analysis/training-comparison.json 保存逐阶段统计与全部配对窗口；
analysis/teacher-benchmark.json 保存数值对照和 HTTP 计时；
analysis/batch-grouping-observation.json 与 teacher-real-rollout-summary.json
记录组批敏感性。每个 runs/<case>/verification.json、checkpoint/ 和 experiment/
分别保存该 case 的验收结果、checkpoint 和候选/音频/指标。平台回执位于 platform/。

## 运行时重构与 32 卡回归（2026-10-13 UTC）

本轮把 Prism 运行时从 examples 移入安装包，拆分模型辅助逻辑，并验证教师请求、
样本观察器、紧凑 replay、权重发布边界、联合 checkpoint 和原生阶段计时。
对照基线为 Slime `fb4a0f6b` / Omni `9e9e67f8`。最终训练代码为 Slime
`b239a1ac` / Omni `50b8b1d8`；MoPD 64 步速度测试使用 `a86079a4`，其后
`b239a1ac` 只补充旧 checkpoint 升级路径的保护和文档，没有修改训练/生成热路径。
每个 case 的完整 SHA、参数、日志和采样均单独保存，运行中的源码保持固定。

所有速度测试使用同一组 32 张 H200，DP32、TP=PP=CP=1、B2048、每卡 20480
时间位置预算、每 rank 一条 packed microbatch；训练与学生推理均常驻。
MoPD 每次抽取 2048 个 prompt、每个生成一次，使用两个独立部署的同权重 v0.4.0
冻结教师副本；GRPO 为 256×8，只使用四副本 ASR 奖励。每条已生成轨迹只训练一次。
实际平均样本长度约 277–278 个时间位置，每卡约 17700–17800；更长数据应重新
核对 packing。64 步 MoPD 会跨过 80000 条 prompt 的 epoch，并重新生成后续轨迹。

下表使用相同的 rollout 3–22，共 20 个稳态窗口。基线各有两轮独立重复，
取两轮均值；窗口从本轮 generate 起点到下一轮起点，包含评分、归档、训练、
清理和发布。初始化、前三步、checkpoint 步骤及末步单列，不删除较慢的稳态步骤。

| 模式与配置 | 基线秒/步 | 重构秒/步 | 步耗时变化 | 重构 samples/s |
| --- | ---: | ---: | ---: | ---: |
| MoPD，教师并发 16，每步清理 | 14.289 | 14.135 | -1.1% | 144.89 |
| MoPD，教师并发 16，每 16 步清理 | 14.289 | 13.665 | -4.4% | 149.87 |
| GRPO，每步清理，两轮均值 | 17.068 | 16.987 | -0.5% | 120.56 |
| GRPO，每 16 步清理 | 17.068 | 16.306 | -4.5% | 125.60 |

GRPO 每步清理时的差异很小，结论为基本持平、未观察到回退。完整 64 步 MoPD
的 59 个稳态窗口为 13.640 秒/步、150.15 samples/s；计入中途 checkpoint 的
60 个窗口为 13.721 秒/步。完整 32 步 GRPO cleanup16 的 28 个稳态窗口为
16.421 秒/步、124.72 samples/s；后面的数据窗口也保留在该均值中。
配对音频帧吞吐分别从 13069 提高到 13666 frames/s（MoPD），以及从 11043
提高到 11547 frames/s（GRPO）。这些短期重复实验不构成任意工作负载下的速度保证。

首次重构候选将每个 rollout worker 的教师 HTTP 并发限制为 4，导致 MoPD 两轮
均值变为 15.061 / 15.177 秒，回退集中在 generate 阶段。保持其余配置不变，
将并发提高到 16 后，32 步消融为 14.139 秒/步。最终默认并发为 16；批量上限
仍为 8，实际平均聚合约 1.3 条，因此不能将其解释为每个请求都攒满 8 条。
MoPD worker 的合计 CPU 时间从每步约 43.5 秒降到 35.2 秒，减少约 19%。

通用 `--resident-cleanup-interval` 默认仍为 1。本轮常驻配置可显式使用
`--resident-cleanup-interval 16`，或设置 `RESIDENT_CLEANUP_INTERVAL=16`。
MoPD 全运行采样显存峰值约 50.2 GiB，GRPO 约 51.0 GiB，未出现持续增长。
32 卡稳态 GPU busy 约为 39.7% / 53.2%；NVML 的一秒采样不是 MFU，且按采样
时刻归属的短阶段可能混入前一阶段。精确阶段耗时使用 driver 的独立计时。
MoPD 每步仍约 11.7 秒用于生成和教师评分，训练约 1.3 秒；重构没有消除这部分
主要耗时，也没有证明可以达到 90% GPU busy。

功能验收包括 4 步新建联合 checkpoint、2 步完整恢复、3 步全卸载/恢复、
两种算法的重复对照、MoPD 64 步和 GRPO 32 步连续运行。随后从用户旧长跑的
checkpoint 999 实际恢复 4 步：optimizer 两组计数 `[1000,1000]` →
`[1004,1004]`，scheduler 2048000 → 2056192，数据游标保持 epoch 25、
offset 48000 → 56192，最终保存 iteration 1003。续训没有启用 finetune、
跳过 optimizer/RNG 加载或重置 optimizer。旧目录保持不变，新目录启用联合提交。
14 个本轮 case 的 checkpoint 均核验了 302 个模型 tensor、三类 optimizer
状态的元素数、实际 optimizer/scheduler 计数和保存策略。累计完成 333 次更新，
包含前一作业成功的 4 步 smoke；各独立实验不代表一条连续 333 步的训练轨迹。

最终 64 条独立 ASR 评估跳过教师评分，全部 WAV 有效；固定 ASR 指纹，
按项目现有归一化口径计算 corpus WER 为 3.19%，平均 reward 为 0.9704，
评估耗时 3.82 秒。它验证评估链路，不代表长期语音质量收益。TensorBoard
实际保留 RVQ/stop、概率差异、domain/teacher 和评分覆盖率；GRPO 的 WER
评分覆盖率为 100%，另抽查各候选首末轮记录、波形和策略版本。

Slime 广泛 CPU 回归为 372 passed / 2 skipped，后续教师和 checkpoint 回归为
126 passed / 1 skipped，旧 checkpoint 升级保护另通过 19 项相关测试。
Omni 的控制、显存和评分测试为 86 passed / 1 skipped；完整 pre-commit 通过。
这些测试集合有重叠，不相加作为独立用例总数。

主作业 `slime-prism-32-refactor-1013-a4` 完成全部训练和评估后，新增检查器误将
首次初始化的 finetune/no_load 标志用于判定续训，导致尾部审计返回失败。
修正规则后，`slime-prism-refactor-audit-1013-a5` 在同一镜像上进行单卡只读复核，
通过全部检查，没有重跑训练。原失败日志和 campaign 返回码没有覆盖。
所有 Job、Serving 条目、前三次启动/配置失败尝试及完整产物均保留。

本轮固定产物根目录为：

```text
/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/runtime-refactor-20261013-182303
```

入口为 `analysis/report.html`；`analysis/comparison.json` 给出相同步号对照，
`run-summary.json` / `run-summary.csv` 保留全部统计与逐步窗口，
`feature-validation.json`、`checkpoint-verification.json` 和 `final-audit.json`
分别记录指标/归档、真实 checkpoint 状态和修正后的只读复核。
该历史恢复点位于 `runs/mopd-eval-final-r1-a4/checkpoint/`，对应当时的联合提交
`rollout/training_commit_1003.json`。这份 manifest 属于上述固定源码的验收产物；
当前 Prism 配方使用 Slime 原有的普通 checkpoint 保存和恢复流程，不再生成或读取
这类联合提交。增加训练轮数时明确使用 checkpoint optimizer scheduler 策略，
见 README 的恢复说明。

## 指标、归档与日志策略合入验收（2026-10-13）

合入 `feat/prism-training-metrics` 与 Omni `fix/prism-log-noise`，保留之前已验证的
GC threshold 70000、object-store tensor buffer、教师 score chunk 1024 和 resident 训推配置。
Slime 的 GPU 验证源码为 `eba5c5ee`，Omni 为 `9e9e67f8`。全部使用原 `train.py`、
Prism v0.4.0 iteration 100000、DP32/TP=PP=CP=1、B2048、每卡 20480 时间行预算，
每个 rank 均为单 packed microbatch。GRPO 为 256 prompts × 8，仅使用四副本 ASR；
MOPD 使用两个相同 v0.4.0 冻结教师副本，每个独立 trial 内不重复使用 prompt。
本轮没有并行申请 64 张训练卡。

**验收边界：GRPO 没有观察到速度退化；MOPD 仍有 2.7%–3.5% 的额外耗时，
没有满足严格的零性能回退要求。** 结果不将小幅回退归为已经消除。

| 同分配对照 | 原基线秒/步 | 全 feature 秒/步 | 步耗时变化 |
| --- | ---: | ---: | ---: |
| GRPO，16 rollout workers | 17.186 | 16.943 | -1.4% |
| MOPD，16 workers，第一组 | 13.909 | 14.281 | +2.7% |
| MOPD，16 workers，复测 | 13.951 | 14.438 | +3.5% |
| MOPD，32 workers，复测 | 13.951 | 14.436 | +3.5% |

保留 16 workers。增加到 32 没有收益，未作为默认配置。每组通常运行18次更新，
前3次用于预热及算子采集，最后一次 checkpoint/额外评估单列；中间14个完整步骤
包含生成、评分、归档、日志、数据发布、训练、清理和 tensor 权重同步，保留所有慢步骤。
同节点配对以 driver 的相邻 generate 起点计时，不把不同实验重新初始化的空隙放入稳态。
报告提供全部逐步数据和成对 block-bootstrap 区间；它只描述短跑波动。

原样合入的 MOPD 平均60.417秒/步，其中归档汇总约13.989秒，暴露了共享盘上的
重复目录遍历、路径解析和小文件操作。修订后使用每进程候选索引，复用已返回的
样本记录；固定目录和同组公共文件有界缓存；准备与评分结果写入使用独立队列。
不生成波形的 MOPD 保存完整的分片候选 JSON，prompt/引用信息保留在记录内。
逐样本原子记录、失败/取消/未选中候选与取消时写入完成的语义均保留。

GRPO 还需要省去 WAV 的中间移动。实验入口的 audio path hook 让客户端直接写入
最终 group 目录；正常插件调用保留原路径行为。完整对照中，单候选归档准备从
1.159秒降到0.010秒；ASR请求均值从0.899秒回到0.474秒。18步的整组日志从
45,479,278字节降到877,255字节（含初始化，不能直接换算成训练收益）。
HTTP成功访问与权重桶消息降噪，错误、警告、训练与发布摘要保留。

RVQ/stop 诊断改为 detached segment reduction，不修改目标函数或梯度；64样本、
365464个选中动作的H200微基准从2.142ms降至0.066ms，最大数值差5.96e-8。
这项局部加速不能解释为整步32倍加速。MOPD剩余开销集中在rollout侧的归档与汇总，
训练、内存清理和权重发布基本不变。

GPU测试同时核验真实sample/group计数、MoPD prompt唯一性、全部rank单microbatch、
有限非零梯度、每轮32副本版本、checkpoint模型/optimizer结构和step计数。
完整归档清点在测速结束后进行；抽查训练WAV头并检查全部独立eval WAV，校验
TensorBoard中的RVQ/stop、覆盖率、domain/teacher等实际输出。独立64样本ASR评估
不调用教师，单列耗时约4秒，不作为语音质量收敛结论。另保留32步MOPD长确认结果。

首个Job因旧版TensorBoard未读取CLI目录、写入容器相对路径而触发磁盘配额错误。
该失败记录保留，未计入速度；重跑对全部32个rank执行了共享盘写入预检。
`mopd32-optimized-001`含有带采样延迟的py-spy诊断，调用栈只用于定性定位；
`grpo32-features-002`与文件审计有部分重叠，仅作诊断参考。回归结论使用单独列出的
基线/最终候选，不混用这些诊断窗口。

全部Job、Serving条目、失败尝试、源码pin和原始产物保存在：

```text
/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/metrics-integration-20261013-113000
```

入口为 `metrics-integration-report.html`，机器可读结果为 `analysis/comparison.json`、
`results.csv`、`feature-validation.json`、`training-validation.json`、
`publication-verification.json`、`checkpoint-verification.json`。
最终远端/ editable 对齐收据与资源释放信息分别在 `source-delivery.json`、
`resources-released.json`。实际复现按每个 `configs/*.json` 的源码和配置执行。


## 32 H200 全流程时间线与数据交接优化（2026-10-13）

本轮继续使用 Prism v0.4.0 iteration 100000、DP32/TP1、B2048、每卡20480 token预算、
一次 packed microbatch，以及两个独立部署的同权重冻结教师。训练和学生常驻，
16个完整 rollout workers；每条 MoPD prompt 在一次运行内只消费一次。
未提交新的64卡作业。本轮完整 CUDA 采集与新增端到端对照针对 MoPD，未外推 GRPO/ASR 收益。

同一组32卡按下表顺序运行。稳态窗口包含生成、评分、数据交接、训练、内存清理和
tensor 权重发布，剔除前两轮、重型 profiler/关闭轮次和最终 checkpoint。
全部慢轮次保留；首尾两组原配置用于观察负载漂移。GPU busy 为全部32卡的一秒NVML
采样均值，不是MFU。每次实验都验证单microbatch，实际平均pack约17800 tokens。

| 配置 | 更新数 / 稳态窗口 | 秒/步 | samples/s | GPU busy |
| --- | ---: | ---: | ---: | ---: |
| 原配置 A | 8 / 3 | 16.441 | 124.56 | 35.31% |
| GC threshold 70000 | 16 / 11 | 16.176 | 126.61 | 37.79% |
| GC + 教师 score_chunk_size 1024 | 16 / 13 | 15.391 | 133.07 | 37.42% |
| GC + 教师分块 + 训练 tensor buffer | 16 / 13 | 14.058 | 145.68 | 41.22% |
| 原配置 B 复测 | 6 / 3 | 16.833 | 121.66 | 34.64% |

首尾基线均值16.637秒，相对组合14.058秒，步耗时降低15.5%、吞吐提高18.3%。
基线窗口较短，报告保留每轮数据和前后复测，不把单项收益相乘。组合的训练数据发布
平均0.785秒，对应上一组的1.803秒；learner调用1.216秒、内存清理0.559秒、权重发布
0.530秒。32卡观测峰值显存50.25GiB，manager稳态RSS峰值4.38GiB；此短跑不是长期内存
稳定性或语音质量收益的证明。

全流程追踪关联每轮2048条生成及2048条教师评分，覆盖所有32个学生的请求事件、
4个代表性学生GPU、两个教师GPU及learner rank0 CUDA活动。时间线串联worker、router、
frontend、AR、教师排队/前向、DP分片、训练和权重发布；提供可导入Perfetto的合并trace。
跨主机时间以最小RTT校准，首次完整采集最大不确定性约±54.8ms。嵌套请求跨度不能相加，
同卡learner与student进程也不能计作独立GPU。

定位与处理：

1. 主进程真实GC暂停约3.24秒/步。提高自动GC阈值后降至约1.29秒，完整回收仍会发生；
   16个worker的GC开销也降低。实际整步只改善约1.6%，说明不少暂停与教师工作重叠。
   `gc.freeze()`在真实CPU复现中没有改善总时间，未采用。
2. 训练分片原先仍逐tensor执行Torch storage序列化。新增`--rollout-tensor-buffer`
   对object-store分片复用buffer reducer，不注册全局tensor serializer。保持dtype、
   shape、值、接收者写入隔离及嵌套ObjectRef所有权；特殊tensor回退原reducer，空tensor
   显式恢复到CPU。旧worker buffer引用的reducer模块路径保留兼容别名。
3. 教师评分head以128个预测位置分块，引入大量小kernel。单GPU固定真实输入实验中，
   chunk1024吞吐100.90→111.77条/秒。128请求输入hash、冻结权重和code logprobs一致，
   stop probability最大绝对差4.47e-7。增加构建线程至4反而降至89.47条/秒；扩大批次至64
   未优于单纯增大分块，均未采用。
4. 训练后clear_memory的约0.55秒主要由每个rank约0.5秒的完整GC组成。本轮保留该清理
   语义；调整其频率仍需独立显存/同步边界验证。教师存在请求时仍有数秒细碎CUDA空隙，
   后续可继续研究减少host launch或图捕获，不能把GPU busy低直接当成算力饱和。

短测试的初始化也显著降低平台观测值：原配置A从首个NVML观测到首次生成约528秒，
包含节点启动差异、Ray、32个Omni实例、模型与graph初始化；整段观测GPU busy约9.97%，
稳态35.31%。生产长跑应单独记录并摊薄冷启动，不能混用两种统计口径。

本轮七组成功case共80次优化器更新、163840条样本，全部32rank单microbatch、有限
非零梯度、MoPD prompt唯一性、当前样本版本和每个学生的全部发布版本通过。
七份checkpoint在GPU环境核验模型/optimizer元数据和计数，均含302个模型tensor，
两组optimizer step与对应8/10/16/6次更新一致，scheduler按B2048推进。
初次诊断的失败case与Job保留：脚本误用trace文件名`{pid}`占位符导致教师stage退出；
修复脚本、前端模板校验和profiler重复start后，实际GPU start/score/stop预检与完整采集通过。

Omni诊断相关285 passed、1 skipped，pre-commit全仓通过；Slime GC/协议/恢复相关191
passed、1 skipped，训练buffer相关38 passed，另覆盖空tensor默认设备、旧reducer引用和
删除原始引用后的真实Ray所有权。CPU恢复测试最初8项因子进程缺少straw失败，隔离环境
补齐后原8项全部通过；保存的日志区分环境失败与补验。

全部产物位于
`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/timeline-profile-20261013-070650`，
入口`timeline-report.html`。原始请求/CUDA、CPU/GPU采样、逐轮表、配置与校验文件均保留。
配对Job `slime-prism-32-gc-ab-1013` 为SUCCEEDED，共62次更新；其余18次来自采集前的两组。
Job与Serving记录全部保留，运行资源已停止。GPU实验Slime使用e923be47及8e60fed2，
Omni使用23a0f19e；最终发布另外包含reducer兼容别名、测试和文档，源码交付SHA见该目录
`analysis/source-delivery.json`。

## 32 H200 rollout 与评分优化（2026-10-13）

本轮按用户要求集中优化 32 卡，未提交新的 64 卡作业。使用同一 v0.4.0 iteration
100000、真实 ASR 奖励与同权重冻结教师；TP=PP=CP=1、DP=32、B2048、每 rank
20480 tokens、一次 packed microbatch。GRPO 每 prompt 8 条，MoPD 每 prompt 1 条且
只消费一次。训练与学生模型驻留，16 个完整 rollout CPU workers；评分服务独立
使用表中数量的 H200。GPU busy 为稳态窗口内全部训练卡的一秒采样均值，不是 MFU。

| 配置 | 更新数 / 稳态窗口 | 秒/步 | samples/s | GPU busy | 含评分 GPU秒/样本 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRPO HTTP 分发基线，ASR×1 | 5 / 2 | 54.79 | 37.38 | 18.4% | 0.883 |
| GRPO 完整 worker，ASR×1 | 5 / 2 | 41.86 | 48.92 | 28.6% | 0.675 |
| GRPO 完整 worker，ASR×2 | 10 / 7 | 26.40 | 77.57 | 35.4% | 0.438 |
| GRPO 完整 worker，ASR×4 | 10 / 7 | 20.78 | 98.55 | 41.6% | 0.365 |
| GRPO 同节点复测，ASR×4，并发16 | 8 / 5 | 19.18 | 106.78 | 46.4% | 0.337 |
| GRPO 同节点，并发32 | 8 / 5 | 23.84 | 85.89 | 37.8% | 0.419 |
| GRPO buffer，ASR×4，并发16 | 12 / 9 | 19.82 | 103.33 | 44.8% | 0.348 |
| MoPD 原教师×1，配对基线 | 5 / 2 | 31.16 | 65.72 | 28.3% | 0.502 |
| MoPD 优化教师×1，配对 | 5 / 2 | 25.22 | 81.20 | 33.5% | 0.406 |
| MoPD 优化教师×2 | 8 / 5 | 17.33 | 118.15 | 33.9% | 0.288 |
| MoPD buffer，优化教师×2 | 12 / 9 | 16.81 | 121.83 | 33.2% | 0.279 |
| MoPD 完成队列配对基线 | 8 / 5 | 16.08 | 127.34 | 35.2% | 0.267 |
| MoPD 完成队列优化 | 12 / 9 | 16.08 | 127.36 | 34.3% | 0.267 |

前两次更新及最终 checkpoint 不计入稳态；完整窗口从一次 generate 开始到下一次
开始，包含评分、数据处理、learner 和 tensor 权重发布。保留所有慢轮次。
HTTP/worker 的单 ASR 对照在同一作业内完成；教师原版/向量化也使用相同分配节点。
并发16/32、双教师与 buffer 最终验证复用另一组相同32卡节点。跨作业的评分副本
梯度实验会受到外部负载及采样长度波动影响，不把各项收益简单相乘。

瓶颈及修改：

1. 只分发 HTTP 仍让 JSON/协议解析、tensor 构建、WAV 落盘、教师 hook 和奖励
   集中在一个进程。完整 group 下放 CPU worker 后，单 ASR 的 GRPO 协调 CPU
   约32.9秒/轮降至约8秒/轮，吞吐从37.38升至48.92条/秒。协调进程继续执行
   选样、过滤与排序；旧轮任务排空后才允许新权重版本，失败评分直接传播。
2. 客户端改善后，ASR 成为主要等待源，1→2→4 副本逐步提高吞吐。继续把生成并发
   16加至32没有稳定收益，出现大批评分请求同时等待的长尾，因此最终保留16。
3. 冻结教师的模型输入校验以 NumPy 批量检查词表、role、successor edge 和 anchor。
   128个真实输入的 hash 与所有构建 tensor 完全一致，32个实际 GPU 教师输出逐项
   一致。CPU 构建约4.93→2.56 ms；实际单教师配对 MoPD 31.16→25.22秒/步。
4. 实际 coordinator profile 一轮GRPO触发24576次、MoPD触发28672次 tensor
   storage加载。worker结果改为buffer传输，普通CPU tensor免去逐storage torch.load；
   接收端复制为独立可写tensor，保留完整样本元数据、dtype和形状，特殊tensor回退
   原reducer。未注册进程级tensor序列化器，不改变权重发布的CUDA IPC路径。

5. MoPD 2048个独立任务逐个结束时，原asyncio.wait反复遍历剩余任务并重建callback；
   新完成队列让每个task只通知一次，并支持补样、取消及异常。纯调度微测中2048个
   完成事件约0.65→0.015秒CPU；4096个约2.99→0.030秒。微测不计模型和网络，
   端到端同节点8/12更新配对约16.083→16.080秒/步，未观察到额外吞吐收益；
   协调CPU平均9.78→8.51秒。教师两副本的NVML稳态busy约13.8%和14.0%，当前
   还受请求处理和端到端调度限制，不能据此声称教师GPU计算已饱和。

单独buffer的GRPO端到端收益未获证实：同节点基线19.18秒/步，12次更新的buffer
测试19.82秒/步（含一次ASR长尾）；中位数18.97→18.65秒。协调CPU平均8.30→6.24秒，
因此只能确认CPU开销下降，不把整步均值声称为提升。评分长尾、短序列AR小batch
及阶段屏障仍限制GPU busy；不以增加无效计算或隐藏梯度累积来提高利用率数字。

Slime源码候选为`f2de700562689f75083702bdee6af14b87b4c195`（完整worker）和
`5814045f268a8e96ba8eb73e0dc106aee62991eb`（buffer）及
`5d36a3f5ae66497aa8cd35c84818d54be6c8dbf6`（完成队列），Omni教师候选为
`402f9affa9a7f1aecf6f546cc6c3602fbd106d6f`；Omni学生仍使用a25fe84基线。
Slime相关回归279 passed、1 skipped；Omni评分29 passed、pre-commit全仓通过。
256个真实MoPD样本的全部字段做过传输前后比较；另有两进程Ray、原生采集、
跨轮版本、取消排空、异常传播与嵌套ObjectRef验证。

完整报告及逐次配置：`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/worker-profile-20261013-031520/optimization-report.html`。
`analysis/training-validation.json` 检查所有rank的单microbatch、有限非零梯度及
MoPD prompt唯一性；`publication-verification.json`检查全部学生副本的发布版本。
早期四组checkpoint目录当前缺失（HTTP基线、ASR1 worker、ASR2 worker、初始单教师），
虽有成功保存日志，不把它们声称为tensor内容核验通过；`checkpoint-verification.json`
明确列出可读通过与缺失目录。最终GRPO与MoPD各12更新的checkpoint均核验完整参数与Adam
moments，optimizer进度为[12,12]，scheduler消费24576条。全部14组共113次更新、
231424条样本，所有32个训练rank均保持一个microbatch且梯度有限非零，每条样本
版本与当前发布版本一致，GRPO组完整、MoPD每次运行内prompt不重复。
平台Job全部保留。性能短跑不证明语音质量提升，也不新增跨引擎logprob精度保证。

## 32/64 H200 性能验收（2026-10-12）

使用相同的 Prism 1.7B v0.4.0 native iteration 100000、统一 editable 镜像和原
`train.py`。TP=PP=CP=1，DP 等于训推共卡数量；GRPO 每 prompt 8 条轨迹，
仅用 Qwen3-ASR-1.7B 的实际转写计算奖励；MOPD 每 prompt 一条轨迹，独立部署
同一 v0.4.0 权重的冻结教师。ASR、教师分别占用额外一张 H200。

生产配置使用每 rank 一次 packed microbatch、20480 tokens 预算、FLOPs 均衡、
distributed optimizer、256 MiB 权重桶，以及 `--no-offload-train --no-offload-rollout`。
Omni KV pool 为 65536 tokens，max running requests=32，生成并发=16/副本。
每组完成 5 次 optimizer update；表中排除最初两轮与末轮 checkpoint，记录两个
完整稳态窗口的均值。生产测试关闭完整 rollout tensor dump；原始 Job、日志、
逐样本统计、GPU 采样和完整 checkpoint 保留。诊断配置同时启用 durable replay
恢复日志，生产配置未启用；checkpoint 分别采用 fully-reshardable 和 dp-reshardable
optimizer 存储布局，两者的训练计数及参数/moments 总元素数均已检查。

| 算法 | DP | Batch | 秒/步 | samples/s | tokens/s | 平均 pack tokens | GPU 平均利用率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRPO | 32 | 2048 | 68.79 | 29.77 | 8297 | 17837 | 23.9% |
| GRPO | 64 | 2048 | 77.32 | 26.49 | 7387 | 8924 | 9.5% |
| GRPO | 64 | 4096 | 151.86 | 26.97 | 7506 | 17809 | 12.3% |
| MOPD | 32 | 2048 | 45.74 | 44.77 | 12487 | 17849 | 33.0% |
| MOPD | 64 | 4096 | 91.57 | 44.73 | 12431 | 17786 | 19.6% |

所有 rank 的实际 `num_microbatches` 均为 `[1]`。完成 checkpoint 的 optimizer
step、scheduler 消费样本数、302 个模型张量和完整 Adam moments 元素数均核验；每次学生发布的
版本覆盖全部副本。MOPD 各次运行中 prompt_id 不重复，每条轨迹只参与一次
optimizer update；旧学生与教师的评分前向仍按原生 OPD 流程执行。

本数据平均 L 约 278，包含文本、参考音频与生成时间行。因此 B2048/DP64
仅约 43% pack 填充，B4096 约 87%；24 路 RVQ 不乘入时间长度。增加 batch
提高填充率，但整体吞吐还受 rollout CPU 处理、HTTP 转发和评分等待限制。

保留完整 tensor 归档的受控对照中，DP32 GRPO B1024 约 64.62 s/step，
单独把并发4提高到16约65.86 s/step，没有观察到收益；驻留与配套优化后
B2048约90.08 s/step，samples/s提高约43%。DP32 MOPD从B1024约68.04 s/step
到B2048约67.87 s/step，吞吐约翻倍。完整 tensor dump 本身在B1024下每轮
约9–10秒、B2048下约16–20秒，故与上表生产结果分开记录。

CPU修复包括NumPy2兼容、奖励服务Bearer鉴权、stage mapping配置检查、
DP/VPP对齐后的packing、保持隔离的协议快照复制，以及客户端/Router按
服务和worker拆分HTTP连接池。128条真实响应的解析结果和教师请求hash
逐项一致；相关回归为231 passed、1 skipped。数值目标、教师权重和数据
版本保持固定；本验收不测训练质量提升，也不建立新的跨引擎logprob误差保证。

完整实验归档：`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/scale-profile-20261012-193554`。
`profile-report.html` 汇总CSV、阶段图、验证结果和保留Job清单；
`analysis/failures-and-fixes.json` 记录启动、网络与配置修复。失败和重试
Job也保留，未删除平台记录。

### 利用率瓶颈与 HTTP 对照（2026-10-13）

生产配置的 rollout/评分占整步 94%–98%；DP32 GRPO 的 learner 调用约1.35秒，
权重发布约0.55秒。DP64/B4096 GRPO的一轮里，约第90秒之后学生GPU已基本
无活动，rollout到第151秒才结束，客户端仍在处理生成响应与ASR积压。
平均pack填充到87%只改善短暂的训练段，不能消除生成和评分等待。

原生异步请求之后的JSON解码、协议解析、tensor转换、base64音频解码及
WAV落盘仍有同步执行路径。同一rollout进程的CPU/事件循环限制了请求分发。
DP32 GRPO decode日志平均只有1.33个running request，DP64/B4096只有1.19，
P95均为2，远低于配置的每副本16并发。日志统计以实际decode事件为条件。

使用真实生成WAV/轨迹独立压测：ASR约55–62条/秒，冻结教师约66–81条/秒，
高于当前端到端吞吐。当前优先改善客户端调度和数据处理，再扩评分副本。

同一组32张H200、B2048、五次更新，仅切换原生HTTP分发，稳态各取两轮：

| HTTP方式 | 秒/步 | samples/s | GPU busy | rollout进程CPU秒/轮 | 事件循环延迟P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 本地 | 65.78 | 31.14 | 26.5% | 50.02 | 482 ms |
| 分布式POST | 57.46 | 35.64 | 16.8% | 33.98 | 1.97 ms |

`--use-distributed-post`在这组GRPO中提高吞吐约14.5%，但没有消除集中处理：
少量事件循环阻塞仍达约5秒。平均decode batch从约1.34增至2.83；GPU busy
虽下降，每条样本消耗的训推GPU秒仍从1.028降至0.898。因此同时用samples/s、
GPU-seconds/sample、阶段耗时和batch统计判断效率，不能只优化nvidia-smi百分比。
64卡与MoPD尚未测这个HTTP开关的收益，前表使用本地HTTP，不外推14.5%。

后续优先按worker分散完整生成/解析/音频/评分流水线，保留GRPO组及权重版本
屏障；随后减少大JSON与base64往返，使用紧凑tensor传输和worker生成的共享音频
路径。教师服务已有continuous batching，可另测客户端多样本RPC减少HTTP开销。
本轮保持MoPD轨迹一次消费、每rank一个microbatch，没有引入跨版本异步训练。

## 单环境 editable 镜像验收（2026-10-12）

`slime-ttsrl:editable-v1` 已从新镜像在一张 H200 上完成原 `train.py` 的两轮共卡
GRPO，并从完整 checkpoint 恢复继续第三轮。镜像 digest 为
`sha256:008ecc97fd306fc13c351d38a0e46ea2592a0ece0a8e8ba3102a0a06a712cd6c`，
单层压缩大小 15,629,682,043 bytes。Ray、learner、Omni 均使用
`/opt/venvs/core/bin/python`，训练进程实际记录 NumPy 2.3.5、Transformers 5.12.1。

三份共享源码使用 editable 安装，隔离模式 Python 的 import 路径与安装元数据均已检查；
对三份源码分别修改临时模块并启动新进程，确认两次修改无需重新安装即可生效。
临时模块随后移除。验收代码为 slime `4909f8121d6af1ae0495f306f799929b35e347f0`、
Omni `a25fe84ec01476799fbb637a8d1e4b5765dc9204`、Megatron
`3f49de20de81c1ad5dfc10efa0f7d0c8098f1ce4`；目录与镜像使用方式见
[INSPIRE.md](../../../INSPIRE.md)。

| 阶段 | Optimizer / scheduler | 学生发布版本 | 梯度范数 | Tensor buckets | WAV |
| --- | --- | --- | --- | --- | --- |
| 首次初始化、两轮训练 | `[0,0]` → `[2,2]`；0 → 8 | 1、2、3 | 11.9749088、10.7919912 | 204 | 8 |
| 完整恢复、第三轮训练 | `[2,2]` → `[3,3]`；8 → 12 | 3、4 | 9.9911604 | 136 | 4 |

各阶段最终 checkpoint 均检查 optimizer 张量及进度；恢复阶段明确从新进程读取原 optimizer、
scheduler。真实 CUDA IPC 探针逐元素一致。CPU 回归为 slime 263 passed、4 skipped，
Omni 175 passed。每轮仍为四条八帧轨迹和 ±1 工程奖励，验证工程闭环，不评价语音质量提升。

NumPy 2 兼容包含 backend 入口的 `np.product = np.prod` 别名及移除初始化的 NumPy 1.x
硬限制。第一次镜像实跑发现该旧断言，修复后使用同一镜像重新完成验收；失败记录保留在
QB 验收根目录的 `image-editable-v1-001/`，完整成功报告与训练产物位于
`image-editable-v1-002/verification.json` 及其 `fresh/`、`resume/` 子目录。
镜像记录的是依赖环境及初始构建源码，后续 editable 兼容修复以本节验收 SHA 为准。

另将 PyAV 固定为 16.1.0，修复基底 PyAV 19 移除 `metadata_errors` 导致的 M4A 时长
探测失败。补齐 slime 声明的缺失依赖，保留 Torch/CUDA/TE 等既有版本；基底的依赖
元数据冲突仍单独记录，未将 `pip check` 声称为通过。构建、清理和依赖审计记录在
`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-builds/prism-editable-v1-20261012-1801/`。

## Checkpoint 驻留优化对照（2026-10-12）

在同一作业、同一张 H200 上，从 `colocate-006/resume/checkpoint/iter_0000002`
分别恢复旧流程与优化后的流程，各运行 rollout 3。两组均使用四条八帧轨迹、工程奖励和
完整 checkpoint 保存，Omni source 固定为 `bb73d42c16f246ff493e676b0d9242b61f9d89f3`。
旧 slime source 为 `8bcae7e898efeaaed9218ad02c458dfb2b5d66b0`，优化后的 source 为
`afe9181768c99f931cd2ac1daff14be4c645c316`。

| 训练完成至权重发布之间的操作 | 旧流程 | 优化后 |
| --- | --- | --- |
| 训练结束后的第一次 sleep | 7.7 s | 省去 |
| 为保存执行 wake_up | 2.7 s | 省去 |
| 保存完成后的 sleep | 7.7 s | 8.5 s |
| 状态切换合计 | 18.1 s | 8.5 s |
| Checkpoint 工作时间，扣除嵌套 sleep/wake 计时 | 53.9 s | 53.7 s |

本次保存点的状态切换净减少 **9.6 秒**。优化让需要保存的轮次执行
`train → save → sleep → publish`，保存前仍同步 CUDA 并清理临时分配。模型驻留状态由
worker 记录；普通训练调用默认仍在结束时 offload。Actor/critic 顺序、异步保存的
`force_sync` 和模型/rollout 联合提交边界有 CPU 回归覆盖。

两组 checkpoint 的 optimizer step 均从 `[3,3]` 变为 `[4,4]`，scheduler 从 12 变为
16；每份 checkpoint 的 `param`、`exp_avg`、`exp_avg_sq` 均为 302 项，学生版本均从
4 发布至 5。两组各生成四个 WAV。此次对照保持模型、batch 形状与 GPU 相同，分别重新
随机采样轨迹；性能结论采用独立的 sleep/wake 计时，不作逐数值训练等价声明。

联合报告在本文 QB 根目录的 `checkpoint-residency-001/verification.json`，两组完整
产物分别在 `baseline/` 和 `optimized/` 下。相关训练/保存/恢复 CPU 回归为 195 passed。

## 单卡共卡训练与 tensor IPC 验收（2026-10-12）

Prism 1.7B v0.4.0 native iteration 100000 在一张 H200 上通过原 `train.py` 完成了
两轮 GRPO，并从完整 checkpoint 恢复后继续第三轮。learner 与完整 Omni pipeline
使用同一个 Ray GPU bundle；每个阶段均核验只有一张可见 GPU，启用 `colocate`、
`offload_train` 和 `offload_rollout`。运行采用明确标注的 ±1 工程奖励，每轮一个
prompt group、四条轨迹，每条八帧，并实际输出 WAV。

| 阶段 | RL rollout | optimizer / scheduler | 学生发布版本 | 梯度范数 | WAV |
| --- | --- | --- | --- | --- | --- |
| 首次初始化与连续训练 | 0、1 | `[0,0]` → `[2,2]`；0 → 8 | 1、2、3 | 11.7136135、11.6547775 | 8 |
| 标准完整恢复 | 2 | `[2,2]` → `[3,3]`；8 → 12 | 3、4 | 11.8521719 | 4 |

每次发布经 `UpdateWeightFromTensor` 发送 68 个 SGLang flattened buckets，五次发布
共确认 340 次 tensor 更新。接收端在确认前同步 CUDA，逐桶保持暂停，所有权重安装完成
且 KV/graph 内存恢复后才开放生成。首次与后续发布在本次短轨迹运行中约需 0.7–0.8 秒。

两秒间隔的 NVML 采样记录中，AR/preprocessing 进程最大观测值为 96,764 MiB，
offload 窗口内为 950–1,048 MiB；vocoder 最大观测值为 2,866 MiB，offload 时为
610–720 MiB。learner 最大观测值为 58,926 MiB。这些是该测试配置的观测值。

当前共卡使用 eager prefill、eager vocoder 与 AR decode CUDA graph。SGLang 的
breakable prefill graph 不支持 memory saver，因此 adapter 显式禁用该 prefill 后端。
推理进程与 learner 都在可选库导入后选择匹配的 preload allocator；独立 GPU IPC
探针复现了错误 allocator 模式的失败，并验证修复后的跨进程 tensor 逐元素一致。

大文件与证据仍位于本文下述 QB 根目录：

- `colocate-005/fresh/`：两轮训练、`iter_0000000` / `iter_0000001`、debug dump 和音频。
- `colocate-005/memory-observations.json`：进程显存采样摘要。
- `colocate-006/resume/`：恢复后的训练、`iter_0000002`、debug dump 和音频。
- `colocate-006/verification.json`：联合通过报告，分别记录两个阶段的源码与产物来源。

fresh 的 slime source 为 `deb731cae89296b5d857f1039fe4d0631d304ea2`，Omni source 为
`032eda473bf6089bfcf0562fa19089e1217b5217`。resume 使用同一 slime source，Omni source
为 `bb73d42c16f246ff493e676b0d9242b61f9d89f3`。首次同容器立即恢复时，旧 HTTP 服务已
正常退出，但端口探测因 `TIME_WAIT` 误拒绝。修复探测与 listener 的复用语义后，从原
已保存 checkpoint 重新验证恢复；原失败日志保留，联合报告明确区分两次作业。

该验收覆盖共卡资源、真实 IPC、音频生成、原生训练、完整保存与恢复。未评估真实
WER/SIM/JudgeCLAP 收益、多 learner DP、长语音或严格跨引擎概率一致性。

验收使用用户指定的 `CQ-科研驾驶舱` 工作空间、H200 GPU 和 `voxrl:editable-v1-slim` 镜像。学生资产是 Prism 1.7B v0.4.0 native iteration 100000；固定路径及训练/Omni 解释器合同见 [INSPIRE.md](../../../INSPIRE.md)。实际 learner 为单卡，TP=PP=CP=DP=1。结果保留在共享 QB 盘，完整 checkpoint、原始响应和 debug dump 没有写入 Git。

验证目标是原 `train.py` 的训练、保存、恢复与权重发布闭环。严格跨引擎概率一致性、语音质量提升和真实奖励模型联合验收独立处理。

当前 Prism attention 直接使用 Megatron 原生 TE layer spec 与默认 `auto` 后端选择。下表前三场验收来自各自固定的历史源码，使用先前的自定义 FA4 wrapper；其余验收使用原生 TE。旧报告继续保留为历史证据，各次实现和接口以固定 source SHA 为准。

## 已完成的原生训练

| 场景 | 实际执行与结果 | 验收报告 |
| --- | --- | --- |
| 单学生副本 GRPO 工程闭环 | 4 条轨迹各 8 帧，24 RVQ 和 sampled stop；显式工程 reward ±1；原组优势 ±0.8660247；梯度范数 11.4607334；完整 model/optimizer/scheduler checkpoint `iter_0000000`；学生发布版本 1、2 | `ray-attempt-005/verification.json` |
| 两 domain MOPD 工程闭环 | 按 `ch` / `en` 分别直连两独立冻结教师的 `/score_actions`；每条轨迹 192 个 RVQ 动作、8 个 sampled continue；task reward 0；原生 OPD reverse KL 0.00248493，梯度范数 0.7008069；完整 checkpoint `iter_0000000`；学生发布版本 1、2；训练后教师实际 tensor checksum 与启动 pin 一致 | `mopd-smoke/audit-v2-attempt-20261012T113833Z/verification.json` |
| 完整断点恢复与两个学生副本 | 原标准 loader 恢复单副本 checkpoint；训练前 optimizer steps `[1,1]`、scheduler 4；rollout 1 的 4 条轨迹各 8 帧，数据游标继续到样本 4–7；梯度范数 11.5324316；完整保存 `iter_0000001` 后 optimizer steps `[2,2]`、scheduler 8；两个独立 AR 接收 rank 均完成版本 2、3 安装 | `ray-resume-dual-007/verification.json` |
| 原生 TE attention 与完整断点续训 | 实际 28 个 core 均为 `TEDotProductAttention`；默认 `auto`，运行时 TE 选择 cuDNN `FusedAttention` sub-backend 1；标准 loader 恢复旧 FA4 路径保存的完整 RL checkpoint；optimizer steps `[1,1]` → `[2,2]`，scheduler 4 → 8；梯度范数 11.6026993；完整保存 `iter_0000001`；单 Omni 副本完成版本 2、3 安装 | `native-te-resume-008/verification.json` |
| 通用 initializer 首次模型初始化 | 插件从 native iteration 100000 只导入模型，RL 从 rollout 0 开始；通用 hook、`pretrained_checkpoint`，无显式 `load`；更新前 optimizer `[0,0]`、scheduler 0，保存后 `[1,1]`、scheduler 4；梯度范数 12.0528440；完整保存 `iter_0000000`；学生版本 1、2 | `initializer-refactor-010/fresh/verification.json` |
| 通用加载边界的完整恢复 | 只通过标准 `--load` 恢复上一行的新 checkpoint，initializer、预训练来源和 step 都为空；optimizer `[1,1]` → `[2,2]`，scheduler 4 → 8；数据游标继续到样本 4–7；梯度范数 11.1523933；完整保存 `iter_0000001`；学生版本 2、3 | `initializer-refactor-010/resume/verification.json` |

报告目录根为：

```text
/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/
```

单副本 source 为 `f06e7e0c81641baa2cdfdc2ca966509efb2c58b6`；MOPD source 为 `dcd77d4e91382d1ae0e5cdc2117d27be66f299f6`；双副本恢复 source 为 `a31d82f299cfcb2bc32a1af4267f549a8255df51`。每次作业从已提交的 Git SHA 导出独立源码 snapshot，运行时不修改。各自的 source manifest 和完整日志保留在结果目录。

原生 TE 验收 source 为 `676d90ebf781f85d430aa2e2372d4e836bfab8d6`，使用一张 learner GPU 与一张完整 Omni pipeline GPU。源码已删除自定义 FA4 wrapper 和 core attention 替换函数，recipe 与 publication smoke 也沿用 Megatron 默认后端选择。运行时先检查真实 module 类，再记录 TE 的实际 forward backend 日志；底层 cuDNN 后端由 TE 自动选择。旧 checkpoint 的空 attention 辅助状态通过原标准加载路径兼容处理，未重置 optimizer 或进行 finetune。

通用 initializer 两阶段验收 source 为 `e231fa349cdc754822cd915022f0bd409cec2167`，联合报告为 `initializer-refactor-010/verification.json`。两阶段都调用原 `train.py`，使用一张 learner GPU、一张 Omni GPU，各自创建和清理独立 Ray head。真实 28 个 attention core 都是 `TEDotProductAttention`，TE 自动选择 `FusedAttention` sub-backend 1；两份 checkpoint 的 optimizer `param`、`exp_avg`、`exp_avg_sq` 各 302 项。来源的预训练 iteration 100000 与 RL rollout 0/1 分开记录，source pin 没有污染恢复进度。

初始化主包不再识别 Prism 模型类型、mossLite 目录或 native manifest，旧的 Prism checkpoint flags 已移除。源格式处理留在模型插件，主包仅在无显式 `--load` 且配置通用 initializer 时调用它，统一同步低精度 optimizer master 参数。示例先生成普通 Megatron 架构 CLI，再进入原 HF 校验；错误的显式恢复路径报错，不回退成重新初始化。

旧 `initializer-refactor-009` 在首次发布版本 1 后，由复制自恢复用例的 prompt 中静态期望版本 2 误拒绝，尚未执行更新。其失败源码与日志保留；010 使用新独立目录和分阶段的版本 1/2 prompt，并在预检中核验阶段对应关系，生产版本校验未放宽。

双副本恢复使用一张 learner GPU 与两张完整 Omni pipeline GPU，申请四 GPU 配额，其中一张空闲。生成共用一个 Python router，NCCL world size 为 3，两个 AR worker 分别使用独立接收 rank。标准 resume 未设置 native 预训练初始化、finetune 或跳过 optimizer 加载；Megatron 的 `no_load_optim` 默认值为 `None`，检查器按其 falsey 语义判断，同时保留原值。首次恢复测试发现 group 提前传入的版本 0 阻止了 actor 根据实际加载进度推导版本；修复 actor/group 的共同起点后，第二次运行实际验证了首发版本 2 和更新后版本 3。原失败收据保留。

MOPD 使用同一份基座制品启动两个独立教师服务，验证真实 domain 路由、冻结约束和原生 OPD 接口；它没有证明两个不同 domain 专家模型的训练收益。教师训练前后 checksum 比较是一次工程验收，正常训练不增加每步全模型 hash。

MOPD 原作业完成训练、保存和版本 2 发布后，其首版检查器误把原框架从 `iteration 0` 保存视为未完成，并假设了不存在的 model key 前缀。原失败日志保留。第二版检查器只读同一份结果，核验实际保存日志、一次 optimizer 更新、scheduler sample steps 2、302 个模型 tensor 与 302 组 optimizer 参数及两类 moments 的名称/形状，以及完整 DCP storage ranges；没有修改模型、算法、checkpoint 或原始日志，也没有补跑 GPU。

## 权重与数值监控

native 302 个 tensor 到 HF 390 个 tensor 的完整映射使用真实 DCP 与转换制品逐项 `torch.equal` 验证。GPU native 模型转换后的全部 HF tensor 也与实际制品一致。一次完整 NCCL 安装覆盖 390 个 tensor、3,959,756,800 字节。

在先前 FA4 wrapper 的一条短轨迹上，BF16 Megatron 与 Omni 的 RVQ selected-action logprob 平均绝对误差约 0.02727、最大约 0.20549，stop 最大误差约 3.58e-7。该结果没有达到独立诊断脚本选用的 0.05 最大误差门限。训练以更新前 Megatron 重放分数作为 PPO 的旧学生分数，Omni 分数持续用于诊断；这些历史误差数值不作为其他 attention 后端的保证或验收门限。

正常训练记录逐动作平均绝对误差和 `mean_sample_train_rollout_logprob_max_abs_diff`。后者先计算每条轨迹的真实动作最大误差，再沿原 microbatch/DP 报表平均，不能解释为全 DP global max。逐 codebook 与 stop 原始分数可通过 debug dump 检查。

原生 TE 续训 step 1 实际记录平均绝对误差 0.0235308129、轨迹最大误差的平均 0.1307324171。该验收报告仍明确记录 `policy_score_alignment_verified: false`，不将工程闭环通过当作严格概率对齐证明。

## CPU 检查与实际范围

模型结构/replay、Omni 生成与教师协议、三个奖励客户端、多动作 GRPO/OPD 归约、native checkpoint dispatch、完整权重映射、Ray/Omni 生命周期和发布故障边界的聚焦测试均已通过。标准 checkpoint 恢复的版本修复另通过 45 项 factory、disk delta、NCCL lifecycle 与 native dispatch 回归，并覆盖所有 rank 恢复边界一致性。

原生 TE 改动另通过 22 项模型回归及 29 项配置、checkpoint dispatch、数值诊断回归。构造边界测试覆盖 `auto`、`flash`、`fused` 都保留原生 TE spec，以及 MLP-only 层不会改变其他层的 attention。原生 attention 接线完成后，当前 TP/PP/CP 与 SP 的首版限制继续保留；Prism 输入和动作 loss 的 CP 切分仍需后续适配。

通用 initializer 重构通过 118 项加载、CLI 和架构配置组合用例，27 项模型用例及 124 项原 checkpoint/recovery/version 回归。测试覆盖 initializer 的来源与返回契约、master 参数同步、标准恢复优先、reference/HF 加载、无效显式路径、step 0、普通模型行为，以及无 CUDA driver 时配置生成可用而实际训练明确失败。配置失败记录与修复前 source snapshot 也保留，实际 CPU fresh/resume 命令预检通过后才提交 GPU 运行。

上述历史首轮 GPU GRPO 使用明确标注的工程 reward，未调用真实 ASR、speaker embedding 或 JudgeCLAP 三服务联合评分，不能据此判断 WER/SIM/指令遵循收益。该历史首轮范围不覆盖多 learner DP 等扩展；后续共卡、多卡和真实服务验收见本文顶部。
