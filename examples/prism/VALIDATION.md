# Prism 训练验收与性能记录

## Prism 4B v0.0.0(release iter_0095000)32 H200 MOPD / GRPO 计时与 ASR 副本对照(2026-10-18)

学生为 mossLite release 线 Prism 4B v0.0.0 第 95000 步(native DCP:
`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite/RUNs/release/prism_release_4B_v0.0.0/checkpoints/iter_0095000`,tracker=95000;manifest schema 2 / identity schema 3)。HF/Omni 制品由官方转换器新代码导出(native→safetensors,9 分片 8.11GB BF16,434 张量,H2560/L36/头32组8/FFN9728,RVQ24,loop 4 组 26-重3+3 组 4 重),但 runtime assets 固定用 converter@17b2374(与 1.7B v0.4.0 制品逐 SHA 一致:
modeling=96b7ae0b…, processor=96eb72c4…, prompt_protocol=14e8fd32…),部署在
`artifacts/prism4b-v000/hf-iter-0095000`。配置与 1.7B v0.4.0 32 卡生产计时同构:4 节点×8 H200、
DP32/TP=PP=CP=1、resident colocate(`--no-offload-train --no-offload-rollout`)、batch 2048、
max_frames 1024、训练序列 20480、生成并发 16/副本、Omni KV 65536 tokens、`--rollout-data-transport object-store`、
`--use-dynamic-batch-size --max-tokens-per-gpu 20480 --balance-data --balance-by-flops`、
`--rollout-execution-workers 8 --rollout-gc-threshold 70000 --rollout-tensor-buffer --resident-cleanup-interval 16`、
256 MiB 权重桶、种子 6211、同一 v2-fourteacher 60 万 prompt 混合;每组 24 次优化器更新,
不运行评估,不保存中间 checkpoint(save_interval=100000),计时口径为逐 update 核心循环
(generate+train+权重发布+cleanup),剔除前 2 轮预热,剔除末轮一次性 checkpoint(约 4.8-5.0s)。
Slime 源码 `f549efa2828bd663d6e5b12a941105de3004f156`(本分支基址,代码与 pilot r2 相同),
Megatron `3f49de20`,Omni `c2f8c453`,镜像 `slime-ttsrl:editable-v1`。MoPD 组冻结教师为四个
v0.0.0/iter_0060000 领域服务(instruction/asmr/dialect/sing,各 1 副本);GRPO 组奖励为冻结
Qwen3-ASR 的 wer(fingerprint 8d55fe2a…,`slime-prism-profile-asr-20261012-1935`)。

| 组 | 评分/奖励 | 完成更新 | 核心秒/步(均值) | 中位 | samples/s | node0 GPU 均载 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MoPD | 四领域教师 ×1 | 24/24 | 15.87 | 15.52 | 129.0 | 27.0% |
| GRPO 复现一 | ASR ×4 | 24/24 | 25.02 | 25.75 | 81.8 | 25.5% |
| GRPO 复现二 | ASR ×4 | 24/24 | 25.34 | 25.36 | 80.8 | 36.8% |
| GRPO | ASR ×1 | 24/24 | 44.77 | 44.80 | 45.8 | — |

阶段均值(秒,预热已剔):

| 阶段 | MoPD | GRPO ASR×4(复现一/二) | GRPO ASR×1 |
| --- | ---: | ---: | ---: |
| generate(生成+评分/奖励) | 12.49 | 21.73 / 22.07 | 41.48 |
| async_train(learner) | 2.42 | 2.25 / 2.26 | 2.26 |
| update_weights(发布) | 0.92 | 1.00 / 0.96 | 0.99 |

对照 1.7B v0.4.0 同规格记录:scale-profile(2026-10-12,生产档)GRPO 68.79 / MoPD 45.74 s/step;
优化后(2026-10-13)GRPO 16.42 / MoPD 13.64 s/step;IM-MOPD pilot r2(2026-10-15,当前代码)MoPD 核心 14.39 s/step。
结论:换 4B@95000 后原 32 卡配置可直接运行,但**两算法不再"差不多",差距拉开**——MoPD 13.64→15.87(+16%),
GRPO 16.42→25.0~25.3(+53%),GRPO/MoPD 由约 1.20× 扩到约 1.58×。learner 时间两算法几乎相等且都约为
1.7B 的 1.8×(2.25-2.42s vs 约 1.3s);差距全部来自生成/评分段:GRPO 每步生成 2048 条轨迹(256 prompts×8),
MoPD 仅 2048×1;4B 解码吞吐低于 1.7B 而 RVQ 帧结构不变,生成段 MoPD +7%(12.49 vs 11.7)、GRPO 接近翻倍。
四个 run 的 grad_norm(GRPO 均值约 0.12)与奖励抽样均正常,每步样本/发布版本和权重更新覆盖完好。

ASR 副本对照(4B GRPO):1 副本时奖励段成为硬瓶颈——Qwen3-ASR 单副本实测约 167 样本/s,
每步 2048 条 WER 需要 12.3 秒以上的串行下界,叠加排队后 generate 段 41.48s(×4 副本 21.7-22.1s),
整步 44.77s 对比 25.0s(-43%)。与 reward-services 报告一致:4B GRPO 生产配置 ASR 建议 ≥2 副本,
保持历史可比则用 4 副本。

本次 RCA 与过程记录(三节,均不影响落盘计时数据的有效性):
① 首版 4B HF 制品用最新转换器 runtime(renderer v7)导出,`prompt_protocol_sha256` 与冻结教师
记录的 v6 身份不符,首个教师在评分请求处触发 `attach_teacher_scores` 拒绝(teacher.py:307,
"incompatible replay identity");以 17b2374 版 runtime 重出制品修复,rev7 版留存
`artifacts/prism4b-v000/hf-iter-0095000-rev7runtime/`。4B pretrain 数据侧为 renderer v7,
本验收 rollout runtime 用 v6 与 1.7B 基线/教师同构;后续以质量为目标的 4B RL 需另行对齐 rev7
runtime 与教师链(仅计时不受影响)。
② campaign 式多 case 串行在 case 边界 Ray 重组失败一次(rank0 900s 超时"did not acquire every
allocated GPU";rank>0 早期退出但其平台侧日志缺失);改用逐 case 直跑 `slime_plugins/prism/cluster.py`
后稳定(pilot r2 五连跑与本次 MoPD/三组 GRPO 均靠同入口完成)。
③ 一次平台记录在案的 Job 命令指向了内容过时/非常规出现的 campaign 文件(实际读取了含旧输出目录的
manifest),另一次恰好撞上 ASR 网关 503 瞬时窗口启动失败;前者归档在实验根 `platform/ghosts-1011/`,
后者属服务端点瞬间缺副本所致。两次幽灵/413 过程均未产生计时数据。

计时有效性边界:本验收只测时间线,不做质量评测;两次 GRPO ASR×4 复现在共享服务上存在同时段并发
(合计需求峰值约 164 样本/s,低于该部署 4 副本实测约 283 样本/s 容量),与独占期数值一致。
平台 Job status 查询间歇返回 UNKNOWN,一律以 runs/ 下 result.json 与 driver-phases.jsonl 为准。

完整实验根:`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism4b-v000-mopd-grpo-20261018`
(configs/ 全部 manifest、analysis/timing-summary.json、scripts/ 提交与分析脚本、platform/ 过程记录;
mobd 组 runs/mopd4b-002,GRPO×4 runs/grpo4b-asr4-002 与 -003,GRPO×1 runs/grpo4b-asr1-002)。
平台 Job 条目保留:prism4b-mopd-grpo-1018(MoPD 成功的那次)、prism4b-grpo-asr4-1018、
prism4b-grpo-asrx4-1018-r2、prism4b-grpo-asr1-1018-r2 均 SUCCEEDED;失败与中途清理的条目未删除。

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
新恢复点位于 `runs/mopd-eval-final-r1-a4/checkpoint/`，对应联合提交
`rollout/training_commit_1003.json`。增加训练轮数时明确使用 checkpoint
optimizer scheduler 策略，见 README 的恢复说明。

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
[INSPIRE.md](../../INSPIRE.md)。

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

验收使用用户指定的 `CQ-科研驾驶舱` 工作空间、H200 GPU 和 `voxrl:editable-v1-slim` 镜像。学生资产是 Prism 1.7B v0.4.0 native iteration 100000；固定路径及训练/Omni 解释器合同见 [INSPIRE.md](../../INSPIRE.md)。实际 learner 为单卡，TP=PP=CP=DP=1。结果保留在共享 QB 盘，完整 checkpoint、原始响应和 debug dump 没有写入 Git。

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
