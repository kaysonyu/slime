# TTS RL 验证记录

验证于 **2026-09-16 至 2026-09-17（Asia/Shanghai）**，使用用户指定的
CQ-科研驾驶舱 / CQ项目 / `train` Notebook（`qb-prod-gpu2035`，8 × H200）。
原始 MOSS HF 权重为 `local_pretrain_v0.1.1/hf_ckpts/shared/0020000`，完整路径见
[INSPIRE.md](../INSPIRE.md)。运行版本与上游锚点见
[tts-runtime-versions.json](../docker/tts-runtime-versions.json)。

最终验证文件根目录：

```text
/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/outputs/slime-tts-rl-20260916/
```

该目录中的 `validation_summary.json` 汇总成功运行、真实指标、checkpoint 导出和占卡恢复记录；
`validated_source_manifest.json` 记录最终 slime 与 Omni 源码哈希。
训练验证各自保留了 `source/`、`source_manifest.json`、命令参数、日志、音频和原生轨迹。
历史失败运行保留诊断日志，不计作通过。

## 完成的检查

| 检查 | 结果 | 证据目录 |
| --- | --- | --- |
| Slime CPU 测试 | 113 passed；包含真实 CPU Gloo 归约及新的轨迹/奖励/教师协议测试 | `cpu_tests/` |
| Omni 相关 CPU 测试 | 34 passed；包括采样参数传递、NeoX 兼容、结构化 prompt、Higgs 评分输入、冻结教师拒绝更新 | `omni_cpu_final/` |
| MOSS 原始 HF 加载与生成 | 原始权重成功生成 48 kHz 音频及可回放轨迹 | 本仓库 `outputs/tts_inference_probe_2/` |
| MOSS WER GRPO | 2 次正常迭代 + 新进程加载 checkpoint 后第 3 次迭代；实际 ASR、反向、完整权重回传与保存成功 | `moss_final/` |
| MOSS 双路 MOPD | 2 次迭代；两独立冻结服务按 domain 评分原始学生动作，梯度及回传成功 | `moss_mopd/` |
| Higgs WER，128 行上限 | 2 次迭代；此批短文本 WER 全零，GRPO 优势/梯度也为零 | `higgs_wer_2/` |
| Higgs WER，64 行截断验证 | 2 次迭代；真实 ASR 产生组内差异，梯度非零，回传成功 | `higgs_wer_truncated/` |
| Higgs 双路 MOPD | 2 次迭代；独立教师服务、原始延迟码本评分、反向与回传成功 | `higgs_mopd/` |
| Megatron → HF 导出 | 从 DP=2 保存的 checkpoint 在单卡恢复并导出；438 张量、9 个分片、9,100,807,168 字节 | `moss_export/` |
| 导出 HF 的实际加载 | Omni 加载后生成 41 帧、534 个动作、3.28 秒、48 kHz 音频，自然停止 | `moss_export_load/` |

MOSS 导出的 `local_text_lm_head.weight` 有 390 / 5120 个元素相比原始 HF 改变，
最大绝对变化为 `1.52587890625e-05`。它是训练后权重的导出；导出结果已实际加载运行。

MOPD 的两教师服务使用同一基座的两个副本，用于验证路由及训练连接；这里没有真实领域教师数据，
也没有把这次检查描述为领域能力融合成功。BF16 下同权重服务的生成与 teacher-forcing 分数也有
数值差异，因此这里的非零 MOPD 梯度不能解释为学习到了新的领域知识。

## 真实梯度与 WER

MOSS 最终三轮的平均 WER 为 `0.04167 / 0 / 0.0375`，梯度范数为
`3.58045 / 0 / 3.32895`。第二轮所有组奖励相同，零优势和零梯度符合 GRPO 定义。

Higgs 的截断验证把生成上限显式设为 64 行，截断率 100%。两轮平均 WER 为
`0.43056 / 0.50536`，梯度范数为 `3.98298 / 4.34388`。
这验证了截断轨迹的真实 WER 信号和更新路径；它不是推荐的正式训练长度设置。

这些迭代使用不同 prompt 的小规模合成数据，**不能据此推断 WER 提升或训练收敛**。
正式实验应使用真实领域训练数据及固定独立验证集。

## 并行验证与数值限制

固定 MOSS 实际生成轨迹，在相同初始化下比较单卡、CP=2、TP=2 的真实 Megatron 前向、反向与更新：

| 配置 | 梯度范数 | 相对单卡差异 | decision head 最大变化 |
| --- | ---: | ---: | ---: |
| TP=1, CP=1 | 11.284761 | — | 0.0001220703125 |
| TP=1, CP=2 | 11.292996 | 0.0730% | 0.0001220703125 |
| TP=2, CP=1 | 11.158447 | 1.1193% | 0.0001220703125 |

预先设定的梯度范数容差是 2%。CP=2 的这条样本包含一个没有有效动作、只有 prompt/padding 的 rank，
验证了公共分支维持 backward/DDP 同步的情况。证据在本仓库 `outputs/tts_parallel_probe/`。
这项检查比较所选分数、梯度范数和实际参数更新；没有宣称全梯度逐元素或逐位相等。
Higgs 实际训练验证为 TP=CP=1、DP=2，尚未单独验证其 TP/CP 组合或 checkpoint 导出。

BF16 下，MOSS 的训练/行为 logprob 平均绝对差约 `0.047–0.051`，Higgs 约 `0.093–0.099`。
Higgs 独立 HF 回放也观察到约 `0.094–0.111` 的平均差异。
MOSS 默认检查最大值 1.0、均值 0.1；Higgs 冒烟验证显式使用 1.5 / 0.15。
MOSS 初始裁剪比例约 4%，Higgs 约 12–14%。这些差异会影响 RL surrogate，应作为后续精度工作重点。
原始 behavior logprob 从未被训练重算分数替换；指标保留真实差异。详见 [概率说明](tts_rl.md)。

## 运行收尾

每个 GPU 验证都通过 `tools/tts_gpu_guard.py` 暂停原有 `gpu-occupy` 并在退出后恢复。
成功运行目录均有 `guard.json`，包含原 PID、恢复 PID 和退出码。
最后一次验证退出码为 0，恢复后的占卡 supervisor PID 为 `671897`。
最后核对 8 张 GPU 均回到约 131093 MiB 占用、99% 利用率的原占卡状态。
Notebook 保持 RUNNING；没有创建或停止平台实例。

CPU 契约测试已注册到 `.github/workflows/pr-test.yml.j2` 并生成对应工作流。
H200/HF 集成验证依赖私有共享资产，通过 Inspire 单独运行；未声称已在 GitHub 执行远程 CI。
