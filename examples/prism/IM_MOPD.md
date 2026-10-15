# Prism IM-MOPD 实验

本分支实现 [IM-MOPD](https://arxiv.org/abs/2609.34745) 的任务向量初始化、
验证驱动的迭代合并，以及论文使用的优势裁剪和截断样本屏蔽。
它是独立实验功能，默认不启用，不能把下文的工程验收当作音质提升结论。

## 算法及 Prism 适配

四个专用教师共同从 v0.1.0 / iter_0200000 初始化，因此固定参考权重为这个版本，
任务向量为 `tau_i = teacher_i - reference_v01`。所有干预均累加这个固定向量；
不会重新计算 `teacher_i - current_student`，也不会归一化本轮选中教师的系数。

论文初始化为 `reference + sum(tau_i) / K`，等价于四个教师直接等权平均。
如果把 v0.4.0 当参考做同样的平均，v0.4.0 也会代数消去，不能称为保留学生能力。
保留 v0.4.0 的独立扩展为 `v04 + 0.025 * sum(teacher_i - reference_v01)`；
它的总初始增量比例为 0.1，明确区别于论文原始初始化。

每个指定更新边界计算
`recovery_i = (score_i(student) - score_i(reference)) / (score_i(teacher_i) - score_i(reference))`。
满足 `recovery_i <= gamma` 时向当前学生加 `delta * tau_i`。锚点必须有正的、可分辨的
教师增益，否则配置拒绝启动。论文 1.7B 设置为 `gamma=0.4, delta=0.3`，100 次更新，
在完成 25、50、75 次更新后合并；不能未经测试把这个频率重复套用到 2000 步训练。

## 权重与恢复合同

- `slime_plugins.prism.task_vectors` 从 HF safetensors 构造 FP32 增量。QKV 按 Megatron
  query group 布局还原，SwiGLU gate/up 拼接，覆盖所有 RVQ embedding/head、prediction norm、
  stop head 及 backbone。重复执行的物理层只合并一次；参数名和形状必须完整匹配。
- 每个物理参数有独立的平坦 safetensors 文件。向量索引记录原模型身份、每个 payload 的
  SHA-256 和范数；训练启动时 rank 0 校验全部 payload 并广播校验结果。
- `MergeOwner` 根据 DistributedOptimizer 的真实 shard range，只更新每个 rank 所有的
  FP32 master shard。保留 Adam moments、step 和学习率调度器，再同步 BF16 模型与 DP replicas。
  每个 rank 检查自己的 BF16 参数切片与 FP32 master cast 一致。
- 更新后刷新 actor weight backuper，确保下一次 tensor publication 发送合并后的权重。
- 事件身份、累积历史和计划摘要写入 `iter_N/task_vector_state.json`，并纳入模型/数据联合
  checkpoint 的校验清单。恢复时计划不匹配会失败；已经提交的事件不会重复加，变更同名事件的
  系数也会失败。干预点额外保存同步 checkpoint，便于准确恢复合并边界。

更新顺序为：

```mermaid
flowchart LR
    T[完成一次训练更新] --> P[暂停新 rollout 并发布当前权重]
    P --> E[固定验证集评估]
    E --> D[计算恢复率并选择教师]
    D --> M[更新 FP32 master 并同步 BF16]
    M --> C[提交模型、数据和合并状态 checkpoint]
    C --> W[发布合并后的 tensor 权重]
    W --> R[开始下一批 rollout]
```

当前验收范围是同步、常驻的 Prism actor，DP 任意、TP=PP=CP=1，标准 BF16 Megatron
DistributedOptimizer。没有验收 FSDP、precision-aware optimizer、overlap param gather、
异步 rollout、critic 或训练卸载/重建模式。部分不支持的组合会显式拒绝；不要仅凭参数可解析
推断其它组合也已支持。合并 I/O 失败会终止当前尝试，应从上一个已提交 checkpoint 恢复；
没有实现运行中部分合并的回滚。

## 启动入口

先对每个教师生成一次固定任务向量：

```bash
python -m slime_plugins.prism.task_vectors \
  --base /absolute/path/reference-v01-hf \
  --teacher /absolute/path/domain-teacher-hf \
  --output /absolute/path/vectors/domain
```

`--im-mopd-config` 接收 JSON 计划。以下为两教师结构示例，四领域实验应列出四个教师，
并把初始系数设为每个 0.25：

```json
{
  "version": 1,
  "vectors": {
    "domain_a": {"path": "/absolute/path/vectors/a", "sha256": "SHA256_OF_INDEX_JSON"},
    "domain_b": {"path": "/absolute/path/vectors/b", "sha256": "SHA256_OF_INDEX_JSON"}
  },
  "initial": {"domain_a": 0.5, "domain_b": 0.5},
  "steps": [25, 50, 75],
  "mode": "recovery",
  "delta": 0.3,
  "gamma": 0.4,
  "min_anchor_gap": 0.01,
  "anchors": {
    "domain_a": {"base": 0.2, "teacher": 0.8},
    "domain_b": {"base": 0.1, "teacher": 0.7}
  },
  "datasets": {"domain_a": "eval_a", "domain_b": "eval_b"}
}
```

锚点数字仅示意格式，实际必须使用同一验证集和生成/评分协议测量。
`datasets` 的值对应 `--eval-config` 中的数据集名。`steps` 是已完成的更新次数，
而 checkpoint / rollout 日志使用从零开始的编号，因此完成第 25 次更新对应 `iter_0000024`。
所有合并点必须早于最后一次更新，给后续蒸馏留下更新机会。
干预边界的评估用于选择教师，因此记录的是本轮合并前的模型；同一步额外保存的
checkpoint 包含合并后的模型。比较这两个产物时须结合 policy version 和 merge event，
不能把边界评估分数直接当成合并后 checkpoint 的分数。最终更新没有合并，最终评估与
最终 checkpoint 对应。

在既有 Prism MoPD 启动参数后添加：

```bash
--im-mopd-config /absolute/path/merge-plan.json \
--opd-advantage-clip 5 \
--opd-mask-truncated \
--no-offload-train --no-offload-rollout
```

裁剪只作用于 OPD log-ratio 产生的优势校正项；原始 reverse-KL 指标保留。
截断屏蔽同时覆盖 RVQ 与 stop 分支，整条轨迹不贡献梯度。它不自动补采替代轨迹，
因此 nominal batch 与有效训练样本数必须一起报告。生成侧停止决策、完整 stop probability
trace 和教师 selected-action 评分协议不受此功能改变。

静态初始化对照使用 `mode="fixed"`，各边界的 `schedule` 均为 `[]`；这样能保留相同的
评估和 checkpoint 节奏，只关闭迭代增量。固定非空 schedule 仅用于明确标注的消融或工程验证。
完整续训必须同时使用原始计划与 `--load` 的 Slime 联合 checkpoint。

## 固定实验资产

适用平台：账号 `kayson`，工作空间 `CQ-科研驾驶舱`，项目 `多模态自进化科学模型`。
实验根目录：

```text
/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism-immopd-20261015-044554
```

- `configs/plan.json`：参考、学生、四教师及数据来源。
- `vectors/*/index.json`：固定任务向量及源权重校验值。
- `analysis/initial-evaluation-summary.json`：每领域 128 条、共 512 条固定无长度控制样本的初始化评估。
- `analysis/dp2-{smoke,resume}-verification.json`：真实 GPU 合并与中途恢复验收。
- `configs/pilot-campaign-r2.json`：五组 DP32/B2048 短程对照及具体差异。
- `runs/`、`platform/`：原始音频、停止概率、训练日志、checkpoint、平台提交和失败/停止记录。

五组短程对照分别为原 v0.4.0 MoPD、仅论文裁剪/屏蔽、教师平均后静态 MoPD、教师平均后
迭代合并、保留 v0.4.0 的保守增量初始化。每组 32 次更新，干预点为 8/16/24，
固定验证集为每领域 32 条。这是先导实验，区别于论文的文本任务、均匀领域采样、
B128/100 步及其原始验证指标。生产数据仍为四领域既有混合分布，约 90% 无长度控制。

验证分数为 `1/(1+WER)`，触顶或空输出记零；使用默认停止阈值 0.1、上限 512 帧。
它只衡量内容一致性与自然终止，不能衡量耳语/ASMR、方言真实性、唱法、指令遵从或音色保真。
v0.1.0 在此协议下全部触顶，令归一化恢复率的基线为零；不能将这解释为该模型所有能力为零。
v0.4.0 已接近教师的 ASR 分数，原论文 `gamma=0.4` 对它很可能不触发；
不要为了制造合并事件而暗中改变阈值或把 ASR 指标当作风格能力指标。

初始化检查已经显示，教师等权平均会明显破坏当前 Prism 的内容和终止行为。
保守增量没有触顶但内容分数下降。是否存在训练后的改善，须查实际先导实验结果，
不能把“FP32 合并正确且能继续训练”当成“合并提升模型能力”。

### 先导实验暴露的验证指标缺陷

本轮控制器的 `1/(1+WER)` 在 WER=1 时仍给 0.5 分。因此非空但极短、几乎完全漏读的
输出可能超过 `gamma=0.4` 的恢复率阈值。真实迭代试验中，第二次合并后出现提前停止和
空输出；第三次边界却因四个领域的代理恢复率都高于阈值而不再合并。这是本轮 Prism
代理指标选择的缺陷，不能归因于论文原始验证指标，也不能把代理分数上涨当成方法有效。

`analysis/pilot-content-completeness.json` 使用保留的 ASR 输出补算
`max(0, 1-WER)`，同样对空输出/触顶记零，并同时统计空音频、空 ASR、WER>=1 和
输出/参考帧数比。补算发生在训练后，不更改已发生的控制器决策；不是对原试验的追溯性修正。
参考帧数只用于诊断异常缩短，不作为每个风格应遵循的准确时长。

后续试验必须使用内容完整性、空输出和自然终止的独立验收条件，再考虑适合各领域的
指令遵从/风格指标。单独换成另一个 WER 变换仍不足以证明风格迁移有效。

### 五组 32 卡先导结果

`prism-immopd-pilot32-r2-1015` 完成全部五组，每组 32 次更新、B2048、DP32/TP1。
实际运行代码固定为 `0af54c1d959b60e346c033fd46354ebd8ef703c4`；后续提交只增加
驱动回归测试、文档和结果。五组逐步 prompt hash 一致，每组 65,536 条唯一训练数据，
与验证集无重叠；所有 rank 每步一个 microbatch。四教师和 ASR 服务在实验后停止，
所有 Job、Serving 条目和模型/数据产物保留。

| 方法 | 补算内容分数 | 原控制器分数 | 触顶/128 | 空输出/128 | 核心循环秒/步 | 梯度范数峰值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A：v0.4.0 原 MoPD | 0.5856 | 0.7149 | 0 | 0 | 14.39 | 0.48 |
| B：v0.4.0 + 裁剪/截断屏蔽 | 0.5933 | 0.7055 | 4 | 0 | 14.27 | 0.48 |
| C：教师平均 + 静态 MoPD | 0.0341 | 0.2170 | 55 | 0 | 19.53 | 175.74 |
| D：教师平均 + 迭代合并 | 0.0057 | 0.4406 | 0 | 14 | 14.34 | 4804.82 |
| E：v0.4.0 + 10% 初始增量 + 静态 MoPD | 0.6248 | 0.7318 | 5 | 0 | 13.85 | 5.39 |

补算内容分数为 `max(0,1-WER)`，空输出/触顶记零。时间排除前两步、初始化、验证、
checkpoint 和额外合并边界操作，包含生成、教师评分、训练与正常权重发布。
梯度范数为裁剪前值，实际梯度裁剪阈值为 1。D 的后期变快来自严重缩短输出，不能算作
效率提升；未被 loss 屏蔽的样本也不等于内容合格样本。

D 在完成 8 次更新后向 instruction/sing 各加 0.3，完成 16 次更新后向
instruction/dialect/sing 各加 0.3，完成 24 次更新后未再加入。两次非空合并实测
0.207 / 0.240 秒，第三次空事件0.117秒。累积向量系数总和为2.5，是对共同参考的外推。
第17次更新起梯度范数大幅上升，最终111/128条不足参考帧数一半，内容几乎完全失败。
同期 learner/rollout 的平均 selected-action logprob 差异从前16步的0.024增至后16步的
1.668，而A约为0.021–0.022。大幅合并后还存在明显的训推数值一致性问题；本轮D的
结果不能作为只改变算法、数值条件完全可比的质量实验。参数同步与恢复的工程检查
不等价于两个前向实现输出一致，需结合补充权重校验及后续数值定位解释。

补充的 2 卡诊断从 D 的第 16 次更新 checkpoint 恢复，使用正常 tensor 发布流程，
对 Omni 和 learner 导出的 302 个模型参数逐项做字节级 SHA-256 比较；两份副本均无
缺失参数或不同参数。Omni 额外报告的 `model.codes`、`model.mask` 和 rotary cache
是运行时 buffer，不属于 checkpoint 参数。这个结果排除了本轮已检查路径中的漏发/错发
权重，但不能解释前向实现之间的大幅 logprob 差异；第一次以 batch64 恢复的诊断因
optimizer scheduler 的总迭代数不匹配而失败，记录仍保留。

E 在前8–16次更新有改善迹象，随后回落，最终仍有5条触顶。相对B的最终补算内容分数
差值为+0.0315，但按配对prompt bootstrap得到的95%区间为[-0.0238,0.0887]，
且这不包含重新训练的随机性；尚不足以宣称稳定收益。A/B循环差约1%，没有看到新增
裁剪/屏蔽的明显开销。核心阶段GPU busy各组约35%，不是显著改善GPU利用率的方案。

验收包括：受影响CPU套件84通过/1跳过，新增驱动顺序/恢复后的相关24项通过；
2卡真实合并、tensor发布、联合checkpoint及中途恢复通过；32卡rank间事件与
master/BF16一致性、最终sidecar摘要核验通过。结果报告为实验根目录下
`analysis/experiment-report.md`，图表为 `analysis/pilot-results.png` 和
`analysis/pilot-gradients.png`。

本轮足以否定直接将此配置扩为2000步生产训练。它是单seed短程适配，存在任务、采样、
batch、干预间隔和验证指标差异，不能作为论文方法在其它条件下一般无效的证据。
