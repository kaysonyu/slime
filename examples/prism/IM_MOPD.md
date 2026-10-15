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
