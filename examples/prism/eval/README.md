# Prism 七领域小评测

基于 Slime `0486696e0aa75d10b3a517df5f23e814f289d2f2`。通过已有的 eval dataset、
custom generate、custom RM 和 eval log hook 接入训练循环。使用当前同步到 rollout 引擎的学生权重，
不需要轮询磁盘 checkpoint；大评测的 checkpoint watcher 和 auto-eval 提交仍由外部脚本负责。

服务连接复用 `../configs/rewards.yaml`：Qwen3-ASR-1.7B、ReDimNet2、JudgeCLAP-v1012。
模型、codec、服务地址暂时保留占位；无需为小评测部署其他模型。
评分过程与训练奖励独立：eval 不读取训练的 `weights` / `scorers`，不调用教师，不改变训练的波形开关。

## 已冻结数据

路径：`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/data/prism-rl/eval/small-v1`。
七领域共 **840 条生成记录**；同一文本的 plain/clone 条件分别算一条。28 条 smoke 是这 840 条的子集。
参考音和 GT 音频已复制到内容哈希命名的 `assets/`，不是指向源评测工作区的软链接。

| 领域 | 条数 | 来源与分层 | WER/CER | SIM | JudgeCLAP |
| --- | ---: | --- | ---: | ---: | ---: |
| base | 128 | Seed zh/en 各 32；CV3 ja/ko/de/es/fr/it/ru 各 4；Voice Taxonomy 六类 × 中英 × 3 | 128 | 128 | — |
| instruction | 160 | ITTS v1 48；v2 60；MINT 24；MOSS v2 20；Breeze 8 | 160 | — | 160 |
| clone_w_instruction | 96 | IVC 十个任务各 8；mossv1-zh、breezeblue-en 各补 8，中英各 48 | 96 | 96 | 96 |
| dialect | 200 | 50 类 × plain / 同方言参考 / 普通话参考 / 远方言参考 | 200 | 150 | — |
| asmr | 48 | 中英各 12 文本 × plain/clone | 48 | 24 | — |
| sing | 48 | 中英各 12 文本 × plain/clone | 48 | 24 | — |
| event | 160 | 34 单事件 × 中英 × plain/clone；每语言再补 6 多事件文本 × 两条件 | 160 | 80 | — |
| 合计 | **840** | | **840** | **502** | **256** |

ITTS v2 每语言：global 8、local 8（4 对 description/local 协议配对）、special 6、combined 4、boundary 4。
MOSS v2 四子集各 5。MINT 使用剔除 `special/extra_vocal/` 后的有效集合，覆盖不同难度/类型。
方言 plain 与同方言参考使用相同原始样本；普通话参考、远方言参考遵循平台各自原有样本，
因此四条件不应全部解释成逐文本配对实验。ASMR、sing、event 的 plain/clone 按原始 source ID 严格配对。

选择种子 `20261020`。先物化完整官方 manifest，再分层抽样，保留原始行位置产生的 ID 和参考音分配。
按已有时长估计或文本长度估计分 short ≤10s、medium ≤30s、long ≤60s；默认排除估计超过 60s 的文本，
每个抽样层最多一个 long 哨兵。实际音频时长会在 rollout 后记录，估计值不当作实测值。
冻结版包含 short 513 / medium 303 / long 24 条；以 `manifest.json`、逐样本 metadata 和校验报告为准。

数据文件：

- `all.jsonl`、七个领域 JSONL、`smoke.jsonl`：可直接由 Slime Dataset 读取。
- `small.yaml` / `smoke.yaml`：已生成的 eval 配置，单 prompt 单次采样，max_new_tokens 4096。
- `assets/`：459 份去重音频，包含 20 份 MOSS v2 GT，约 160 MB。
- `manifest.json`：清单/资产 SHA256、平台实际 Python 文件哈希、完整 canonical manifest 哈希、
  官方 query 文件哈希、筛选代码哈希、配额和数据包指纹。
- 每条 `metadata.prism_eval`：原始 ID、语言、领域、任务、参考条件、原始 metadata、
  官方查询、保留查询、排除原因、GT 路径、固定生成 seed、TN 版本与原始台本。

送模侧使用平台结构化 profile；Breeze 的送模 profile 和评分侧原始查询分别保留。
规则 TN v1.16 只作用送模台本，随后冻结平台文本/控制标签清洗；WER 目标仍来自原始标签。
local 指令先定位原文 span，再逐段 TN/清洗，避免数字转换使 span 无法定位。
ASMR 不注入 Language；sing plain 只设置 sing 模态，不补造 singing 指令。

## 指标口径和范围

所有指标均保留有效样本数、应评数量、失败数、适用性和覆盖率。服务失败不会变成 0 分，
也不会悄悄缩小应评样本分母；漏回或被框架过滤的样本会补成 `missing` 记录。

- **WER/CER**：Qwen3-ASR 自动语言识别，使用原生 `/v1/audio/transcriptions`。
  zh/ja/ko 按字，其余按词；记录编辑数与参考单位数。微平均是总编辑数 / 总参考单位数，
  样本平均另外报告；错误率可超过 1，不截断。成功的空转写记全删除，服务错误单独记失败。
  指令类用 NFKC、zhconv 简体、标点和大小写归一化；方言用 OpenCC t2s；
  Taxonomy 保留幺/1→一等专用规则；其余使用平台 pair-v2 标点/分词规则。
  各 normalization profile 和语言单位分开汇总，不混成一个跨语言 WER。
- **SIM**：ReDimNet2 的参考音/生成音说话人向量余弦相似度，保留原始值（可负），
  不应用训练奖励的截断/加权。只评有 timbre 参考的记录。
- **JudgeCLAP**：使用冻结的官方原子查询；每个属性 `p > 0.5` 判通过，等于 0.5 不通过。
  主指标为属性微平均，通过属性数 / 有效属性数，同时报告样本平均和分类目通过率。
  MOSS v2 先用**同一个服务身份**对 GT 打分，仅将 GT `p > 0.5` 的属性计入分母。
  GT 全不通过记 `not_applicable`；GT 请求失败记评分失败，不回退为无门控分数。
  当前包保留 4,403 条属性查询、明确排除 50 条超出 renderer 或模型类目范围的查询。
  GT 结果、参考向量和文本向量会在进程内按服务版本缓存，跨训练评测步复用。

这里是 **三服务小评测代理指标**。CPU 提示词/归一化对拍不等于真实音频评分与平台完全等价。
尤其 ASR 服务的切窗方式、音频预处理，以及 Taxonomy 空转写处理可能与平台入口不同。
JudgeCLAP 服务默认取前 30 秒，长样本的指令覆盖不能据此解释成整段都满足；
local 查询目前也是整音频属性诊断，没有额外强制对齐/逐 span 打分。

不报告 dialect 分类准确率、耳语率、判唱率、事件分类/位置/计数准确率。
这些能力目前用输出音频试听；ASR 只反映字词可识别度，SIM 只反映说话人相似度。
方言和 ASMR 的额外 Judge 诊断默认没有启用。JudgeCLAP-v1012 的具体部署由服务脚本的
`JUDGE_ADAPTER` / `JUDGE_ATOM_STATS` 指定，不能仅凭通用 `judge-clap` 服务名证明检查点版本；
结果会保存 embedding_version 和完整服务 identity，换服务版本应开新对照实验。

## 接入训练

在 Slime 的 Python 环境安装 CPU 依赖：

```bash
python -m pip install -r examples/prism/eval/requirements.txt
```

填写原有 `configs/student.yaml` 的 codec 路径，以及 `run-mopd.sh` 的学生 HF/native checkpoint、
教师配置；也可继续用原脚本支持的 CLI 覆盖。填写共享 `configs/rewards.yaml` 的三个服务地址与认证。
鉴权仍通过 `auth_token_env`，配置和归档不保存 token。`.invalid` 地址会在生成开始前明确报错。
当前没有请求提交 GPU 作业、启动评分模型或验证这些占位路径。

先用 smoke 测量实际开销：

```bash
PRISM_EVAL_SIZE=smoke PRISM_EVAL_INTERVAL=20 \
PRISM_REWARD_CONFIG=/absolute/shared-rewards.yaml \
bash examples/prism/eval/run-mopd-eval.sh \
  --hf-checkpoint /absolute/student-hf \
  --pretrained-checkpoint /absolute/student-native/checkpoints/iter_XXXXXXX
```

`PRISM_EVAL_SIZE=small`（默认）切到 840 条；`PRISM_EVAL_DATA` 可指定未来版本目录。
间隔 20 是启动示例，不代表已测得的推荐频率；先量 smoke/完整 eval 耗时，再把训练开销控制在目标范围。
沿用原 MOPD 配方的 eval-before-train 行为。多节点时和 `run-mopd.sh` 一样在每个节点调用。
wrapper 只追加 eval 参数并调用原训练脚本；数据由 Slime 按 eval 配置加载。

已有启动脚本也可直接追加以下参数：

```bash
--eval-config /absolute/small-v1/small.yaml \
--eval-interval 20 \
--omni-reward-config /absolute/shared-rewards.yaml \
--omni-eval-return-audio \
--no-opd-eval-teacher-scoring \
--custom-eval-rollout-log-function-path examples.prism.eval.runtime.log_eval
```

最后一个 hook 是必需项：Slime 的 reward 字段仅返回流程占位值，真实指标在 `prism_eval_metrics`；
自定义 logger 会阻止默认的 reward 均值/accuracy 被误当成评测质量。`group_rm` 不支持小评测。
生成 seed 由冻结样本 ID 派生，不依赖训练 step 或并发完成顺序；重复采样加 repeat index。
固定 seed 不保证不同并发、硬件或服务实现下逐比特一致。

## 产物和重评分

产物保存在 `--omni-audio-dir/small-eval/`：

- `<eval-run-hash>/audio/*.wav`：每条生成音频；运行 ID 和重复采样 ID 隔离，续训同 step 不覆盖。
- `<eval-run-hash>/records/*.json`：生成后立即写一版，评分后原子替换；保留实际 weight_versions、
  seed、prompt、转写、分数、服务 identity、GT 门控明细和失败状态。
- `summaries/step-<rollout-id>-<uuid>/samples.jsonl`：完整可重评分记录。
- 同目录 `listening.csv`：生成音/参考音/GT 路径、台本、领域、条件、指标与状态。
- 同目录 `summary.json`：分领域、benchmark、语言、条件、task 的计数与标量。

TensorBoard 只写 `eval/prism/...` 标量，不调用 `add_audio`，不把 WAV 内容写入 event 文件。
例如 `eval/prism/asmr/cer/pair-v2/micro`、`eval/prism/clone_w_instruction/sim/mean`、
`eval/prism/instruction/judge_clap/pass_micro`、`eval/prism/instruction/judge_clap/coverage`。
所有比率按 0–1 存储，WER/CER 可 >1。`generation_returned` 是生成协议成功返回数，
`generation_success` 额外要求非零帧和音频路径；零帧、截断分别记录。

服务就绪后可先核对 embedding 服务身份：

```bash
python -m examples.prism.eval.cli preflight --services /absolute/shared-rewards.yaml
```

该检查读取 SIM/Judge `/info` 并检查 ASR 配置；ASR 的真实转写需随后 smoke 验证。
用已有 WAV 重评分（不加载学生模型）：

```bash
python -m examples.prism.eval.cli rescore \
  --services /absolute/shared-rewards.yaml \
  --records /absolute/summaries/step-XX-UUID/samples.jsonl \
  --output /absolute/rescored/new-run --concurrency 8
```

`--records` 也接受单次 eval 的 `records/` 目录，用于中断后的已生成样本；目录恢复只包含已有记录，
不代表未生成的完整 840 条。输出目录必须不存在，以免混入前一次评分结果。

## 重建与验证

重建工具只在构建期依赖 moss-eval，运行时不导入该仓库，也不依赖构建环境。
完整物化输出较大，和精简冻结包分开放：

```bash
# 在 moss-eval metrics 环境执行，输出到自己的目录。
python -m examples.prism.eval.materialize \
  --moss-eval /absolute/moss-eval \
  --datasets /absolute/dataset \
  --cv3-prepared /absolute/moss-eval-runs/cache/cv3 \
  --output /absolute/build-new/canonical

python -m examples.prism.eval.build \
  --canonical /absolute/build-new/canonical \
  --moss-eval /absolute/moss-eval \
  --datasets /absolute/dataset \
  --output /absolute/eval/small-v2
```

CV3 读取现有 prepared cache，避免触发平台的共享缓存写入。
ITTS / MINT 读取本地结构化 profile，允许其不是上游原始版本，但对真实输入 manifest 和查询留哈希。
已含 `manifest.json` 的版本目录禁止覆盖。冻结包 JSONL 的资产路径和 YAML 路径为绝对路径，
换根目录时应重建新版本，不直接移动后继续使用。

完整数据校验用于构建交付、重建或排查数据损坏时手动执行，不在训练启动时扫描数据包和音频：

```bash
python -m examples.prism.eval.cli validate \
  /inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/data/prism-rl/eval/small-v1
```

```bash
python -m pytest tests/test_prism_small_eval.py tests/test_eval_config.py \
  tests/test_omni_rollout_plugin.py tests/test_audio_reward_clients.py tests/test_audio_reward_lifecycle.py

# 可选：CPU 提示词与文本归一化对拍，protocol 指向选定 Prism 运行时文件。
python -m examples.prism.eval.audit \
  --root /absolute/eval/small-v1 \
  --moss-eval /absolute/moss-eval \
  --protocol /absolute/prism-hf/prompt_protocol.py \
  --report /absolute/audit.json
```

本次已验证冻结数据完整性、840 条 prompt 渲染、1,680 组归一化，以及相关 CPU 回归。
真实学生+codec rollout、Qwen3-ASR/ReDimNet2/JudgeCLAP 服务联调、同 WAV 平台分数对拍和训练开销
均需在占位配置填好后验收。不要把 CPU mock 测试当成 GPU smoke 通过。
