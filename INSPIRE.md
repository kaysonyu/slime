# Prism RL 资产

平台训练入口：[`run-grpo.sh`](examples/prism/run-grpo.sh) / [`run-mopd.sh`](examples/prism/run-mopd.sh)，
在各节点直接使用 Inspire 注入的分布式环境并管理 Ray 启停。输出按实验、任务和 running round 分层，
见 [`RAY.md`](examples/prism/docs/RAY.md)；原生参数与最小命令见
[`examples/prism/README.md`](examples/prism/README.md)。下文保留稳定资产与历史验收。

适用范围：账号 `kayson`，用户指定的 `CQ-科研驾驶舱` 工作空间，项目 `多模态自进化科学模型`。调度配额、可用 GPU 和平台状态每次从 Inspire Live 查询；本文件只记录共享资产及环境合同。

当前 ttsrl 的 32 卡配对测速产物位于
`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/ttsrl-32-ab-20261016-183137/attempt-002`。
适用同一项目与工作空间；`manifest.json` 固定基线/当前源码及教师接口兼容补丁，
`configs/` 与 `scripts/` 保存同权重、同数据、共用教师/ASR 的对照方法，
`analysis/report.json`、`analysis/steps.csv`、`analysis/verification.json` 保存完整统计和核验。
该轮记录还保存节点 hostname 回环解析问题及可路由 `SLIME_HOST_IP` 的选择，
详见 `examples/prism/docs/VALIDATION.md`。平台资源实时状态仍以 Live 查询为准。

当前 Slime 与 Omni ttsrl 的 renderer 7 功能验收资产位于
`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/ttsrl-grpo-mopd-20261016-180409`。
适用同一项目和工作空间；`manifest.json`、`source/` 固定源码和工作区补丁，
`analysis/verification.json` 汇总 GRPO/MoPD 的 DP2 更新、完整 checkpoint 恢复、
真实 ASR 和独立教师 Serving 核验；逐次配置、日志、音频和 checkpoint 保留在
`attempt-002`、`attempt-003`，范围见 `examples/prism/docs/VALIDATION.md`。

Policy streams 与 Omni deployment 重构的配对复现资产位于
`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/latest-refactors-ab-20261015-190912`。
适用上述项目和工作空间；`manifest.json` 与 `source/` 固定干净基线 `b73f0fd4`、
最新提交 `e99476bd`、Omni `c2f8c453`、Megatron `3f49de20` 及内容指纹。
`configs/`、`scripts/`、`runs/` 保存复现参数、程序和 checkpoint，
`analysis/report.json` 与 `analysis/verification.json` 为统计和核验入口；
性能结果、测量范围及保留记录见 `examples/prism/docs/VALIDATION.md`
中“Policy streams 与 Omni deployment 重构配对验收”。

运行时重构的固定验收产物位于
`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/runtime-refactor-20261013-182303`。
入口为 `analysis/report.html`；每项配置、源码 SHA、checkpoint、归档及修正后的
只读复核记录保留在该目录。具体性能数据和验收边界见
`examples/prism/docs/VALIDATION.md`；实际运行版本以每个 case 的 source pin 为准。

兼容层清理的固定验收产物位于
`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/compat-cleanup-20261014-024941`。
该轮使用成对升级的 Prism teacher version 2 协议；2 卡回归、真实评分对照及
组批敏感性记录见 `examples/prism/docs/VALIDATION.md`，没有替代此前的 32 卡验收。

公共动作流分支的固定复现资产位于
`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/policy-streams-25step-20261015-085228`，
适用同一项目和工作空间。`manifest.json` 与 `source/` 固定 Slime `47b00de9` 及当时
工作区补丁、Omni `c2f8c453` 和源码内容指纹；`configs/`、`scripts/` 与 `runs/`
保存输入配置、复现程序和 checkpoint。该记录的验证范围与结果见
`examples/prism/docs/VALIDATION.md` 中“公共动作流的 25 步 DP32 测速”，不能只用
基线 commit 代替包含补丁的完整源码身份。

Distributed optimizer 与完整重计算的配置消融资产位于
`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/memory-ablation-20261015-094424`。
`source/` 引用上述公共动作流的固定源码快照，`manifest.json` 记录来源；
`configs/`、`scripts/`、`runs/` 保存完整 recipe、测量与核验程序及 checkpoint。
配置生效方式、显存与速度结果及验收边界见 `examples/prism/docs/VALIDATION.md`
中“Distributed optimizer 与完整重计算消融”，完整统计入口为 `analysis/report.json`。

统一 request / renderer 7 的 Prism 制品位于 `/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/prism/prism17b-v04-iter-0100000-r7`。
适用同一项目和工作空间；`native/` 保存源 RUN tracker 选中的 iteration 100000
完整 checkpoint 副本（model 与 training_state），`hf/` 为 mossLite `1c97ef5`
官方 converter 导出的 renderer 7 制品。`manifest.json` 与 `provenance/` 记录源位置、
逐文件复制校验、转换命令和权重一致性；作为当前 Prism RL 的模型输入资产。
模型参数、tokenizer 与旧 renderer 6 制品一致，格式与运行时代码按新制品固定。

| 资产 | 固定位置 / 身份 |
| --- | --- |
| Native Prism 预训练 RUN | `/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/yangxiaogui/mossLite/RUNs/delivery/pretrain/prism_pretrain_1.7B_v0.4.0` |
| 本次导出 iteration | `/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/prism/prism17b-v04-iter-0100000-r7/native/iter_0100000`，模型组件为其 `model/` DCP |
| 对应 HF/Omni 制品 | `/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/prism/prism17b-v04-iter-0100000-r7/hf`；renderer revision 7 |
| Codec | `/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/MOSS-Audio-Tokenizer-v2` |
| 转换命令与检查记录 | `/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/prism/prism17b-v04-iter-0100000-r7/provenance/` 中的 `conversion-command.json`、`conversion-result.json` 与 `hf-weight-comparison.json` |
| GPU 容器镜像 | `slime-ttsrl:editable-v1`；地址 `docker.sii.shaipower.online/inspire-studio/slime-ttsrl:editable-v1`；digest `sha256:008ecc97fd306fc13c351d38a0e46ea2592a0ece0a8e8ba3102a0a06a712cd6c` |
| 训练、Ray、Omni Python | 统一使用镜像内 `/opt/venvs/core/bin/python`；环境入口 `/opt/slime-ttsrl/env.sh` |
| Editable slime | `/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable/slime`；分支 `refactor/multichannel-policy-streams`；本轮验证为 `43683a89` 加工作区补丁，完整身份见上述 ttsrl 验收 manifest |
| Editable SGLang-Omni | `/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable/sglang-omni`；仓库 `https://github.com/kaysonyu/sglang-omni`，分支 `ttsrl`；本轮验证 `0cc51ec4e991d0311dc018a6cf67ec46b730f753` |
| Editable Megatron-LM | `/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable/Megatron-LM`；`3f49de20de81c1ad5dfc10efa0f7d0c8098f1ce4` |

HF 制品由 mossLite 的官方 native 转换工具在 CPU 上生成。结构为 24 RVQ、32 物理层、H2048、MLP-only 28–31、BF16；执行中的重复层保留物理参数共享。模型初始化使用 native DCP 模型组件，RL 续训使用 slime 自己的完整 checkpoint。

首次初始化通过通用 `--custom-megatron-model-initializer-path` 指定 Prism 插件，来源使用 `--pretrained-checkpoint`，可选 step 使用 `--ckpt-step`。源 checkpoint 格式识别留在插件中；显式 `--load` 完整恢复优先。示例环境变量仍使用 `PRISM_PRETRAINED_CHECKPOINT` / `PRISM_PRETRAINED_STEP`，训练主包没有对应的 Prism 专属参数。

Prism learner 使用 Megatron 原生 TE layer spec 与 `TEDotProductAttention`，后端沿用 Megatron 默认 `auto` 选择。模型包不再直接依赖 `flash_attn.cute` wrapper；TE 根据实际环境选择底层 attention kernel。Omni 的推理后端由其独立配置和运行时选择。

统一环境使用 NumPy 2.3.5、Transformers 5.12.1、PyArrow 26.0.0、Ray 2.56.0、Torch 2.13.0+cu130 和 TE 2.17.0。Megatron Core 为 0.19.0，editable 安装元数据带源码后缀 `+3f49de2`。slime 在 Megatron backend 初始化时补齐 NumPy 2 移除的 `np.product` 别名，并允许 NumPy 1/2 初始化。PyAV 固定为 16.1.0，因为当前 Omni 的音频时长探测使用了 PyAV 19 已移除的 `metadata_errors` 参数。

三个项目均通过 `pip install -e` 绑定上表共享目录；镜像版本固定运行依赖，源码修改对新进程直接生效。镜像内 `/opt/slime-ttsrl/manifest.json` 记录初始构建版本，实际运行版本由各次验收 manifest 和 Git HEAD 记录。运行前使用 `source /opt/slime-ttsrl/env.sh`，或将 `TRAIN_PYTHON` 和 `OMNI_PYTHON` 都设为 `/opt/venvs/core/bin/python`。Ray head/node 也使用该解释器。

每次 GPU 验收记录完整源码 SHA 并使用独立结果目录，保留响应、逐动作分数、发布版本和清理结果；运行期间不修改源码。历史 snapshot 验收使用 `git archive` 导出源码，新镜像验收直接检查 editable 目录、安装元数据和真实 import 路径。

镜像构建、包版本清单、依赖元数据差异和 editable 修改生效检查保存在 `/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-builds/prism-editable-v1-20261012-1801/`。除固定 PyAV 外，已有依赖版本保持不变；基底镜像已有的 cuDNN、NIXL 等依赖元数据冲突记录在 `dependency-audit.json`，运行验收范围见 `examples/prism/docs/VALIDATION.md`。

本项目后续 GPU 验收的大文件输出放在 `/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/`。其中 `ray-attempt-005/checkpoint/` 是使用明确标注的工程奖励产生的完整 RL checkpoint，作为 optimizer 与断点恢复的验证 fixture；它不代表语音质量训练结果。CPU 工具、HF 制品与代码入口的原共享路径保留。

共卡运行设置 `COLOCATE=1`，单卡 Ray 使用一个 GPU，并设置
`ACTOR_GPUS_PER_NODE=ROLLOUT_GPUS=GPUS_PER_NODE=1`。训练与 Omni 共用 core Python。

历史 `colocate-005` / `colocate-006` 与 `checkpoint-residency-001` 使用
`voxrl:editable-v1-slim`（digest `sha256:f98aa62c7a9926dd7ab42925953459dc33f759f78654692cbea8293abf2dc484`），
以及共享盘 `artifacts/prism17b-v04/gpu-validation/training-env/bin/python` 中的
NumPy 1.26.4 / Transformers 5.0.0 训练环境。对应的固定 Omni 源码入口为
`/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/src/vox/artifacts/prism17b-v04/gpu-validation/colocate-smoke/attempt-006/omni-python`；
该 wrapper 在原 `/opt/venvs/core/bin/python` 前固定 Omni 源码路径，源码 SHA 记录在同目录
`sources.json`。这些入口保留用于复现历史记录；新镜像使用上文的统一环境。

QB 根目录中的 `colocate-005/fresh/checkpoint/` 与 `colocate-006/resume/checkpoint/`
是工程奖励下的共卡完整 RL checkpoint fixture；联合报告为 `colocate-006/verification.json`。
端口、Ray head 和平台 workload 均为临时资源，后续运行重新分配。

32/64 H200 性能验收与完整复现产物位于 `/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/scale-profile-20261012-193554`；`profile-report.html`、`analysis/results.csv` 和 `analysis/jobs-retained.csv` 记录测试范围、结果和保留的Job。实际生产配置与测量方法见 `examples/prism/docs/VALIDATION.md`。

后续 32 H200 rollout worker、tensor 传输及冻结评分优化的复现产物位于
`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/worker-profile-20261013-031520`。
入口为 `optimization-report.html`；配置、逐步统计、CPU profiles、GPU 采样和验证结果
分别位于 `configs/`、`runs/`、`analysis/`。Job 记录全部保留；checkpoint 核验报告
明确列出可读并通过的目录和缺失目录。每次实验使用的 Slime 与 Omni 源码 SHA 以
该目录 manifest 为准，64 卡扩展按用户要求延期。

2026-10-13 的 32 H200 全流程请求/CUDA 时间线保存在
`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/timeline-profile-20261013-070650`。
入口为 `timeline-report.html`；成功 MoPD trace 关联 2048 条生成与 2048 条教师评分，
包含 worker/router/frontend、AR、冻结教师、DP 数据发布、learner 和 tensor 权重同步。
`progress.json` 与 `platform/` 记录状态，原始 trace 和失败采集记录均保留。

GC 阈值与训练 tensor buffer 候选在远端 `perf/prism-training-buffer-20261013`，
Omni 诊断增强在远端 `perf/prism-timeline-20261013`。32 卡配对 Job
`slime-prism-32-gc-ab-1013` 已 SUCCEEDED，完成62次更新；本轮七组成功case累计80次更新。
前后原配置均值16.637秒/步，组合14.058秒/步，吞吐提高18.3%；32卡GPU busy约35%→41.2%。
所有rank单microbatch、样本/发布版本、梯度及checkpoint检查通过。
`scripts/watch-campaign-services.py` 在节点分配后恢复并核验冻结服务，结束后停止服务
并生成检查报告。运行资源已停止，Job/Serving条目、失败证据和所有逐次配置均保留。
改动合入 `feat/prism-omni-colocate-rl` 与 `rl/moss-tts-prism`；最终源码与 editable
同步收据见 `analysis/source-delivery.json`，实际GPU实验仍按各case原worktree/SHA复现。

指标、实验归档与 HTTP 日志策略合入的固定验收产物位于
`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism17b-v04/metrics-integration-20261013-113000`。
入口为 `metrics-integration-report.html`；每次源码/config pin、训练和归档核验、性能数据、
源码交付收据保存在 `configs/`、`runs/`、`analysis/`。性能结论及限制见
`examples/prism/docs/VALIDATION.md`，不从镜像构建版本推断运行源码版本。


六领域 MoPD 数据与学生 v0.1.0 的资产合同：

- 数据根目录：`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/data/tts2.0-mopd`。
  固定混合训练入口 `mixtures/v1/train.jsonl` 为 100 万条，`eval.jsonl` 为 6000 条；
  各领域分片、Lance snapshot、原始 reference 路径和构造规则见该目录 README 与 manifest。
- 模型导出、数据准备与校验脚本、六领域 smoke 记录、启动配置的固定根目录：
  `/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/vox-slime-validation/prism-mopd-1m-20261014-043616`。
- 学生：`prism_pretrain_1.7B_v0.1.0/checkpoints/iter_0200000`，通过 native 模型初始化入口加载。
  专用 instruction / asmr / dialect / sing 教师固定各自 v0.0.0 的 iter_0060000；
  fullclone / multilingual 共用 v0.4.0 的 iter_0100000。所有服务采用同一 revision 6 runtime/tokenizer。
- 32 卡、B2048、2000 步的可复现 manifest 为上述实验根目录下 `configs/train-32-2000.json`。
  配置与实际校验见 `analysis/preflight-summary.json`；运行状态从平台 live 查询。
  `scripts/watch-training.py` 管理本轮冻结教师生命周期，Job 与 Serving 条目始终保留。
