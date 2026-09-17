# Canonical Local HF 与 torch_dist 迁移

本轮以用户提供的 `sglang-omni-tmp` 中 MOSS Local 模型、processor 调用和原生 codec 实现为基线，
在其上保留/补充 RL 所选动作分数、原始轨迹、版本检查和冻结教师评分。
Omni 实现提交为 `e8c998291c9ae2ec1e48d18f1102952d9f0412a3`。
参考快照标注版本为 `92a53c268a167e153cbe7f36b1d916b8a72c0b4f`。
该快照没有 `.git`；Omni 仓库的 `docs/developer_reference/moss_local_reference.json` 记录实际参考文件哈希。

## 已生成的权重

原始 HF（只读转换输入）：

```text
/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/zczhang/moss-eval-runs/automations/moss_checkpoint_watch/local_pretrain_v0.1.1/hf_ckpts/shared/0020000
```

Omni Local HF：

```text
/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/MOSS-TTS-Local-0020000-omni
```

- 437 个张量，8,322,894,848 字节；保留原来 10 个分片文件。
- 423 个张量直接保留，14 个 Local 张量改名，其中 QKV weight/bias 共 2 个张量重排。
- 去掉的 text head 与文本 embedding 经过逐元素相等检查；12 个独立 audio heads 全部保留。
- 原 text head 独占的分片为空，index 不引用它，实际 Omni 加载通过。
- `gpt2_config`、Local 布局标记和 processor shim 已写入。全部输出张量回读后与源值或精确置换结果按字节检查。
- `sglang_omni.yaml` 是学生服务配置；`sglang_omni_teacher.yaml` 是冻结评分服务配置，codec 路径已填写。

Megatron 初始化权重：

```text
/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/MOSS-TTS-Local-0020000-torch_dist
```

这是 `torch_dist` 的 iteration 0、model-only checkpoint，TP=CP=1，不含 optimizer/RNG 状态。
`slime_checkpoint.json` 标记为 initialization，首次训练使用新优化器并从 rollout 0 开始。
训练 checkpoint 应写入单独输出目录；普通 resume 恢复 optimizer、scheduler 与对应 sampler 状态。
已实际验证从该单卡初始化目录加载为 DP=2、CP=2 或 TP=2 的模型。

## 代码边界

- 原始 NeoX HF 的改名、每头 QKV regroup 和 RoPE halves→interleaved 置换，只在
  `tools/convert_moss_tts_2_0_for_sglang_omni.py` 中执行。
- Slime 的 Local HF loader/exporter 使用规范 Local 名称；训练入口不再隐式接受 HF 作为 `--load`。
- `tools/convert_hf_to_torch_dist.py` 创建初始化目录；`tools/export_tts_to_hf.py` 将训练后权重导出为同一 Local HF 布局。
- Omni 删除运行时 `neox_compat`、手写 v2 prompt renderer 和旧 Local codec wrapper；使用参考实现的 processor 路径和原生 codec。
- HF artifact 自带的原始模型/config Python 文件按用户转换方案保留，供 processor/config 加载。
  Local 布局权重用于 Omni 的专有模型，不宣称可由原始 NeoX `AutoModel` 直接加载。
- Omni 的 RL trace 保持 v2：原始提示行、continue/stop、二维码本、实际所选 logprob、终止原因和 admission weight version。
- 教师只评分学生原始动作；冻结服务拒绝权重更新。冗余的 updatable student-score pipeline 已移除。

## 实际验证

验证产物根目录：

```text
/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/outputs/slime-tts-local-layout-20260917
```

| 项目 | 结果 | 证据 |
| --- | --- | --- |
| 全量转换回读 | 所有输出张量按字节通过 | HF 目录 `conversion_manifest.json`；`hf_conversion.log` |
| 真实 Local FP32 对比 | 原 NeoX vs Omni step 最大差 `0.00010860`；vs 训练 Local 最大差 `0.00012338` | `offline_verify/result.json` |
| Processor 契约 | 输入 `[1,38,13]`；audio_start=151652、audio_pad=1024；language context=40960 | `offline_verify/result.json` |
| Slime CPU suite | 115 passed | `slime_cpu_tests.log` |
| Omni 与共享 codec CPU suite | 363 passed，33 skipped（GPU/平台条件） | `omni_final_tests/test.log` |
| 规范 HF 的实际生成 | 47 帧、612 个动作、48 kHz、3.76 秒，自然停止 | `inference/result.json` |
| WER GRPO + resume | 从新 torch_dist 初始化；2 轮后由新进程继续第 3 轮，完整回传并保存 checkpoint | `wer_e2e/result.json` |
| 同模型双路 MOPD | 2 个冻结服务，domain 路由、原始动作评分、2 次更新和回传通过 | `mopd_e2e/result.json` |
| CP/TP | CP=2 梯度范数相对单卡差 0.0720%；TP=2 差 1.1219%；三组均有实际参数更新 | `parallel/result.json` |
| 训练后导出 | 437 张量、8 个分片，保持规范 Local 名称 | `trained_export/omni/slime_export.json` |
| 导出模型与普通 Speech API | 实际重载；参考音频 data URI；WAV 2.4 秒；流式 PCM 4 个非空块，共 230400 字节，48 kHz / mono / 16-bit | `speech_api/result.json` |

FP32 对比使用 CPU、真实 Local 权重和 13 个 Local positions，预设 `atol=2e-4, rtol=2e-5`。
QKV 置换本身没有浮点计算；以上差异来自不同前向执行顺序的浮点舍入。
全模型 BF16 的训练/推理差异仍约为平均 0.05，保持既有检查与实际 behavior logprob，不宣称逐位对齐。

WER 三轮分别为 0.04167 / 0 / 0，第一轮梯度范数 3.8062；后两轮组内奖励相同，GRPO 梯度为零。
MOPD 梯度范数为 0.43976 / 0.47188；此次两教师为同基座副本，验证路由与训练连接，不代表领域知识融合。
使用的是小规模合成文本，各轮 prompt 不同，不据此推断 WER 提升。

GPU 验证使用原有 `train` Notebook 的训练镜像与当前 editable 源码，Torch 2.13.0+cu130、
Transformers 5.12.1、SGLang 0.5.18。用户给出的镜像用于标明参考实现来源，未声称另行验证了该镜像二进制。
所有 GPU 命令通过 `tts_gpu_guard.py` 暂停并恢复原有 `gpu-occupy`，各运行目录的 `guard.json` 保存退出状态。
