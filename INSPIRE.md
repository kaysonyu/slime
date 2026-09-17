# TTS RL validation assets

Scope: project **CQ项目**, workspace **CQ-科研驾驶舱**.

- The user-designated validation Notebook is named `train`. Resolve it using
  live status before execution; there are multiple stopped instances with this
  name. Do not create or stop platform instances as part of the validation.
- Runtime image: `miles-moss-tts-local-env:20260908-cu130-v1`.
- Shared source checkouts:
  - slime: `/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/src/myomni/slime`
  - Omni: `/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable/sglang-omni`
  - Megatron: `/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/editable/Megatron-LM`
- The user requires stopping the existing
  `/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/bin/gpu-occupy`
  supervisor before GPU validation, and restoring it afterward. Use
  `tools/tts_gpu_guard.py`, which preserves the process's original configuration
  and restores it even when validation fails.
- Original MOSS Local HF conversion input:
  `/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/zczhang/moss-eval-runs/automations/moss_checkpoint_watch/local_pretrain_v0.1.1/hf_ckpts/shared/0020000`
- Codec:
  `/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/MOSS-Audio-Tokenizer-v2`
- Local ASR asset for WER validation:
  `/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/Qwen3-ASR-1.7B`
- Higgs asset:
  `/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/Higgs-TTS-3-4B`

The HF model artifacts are inputs. Small diagnostics may use this repository's ignored `outputs/` directory.
Large RL checkpoints and final validation artifacts use
`/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/outputs/slime-tts-rl-20260916/`.
Check available space before checkpoint validation.
GPU Notebook commands read scripts through the shared filesystem; do not pipe
local stdin into restricted Notebook exec.

- Canonical Omni Local HF model:
  `/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/MOSS-TTS-Local-0020000-omni`
- Megatron model-only initialization (`torch_dist`, iteration 0):
  `/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/MOSS-TTS-Local-0020000-torch_dist`
- Local model reference snapshot (no `.git` directory in the supplied copy):
  `/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/src/sglang-omni-tmp`
  User-provided revision: `92a53c268a167e153cbe7f36b1d916b8a72c0b4f`.
- Reference serving image supplied by the user:
  `docker-qb.sii.edu.cn/inspire-studio/sglang-omni:moss-tts-2.0-v1`.
  The existing `train` Notebook retains its training image; validation uses the
  explicitly selected editable source and records that actual environment.
