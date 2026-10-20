# Vendored inference definitions

These files make serving independent of unrelated editable repositories. Runtime
dependencies and model weights remain external; no weight files are vendored.
The ReDimNet2 root MIT license and the Apache-2.0 license referenced by individual
pooling files are both included; original per-file copyright notices are retained.

| Directory | Source | Revision | Local changes |
|---|---|---|---|
| `redimnet2/` | https://github.com/PalabraAI/redimnet2 | `c5bbe0b76e37df698c403f8844e41304ceab6307` | Package imports made relative to this repository; network model loader omitted. MIT license retained in `redimnet2/LICENSE`; individual file notices retained. |
| `judge_clap/` | https://github.com/VoiceAgentGroup/judge_clap | `ae235dbfe1aa9241815336a08b799074d5a4f7e4` | Keep model/token setup and inference collator functions; omit training helpers. Accept CPU-precomputed positions while retaining the original path for comparison. Category list is canonical category map minus Loudness. |

ReDimNet2 SDPA substitution and the masked waveform frontend are implemented in
`../backends.py`, keeping the upstream definitions available for numerical
comparison. JudgeCLAP retains its exact templates, pre-`<emb>` pooling position,
2–30 second audio policy, and unnormalized margin scoring.
