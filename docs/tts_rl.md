# TTS RL on slime v0.3.2

This branch keeps slime's rollout/train loop, Ray placement, Megatron training
schedule, DP planner, checkpoint and observability locations. Native speech
trajectories replace the default text-token batch contract. The inference
service is SGLang-Omni; training uses the separately maintained Megatron-LM.

## Source anchors

| Component | Upstream base | Implementation inspected |
| --- | --- | --- |
| slime | `3778dbf6d1a533ab478ecf5ddaa11449a47752b2` (`v0.3.2`) | branch `tts-omni-rl` |
| SGLang-Omni | `1ddf9c17bb8b211b439e9e91c22f1c722fb3461f` | `e8c998291c9ae2ec1e48d18f1102952d9f0412a3` (committed Local-layout RL integration) |
| Megatron-LM | `3f49de20de81c1ad5dfc10efa0f7d0c8098f1ce4` | same commit |
| mossLite model reference | `43892a9b2797591fd6be35ae95066524ac984b1f` | MOSS Local model, topology, and checkpoint conversion |

The external repositories are maintained as source checkouts. This project does
not generate or apply dependency patch files. A commit alone does not identify
an external checkout with uncommitted changes; validation records must also
identify those changes.

## Runtime checkpoint contract

The original GPTNeoX HF artifact is accepted only by the offline converter.
Omni and the model's HF import/export helpers use `gpt2_config`, GPT-2-style
Local parameter names, q|k|v blocks, and interleaved RoPE. Serving and live
weight publication contain no NeoX conversion branch. Training requires
`--load` pointing to `torch_dist`; `--hf-checkpoint` supplies the converted
artifact's configuration/tokenizer and canonical publication manifest.

`tools/convert_hf_to_torch_dist.py` writes a model-only initialization checkpoint
at iteration 0 with `slime_checkpoint.json`. This marker selects fresh optimizer
initialization and rollout 0. Normal training checkpoints restore optimizer,
scheduler and sampler progress. `tools/export_tts_to_hf.py` returns trained
weights in the same canonical Omni layout.

The Local model/processor/codec base now follows the user-supplied snapshot
reported as `92a53c268a167e153cbe7f36b1d916b8a72c0b4f`. See
[Local layout migration](local_layout_migration.md) for the current artifacts
and validation. Earlier results below describe the prior raw-HF runtime path.

## Data and probability contract

MOSS trajectories keep prompt rows `[P,C+1]`, codes `[T,C]`, and separate
continue/stop decisions `[D]`. A natural stop has `D=T+1`; length truncation has
`D=T`. Selected behavior scores preserve those shapes and identify their
sampling distribution and admission weight version.

Model adapters produce global input rows and the source position of every
prediction. The common Megatron batch path owns CP partitioning. All channels
of a prediction follow its source hidden state, including across a boundary
between an input position and the next row. Complete-sample loss denominators
are computed before CP slicing.

GRPO uses action-level clipped ratios and prompt-group advantages. MOPD uses
domain-routed frozen teachers on the exact student trajectory, with a detached
teacher-minus-student logprob advantage. The shared reward path now uses the
Delay adapter's multilingual WER definition and bounded `1 - min(WER, 1)` reward,
retaining raw WER in metrics. Chinese/Cantonese text is normalized to simplified
Chinese, and character languages use grapheme tokens. Reward services have
separate concurrency limits, bounded retries and complete-group recovery.
An actual empty generation receives zero reward. See
[shared speech GRPO](../examples/tts_grpo/README.md) for composite WER/SIM/Judge,
data preflight, independent evaluation and Inspire deployment.

## Validation stages

The CPU tests cover causal prediction alignment, natural stop/truncation, CP
ownership and gradient reduction, Local QKV/RoPE coordinate conversion against
HF GPT-NeoX, WER semantics, clipping, and MOPD gradient direction.

`tools/tts_inference_probe.py` checks real HF loading, generation, audio, and
the structured trajectory. `tools/tts_megatron_probe.py` replays that saved
trajectory using the actual Megatron schedule. Its positive-advantage update
is a diagnostic optimizer check, not a WER training result.

GPU validation in the user-designated train instance uses
`tools/tts_gpu_guard.py`. It stops only the exact existing `gpu-occupy` supervisor
and restores its argv, working directory, and environment in `finally`.
Credential-bearing process environments are not written to reports.

CPU checks do not establish GPU/TP/CP correctness. A parallel configuration is
validated only after real selected-score, gradient, and update comparisons.
BF16 generation/replay differences are reported separately from exact layout
and parameter-mapping invariants.

The real 0020000 checkpoint's first saved trajectory showed BF16 selected-score
differences of max 0.520 / mean 0.0448 between Megatron and Omni. An independent
HF Qwen3 + official GPT-NeoX reference showed max 0.524 / mean 0.0517 against
Omni as well; the Local FP32 coordinate-conversion reference test passes at
1e-5. This is not bitwise train/inference equivalence. Initial BF16 checks use
explicit maximum (1.0) and mean (0.1) bounds and retain measured mismatch and
policy-ratio metrics. Behavior scores are never overwritten with recomputed
training scores. The diagnostic bounds are configurable and are not a claim
that arbitrary precision/sampling changes preserve the policy.

## Second model and public Omni interfaces

Higgs is an independently sampled delayed-codebook policy. Its audio-validity
mask omits EOC and delayed-tail choices that still affect the subsequent hidden
state. The v2 trace therefore carries a separate sampled-action mask. Replay
keeps original prompt IDs, delayed reference codes, delayed generated rows and
all sampled choices. It never converts the undelayed audio back into a guessed
training sequence.

The public `/generate` request accepts structured prompt dictionaries as well
as text. This lets a model adapter pass reference conditioning without making
common rollout code interpret a model's prompt protocol. `/score_actions` uses
one batch/result envelope, sample IDs, exact-input fingerprints, weight
identities and selected-logprob semantics. The action geometry and causal
teacher-forcing implementation remain model-specific. Adding an unrelated
continuous or diffusion policy would require an appropriate action/probability
contract; the discrete codec interface does not make that automatic.

Both score-only pipelines advertise `supports_weight_update=false`. The common
Omni worker enforces this capability for refit/group initialization and rejects
the destructive checker reset operation. Teacher fingerprints can therefore
identify the fixed parameter set for the service lifetime. A new teacher
requires a new service and a deliberate training configuration change.

Higgs BF16 replay was checked against an independent Transformers Qwen3 model
loaded from the exact original HF tensors: selected-score absolute differences
were max 1.023 / mean 0.0944 for HF vs Megatron and max 0.986 / mean 0.1109 for
HF vs Omni. The first full rollout failed the MOSS default maximum bound at
1.108. Subsequent Higgs smoke tests explicitly use maximum 1.5 and mean 0.15;
they do not silently relax the MOSS defaults. This is evidence of comparable
BF16 execution differences across the three implementations, not proof of
bitwise or distribution-level equivalence.

## Reuse and removal relative to v0.3.2

The retained orchestration is synchronous: publish a complete policy version,
generate one batch, score its actions, train, checkpoint if needed, publish the
next version, then evaluate. Each sample/group is identified explicitly, and a
batch cannot combine behavior versions. The existing DP planner and Megatron
training schedule remain in place. The original CP/DP numerical tests remain
as regression coverage beside the new structured-action tests.

The removed code serves paths outside this TTS scope: original SGLang engine
management/router, agents, critic/PPO value heads, text-only rollout/SFT,
asynchronous/partial rollout, unrelated model converters, low precision,
delta/disk weight synchronization, legacy environment patch installation and
their examples/CI/documentation. The v0.3.2 tag remains the historical source.
The main package and model-plugin directories retain their original locations.

`tools/export_tts_to_hf.py` exports native checkpoints with the same canonical
model mapping used by live refit. It verifies all trainable-policy keys before
copying frozen HF components, writes an exact tensor index, and publishes the
new directory only when complete. Optimizer state stays in the native training
checkpoint. Exported HF directories are directly usable as teacher or student
Omni model paths.
