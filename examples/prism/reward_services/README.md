# Frozen reward services for Prism RL

This directory contains deployment entrypoints, client utilities and validation
tools for Qwen3-ASR, Whisper large-v3, ReDimNet2 and JudgeCLAP. Runtime code is
installed under [`slime_plugins/audio_rewards`](../../../slime_plugins/audio_rewards/).
ASR uses native vLLM OpenAI transcriptions. The other two services expose
versioned embeddings; Slime caches reference/text vectors and computes SIM or
calibrated JudgeCLAP scores locally. Model weights and audio stay outside Git.

For training-side configuration, component scorers, reward composition and
post-reward filters, start with the [reward workflow](../rewards/README.md).
This page owns service deployment, API contracts and validation tools.

For slow joint scoring, see the [latency diagnostics](../docs/REWARD_SERVICES.md#联合评分变慢时怎么定位).
The client records per-component wall time, admission wait, HTTP duration and
cache activity; existing Prism archive hooks report percentiles and the last
completed component in training logs, TensorBoard/W&B and JSON records.
The [configuration reference](CONCURRENCY_CONFIG.md) describes defaults,
total-budget allocation, deployment settings and migration from older settings.

Historical capacity measurements and GRPO/gateway experiments are kept in a
[separate experiment archive](https://github.com/VoiceAgentGroup/slime/blob/e57ed66c1ed6bb35015db134e5a0dc1a508caff3/examples/prism/reward_services/experiments/README.md). Each report pins its tested
source and inputs; the archive can remain unmerged without affecting deployment.

## Code and dependency boundaries

| Layer | Location | Responsibility |
|---|---|---|
| Client | `slime_plugins.audio_rewards.client` | Native ASR/embedding HTTP, version checks, bounded calls, reference/text caches |
| Component math | `slime_plugins.audio_rewards.scoring` | Default text/speaker/JudgeCLAP scores, optional per-component scorer functions |
| Cache reader | `slime_plugins.audio_rewards.cache` | Validate and read precomputed SQLite vectors |
| Prism adapter | `slime_plugins.prism.reward_adapter` | Interpret Prism samples, references and instruction attributes; own per-loop client state |
| Reward recipes | `examples.prism.rewards` | Select components and combine them into the objective |
| GPU service | `slime_plugins.audio_rewards.serving` | Audio processing, batching, model loading and HTTP endpoints |

Clients take explicit audio paths, reference paths and query strings; they do
not interpret Prism samples or import the serving stack. The serving package
does not import training or example modules. Its model definitions, category JSON
and licenses are included in the wheel. The root package initializers are light.
Use the existing example reward hooks (`examples.prism.rewards.*.reward`).

The model server still uses its own Python environment and pinned dependency
file below. Installing service source into a Python package does not install the
GPU stack into the training environment. In a serving image, install the built
wheel with `pip install --no-deps <wheel>` after preparing that environment, then
launch `python -m slime_plugins.audio_rewards.serving.server --help` for CLI
options. The checked-in launch scripts also support a pinned source checkout.

Embedding identities include source hashes. Check the new endpoint identity
after changing deployed code, including this package relocation. Recompute a
SQLite table if its identity differs; mismatched tables are rejected explicitly.

## Container entrypoints

Use an absolute checkout path in Inspire's **模型部署 → 运行命令**. Set `PORT` if
the configured container port differs from 8000. Each command uses one GPU; multiple CPU/model workers can share that GPU. Use a checkout pinned to a commit for reproducible deployments.

| Service | Entrypoint | Runtime | Native output |
|---|---|---|---|
| Qwen3-ASR-1.7B | `start_qwen3_asr.sh` | `vox-vllm-asr:v0.32.0-cu129-audio-20261018` | Transcript |
| Whisper large-v3 | `start_whisper.sh` | Same ASR image | Transcript |
| ReDimNet2 B6 | `start_redimnet2.sh` | CUDA 12 / Python 3.12 / Torch 2.8 runtime | 192-D unit vector |
| JudgeCLAP | `start_judge_clap.sh` | Above + Transformers 4.57.3 / PEFT 0.19.1 / optional FA2 | 3584-D **unnormalized** vector |

For example, after replacing `/shared/slime` with the actual checkout:

```bash
bash /shared/slime/examples/prism/reward_services/start_qwen3_asr.sh
bash /shared/slime/examples/prism/reward_services/start_whisper.sh
PYTHON=/shared/reward-env/bin/python bash /shared/slime/examples/prism/reward_services/start_redimnet2.sh
PYTHON=/shared/reward-env/bin/python bash /shared/slime/examples/prism/reward_services/start_judge_clap.sh
```

ASR startup requires no repository Python package or custom model fork. Both ASR
scripts disable prefix and multimodal processor caches so repeated benchmark
audio cannot inflate capacity estimates. No reference transcript or language hint
is supplied by the RL client. Both ASR scripts default to two API workers and 128 scheduler sequences. Whisper
uses encoder-decoder-compatible eager execution. Set `API_SERVER_COUNT=1` for
a smaller CPU footprint.

The Qwen startup and native ASR client explicitly cap decoding at 1024 output
tokens. Real singing/noise clips otherwise triggered repeated transcripts and
120 s timeouts. This does not crop the input waveform. Set server
`MAX_COMPLETION_TOKENS` and client `max_completion_tokens` together when legitimate
long transcripts need more room; native transcription JSON does not expose a
reliable finish reason for detecting token-limit truncation.

`MODEL_PATH` overrides the ASR/ReDimNet2 checkpoint path. JudgeCLAP accepts
`JUDGE_BASE`, `JUDGE_RUN`, `JUDGE_ADAPTER`, and `JUDGE_ATOM_STATS`; its default run
is the user-selected `mosstts2instr_margin_cal_novirt_fixw` checkpoint from
2026-10-11. All defaults point at the existing shared assets documented in the
repository's `INSPIRE.md`; they never trigger a model download.

For a new embedding runtime, execute `prepare_runtime.sh` in a networked CPU
environment with `RUNTIME_DIR=/shared/reward-env`. `BASE_PYTHON` must remain
accessible inside the selected Serving image. The existing portable Python 3.12
asset is the default. The script creates a new environment, installs the pinned
GPU stack and checksum-pinned FA2 wheel, and refuses to modify an existing env.
Set `WITH_FLASH_ATTN=0` to use the reported SDPA fallback. The tested base image
for shared virtualenv deployments is `ngc-pytorch:25.02-cuda12.8.0-py3`.

## Embedding API

- `GET /health`: ready only after weights load and worker warmup completes.
- `GET /info`: `protocol=voxrl.embedding.v1`, `embedding_version`, `dimension`,
  `dtype=float32`, `normalized`, audio policy, runtime and request limits.
- `POST /embed_audio`: full audio paths or uploads, one or multiple items.
- `POST /embed_text`: JudgeCLAP atomic `Category: value` queries.
- `GET /metrics`: queue, actual batch sizes, forward time and processed counts.

Read `/info` and pin the returned version before sending requests:

```json
{
  "request_id": "rollout-42",
  "expected_embedding_version": "sha256:<value-from-info>",
  "inputs": [
    {"id": "candidate-1", "audio": {"path": "/shared/candidate-1.wav"}},
    {"id": "candidate-2", "audio": {"path": "/shared/candidate-2.wav"}}
  ]
}
```

Uploads replace `audio.path` with `audio.data_b64` and optional `suffix`; exactly
one transport is allowed. WAV/FLAC and other libsndfile-supported formats are
decoded to mono 16 kHz, with a default duration limit of 120 s.
Text inputs use `{"id": "q1", "text": "Pitch: high"}`. Each text
must fit both the 4096-character limit and 1024 tokens after rendering the full
chat template, including the final embedding marker. Overlong texts are rejected
with 422 before GPU batching, so they cannot fail other requests in the batch.

Both encoding routes return the echoed request ID, embedding version, dimension,
dtype, normalization flag and an ordered `results` list. Successful entries have
`id`, `status=success`, and `embedding` (exactly `dimension` finite numbers).
Audio entries also report original/processed duration and whether padding or
truncation occurred. Invalid audio entries contain `status=invalid_audio` and a
reason instead of a vector. Malformed requests fail with 400/422, version changes
with 409, and saturated admission/queues with 503. Technical failures never become
numeric rewards. GPU batches can differ from the HTTP input batch.

ReDimNet2 preserves full utterances. Nonempty audio below 1600 samples at 16 kHz
returns the existing Slime `too_short` outcome. JudgeCLAP follows the pinned
training collator: pad nonempty short audio to 2 s, retain the first 30 s of long
audio. Empty/undecodable audio is reported explicitly.

JudgeCLAP's **27** supported categories come from upstream `ae235db` minus
Loudness. `Appearance`, `Listener context`, and `Physical scene` remain generation
attributes but are excluded from this checkpoint's reward, with exclusions
recorded in reward metadata. Other unknown categories fail. Atomic values split
on ` | ` are encoded separately. Query version is
`moss_tts_v2_revision6_instruction_v1`.

## Inference and numerical behavior

ReDimNet2 uses vendored model definitions, SDPA, and FP16 convolutions/attention;
frontend, pooling and final normalization retain FP32 arithmetic. The masked
batch path preserves every utterance's waveform/feature statistics, GroupNorm
statistics, convolution boundaries, attention keys and attentive pooling. Time
and batch dimensions use finite shape buckets, with masked dummy rows removed
from the response. Worker-local warmup prepares the common shape grid. In compile mode, only
128–4096 padded feature frames use the compiled core. Rarer short/long shapes
use the equivalent eager core so one subsecond clip cannot block the entire
queue with online compilation. Set `TORCHINDUCTOR_CACHE_DIR` to a persistent
shared directory for repeated deployments; compile mode requires additional
startup warmup. The upstream-style exact-length path remains available with
`--batching exact`.

JudgeCLAP uses audio and text encodings from the same pinned Thinker checkpoint.
LoRA remains unmerged by default to retain the checkpoint's adapter path.
Processor masks and CPU-built positions preserve the trained pre-`<emb>` pooling
rule. Audio and text GPU work is serialized; decoding runs in
a bounded CPU pool. `--preprocessing per_audio` overlaps independent CPU fbank
extraction with the GPU worker; it preserves the original processor tensors.
The default `--processes 4 --decode-workers 4` runs four model copies on one H200,
with separate queues and CPU pools. Allow roughly 120 GiB device memory for this
configuration; use `--processes 1` on smaller GPUs. Readiness waits for every process. `/metrics` is
per process (uniquely identified across replicas by `worker_id=hostname:pid`),
not an aggregate across processes.

GPU fbank (`--feature-device cuda`) and BF16 adapters (`--adapter-precision
bfloat16`) are optional settings that can change numerical results; the default
keeps CPU fbank and FP32 adapters. Validate numerical behavior and capacity before
enabling these settings, `--merge-lora`, or JudgeCLAP compilation (`--compile`) for a new runtime.

By default, the client calculates SIM as the dot product of unit speaker vectors. JudgeCLAP
uses FP32 `sigmoid(scale * (a @ t - ||t||²) + bias)` with the checkpoint's effective
scale and bias. Normalizing JudgeCLAP vectors or substituting cosine changes the
checkpoint's calibrated score; custom scorers must make that choice explicitly.
Base weights, adapter, model definitions, preprocessing and relevant runtime
configuration participate in embedding identity.

Both embedding launchers use the measured single-H200 settings by default:
ReDimNet2 runs one model process with compiled masked batching, and JudgeCLAP
runs four model processes without compilation. Both use four preprocessing
threads per process. These remain single-GPU replicas. Use `--no-compile` to
disable ReDimNet2 compilation and `--processes 1` to run one JudgeCLAP model copy.
The `PROFILE` environment variable and `--profile` option have been removed.
Old per-setting launch environment variables are also rejected; use explicit
server CLI overrides.

The CLI retains advanced overrides (`--max-batch`, `--max-cost`,
`--batch-window-ms`, `--precision`, `--processes`, `--decode-workers`). GPU
execution is fixed to one worker per model process; there is no `--gpu-workers`
option. ReDimNet2 uses padded feature frames for batch cost; JudgeCLAP uses an
audio-token estimate. Masked ReDimNet2 requires a power-of-two cost budget.

The server has one admission limit across audio and text, defaulting to eight
audio batches, with a minimum of 64 input items: 256 for ReDimNet2 and 128 for
JudgeCLAP with the default batch sizes. `--max-pending` is an advanced override.
Excess requests receive 503; accepted CPU/GPU work retains its reservation until
it finishes, including after request cancellation. The batcher has no duplicate
queue-capacity setting. `/metrics` reports the limit, active/peak admitted items
and rejected input count, with a replica/process-specific `worker_id`.

## Slime client configuration and precomputation

The shared [reward configuration](../configs/rewards.yaml) selects:

- WER: `api: openai_audio`, `model: Qwen3-ASR-1.7B`, `audio_transport: upload`.
  Whisper uses its own endpoint and `model: whisper-large-v3`.
- SIM and JudgeCLAP: `api: voxrl_embedding`, shared paths or uploads.
- `api` is required. WER uses `openai_audio`; SIM/JudgeCLAP use `voxrl_embedding`.

`services.<name>.concurrency` is the total HTTP request budget for one training
run. Slime divides it across the actual execution workers, distributing any
remainder without exceeding the total. It must be at least the number of workers.
Evaluation or a standalone single-process client uses the full budget. Defaults
are 128 for ASR and 64 for either embedding service, matching the YAML template.
For the measured 2/2/4 H200 deployment, use totals 256/128/256.

Reward clients own their HTTP connections. Admission is enforced once, before
audio upload preparation; the HTTP pool has no separate active-connection gate
and is independent of generation concurrency. Idle connections are bounded by
the local request share. The old `max_inflight` field is rejected, so an old
per-worker value cannot silently become a total budget.

Cancellation and timeouts drain any started upload preparation before releasing
admission or returning to the caller. Requests cancelled during preparation do
not proceed to HTTP. File cleanup can therefore extend beyond `timeout_s`. Closing the client
rejects new requests and cancels complete component operations, queued/in-flight
requests and shared reference/text fills. It finishes only after their work,
including reference hashing and SQLite reads, has drained and HTTP connections
have closed. Repeated or concurrent
close calls share the same cleanup, even if a caller cancels its close operation.

Standalone clients support `async with AudioRewardClient(...)`; adapters expose
`await adapter.aclose()`. Prism reward hooks close their clients through the
existing async-aware hook cleanup. No Slime HTTP-client bootstrap is needed.
Authentication is injected with
`auth_token_env: INF_API_KEY`; neither scripts nor results persist the token.
Inspire can inject the key with `inspire account api-key run <key-name> -- ...`.

Clients coalesce reference misses and cache vectors by content hash/model version;
JudgeCLAP coalesces text misses by model version and query, sharing both results
and failures among current waiters. Cache hits and unrelated queries do not wait
behind a fill; cancelling one waiter leaves the shared fill available to others.
After a failed fill, a later call can retry. Fixed references and query texts can be
precomputed into a read-only table shared by every rollout worker:

```bash
python -m examples.prism.reward_services.bench.precompute --kind audio \
  --endpoint "$SIM_ENDPOINT" --manifest references.jsonl --output references.sqlite
python -m examples.prism.reward_services.bench.precompute --kind text \
  --endpoint "$JUDGE_ENDPOINT" --manifest queries.jsonl --output instruction-texts.sqlite
```

Input rows are `{"audio": "/shared/reference.wav"}` or
`{"text": "Pitch: high"}`. Set the corresponding service's `embedding_cache`
to the resulting SQLite file. Missing entries fall back to remote encoding;
version mismatches fail instead of mixing feature spaces. Precomputation resumes
only against the same version. Cold/warm reference workloads must be measured
separately from model inference capacity.

## Custom component scorers

Component math runs in the training worker after inference. Configure optional
`scorers` alongside `services` in the reward YAML; this does not change service
startup or model deployments. The complete interface, defaults and copied-input
contract now live in the [component scoring guide](../rewards/README.md#组件评分).

Runnable examples are in [`rewards/scorers.py`](../rewards/scorers.py). Category
formulas combine component records; [`rewards/filters.py`](../rewards/filters.py)
uses Slime's native hooks to select prompt groups or mask samples from loss.
See the [complete training example](../rewards/README.md#完整训练示例) for all entrypoints.

## Reproducible benchmarks

From the repository root:

```bash
python -m examples.prism.reward_services.bench.prepare_librispeech --output /shared/librispeech
python -m examples.prism.reward_services.bench.benchmark --kind asr --max-completion-tokens 1024 \
  --endpoint "$ASR_ENDPOINT" --manifest /shared/librispeech/manifest.jsonl \
  --output /shared/results/asr --concurrency 1 8 32 64 128 --require-auth
python -m examples.prism.reward_services.bench.benchmark --kind embedding \
  --endpoint "$EMBED_ENDPOINT" --manifest /shared/librispeech/manifest.jsonl \
  --output /shared/results/embedding --concurrency 1 8 32 64 --require-auth
```

The official test archives are checksum-verified and the full-utterance manifest
is hashed. Reports include the selected length distribution, transport, runtime
identity, warmup, errors, latency and audio-item/audio-second throughput. In-flight
drain is excluded from the throughput window but included in latency statistics.
`--batch-size` controls items per embedding HTTP request, not the GPU batch size.

`bench/validate_gpu.py` compares numerical behavior and fixed-shape compute in an
idle GPU container. `bench/smoke.py` tests all four endpoints and the real Slime
client. `bench/resource_monitor.py` samples GPU and container counters when
platform metrics are unavailable. Save new reports with their source commit,
configuration and input manifest so comparisons use the same workload.

## CPU regression checks

From the repository root, use an environment with the test dependencies:

```bash
python -m pytest tests/test_audio_reward_clients.py tests/test_audio_reward_scorers.py tests/test_audio_reward_concurrency.py \
  tests/test_audio_reward_lifecycle.py tests/test_audio_reward_shared_texts.py \
  tests/test_audio_reward_package.py tests/test_audio_reward_timing.py \
  tests/test_prism_embedding_services.py tests/test_prism_reward_composition.py
```

These checks cover client contracts, custom scorers, admission, cache behavior, cancellation,
batching, package resources and component composition. Use the endpoint smoke
and GPU validation tools above to check a deployed model runtime.
