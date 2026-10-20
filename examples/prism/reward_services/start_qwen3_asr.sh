#!/usr/bin/env bash
# Container entrypoint: native vLLM OpenAI audio transcription API.
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export VLLM_MAX_AUDIO_CLIP_FILESIZE_MB="${VLLM_MAX_AUDIO_CLIP_FILESIZE_MB:-64}"
MODEL_PATH="${MODEL_PATH:-/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/Qwen3-ASR-1.7B}"
test -f "$MODEL_PATH/config.json"
exec vllm serve "$MODEL_PATH" \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" \
  --served-model-name "${SERVED_MODEL_NAME:-Qwen3-ASR-1.7B}" \
  --api-server-count "${API_SERVER_COUNT:-2}" \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}" \
  --max-model-len "${MAX_MODEL_LEN:-16384}" \
  --override-generation-config "{\"max_new_tokens\":${MAX_COMPLETION_TOKENS:-1024}}" \
  --no-enable-prefix-caching --mm-processor-cache-gb 0 \
  --max-num-seqs "${MAX_NUM_SEQS:-128}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}" "$@"
