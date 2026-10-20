#!/usr/bin/env bash
# Container entrypoint: native vLLM OpenAI audio transcription API.
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export VLLM_MAX_AUDIO_CLIP_FILESIZE_MB="${VLLM_MAX_AUDIO_CLIP_FILESIZE_MB:-64}"
MODEL_PATH="${MODEL_PATH:-/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/whisper-large-v3}"
test -f "$MODEL_PATH/config.json"
exec vllm serve "$MODEL_PATH" \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" \
  --served-model-name "${SERVED_MODEL_NAME:-whisper-large-v3}" \
  --api-server-count "${API_SERVER_COUNT:-2}" \
  --dtype float16 --tensor-parallel-size 1 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}" \
  --max-model-len 448 --max-num-seqs "${MAX_NUM_SEQS:-128}" \
  --no-enable-prefix-caching --no-enable-chunked-prefill --mm-processor-cache-gb 0 --enforce-eager "$@"
