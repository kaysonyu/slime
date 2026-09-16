#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?Set MODEL_DIR to a complete MOSS Local HF artifact}"
: "${MEGATRON_DIR:?Set MEGATRON_DIR to the pinned Megatron-LM checkout}"
: "${OMNI_ENDPOINT:?Set OMNI_ENDPOINT to the student Omni service}"
: "${TRAIN_CHECKPOINT:?Set TRAIN_CHECKPOINT to the initial or resumed torch_dist directory}"
: "${PROMPT_DATA:?Set PROMPT_DATA to the TTS JSONL dataset}"
OBJECTIVE="${OBJECTIVE:-grpo}"
if [[ "${OBJECTIVE}" == "grpo" ]]; then
  : "${ASR_ENDPOINT:?Set ASR_ENDPOINT to the audio transcription endpoint}"
fi
ASR_ARGS=()
if [[ -n "${ASR_ENDPOINT:-}" ]]; then
  ASR_ARGS=(--asr-endpoint "${ASR_ENDPOINT}")
fi
OUTPUT_DIR="${OUTPUT_DIR:-outputs/moss-tts-local}"

SLIME_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${MEGATRON_DIR}:${SLIME_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export TOKENIZERS_PARALLELISM=false

exec python "${SLIME_ROOT}/train.py" \
  --model-family moss_tts_local --hf-checkpoint "${MODEL_DIR}" --load "${TRAIN_CHECKPOINT}" \
  --omni-endpoints "${OMNI_ENDPOINT}" "${ASR_ARGS[@]}" \
  --prompt-data "${PROMPT_DATA}" --objective "${OBJECTIVE}" \
  --actor-num-gpus-per-node "${TRAIN_GPUS:-1}" \
  --rollout-batch-size 2 --n-samples-per-prompt 4 --global-batch-size 8 \
  --micro-batch-size 1 --num-rollout "${NUM_ROLLOUTS:-10}" \
  --lr 0.000003 --lr-decay-style constant --weight-decay 0 \
  --save "${OUTPUT_DIR}/checkpoints" --save-interval 5 \
  --metrics-jsonl "${OUTPUT_DIR}/metrics.jsonl" --audio-output-dir "${OUTPUT_DIR}/audio" \
  --no-gradient-accumulation-fusion --no-masked-softmax-fusion \
  --no-rope-fusion --no-persist-layer-norm --attention-backend flash "$@"
