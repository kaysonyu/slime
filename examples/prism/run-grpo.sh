#!/usr/bin/env bash
# Edit the asset paths below. Run this script on every allocated node.
# The recipe starts Ray, runs training on rank 0, and stops local Ray on exit.
set -euo pipefail
for argument in "$@"; do
    if [[ $argument == --dry-run ]]; then
        echo "--dry-run was removed; inspect the recipe or check shell syntax with bash -n." >&2
        exit 2
    fi
done
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${SLIME_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}

RUN_NAME=${RUN_NAME:-prism-grpo}
source "${REPO_ROOT}/examples/prism/ray.sh"
prism_prepare_run
if (( NODE_RANK != 0 )); then
    prism_run
fi

CKPT_ARGS=(
    --hf-checkpoint /absolute/student-hf
    --pretrained-checkpoint /absolute/student-native/checkpoints/iter_0100000
    # Resume: replace --pretrained-checkpoint with --load /absolute/previous/checkpoint.
    --save "${ROUND_PATH}/checkpoint"
    --save-interval 10
)

# Reuse the model's config reader; no Python code is embedded in this shell.
# Include trailing CLI overrides so architecture matches the effective --hf-checkpoint.
MODEL_ARGS_TEXT=$("${TRAIN_PYTHON:-python3}" -m slime_plugins.models.prism_configuration "${CKPT_ARGS[@]}" "$@")
mapfile -t MODEL_ARGS <<< "$MODEL_ARGS_TEXT"
MODEL_ARGS+=(
    --model-name moss_tts_prism
    --custom-model-provider-path slime_plugins.models.prism.model_provider
    --custom-megatron-model-initializer-path slime_plugins.models.prism_checkpoint.load_prism_pretrained
    --custom-megatron-log-prob-function-path slime_plugins.models.prism_policy.score_policy
)

ROLLOUT_ARGS=(
    --prompt-data "${REPO_ROOT}/examples/prism/data/grpo-base.example.jsonl"
    --input-key script
    --metadata-key metadata
    --rollout-shuffle
    --rollout-function-path slime.rollout.sglang_rollout.generate_rollout
    --custom-generate-function-path slime_plugins.omni_rollout.generate
    --num-rollout 100
    --rollout-batch-size "$((2 * TOTAL_GPUS))"
    --n-samples-per-prompt 4
    --global-batch-size "$((8 * TOTAL_GPUS))"
    --rollout-max-response-len 512
    --rollout-temperature 1
    --rollout-top-p 1
    --rollout-top-k -1
    --rollout-seed 6211
    --rollout-data-transport object-store
    --balance-data
)

OMNI_ARGS=(
    --rollout-backend sglang-omni
    --omni-return-audio
    --rollout-num-gpus-per-engine 1
    --omni-config "${REPO_ROOT}/examples/prism/configs/student.yaml"
    --omni-update-stage tts_engine
    --omni-worker-capabilities chat audio_input audio_output
    --omni-audio-dir "${ROUND_PATH}/rollout_audio"
    --omni-router-policy least_request
    --omni-compact-replay
    --sglang-server-concurrency 2
    --router-log-level warn
    --sglang-log-level warning
    --sglang-log-level-http warning
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --loss-type policy_loss
    --kl-coef 0
    --entropy-coef 0
    --eps-clip 0.2
    --eps-clip-high 0.2
    --policy-loss-weights '{"stop": 1}'
    --custom-rm-path examples.prism.rewards.base.reward
    --omni-reward-config "${REPO_ROOT}/examples/prism/configs/rewards.yaml"
    # MixRL: choose examples.prism.rewards.reward and provide metadata.reward_category.
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.98
    --seed 6211
)

# Use the allocated nodes/GPUs with learner and rollout offload; defaults to one GPU.
PERF_ARGS=(
    --actor-num-nodes "$NUM_NODES"
    --actor-num-gpus-per-node "$GPUS_PER_NODE"
    --rollout-num-gpus "$TOTAL_GPUS"
    --num-gpus-per-node "$GPUS_PER_NODE"
    --colocate
    --offload-train
    --offload-rollout
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --micro-batch-size 1
    --seq-length 4096
    --use-dynamic-batch-size
    --max-tokens-per-gpu 4096
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --accumulate-allreduce-grads-in-fp32
    --update-weight-mode full
    --update-weight-transport nccl
)

EVAL_ARGS=(
    # --eval-config /absolute/eval.yaml
    # --eval-interval 10
)

LOGGING_ARGS=(
    --use-tensorboard
    --tensorboard-dir "${ROUND_PATH}/tensorboard"
    --resolved-config-path "${ROUND_PATH}/config/resolved-args.json"
    --phase-trace-path "${ROUND_PATH}/logs/phases.jsonl"
    # Optional artifact/eval configuration: see docs/GUIDE.md and docs/TRAINING_MODES.md.
)

prism_run "${TRAIN_PYTHON:-python3}" "${REPO_ROOT}/train.py" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OMNI_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "$@"
