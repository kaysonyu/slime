#!/usr/bin/env bash
# Configure existing student.yaml / checkpoint args and the shared rewards.yaml.
# Model/codec/service addresses remain explicit placeholders until deployment.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
PRISM_EVAL_DATA=${PRISM_EVAL_DATA:-/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/data/prism-rl/eval/small-v1}
PRISM_EVAL_SIZE=${PRISM_EVAL_SIZE:-small}
PRISM_REWARD_CONFIG=${PRISM_REWARD_CONFIG:-${REPO_ROOT}/examples/prism/configs/rewards.yaml}
PRISM_EVAL_INTERVAL=${PRISM_EVAL_INTERVAL:-20}
if [[ "$PRISM_EVAL_SIZE" != small && "$PRISM_EVAL_SIZE" != smoke ]]; then
    echo 'PRISM_EVAL_SIZE must be small or smoke' >&2
    exit 2
fi
exec bash "${REPO_ROOT}/examples/prism/run-mopd.sh" \
    --eval-config "${PRISM_EVAL_DATA}/${PRISM_EVAL_SIZE}.yaml" \
    --eval-interval "$PRISM_EVAL_INTERVAL" \
    --omni-reward-config "$PRISM_REWARD_CONFIG" \
    --omni-eval-return-audio \
    --no-opd-eval-teacher-scoring \
    --custom-eval-rollout-log-function-path examples.prism.eval.runtime.log_eval \
    "$@"
