#!/usr/bin/env bash
set -euo pipefail
for option in PROFILE PREPROCESSING DECODE_WORKERS MODEL_PROCESSES MAX_BATCH MAX_BATCH_COST BATCH_WINDOW_MS; do
  if [[ -v $option ]]; then
    printf '%s is no longer a launch variable; use explicit server CLI options.\n' "$option" >&2
    exit 2
  fi
done
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
BASE="${JUDGE_BASE:-/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/denglongtai/models/Qwen2.5-Omni-7B}"
RUN="${JUDGE_RUN:-/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/denglongtai/experiments/judge_clap/judge_clap_mosstts2instr_margin_cal_novirt_fixw/judge_clap_mosstts2instr_margin_cal_novirt_fixw-20261011-014703_job-ea14bce9-496d-4579-93b1-ac93c86bb224/running_round_0}"
exec "${PYTHON:-python3}" -m slime_plugins.audio_rewards.serving.server \
  --model judge_clap --base "$BASE" --adapter "${JUDGE_ADAPTER:-$RUN/final}" \
  --atom-stats "${JUDGE_ATOM_STATS:-$RUN/atom_stats.npz}" "$@"
