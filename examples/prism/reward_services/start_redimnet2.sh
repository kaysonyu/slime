#!/usr/bin/env bash
set -euo pipefail
for option in PROFILE BATCHING PRECISION MAX_BATCH MAX_BATCH_COST GPU_WORKERS BATCH_WINDOW_MS; do
  if [[ -v $option ]]; then
    printf '%s is no longer a launch variable; use explicit server CLI options.\n' "$option" >&2
    exit 2
  fi
done
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/cache/prism-reward-inductor}"
MODEL_PATH="${MODEL_PATH:-/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/redimnet2/b6-vb2+vox2+cnc2_v0-lm.pt}"
exec "${PYTHON:-python3}" -m slime_plugins.audio_rewards.serving.server \
  --model redimnet2 --checkpoint "$MODEL_PATH" "$@"
