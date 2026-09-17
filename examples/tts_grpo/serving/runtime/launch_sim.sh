#!/usr/bin/env bash
set -Eeuo pipefail

die() {
   printf 'error: %s\n' "$*" >&2
   exit 1
}

for name in EXPECTED_VISIBLE_GPUS SLIME_REPO SLIME_DEPLOY_SIM_TREE SLIME_DEPLOY_LAUNCHER_BLOB WAVLM_SIM_CHECKPOINT WAVLM_SIM_ALLOWED_ROOTS SERVICE_PORT; do
   [ -n "${!name:-}" ] || die "${name} must be set"
done
case "${EXPECTED_VISIBLE_GPUS}" in
   1|4|8) ;;
   *) die "EXPECTED_VISIBLE_GPUS must be one of 1, 4, or 8" ;;
esac
[ -f "${SLIME_REPO}/slime/serving/tts_sim/__main__.py" ] || \
   die "Slime is missing its vendored SIM service entrypoint"
[ -f "${WAVLM_SIM_CHECKPOINT}" ] || die "WAVLM_SIM_CHECKPOINT is missing"
[[ "${WAVLM_SIM_ALLOWED_ROOTS}" != :* && "${WAVLM_SIM_ALLOWED_ROOTS}" != *: && "${WAVLM_SIM_ALLOWED_ROOTS}" != *::* ]] || \
   die "WAVLM_SIM_ALLOWED_ROOTS must be a colon-separated list of non-empty paths"
IFS=: read -r -a allowed_roots <<<"${WAVLM_SIM_ALLOWED_ROOTS}"
for allowed_root in "${allowed_roots[@]}"; do
   [[ "${allowed_root}" = /inspire/* ]] || die "WAVLM_SIM_ALLOWED_ROOTS entries must be shared /inspire paths"
   [ -d "${allowed_root}" ] || die "WAVLM_SIM_ALLOWED_ROOTS entry is not a directory: ${allowed_root}"
done
actual_tree=$(python3 "${SLIME_REPO}/tools/tts_source_fingerprint.py" "${SLIME_REPO}/slime/serving/tts_sim")
[ "${actual_tree}" = "${SLIME_DEPLOY_SIM_TREE}" ] || die "vendored SIM source does not match the deployment command"
actual_launcher_blob=$(git -C "${SLIME_REPO}" hash-object -- "$(readlink -f -- "$0")")
[ "${actual_launcher_blob}" = "${SLIME_DEPLOY_LAUNCHER_BLOB}" ] || \
   die "SIM runtime launcher does not match the deployment command"


export PYTHONPATH="${SLIME_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export WAVLM_SIM_HOST=0.0.0.0
export WAVLM_SIM_PORT=${SERVICE_PORT}
export WAVLM_SIM_DEVICES=all_cuda
export WAVLM_SIM_EXPECTED_CUDA_DEVICES=${EXPECTED_VISIBLE_GPUS}
export WAVLM_SIM_SOURCE_TREE=${SLIME_DEPLOY_SIM_TREE}
export WAVLM_SIM_ACCESS_LOG=0
export WAVLM_SIM_PER_DEVICE_BATCH=${WAVLM_SIM_PER_DEVICE_BATCH:-16}
export WAVLM_SIM_MAX_ITEMS_PER_REQUEST=${WAVLM_SIM_MAX_ITEMS_PER_REQUEST:-16}
export WAVLM_SIM_DYNAMIC_DELAY_MS=${WAVLM_SIM_DYNAMIC_DELAY_MS:-5}
export WAVLM_SIM_AUDIO_WORKERS=${WAVLM_SIM_AUDIO_WORKERS:-16}
export WAVLM_SIM_MAX_QUEUE_ITEMS=${WAVLM_SIM_MAX_QUEUE_ITEMS:-8192}
export WAVLM_SIM_MAX_INFLIGHT_REQUESTS=${WAVLM_SIM_MAX_INFLIGHT_REQUESTS:-4096}
export WAVLM_SIM_REFERENCE_CACHE_ITEMS=${WAVLM_SIM_REFERENCE_CACHE_ITEMS:-100000}
export WAVLM_SIM_ALLOW_TF32=${WAVLM_SIM_ALLOW_TF32:-1}
exec python3 -m slime.serving.tts_sim
