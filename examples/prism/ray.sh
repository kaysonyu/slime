#!/usr/bin/env bash
# Shared lifecycle for run-grpo.sh and run-mopd.sh on dedicated nodes.
# Source this file; the run scripts own the training command and native arguments.

prism_prepare_run() {
    NODE_RANK=${PET_NODE_RANK:-${RANK:-0}}
    NUM_NODES=${PET_NNODES:-${WORLD_SIZE:-1}}
    GPUS_PER_NODE=${PET_NPROC_PER_NODE:-1}
    if [[ ! $NODE_RANK =~ ^[0-9]+$ || ! $NUM_NODES =~ ^[1-9][0-9]*$ || ! $GPUS_PER_NODE =~ ^[1-9][0-9]*$ ]] ||
        (( NODE_RANK >= NUM_NODES )); then
        echo "Expected a node rank in [0, NUM_NODES) and positive node/GPU counts." >&2
        exit 2
    fi
    TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))
    local master_addr=${MASTER_ADDR:-${PET_MASTER_ADDR:-}} master_port=${MASTER_PORT:-${PET_MASTER_PORT:-6379}}
    if [[ -z $master_addr ]] && (( NUM_NODES > 1 )); then
        echo "Set MASTER_ADDR or PET_MASTER_ADDR to the head node for a multi-node run." >&2
        exit 2
    fi
    export RAY_ADDRESS="${master_addr:-127.0.0.1}:${master_port}"
    RAY_PORT=$master_port
    RAY=("${TRAIN_PYTHON:-python3}" -m ray.scripts.scripts)
    START_TIMEOUT=${RAY_START_TIMEOUT:-900}

    OUTPUT_ROOT=${OUTPUT_ROOT:-"${REPO_ROOT}/outputs"}
    [[ $OUTPUT_ROOT == /* ]] || { echo "OUTPUT_ROOT must be an absolute path shared by all nodes." >&2; exit 2; }
    JOB_NAME=${JOB_NAME:-manual}
    TRAIN_JOB_ID=${TRAIN_JOB_ID:-local}
    RUNNING_ROUND=${RUNNING_ROUND:-0}
    JOB_TAG="${JOB_NAME}_${TRAIN_JOB_ID}"
    RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}/${JOB_TAG}"
    ROUND_PATH="${RUN_ROOT}/running_round_${RUNNING_ROUND}"
    RAY_RUN_DIR="${ROUND_PATH}/.ray"
    mkdir -p -- "${ROUND_PATH}/logs"
    if (( NODE_RANK == 0 )); then
        if ! mkdir -- "$RAY_RUN_DIR"; then
            echo "This round already exists; choose a new JOB_NAME or RUNNING_ROUND." >&2
            exit 2
        fi
    elif [[ -f $RAY_RUN_DIR/exit-code ]]; then
        echo "This round has already finished; choose a new JOB_NAME or RUNNING_ROUND." >&2
        exit 2
    fi

    RAY_STARTED=0
    DRIVER_PID=
    trap 'prism_cleanup' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    exec > >(tee -a "${ROUND_PATH}/logs/rank${NODE_RANK}.log") 2>&1
    echo "RUN_ROOT=${RUN_ROOT}"
    echo "ROUND_PATH=${ROUND_PATH}"
    echo "NODE_RANK=${NODE_RANK}/${NUM_NODES}, GPUS_PER_NODE=${GPUS_PER_NODE}, RAY_ADDRESS=${RAY_ADDRESS}"
}

prism_wait_for_file() {
    local file=$1 deadline=$((SECONDS + $2))
    while [[ ! -f $file ]]; do
        if [[ -f $RAY_RUN_DIR/exit-code ]]; then
            local status
            status=$(< "$RAY_RUN_DIR/exit-code")
            echo "The head exited while waiting for $file" >&2
            return "$((status == 0 ? 1 : status))"
        fi
        if (( SECONDS >= deadline )); then
            echo "Timed out waiting for $file" >&2
            return 1
        fi
        sleep 1
    done
}

# Called by EXIT, including failed startup and interrupted training.
# shellcheck disable=SC2329
prism_cleanup() {
    local status=$? attempt
    trap - EXIT INT TERM
    if [[ -n $DRIVER_PID ]]; then
        kill -TERM -- "-$DRIVER_PID" 2>/dev/null || true
        for (( attempt = 0; attempt < 15; attempt++ )); do
            kill -0 "$DRIVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$DRIVER_PID" 2>/dev/null || true
        wait "$DRIVER_PID" 2>/dev/null || true
    fi
    if (( NODE_RANK == 0 )); then
        printf '%s\n' "$status" > "$RAY_RUN_DIR/exit-code.tmp"
        mv -- "$RAY_RUN_DIR/exit-code.tmp" "$RAY_RUN_DIR/exit-code"
    fi
    if (( RAY_STARTED )); then
        timeout --kill-after=5 40 "${RAY[@]}" stop --force || true
    fi
    exit "$status"
}

prism_run() {
    local ray_args=(--num-gpus "$GPUS_PER_NODE" --disable-usage-stats)
    if (( NODE_RANK == 0 )); then
        ray_args+=(--head --port "$RAY_PORT" --include-dashboard=false)
    else
        prism_wait_for_file "$RAY_RUN_DIR/head-ready" "$START_TIMEOUT"
        ray_args+=(--address "$RAY_ADDRESS")
    fi
    RAY_STARTED=1
    timeout --kill-after=10 "$START_TIMEOUT" "${RAY[@]}" start "${ray_args[@]}"
    if (( NODE_RANK == 0 )); then
        touch "$RAY_RUN_DIR/head-ready"
    fi
    touch "$RAY_RUN_DIR/node-${NODE_RANK}.ready"

    if (( NODE_RANK == 0 )); then
        local node status
        for (( node = 0; node < NUM_NODES; node++ )); do
            prism_wait_for_file "$RAY_RUN_DIR/node-${node}.ready" "$START_TIMEOUT"
        done
        # Keep the shell alive to forward signals and stop local Ray on exit.
        setsid "$@" &
        DRIVER_PID=$!
        if wait "$DRIVER_PID"; then status=0; else status=$?; fi
        DRIVER_PID=
        exit "$status"
    else
        prism_wait_for_file "$RAY_RUN_DIR/exit-code" "${RAY_JOB_TIMEOUT:-172800}"
        exit "$(< "$RAY_RUN_DIR/exit-code")"
    fi
}
