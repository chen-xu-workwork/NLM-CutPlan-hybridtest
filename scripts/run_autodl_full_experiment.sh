#!/usr/bin/env bash

# Formal full-dataset entry point for the native and LLM-assisted anytime runs.
#
# Default schedule:
#   1. scale 10/20/30, eight planner processes in parallel
#   2. scale 40, two planner processes in parallel
#   3. run the native baseline first, then the full live configuration
#   4. allow 1,800 seconds for scale 10/20/30 and 3,600 for scale 40
#
# Useful overrides:
#   RUN_MODE=off|live|both
#   TIME_LIMIT_SECONDS=...         Override both scale groups.
#   TIME_LIMIT_10_30_SECONDS=...   Override only scale 10/20/30.
#   TIME_LIMIT_40_SECONDS=...      Override only scale 40.
#   FULL_OUTPUT_DIR=...   Reuse the same directory to resume completed groups.
#   DETACH=0              Run in the current terminal instead of tmux.

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BATCH_RUNNER="$SCRIPT_DIR/run_autodl_anytime_batch.sh"

EXPECTED_PROJECT_ROOT="${EXPECTED_PROJECT_ROOT:-/root/autodl-tmp/NLM-CutPlan-hybridtest}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/Qwen3_5-9B/dapo/data_260811_resume_193/global_step_350/actor/huggingface}"
DOMAIN_PATH="${DOMAIN_PATH:-/root/PyPACE/data/generated-pddl/depots-numeric-validation-original/domain.pddl}"
PROBLEM_DIR="${PROBLEM_DIR:-/root/PyPACE/data/generated-pddl/depots-numeric-validation-original/problems}"
RESULTS_ROOT="${RESULTS_ROOT:-/root/autodl-tmp/nlm-cutplan-results}"

RUN_MODE="${RUN_MODE:-both}"
TIME_LIMIT_SECONDS="${TIME_LIMIT_SECONDS:-}"
TIME_LIMIT_10_30_SECONDS="${TIME_LIMIT_10_30_SECONDS:-${TIME_LIMIT_SECONDS:-1800}}"
TIME_LIMIT_40_SECONDS="${TIME_LIMIT_40_SECONDS:-${TIME_LIMIT_SECONDS:-3600}}"
MEMORY_LIMIT="${MEMORY_LIMIT:-${NLM_OVERALL_MEMORY_LIMIT:-}}"
LLM_SEED="${LLM_SEED:-${NLM_LLM_SEED:-0}}"
VLLM_BIN="${VLLM_BIN:-${NLM_VLLM_EXECUTABLE:-${VLLM_EXECUTABLE:-vllm}}}"
CXX_RUNTIME_DIR="${CXX_RUNTIME_DIR:-}"
DETACH="${DETACH:-1}"

FULL_RUN_TAG="${FULL_RUN_TAG:-full-anytime-$(date +%Y%m%d-%H%M%S)}"
FULL_OUTPUT_DIR="${FULL_OUTPUT_DIR:-$RESULTS_ROOT/$FULL_RUN_TAG}"
SESSION_NAME="${SESSION_NAME:-nlm-$FULL_RUN_TAG}"
SESSION_NAME="${SESSION_NAME//[^A-Za-z0-9_-]/_}"

fail() {
    echo "[NLM-FULL] ERROR: $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "missing command: $1"
}

[[ "$RUN_MODE" == "off" || "$RUN_MODE" == "live" ||
   "$RUN_MODE" == "both" ]] || fail "RUN_MODE must be off, live, or both"
[[ "$TIME_LIMIT_10_30_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
    fail "TIME_LIMIT_10_30_SECONDS must be a positive number"
[[ ! "$TIME_LIMIT_10_30_SECONDS" =~ ^0+([.]0+)?$ ]] || \
    fail "TIME_LIMIT_10_30_SECONDS must be greater than zero"
[[ "$TIME_LIMIT_40_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
    fail "TIME_LIMIT_40_SECONDS must be a positive number"
[[ ! "$TIME_LIMIT_40_SECONDS" =~ ^0+([.]0+)?$ ]] || \
    fail "TIME_LIMIT_40_SECONDS must be greater than zero"
[[ "$DETACH" == "0" || "$DETACH" == "1" ]] || \
    fail "DETACH must be 0 or 1"
[[ -f "$BATCH_RUNNER" ]] || fail "missing batch runner: $BATCH_RUNNER"

require_command readlink

# Create one supervising tmux session. The two scale groups then run in order,
# so their different planner parallelism and trigger profiles never overlap.
if [[ "$DETACH" == "1" && "${NLM_FULL_TMUX_CHILD:-0}" != "1" ]]; then
    require_command tmux
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        fail "tmux session already exists: $SESSION_NAME"
    fi
    mkdir -p "$FULL_OUTPUT_DIR"
    TMUX_ENV=(env)
    if [[ -v CUDA_VISIBLE_DEVICES ]]; then
        TMUX_ENV+=("CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
    else
        TMUX_ENV+=(-u CUDA_VISIBLE_DEVICES)
    fi
    TMUX_ENV+=(
        "PATH=$PATH"
        "PYTHONPATH=${PYTHONPATH:-}"
        "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
        "CONDA_PREFIX=${CONDA_PREFIX:-}"
        NLM_FULL_TMUX_CHILD=1
        DETACH=0
        "RUN_MODE=$RUN_MODE"
        "TIME_LIMIT_SECONDS=$TIME_LIMIT_SECONDS"
        "TIME_LIMIT_10_30_SECONDS=$TIME_LIMIT_10_30_SECONDS"
        "TIME_LIMIT_40_SECONDS=$TIME_LIMIT_40_SECONDS"
        "MEMORY_LIMIT=$MEMORY_LIMIT"
        "FULL_RUN_TAG=$FULL_RUN_TAG"
        "FULL_OUTPUT_DIR=$FULL_OUTPUT_DIR"
        "MODEL_PATH=$MODEL_PATH"
        "DOMAIN_PATH=$DOMAIN_PATH"
        "PROBLEM_DIR=$PROBLEM_DIR"
        "RESULTS_ROOT=$RESULTS_ROOT"
        "LLM_SEED=$LLM_SEED"
        "VLLM_BIN=$VLLM_BIN"
        "CXX_RUNTIME_DIR=$CXX_RUNTIME_DIR"
        "EXPECTED_PROJECT_ROOT=$EXPECTED_PROJECT_ROOT"
        bash
        "$SCRIPT_PATH"
    )
    printf -v TMUX_COMMAND '%q ' "${TMUX_ENV[@]}"
    tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_ROOT" "$TMUX_COMMAND"
    echo "[NLM-FULL] started tmux session: $SESSION_NAME"
    echo "[NLM-FULL] attach: tmux attach -t $SESSION_NAME"
    echo "[NLM-FULL] output: $FULL_OUTPUT_DIR"
    exit 0
fi

mkdir -p "$FULL_OUTPUT_DIR"
exec > >(tee -a "$FULL_OUTPUT_DIR/full-console.log") 2>&1

STANDARD_TRIGGER_ENV=(
    NLM_LLM_ANALYSIS_INTERVAL=65536
    NLM_LLM_ACTIVITY_WINDOWS=4
    NLM_LLM_GROWTH_CONFIRM_WINDOWS=2
    NLM_LLM_LAYER_RESET_WINDOWS=4
    NLM_LLM_LAYER_MIN_RECENT_EXPANDED=4096
    NLM_LLM_LAYER_MIN_RECENT_NET_GROWTH=1024
    NLM_LLM_LAYER_MIN_SINCE_REQUEST_EXPANDED=8192
    NLM_LLM_LAYER_MIN_SINCE_REQUEST_NET_GROWTH=2048
    NLM_LLM_PLATEAU_GROWTH_RATIO=1.05
    NLM_LLM_STALL_EXPANSIONS=500000
    NLM_LLM_ANCESTOR_CHECK_INTERVAL=100000
    NLM_LLM_ANCESTOR_DEPTH=20
    NLM_LLM_MIN_DEPTH=30
    NLM_LLM_MIN_REQUEST_GAP_EXPANSIONS=100000
    NLM_LLM_PER_LAYER_REQUEST_GAP_EXPANSIONS=500000
    NLM_LLM_CANDIDATE_LAYERS=3
    NLM_LLM_REQUESTS_PER_SLOT=1
    NLM_LLM_HEARTBEAT_INTERVAL=100000
    NLM_LLM_MAX_REQUESTS=10
    NLM_LLM_H_EPSILON=0.001
    NLM_LLM_H_RELATIVE_EPSILON=0.005
)

# Scale 40 expands fewer states per wall-clock minute because grounding and
# successor evaluation are more expensive. Halve expansion-count time scales,
# while retaining action-depth and statistical-ratio semantics.
SCALE40_TRIGGER_ENV=(
    NLM_LLM_ANALYSIS_INTERVAL=32768
    NLM_LLM_ACTIVITY_WINDOWS=4
    NLM_LLM_GROWTH_CONFIRM_WINDOWS=2
    NLM_LLM_LAYER_RESET_WINDOWS=4
    NLM_LLM_LAYER_MIN_RECENT_EXPANDED=2048
    NLM_LLM_LAYER_MIN_RECENT_NET_GROWTH=512
    NLM_LLM_LAYER_MIN_SINCE_REQUEST_EXPANDED=4096
    NLM_LLM_LAYER_MIN_SINCE_REQUEST_NET_GROWTH=1024
    NLM_LLM_PLATEAU_GROWTH_RATIO=1.05
    NLM_LLM_STALL_EXPANSIONS=250000
    NLM_LLM_ANCESTOR_CHECK_INTERVAL=50000
    NLM_LLM_ANCESTOR_DEPTH=20
    NLM_LLM_MIN_DEPTH=30
    NLM_LLM_MIN_REQUEST_GAP_EXPANSIONS=50000
    NLM_LLM_PER_LAYER_REQUEST_GAP_EXPANSIONS=250000
    NLM_LLM_CANDIDATE_LAYERS=3
    NLM_LLM_REQUESTS_PER_SLOT=1
    NLM_LLM_HEARTBEAT_INTERVAL=50000
    NLM_LLM_MAX_REQUESTS=10
    NLM_LLM_H_EPSILON=0.001
    NLM_LLM_H_RELATIVE_EPSILON=0.005
)

{
    printf 'run_mode=%s\n' "$RUN_MODE"
    printf 'scales_10_20_30_time_limit_seconds=%s\n' "$TIME_LIMIT_10_30_SECONDS"
    printf 'scale_40_time_limit_seconds=%s\n' "$TIME_LIMIT_40_SECONDS"
    printf 'scales_10_20_30_parallelism=8\n'
    printf 'scale_40_parallelism=2\n'
    printf 'standard_min_request_gap_expansions=100000\n'
    printf 'scale_40_min_request_gap_expansions=50000\n'
} >"$FULL_OUTPUT_DIR/full_experiment_config.txt"

run_group() {
    local experiment_mode="$1"
    local scale_spec="$2"
    local parallelism="$3"
    local profile_name="$4"
    local time_limit_seconds="$5"
    local output_dir="$FULL_OUTPUT_DIR/$experiment_mode/$profile_name"
    local -a trigger_environment
    if [[ "$profile_name" == "scale_40" ]]; then
        trigger_environment=("${SCALE40_TRIGGER_ENV[@]}")
    else
        trigger_environment=("${STANDARD_TRIGGER_ENV[@]}")
    fi

    echo "[NLM-FULL] starting mode=$experiment_mode scales=$scale_spec parallelism=$parallelism time_limit=${time_limit_seconds}s profile=$profile_name"
    env "${trigger_environment[@]}" \
        MODE="$experiment_mode" \
        SCALE="$scale_spec" \
        LIMIT=0 \
        PARALLELISM="$parallelism" \
        TIME_LIMIT_SECONDS="$time_limit_seconds" \
        MEMORY_LIMIT="$MEMORY_LIMIT" \
        DETACH=0 \
        RUN_TAG="$FULL_RUN_TAG-$experiment_mode-$profile_name" \
        OUTPUT_DIR="$output_dir" \
        MODEL_PATH="$MODEL_PATH" \
        DOMAIN_PATH="$DOMAIN_PATH" \
        PROBLEM_DIR="$PROBLEM_DIR" \
        RESULTS_ROOT="$RESULTS_ROOT" \
        LLM_SEED="$LLM_SEED" \
        VLLM_BIN="$VLLM_BIN" \
        CXX_RUNTIME_DIR="$CXX_RUNTIME_DIR" \
        EXPECTED_PROJECT_ROOT="$EXPECTED_PROJECT_ROOT" \
        bash "$BATCH_RUNNER"
    echo "[NLM-FULL] completed mode=$experiment_mode profile=$profile_name"
}

if [[ "$RUN_MODE" == "both" ]]; then
    MODES=(off live)
else
    MODES=("$RUN_MODE")
fi

echo "[NLM-FULL] output root: $FULL_OUTPUT_DIR"
echo "[NLM-FULL] scale 10/20/30 limit: ${TIME_LIMIT_10_30_SECONDS}s per problem"
echo "[NLM-FULL] scale 40 limit: ${TIME_LIMIT_40_SECONDS}s per problem"
for experiment_mode in "${MODES[@]}"; do
    run_group "$experiment_mode" "10,20,30" 8 "scales_10_20_30" \
        "$TIME_LIMIT_10_30_SECONDS"
    run_group "$experiment_mode" "40" 2 "scale_40" \
        "$TIME_LIMIT_40_SECONDS"
done
echo "[NLM-FULL] all requested experiment groups completed"
