#!/usr/bin/env bash

# AutoDL container entry point for NLM-CutPlan anytime experiments.
#
# Default pilot:
#   - live LLM mode
#   - two deterministic scale-30 problems
#   - two planner jobs in parallel
#   - 600 seconds of anytime search per problem
#   - one vLLM process shared by both planners
#   - detached tmux execution
#
# Useful overrides:
#   MODE=off       Disable every LLM/trigger/bridge component.
#   LIMIT=0        Run every matching problem instead of the first two.
#   SCALE=all      Select all problem scales.
#   DETACH=0       Run in the current terminal instead of tmux.
#   OUTPUT_DIR=... Resume or inspect one explicitly named result directory.

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EXPECTED_PROJECT_ROOT="${EXPECTED_PROJECT_ROOT:-/root/autodl-tmp/NLM-CutPlan-hybridtest}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/Qwen3_5-9B/dapo/data_260811_resume_193/global_step_350/actor/huggingface}"
DOMAIN_PATH="${DOMAIN_PATH:-/root/PyPACE/data/generated-pddl/depots-numeric-validation-original/domain.pddl}"
PROBLEM_DIR="${PROBLEM_DIR:-/root/PyPACE/data/generated-pddl/depots-numeric-validation-original/problems}"
RESULTS_ROOT="${RESULTS_ROOT:-/root/autodl-tmp/nlm-cutplan-results}"

MODE="${MODE:-live}"
SCALE="${SCALE:-30}"
LIMIT="${LIMIT:-2}"
PARALLELISM="${PARALLELISM:-2}"
TIME_LIMIT_SECONDS="${TIME_LIMIT_SECONDS:-600}"
MEMORY_LIMIT="${MEMORY_LIMIT:-${NLM_OVERALL_MEMORY_LIMIT:-}}"
LLM_MODEL_NAME="${LLM_MODEL_NAME:-$(basename "$MODEL_PATH")}"
LLM_SEED="${LLM_SEED:-${NLM_LLM_SEED:-0}}"
VLLM_BIN="${VLLM_BIN:-${NLM_VLLM_EXECUTABLE:-${VLLM_EXECUTABLE:-vllm}}}"
# vLLM warns about unknown inherited variables whose names begin with VLLM_.
# Keep the executable override under a launcher-specific name instead.
unset VLLM_EXECUTABLE
DETACH="${DETACH:-1}"

RUN_TAG="${RUN_TAG:-pilot-scale${SCALE}-${MODE}-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_ROOT/$RUN_TAG}"
SESSION_NAME="${SESSION_NAME:-nlm-${RUN_TAG}}"
SESSION_NAME="${SESSION_NAME//[^A-Za-z0-9_-]/_}"

fail() {
    echo "[NLM-AUTODL] ERROR: $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "missing command: $1"
}

require_file() {
    [[ -f "$1" ]] || fail "missing file: $1"
}

require_executable_file() {
    [[ -x "$1" ]] || fail "missing executable file: $1"
}

require_directory() {
    [[ -d "$1" ]] || fail "missing directory: $1"
}

[[ "$MODE" == "live" || "$MODE" == "off" ]] || \
    fail "MODE must be live or off"
[[ "$SCALE" == "all" || "$SCALE" =~ ^[1-9][0-9]*$ ]] || \
    fail "SCALE must be a positive integer or all"
[[ "$LIMIT" =~ ^(0|[1-9][0-9]*)$ ]] || \
    fail "LIMIT must be a non-negative integer without leading zeroes"
[[ "$PARALLELISM" =~ ^[1-9][0-9]*$ ]] || \
    fail "PARALLELISM must be a positive integer"
[[ "$TIME_LIMIT_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
    fail "TIME_LIMIT_SECONDS must be a positive number"
[[ ! "$TIME_LIMIT_SECONDS" =~ ^0+([.]0+)?$ ]] || \
    fail "TIME_LIMIT_SECONDS must be greater than zero"
[[ "$DETACH" == "0" || "$DETACH" == "1" ]] || \
    fail "DETACH must be 0 or 1"

require_command python3
require_command python2
require_command readlink
require_file "$PROJECT_ROOT/fast-downward.py"
require_executable_file "$PROJECT_ROOT/builds/release64/bin/downward"
require_file "$DOMAIN_PATH"
require_directory "$PROBLEM_DIR"

if [[ "$PROJECT_ROOT" != "$EXPECTED_PROJECT_ROOT" ]]; then
    echo "[NLM-AUTODL] note: project is running from $PROJECT_ROOT" >&2
    echo "[NLM-AUTODL] expected container path was $EXPECTED_PROJECT_ROOT" >&2
fi

if [[ "$MODE" == "live" ]]; then
    # An explicitly empty value hides every GPU from CUDA. Treat it like an
    # unset variable for live runs so the container's normal GPU discovery is
    # preserved.
    if [[ -v CUDA_VISIBLE_DEVICES && -z "$CUDA_VISIBLE_DEVICES" ]]; then
        unset CUDA_VISIBLE_DEVICES
    fi
    require_directory "$MODEL_PATH"
    require_command "$VLLM_BIN"
    python3 -c 'import aiohttp, pddl, unified_planning' >/dev/null 2>&1 || \
        fail "missing Python live dependencies; install requirements/hybrid.txt"
    GPU_COUNT="$(python3 -c 'import torch; print(torch.cuda.device_count())')" || \
        fail "failed to query CUDA devices through PyTorch"
    [[ "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]] || \
        fail "no CUDA device is visible to PyTorch"
fi

if [[ "$SCALE" == "all" ]]; then
    PROBLEM_PATTERN='problem_scale_*_id_*.pddl'
else
    PROBLEM_PATTERN="problem_scale_${SCALE}_id_*.pddl"
fi

mapfile -t ALL_PROBLEMS < <(
    find "$PROBLEM_DIR" -maxdepth 1 -type f -name "$PROBLEM_PATTERN" -print |
        LC_ALL=C sort
)
(( ${#ALL_PROBLEMS[@]} > 0 )) || \
    fail "no problems matched $PROBLEM_DIR/$PROBLEM_PATTERN"

if (( LIMIT == 0 || LIMIT >= ${#ALL_PROBLEMS[@]} )); then
    PROBLEMS=("${ALL_PROBLEMS[@]}")
else
    PROBLEMS=("${ALL_PROBLEMS[@]:0:LIMIT}")
fi

if (( PARALLELISM > ${#PROBLEMS[@]} )); then
    PARALLELISM="${#PROBLEMS[@]}"
fi

# Existing tmux servers can retain an old conda/PATH environment. Embed the
# essential runtime and experiment values into the child command explicitly.
if [[ "$DETACH" == "1" && "${NLM_TMUX_CHILD:-0}" != "1" ]]; then
    require_command tmux
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        fail "tmux session already exists: $SESSION_NAME"
    fi
    TMUX_ENV=(env)
    if [[ -v CUDA_VISIBLE_DEVICES ]]; then
        TMUX_ENV+=("CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
    else
        # An existing tmux server may have retained an obsolete empty value.
        TMUX_ENV+=(-u CUDA_VISIBLE_DEVICES)
    fi
    TMUX_ENV+=(
        "PATH=$PATH"
        "PYTHONPATH=${PYTHONPATH:-}"
        "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
        "CONDA_PREFIX=${CONDA_PREFIX:-}"
        NLM_TMUX_CHILD=1
        DETACH=0
        "RUN_TAG=$RUN_TAG"
        "OUTPUT_DIR=$OUTPUT_DIR"
        "MODE=$MODE"
        "SCALE=$SCALE"
        "LIMIT=$LIMIT"
        "PARALLELISM=$PARALLELISM"
        "TIME_LIMIT_SECONDS=$TIME_LIMIT_SECONDS"
        "MEMORY_LIMIT=$MEMORY_LIMIT"
        "MODEL_PATH=$MODEL_PATH"
        "DOMAIN_PATH=$DOMAIN_PATH"
        "PROBLEM_DIR=$PROBLEM_DIR"
        "RESULTS_ROOT=$RESULTS_ROOT"
        "LLM_MODEL_NAME=$LLM_MODEL_NAME"
        "LLM_SEED=$LLM_SEED"
        "VLLM_BIN=$VLLM_BIN"
        "EXPECTED_PROJECT_ROOT=$EXPECTED_PROJECT_ROOT"
        bash
        "$SCRIPT_PATH"
    )
    printf -v TMUX_COMMAND '%q ' "${TMUX_ENV[@]}"
    tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_ROOT" "$TMUX_COMMAND"
    echo "[NLM-AUTODL] started tmux session: $SESSION_NAME"
    echo "[NLM-AUTODL] attach: tmux attach -t $SESSION_NAME"
    echo "[NLM-AUTODL] output: $OUTPUT_DIR"
    exit 0
fi

mkdir -p "$OUTPUT_DIR"
printf '%s\n' "${PROBLEMS[@]}" >"$OUTPUT_DIR/selected_problems.txt"

echo "[NLM-AUTODL] project: $PROJECT_ROOT"
echo "[NLM-AUTODL] mode: $MODE"
echo "[NLM-AUTODL] selected problems: ${#PROBLEMS[@]}"
echo "[NLM-AUTODL] parallel planners: $PARALLELISM"
echo "[NLM-AUTODL] anytime limit per problem: ${TIME_LIMIT_SECONDS}s"
echo "[NLM-AUTODL] output: $OUTPUT_DIR"
if [[ "$MODE" == "live" ]]; then
    echo "[NLM-AUTODL] model path: $MODEL_PATH"
    echo "[NLM-AUTODL] served model name: $LLM_MODEL_NAME"
fi
printf '[NLM-AUTODL] problem: %s\n' "${PROBLEMS[@]}"

COMMAND=(
    python3 -m hybrid_planner.batch_console
    "$DOMAIN_PATH"
    "${PROBLEMS[@]}"
    --default-mode "$MODE"
    --output-dir "$OUTPUT_DIR"
    --parallelism "$PARALLELISM"
    --search-time-limit "$TIME_LIMIT_SECONDS"
    --build release64
    --planner-python python2
    --llm-seed "$LLM_SEED"
    --resume
)

if [[ -n "$MEMORY_LIMIT" ]]; then
    COMMAND+=(--overall-memory-limit "$MEMORY_LIMIT")
fi

if [[ "$MODE" == "live" ]]; then
    COMMAND+=(
        --vllm-model-path "$MODEL_PATH"
        --vllm-executable "$VLLM_BIN"
        --llm-model "$LLM_MODEL_NAME"
    )
fi

set +e
"${COMMAND[@]}" 2>&1 | tee "$OUTPUT_DIR/batch-console.log"
EXIT_CODE="${PIPESTATUS[0]}"
set -e

printf '%s\n' "$EXIT_CODE" >"$OUTPUT_DIR/launcher.exit_code"
if (( EXIT_CODE == 0 )); then
    echo "[NLM-AUTODL] batch completed successfully"
else
    echo "[NLM-AUTODL] batch failed with exit code $EXIT_CODE" >&2
fi
echo "[NLM-AUTODL] results: $OUTPUT_DIR"
exit "$EXIT_CODE"
