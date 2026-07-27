#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_hybrid_live_linux.sh /absolute/path/to/model [console options]

Environment overrides:
  NLM_DOMAIN_PATH                    Solver/validation domain PDDL
  NLM_PROBLEM_PATH                   Initial-state test problem PDDL
  NLM_PLAN_PATH                      Output plan path
  NLM_LLM_MODEL                      Name exposed by vLLM and sent by the client
  NLM_VLLM_GPUS                      Optional CUDA device list, for example 0,1
  NLM_VLLM_TENSOR_PARALLEL_SIZE      Tensor-parallel GPU count, default 1
  NLM_VLLM_GPU_MEMORY_UTILIZATION    GPU memory fraction, default 0.90
  NLM_VLLM_MAX_MODEL_LEN             Maximum model context, default 32768
  NLM_VLLM_DTYPE                     Model dtype, default bfloat16
  NLM_LLM_TIMEOUT                    Total generation timeout, default 600 seconds

All arguments after MODEL_PATH are forwarded to hybrid_planner.console.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

MODEL_PATH="${1:-${NLM_VLLM_MODEL_PATH:-}}"
if [[ -z "$MODEL_PATH" ]]; then
    usage >&2
    echo "Missing model path." >&2
    exit 2
fi
if [[ $# -gt 0 ]]; then
    shift
fi

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Missing required file: $1" >&2
        exit 1
    fi
}

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This entry point is intended for a Linux container." >&2
    exit 1
fi

require_command python3
require_command python2
require_command "${NLM_VLLM_EXECUTABLE:-vllm}"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Model directory does not exist: $MODEL_PATH" >&2
    exit 1
fi

DOMAIN_PATH="${NLM_DOMAIN_PATH:-$PROJECT_ROOT/../pddl/domain.pddl}"
PROBLEM_PATH="${NLM_PROBLEM_PATH:-$PROJECT_ROOT/../pddl/problem_scale_10_id_1.pddl}"
PLAN_PATH="${NLM_PLAN_PATH:-$PROJECT_ROOT/../pddl/nlm_hybrid_live.plan}"
MODEL_BASENAME="$(basename "${MODEL_PATH%/}")"
MODEL_NAME="${NLM_LLM_MODEL:-$MODEL_BASENAME}"
VLLM_HOST="${NLM_VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${NLM_VLLM_PORT:-8091}"
TENSOR_PARALLEL_SIZE="${NLM_VLLM_TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${NLM_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${NLM_VLLM_MAX_MODEL_LEN:-32768}"
VLLM_DTYPE="${NLM_VLLM_DTYPE:-bfloat16}"
LLM_TIMEOUT="${NLM_LLM_TIMEOUT:-600}"
STARTUP_TIMEOUT="${NLM_VLLM_STARTUP_TIMEOUT:-1200}"
DEBUG_DIR="${NLM_LLM_PROMPT_DEBUG_DIR:-$PROJECT_ROOT/logs/live_initial_debug}"
VLLM_LOG="${NLM_VLLM_LOG:-$PROJECT_ROOT/logs/vllm-live-initial.log}"

require_file "$DOMAIN_PATH"
require_file "$PROBLEM_PATH"
require_file "$PROJECT_ROOT/fast-downward.py"
require_file "$PROJECT_ROOT/builds/release64/bin/downward"

mkdir -p "$(dirname "$PLAN_PATH")" "$DEBUG_DIR" "$(dirname "$VLLM_LOG")"

# This test intentionally asks only for state #0. The currently trained model
# is not expected to handle predicates that appear only in intermediate states.
export NLM_LLM_TRIGGER=1
export NLM_LLM_REQUEST_INITIAL=1
export NLM_LLM_MAX_PENDING=1
export NLM_LLM_CHECK_INTERVAL=2147483647
export NLM_LLM_STALL_EXPANSIONS=0
export NLM_LLM_MIN_DEPTH=2147483647
export NLM_LLM_EMIT_STATE=1

command=(
    python3 -m hybrid_planner.console
    "$DOMAIN_PATH"
    "$PROBLEM_PATH"
    "$PLAN_PATH"
    --llm-mode live
    --llm-model "$MODEL_NAME"
    --llm-max-concurrency 1
    --llm-timeout "$LLM_TIMEOUT"
    --pending-behavior skip
    --http-workers 1
    --emit-state 1
    --vllm-model-path "$MODEL_PATH"
    --vllm-host "$VLLM_HOST"
    --vllm-port "$VLLM_PORT"
    --vllm-tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --vllm-gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --vllm-max-model-len "$MAX_MODEL_LEN"
    --vllm-dtype "$VLLM_DTYPE"
    --vllm-startup-timeout "$STARTUP_TIMEOUT"
    --vllm-log "$VLLM_LOG"
    --prompt-debug-dir "$DEBUG_DIR"
)

if [[ -n "${NLM_VLLM_GPUS:-}" ]]; then
    command+=(--vllm-gpus "$NLM_VLLM_GPUS")
fi

exec "${command[@]}" "$@"
