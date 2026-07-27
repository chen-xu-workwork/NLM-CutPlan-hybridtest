#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

REPLAY_OUTPUT="${NLM_LLM_REPLAY_OUTPUT:-$PROJECT_ROOT/tests/fixtures/depots_initial_prefix.txt}"

if [ ! -f "$REPLAY_OUTPUT" ]; then
    echo "Missing replay model output: $REPLAY_OUTPUT" >&2
    exit 1
fi

# Isolate one deterministic initial-state request. This verifies the complete
# bridge and Open List injection path without starting vLLM or using a GPU.
export NLM_LLM_REQUEST_INITIAL="${NLM_LLM_REQUEST_INITIAL:-1}"
export NLM_LLM_MAX_PENDING="${NLM_LLM_MAX_PENDING:-1}"
export NLM_LLM_CHECK_INTERVAL="${NLM_LLM_CHECK_INTERVAL:-2147483647}"
export NLM_LLM_STALL_EXPANSIONS="${NLM_LLM_STALL_EXPANSIONS:-0}"
export NLM_LLM_MIN_DEPTH="${NLM_LLM_MIN_DEPTH:-2147483647}"
export NLM_LLM_EMIT_STATE="${NLM_LLM_EMIT_STATE:-1}"

exec python3 -m hybrid_planner.console \
    --llm-mode replay \
    --replay-model-output "$REPLAY_OUTPUT" \
    --pending-behavior skip \
    --http-workers 1 \
    "$@"
