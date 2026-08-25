#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# The checkpoint is intentionally supplied by the runtime environment. Model
# training and checkpoint management remain outside this repository.
: "${NLM_VLLM_MODEL_PATH:?Set NLM_VLLM_MODEL_PATH to the trained checkpoint directory}"

MODEL_NAME="${NLM_LLM_MODEL:-Qwen3.5-9B}"
VLLM_HOST="${NLM_VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${NLM_VLLM_PORT:-8091}"
VLLM_GPUS="${NLM_VLLM_GPUS:-0}"
LLM_CONCURRENCY="${NLM_LLM_MAX_CONCURRENCY:-100}"
SAMPLES_PER_STATE="${NLM_LLM_SAMPLES_PER_STATE:-3}"
PENDING_BEHAVIOR="${NLM_LLM_PENDING_BEHAVIOR:-normal}"

# Formal online mode uses the three search-state triggers. The initial-state
# request remains available as an explicit smoke-test override.
export NLM_LLM_REQUEST_INITIAL="${NLM_LLM_REQUEST_INITIAL:-0}"
export NLM_LLM_MAX_REQUESTS="${NLM_LLM_MAX_REQUESTS:-10}"

exec python3 -m hybrid_planner.console \
    --llm-mode live \
    --llm-model "$MODEL_NAME" \
    --llm-max-concurrency "$LLM_CONCURRENCY" \
    --llm-samples-per-state "$SAMPLES_PER_STATE" \
    --pending-behavior "$PENDING_BEHAVIOR" \
    --vllm-model-path "$NLM_VLLM_MODEL_PATH" \
    --vllm-host "$VLLM_HOST" \
    --vllm-port "$VLLM_PORT" \
    --vllm-gpus "$VLLM_GPUS" \
    "$@"
