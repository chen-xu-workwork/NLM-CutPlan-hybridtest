#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

show_usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_trigger_calibration.sh DOMAIN.pddl ORIGINAL_PROBLEM.pddl [PLAN]

Runs the main-line eager-greedy satisficing NLM search with trigger detection
in log-only mode. No Python controller or LLM is started, and no state is
removed from the Open List.

Read the final [NLM-LLM-TRIGGER-STATS] submitted=<N> as the number of distinct
state requests that the current parameters would have offered to the LLM.
Per-trigger submitted counts use [NLM-LLM-TRIGGER-REASON-STATS].

NLM_LLM_MAX_REQUESTS is forced to 0 here so calibration observes the
untruncated count. Other NLM_LLM_* environment variables may be overridden
normally.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_usage
    exit 0
fi
if [[ $# -lt 2 ]]; then
    show_usage >&2
    exit 2
fi

# Observe trigger behavior without model, prompt, or full state output.
# The monitor emits only sparse diagnostic heartbeats and request summaries.
export NLM_LLM_TRIGGER=1
export NLM_LLM_COMM_MODE=log
export NLM_LLM_PENDING_BEHAVIOR=normal
export NLM_LLM_REQUEST_INITIAL="${NLM_LLM_REQUEST_INITIAL:-0}"
export NLM_LLM_EMIT_STATE="${NLM_LLM_EMIT_STATE:-0}"
export NLM_LLM_LOG_RESPONSE_BODY=0
export NLM_LLM_ENABLE_ANCESTOR_STAGNATION="${NLM_LLM_ENABLE_ANCESTOR_STAGNATION:-1}"
export NLM_LLM_ENABLE_FRONTIER_PLATEAU="${NLM_LLM_ENABLE_FRONTIER_PLATEAU:-1}"
export NLM_LLM_ENABLE_GLOBAL_STALL="${NLM_LLM_ENABLE_GLOBAL_STALL:-1}"
export NLM_LLM_MAX_PENDING=0
# This runner exists specifically to measure the uncapped opportunity count.
# Ignore a value such as 10 inherited from a previous live-run shell.
export NLM_LLM_MAX_REQUESTS=0
export NLM_LLM_ANALYSIS_INTERVAL="${NLM_LLM_ANALYSIS_INTERVAL:-65536}"
export NLM_LLM_ACTIVITY_WINDOWS="${NLM_LLM_ACTIVITY_WINDOWS:-4}"
export NLM_LLM_GROWTH_CONFIRM_WINDOWS="${NLM_LLM_GROWTH_CONFIRM_WINDOWS:-2}"
export NLM_LLM_LAYER_RESET_WINDOWS="${NLM_LLM_LAYER_RESET_WINDOWS:-4}"
export NLM_LLM_LAYER_MIN_RECENT_EXPANDED="${NLM_LLM_LAYER_MIN_RECENT_EXPANDED:-4096}"
export NLM_LLM_LAYER_MIN_RECENT_NET_GROWTH="${NLM_LLM_LAYER_MIN_RECENT_NET_GROWTH:-1024}"
export NLM_LLM_LAYER_MIN_SINCE_REQUEST_EXPANDED="${NLM_LLM_LAYER_MIN_SINCE_REQUEST_EXPANDED:-8192}"
export NLM_LLM_LAYER_MIN_SINCE_REQUEST_NET_GROWTH="${NLM_LLM_LAYER_MIN_SINCE_REQUEST_NET_GROWTH:-2048}"
export NLM_LLM_PLATEAU_GROWTH_RATIO="${NLM_LLM_PLATEAU_GROWTH_RATIO:-1.05}"
export NLM_LLM_STALL_EXPANSIONS="${NLM_LLM_STALL_EXPANSIONS:-500000}"
export NLM_LLM_ANCESTOR_CHECK_INTERVAL="${NLM_LLM_ANCESTOR_CHECK_INTERVAL:-100000}"
export NLM_LLM_ANCESTOR_DEPTH="${NLM_LLM_ANCESTOR_DEPTH:-20}"
export NLM_LLM_MIN_DEPTH="${NLM_LLM_MIN_DEPTH:-30}"
export NLM_LLM_MIN_REQUEST_GAP_EXPANSIONS="${NLM_LLM_MIN_REQUEST_GAP_EXPANSIONS:-100000}"
export NLM_LLM_PER_LAYER_REQUEST_GAP_EXPANSIONS="${NLM_LLM_PER_LAYER_REQUEST_GAP_EXPANSIONS:-500000}"
export NLM_LLM_CANDIDATE_LAYERS="${NLM_LLM_CANDIDATE_LAYERS:-3}"
export NLM_LLM_REQUESTS_PER_SLOT="${NLM_LLM_REQUESTS_PER_SLOT:-1}"
export NLM_LLM_HEARTBEAT_INTERVAL="${NLM_LLM_HEARTBEAT_INTERVAL:-100000}"
export NLM_LLM_H_EPSILON="${NLM_LLM_H_EPSILON:-0.001}"
export NLM_LLM_H_RELATIVE_EPSILON="${NLM_LLM_H_RELATIVE_EPSILON:-0.005}"

exec "$SCRIPT_DIR/run_nlm_search_wsl.sh" "$@"
