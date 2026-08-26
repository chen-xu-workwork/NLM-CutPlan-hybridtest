#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

show_usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_nlm_search_wsl.sh DOMAIN.pddl PROBLEM.pddl [PLAN]

Runs the compiled NLM searcher in WSL. Trigger behavior is controlled only by
the NLM_LLM_* environment variables supplied by the caller. For a bounded
repeated-last anytime run, either provide an outer max_time in SEARCH or leave
SEARCH unset and set NLM_SEARCH_TIME_LIMIT_SECONDS (for example, 7200).
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

: "${CPLEX_HOME:=/opt/ibm/ILOG/CPLEX_Studio_Community222}"
export DOWNWARD_CPLEX_ROOT="${DOWNWARD_CPLEX_ROOT:-$CPLEX_HOME/cplex}"
export DOWNWARD_CONCERT_ROOT="${DOWNWARD_CONCERT_ROOT:-$CPLEX_HOME/concert}"
export DOWNWARD_COIN_ROOT="${DOWNWARD_COIN_ROOT:-/opt/osi}"
export LD_LIBRARY_PATH="$DOWNWARD_COIN_ROOT/lib:$DOWNWARD_CPLEX_ROOT/lib/x86-64_linux/static_pic:$DOWNWARD_CONCERT_ROOT/lib/x86-64_linux/static_pic:${LD_LIBRARY_PATH:-}"

DOMAIN="$1"
PROBLEM="$2"
PLAN="${3:-$PROJECT_ROOT/plans/nlm-search.plan}"
HEURISTIC="${HEURISTIC:-nlm_h=lmcutnumeric(use_second_order_simple=true, bound_iterations=10, ceiling_less_than_one=true)}"
DEFAULT_EAGER_SEARCH="eager(tiebreaking([nlm_h, goalcount()]), reopen_closed=false, llm_h=nlm_h, llm_h_open_list_key_index=0)"
if [[ -n "${NLM_SEARCH_TIME_LIMIT_SECONDS:-}" ]]; then
    if [[ ! "$NLM_SEARCH_TIME_LIMIT_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
       [[ "$NLM_SEARCH_TIME_LIMIT_SECONDS" =~ ^0+([.]0+)?$ ]]; then
        echo "NLM_SEARCH_TIME_LIMIT_SECONDS must be a positive number." >&2
        exit 2
    fi
    DEFAULT_SEARCH="iterated([$DEFAULT_EAGER_SEARCH], pass_bound=true, repeat_last=true, continue_on_solve=true, continue_on_fail=false, max_time=$NLM_SEARCH_TIME_LIMIT_SECONDS)"
else
    DEFAULT_SEARCH="$DEFAULT_EAGER_SEARCH"
fi
SEARCH="${SEARCH:-$DEFAULT_SEARCH}"

if ! command -v python2 >/dev/null 2>&1; then
    echo "Missing required command: python2" >&2
    echo "The NLM-CutPlan translator uses Python 2 style imports." >&2
    exit 1
fi
if [[ ! -x builds/release64/bin/downward ]]; then
    echo "Missing builds/release64/bin/downward." >&2
    echo "Run: bash scripts/compile_windows_source_wsl.sh" >&2
    exit 1
fi
if [[ ! -f "$DOMAIN" ]]; then
    echo "Missing domain file: $DOMAIN" >&2
    exit 1
fi
if [[ ! -f "$PROBLEM" ]]; then
    echo "Missing problem file: $PROBLEM" >&2
    exit 1
fi

mkdir -p "$(dirname "$PLAN")"

echo "Running NLM-CutPlan from: $PROJECT_ROOT"
echo "Domain: $DOMAIN"
echo "Problem: $PROBLEM"
echo "Plan output: $PLAN"
echo "Heuristic: $HEURISTIC"
echo "Search: $SEARCH"
if [[ -n "${NLM_SEARCH_TIME_LIMIT_SECONDS:-}" ]]; then
    echo "Total search wall-time limit: $NLM_SEARCH_TIME_LIMIT_SECONDS seconds"
fi
for name in \
    NLM_LLM_TRIGGER \
    NLM_LLM_COMM_MODE \
    NLM_LLM_REQUEST_INITIAL \
    NLM_LLM_ENABLE_ANCESTOR_STAGNATION \
    NLM_LLM_ENABLE_FRONTIER_PLATEAU \
    NLM_LLM_ENABLE_GLOBAL_STALL \
    NLM_LLM_ANALYSIS_INTERVAL \
    NLM_LLM_ACTIVITY_WINDOWS \
    NLM_LLM_GROWTH_CONFIRM_WINDOWS \
    NLM_LLM_LAYER_RESET_WINDOWS \
    NLM_LLM_LAYER_MIN_RECENT_EXPANDED \
    NLM_LLM_LAYER_MIN_RECENT_NET_GROWTH \
    NLM_LLM_LAYER_MIN_SINCE_REQUEST_EXPANDED \
    NLM_LLM_LAYER_MIN_SINCE_REQUEST_NET_GROWTH \
    NLM_LLM_PLATEAU_GROWTH_RATIO \
    NLM_LLM_STALL_EXPANSIONS \
    NLM_LLM_ANCESTOR_CHECK_INTERVAL \
    NLM_LLM_ANCESTOR_DEPTH \
    NLM_LLM_MIN_DEPTH \
    NLM_LLM_MIN_REQUEST_GAP_EXPANSIONS \
    NLM_LLM_PER_LAYER_REQUEST_GAP_EXPANSIONS \
    NLM_LLM_CANDIDATE_LAYERS \
    NLM_LLM_REQUESTS_PER_SLOT \
    NLM_LLM_HEARTBEAT_INTERVAL \
    NLM_LLM_MAX_PENDING \
    NLM_LLM_MAX_REQUESTS \
    NLM_LLM_H_EPSILON \
    NLM_LLM_H_RELATIVE_EPSILON
do
    printf '%s=%s\n' "$name" "${!name:-<unset>}"
done

exec python2 fast-downward.py --build release64 \
    --plan-file "$PLAN" \
    "$DOMAIN" \
    "$PROBLEM" \
    --heuristic "$HEURISTIC" \
    --search "$SEARCH"
