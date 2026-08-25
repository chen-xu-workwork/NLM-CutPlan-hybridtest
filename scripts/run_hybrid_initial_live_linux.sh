#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Preserve the deterministic state-#0 live smoke test as a separate entry
# point. The main Linux runner now defaults to the formal three-trigger flow.
export NLM_LLM_INITIAL_ONLY=1
exec "$SCRIPT_DIR/run_hybrid_live_linux.sh" "$@"
