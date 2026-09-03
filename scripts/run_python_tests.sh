#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_COMMAND="${PYTHON_COMMAND:-python3}"
TEST_PATTERN="${TEST_PATTERN:-test_*.py}"

if ! command -v "$PYTHON_COMMAND" >/dev/null 2>&1; then
    echo "Missing required command: $PYTHON_COMMAND" >&2
    exit 1
fi

exec "$PYTHON_COMMAND" -m unittest discover \
    -s tests \
    -p "$TEST_PATTERN" \
    "$@"
