#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

: "${CPLEX_HOME:=/opt/ibm/ILOG/CPLEX_Studio_Community222}"
export DOWNWARD_CPLEX_ROOT="${DOWNWARD_CPLEX_ROOT:-$CPLEX_HOME/cplex}"
export DOWNWARD_CONCERT_ROOT="${DOWNWARD_CONCERT_ROOT:-$CPLEX_HOME/concert}"
export DOWNWARD_COIN_ROOT="${DOWNWARD_COIN_ROOT:-/opt/osi}"
export LD_LIBRARY_PATH="$DOWNWARD_COIN_ROOT/lib:$DOWNWARD_CPLEX_ROOT/lib/x86-64_linux/static_pic:$DOWNWARD_CONCERT_ROOT/lib/x86-64_linux/static_pic:${LD_LIBRARY_PATH:-}"

JOBS="${JOBS:-$(nproc)}"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

require_path() {
    if [[ ! -e "$1" ]]; then
        echo "Missing required path: $1" >&2
        exit 1
    fi
}

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This build helper requires Linux." >&2
    exit 1
fi

require_command cmake
require_command g++
require_command make
require_command python3

require_path "$DOWNWARD_CPLEX_ROOT/include/ilcplex/cplex.h"
require_path "$DOWNWARD_CONCERT_ROOT/include/ilconcert/iloenv.h"
require_path "$DOWNWARD_CPLEX_ROOT/lib/x86-64_linux/static_pic/libcplex.a"
require_path "$DOWNWARD_CONCERT_ROOT/lib/x86-64_linux/static_pic/libconcert.a"
require_path "$DOWNWARD_COIN_ROOT/include/coin/OsiSolverInterface.hpp"

echo "Building NLM-CutPlan on Linux from: $PROJECT_ROOT"
echo "Using $JOBS parallel jobs"

make -C src/search/bliss-0.73 -j"$JOBS"
python3 build.py release64 -j"$JOBS"
