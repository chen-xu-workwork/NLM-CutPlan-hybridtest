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

# 旧的状态探针：用于直接观察 eager search 展开的状态。
# 1 表示开启；0 表示关闭。正式测速时通常应关闭。
export NLM_EAGER_PROBES="${NLM_EAGER_PROBES:-1}"
# 最多打印多少个被正常展开的状态；设为 -1 可打印全部，设为 0 则不打印状态。
export NLM_EAGER_STATE_PROBE_LIMIT="${NLM_EAGER_STATE_PROBE_LIMIT:-3}"
# 状态打印步长；1 表示每个状态都检查，10 表示每 10 个展开状态打印一次。
export NLM_EAGER_STATE_PROBE_STRIDE="${NLM_EAGER_STATE_PROBE_STRIDE:-1}"

# LLM 介入触发器总开关：1 开启候选检测，0 完全关闭。
export NLM_LLM_TRIGGER="${NLM_LLM_TRIGGER:-1}"
# 通信模式：log 只在 C++ 内部打印请求；http 会启动 C++ 后台线程，把请求发给 Python 主控。
# 直接运行本脚本时默认保持 log；用 python3 -m hybrid_planner.console 启动时会自动覆盖成 http。
export NLM_LLM_COMM_MODE="${NLM_LLM_COMM_MODE:-log}"
export NLM_LLM_HTTP_HOST="${NLM_LLM_HTTP_HOST:-127.0.0.1}"
export NLM_LLM_HTTP_PORT="${NLM_LLM_HTTP_PORT:-8765}"
export NLM_LLM_HTTP_PATH="${NLM_LLM_HTTP_PATH:-/llm/request}"
export NLM_LLM_HTTP_TIMEOUT_MS="${NLM_LLM_HTTP_TIMEOUT_MS:-30000}"
# 并发 HTTP worker 数量。每个 worker 同时维护一个在途请求；例如设为 8 时，
# 最多可让 8 个状态同时进入 Python/vLLM，从而利用 vLLM 的连续批处理能力。
# 该值限制并发 socket/线程数量，不限制等待队列长度。
export NLM_LLM_HTTP_WORKERS="${NLM_LLM_HTTP_WORKERS:-8}"
# 等待 worker 领取的 backlog 上限；0 表示不限制。它不包含正在处理的请求。
export NLM_LLM_HTTP_MAX_QUEUE="${NLM_LLM_HTTP_MAX_QUEUE:-0}"
export NLM_LLM_PROBLEM_ID="${NLM_LLM_PROBLEM_ID:-}"
# normal 会在请求在途时继续展开源状态；skip 会在源状态出队时暂时挂起，
# HTTP 返回后恢复源状态，并把合法 LLM 动作链作为额外分支加入 Open List。
# log 通信模式没有返回包，因此只记录触发，不会真正挂起状态。
export NLM_LLM_PENDING_BEHAVIOR="${NLM_LLM_PENDING_BEHAVIOR:-normal}"
# 1 表示在诊断日志中一起打印完整 (:init ...)；0 只打印元信息。
# HTTP 请求无论此值如何都会携带 init；正式测速时建议设为 0，避免重复控制台 I/O。
export NLM_LLM_EMIT_STATE="${NLM_LLM_EMIT_STATE:-1}"
# 每次检查 openlist/frontier 时观察前 K 个最优候选。
# 调大能更稳定地判断 plateau，也更适合给 LLM 组 batch；调小则开销更低、触发更敏感。
# 当前默认值偏向小问题 smoke test；正式实验可调回 64 或更大。
export NLM_LLM_FRONTIER_K="${NLM_LLM_FRONTIER_K:-16}"
# 单次触发最多发出多少个候选状态。
# 这个值可以贴近 vLLM 的并行 batch 能力；调大吞吐更好，但也可能塞入较多质量相近的请求。
export NLM_LLM_BATCH_SIZE="${NLM_LLM_BATCH_SIZE:-8}"
# 每隔多少次状态展开检查一次 frontier plateau / global stall。
# 调小响应更快但检查更频繁；调大开销更低但可能错过较早的介入点。
# 当前默认 1 是为了在小测试用例中尽快触发；正式实验建议从 50 起调。
export NLM_LLM_CHECK_INTERVAL="${NLM_LLM_CHECK_INTERVAL:-1}"
# global stall 阈值：全局最佳 h 连续多少次展开没有改善时，认为搜索可能停滞。
# 调小更容易触发 LLM；调大更保守，减少不必要请求。
# 当前默认 20 是为了不到 200 个状态的小问题也有机会触发；正式实验建议 500 起。
export NLM_LLM_STALL_EXPANSIONS="${NLM_LLM_STALL_EXPANSIONS:-20}"
# ancestor stagnation 深度：比较当前状态和多少级父节点的 h 变化。
# 如果多步动作后 h 基本不下降，就认为这条路径可能在局部打转。
# 当前默认 2 是为了 smoke test 更容易触发；正式实验可调到 4-6。
export NLM_LLM_ANCESTOR_DEPTH="${NLM_LLM_ANCESTOR_DEPTH:-2}"
# 最小深度：深度低于该值的状态不触发 ancestor stagnation。
# 用来避免刚开始搜索时信息太少就过早请求 LLM。
# 当前默认 2 是为了小问题验通；正式实验可调到 4 或更高。
export NLM_LLM_MIN_DEPTH="${NLM_LLM_MIN_DEPTH:-2}"
# 最大挂起数：HTTP + skip 模式下最多允许多少个状态等待 LLM。
# 0 表示不限制；可用它限制请求积压及被暂时移出 Open List 的状态数量。
export NLM_LLM_MAX_PENDING="${NLM_LLM_MAX_PENDING:-0}"
# h 值改善的绝对下限：用于 h 很接近 0、浮点误差、或内部 g 值陈旧检查的兜底。
# 不同问题的 h 量纲可能不同，所以不要主要依赖这个绝对值做停滞判定。
export NLM_LLM_H_EPSILON="${NLM_LLM_H_EPSILON:-0.001}"
# h 值改善的相对阈值：父链停滞和全局 best-h 改善主要看这个比例。
# 例如 0.01 表示 h 下降不超过约 1% 时，视作“几乎没改善”；调大更容易触发 LLM。
# 当前默认 0.5 很激进，只适合 smoke test；正式实验建议从 0.01-0.05 起。
export NLM_LLM_H_RELATIVE_EPSILON="${NLM_LLM_H_RELATIVE_EPSILON:-0.5}"
# frontier 前 K 个状态的 h 变异系数阈值，计算方式是 stddev(h) / mean(h)。
# 它本身就是相对量，不同问题的 h 尺度差异影响较小；越小越严格，越大越容易触发。
# 当前默认 10 几乎是强制放宽 plateau 判定，只适合确认机制能否触发。
export NLM_LLM_PLATEAU_H_CV="${NLM_LLM_PLATEAU_H_CV:-10}"
# frontier 前 K 个状态的 f=g+h 变异系数阈值，计算方式是 stddev(f) / mean(f)。
# 和 h 阈值一起使用，避免只看 h 时把代价层次差异很大的状态误判为同一平台。
# 当前默认 10 几乎是强制放宽 plateau 判定，只适合确认机制能否触发。
export NLM_LLM_PLATEAU_F_CV="${NLM_LLM_PLATEAU_F_CV:-10}"


DOMAIN="${1:-../pddl/domain.pddl}"
PROBLEM="${2:-../pddl/problem_scale_10_id_1.pddl}"
PLAN="${3:-../pddl/nlm_probe_test.plan}"
SEARCH="${SEARCH:-astar(lmcutnumeric(use_second_order_simple=true, bound_iterations=10, ceiling_less_than_one=true))}"

if ! command -v python2 >/dev/null 2>&1; then
    echo "Missing required command: python2" >&2
    echo "The NLM-CutPlan translator uses Python 2 style imports." >&2
    exit 1
fi

if [ ! -x builds/release64/bin/downward ]; then
    echo "Missing builds/release64/bin/downward." >&2
    echo "Run: bash scripts/compile_windows_source_wsl.sh" >&2
    exit 1
fi

if [ ! -f "$DOMAIN" ]; then
    echo "Missing domain file: $DOMAIN" >&2
    exit 1
fi

if [ ! -f "$PROBLEM" ]; then
    echo "Missing problem file: $PROBLEM" >&2
    exit 1
fi

echo "Running NLM-CutPlan from: $PROJECT_ROOT"
echo "Domain: $DOMAIN"
echo "Problem: $PROBLEM"
echo "Plan output: $PLAN"
echo "NLM_EAGER_PROBES=$NLM_EAGER_PROBES"
echo "NLM_EAGER_STATE_PROBE_LIMIT=$NLM_EAGER_STATE_PROBE_LIMIT"
echo "NLM_EAGER_STATE_PROBE_STRIDE=$NLM_EAGER_STATE_PROBE_STRIDE"
echo "NLM_LLM_TRIGGER=$NLM_LLM_TRIGGER"
echo "NLM_LLM_COMM_MODE=$NLM_LLM_COMM_MODE"
echo "NLM_LLM_HTTP_HOST=$NLM_LLM_HTTP_HOST"
echo "NLM_LLM_HTTP_PORT=$NLM_LLM_HTTP_PORT"
echo "NLM_LLM_HTTP_PATH=$NLM_LLM_HTTP_PATH"
echo "NLM_LLM_HTTP_TIMEOUT_MS=$NLM_LLM_HTTP_TIMEOUT_MS"
echo "NLM_LLM_HTTP_WORKERS=$NLM_LLM_HTTP_WORKERS"
echo "NLM_LLM_HTTP_MAX_QUEUE=$NLM_LLM_HTTP_MAX_QUEUE"
echo "NLM_LLM_PROBLEM_ID=$NLM_LLM_PROBLEM_ID"
echo "NLM_LLM_PENDING_BEHAVIOR=$NLM_LLM_PENDING_BEHAVIOR"
echo "NLM_LLM_EMIT_STATE=$NLM_LLM_EMIT_STATE"
echo "NLM_LLM_FRONTIER_K=$NLM_LLM_FRONTIER_K"
echo "NLM_LLM_BATCH_SIZE=$NLM_LLM_BATCH_SIZE"
echo "NLM_LLM_CHECK_INTERVAL=$NLM_LLM_CHECK_INTERVAL"
echo "NLM_LLM_STALL_EXPANSIONS=$NLM_LLM_STALL_EXPANSIONS"
echo "NLM_LLM_ANCESTOR_DEPTH=$NLM_LLM_ANCESTOR_DEPTH"
echo "NLM_LLM_MIN_DEPTH=$NLM_LLM_MIN_DEPTH"
echo "NLM_LLM_MAX_PENDING=$NLM_LLM_MAX_PENDING"
echo "NLM_LLM_H_EPSILON=$NLM_LLM_H_EPSILON"
echo "NLM_LLM_H_RELATIVE_EPSILON=$NLM_LLM_H_RELATIVE_EPSILON"
echo "NLM_LLM_PLATEAU_H_CV=$NLM_LLM_PLATEAU_H_CV"
echo "NLM_LLM_PLATEAU_F_CV=$NLM_LLM_PLATEAU_F_CV"

python2 fast-downward.py --build release64 \
    --plan-file "$PLAN" \
    "$DOMAIN" \
    "$PROBLEM" \
    --search "$SEARCH"
