# Team 1 -- Numeric Fast Downward

## Hybrid LLM integration

For a reviewer-oriented description of the changes to the original NLM,
including state export, trigger semantics, asynchronous communication, Open
List injection, and the trajectory scorer boundary, see
[`docs/nlm_modification_review.md`](docs/nlm_modification_review.md).

The online planner runtime is intentionally independent from the PyPACE
training repository. The Python control plane is grouped by runtime role:

- `hybrid_planner/llm/`: vLLM lifecycle and concurrent model requests.
- `hybrid_planner/prompting/`: prompt construction, templates, PDDL
  translation, and domain resources.
- `hybrid_planner/validation/`: model-output parsing and legal-prefix
  validation.
- `hybrid_planner/console.py`: stable entry point coordinating the complete
  workflow.

Install the runtime dependencies in the Python 3 environment used by WSL or
the Linux GPU container:

```bash
python3 -m pip install -r requirements/hybrid.txt
```

### Linux GPU container: initial-state live test

The preserved cloud-container smoke test starts a local vLLM server from a
model directory and deliberately sends only the planner's initial state. Use it
as a focused communication/injection regression; the formal three-trigger
search entry point is described under **Live model** below.

The Linux environment needs:

- the `vllm` command in the active Python 3 environment;
- the packages in `requirements/hybrid.txt`;
- a separate `python2` command for the legacy Fast Downward translator;
- CPLEX/Concert and OSI when compiling the solver.

Compile in the container after setting non-default dependency roots if needed:

```bash
export CPLEX_HOME=/opt/ibm/ILOG/CPLEX_Studio_Community222
export DOWNWARD_COIN_ROOT=/opt/osi
bash scripts/compile_linux.sh
```

Run the complete initial-state test with an absolute model path:

```bash
bash scripts/run_hybrid_initial_live_linux.sh /models/depots-checkpoint
```

The default repository layout expects `domain.pddl` and
`problem_scale_10_id_1.pddl` in the sibling `../pddl` directory. Override them
when the container uses another layout:

```bash
export NLM_DOMAIN_PATH=/workspace/pddl/domain.pddl
export NLM_PROBLEM_PATH=/workspace/pddl/problem_scale_10_id_1.pddl
export NLM_PLAN_PATH=/workspace/results/depots.plan
bash scripts/run_hybrid_initial_live_linux.sh /models/depots-checkpoint
```

By default the owned vLLM process inherits the container's existing
`CUDA_VISIBLE_DEVICES`; this avoids replacing device assignments made by the
cloud platform. For a two-GPU model shard:

```bash
export NLM_VLLM_TENSOR_PARALLEL_SIZE=2
bash scripts/run_hybrid_initial_live_linux.sh /models/depots-checkpoint
```

Set `NLM_VLLM_GPUS=0,1` only when an explicit subset is required. Other useful
overrides include `NLM_LLM_MODEL`, `NLM_VLLM_MAX_MODEL_LEN`,
`NLM_VLLM_GPU_MEMORY_UTILIZATION`, `NLM_VLLM_DTYPE`, and
`NLM_LLM_TIMEOUT`. Additional console options can follow the model path, for
example `--vllm-extra-arg=--enable-prefix-caching`.

This script forces `NLM_LLM_REQUEST_INITIAL=1` and disables frontier, global
stall, and ancestor triggers. It starts vLLM, waits for `/v1/models`, starts the
planner, sends state `#0`, parses the real completion, validates the longest
legal prefix, and injects it into the Open List. Runtime artifacts are written
to:

```text
logs/vllm-live-initial.log
logs/live_initial_debug/*.json
```

If the model emits a complete correct plan, validation stops at the first
state satisfying the goal. The response has `status=ok` and
`goal_reached=true`; only the goal-reaching prefix is returned to C++, so any
trailing model actions cannot move the plan away from the goal.

Expected milestone logs are:

```text
[NLM-PY-CONSOLE] vLLM ready models=...
[NLM-LLM-BRIDGE] submitted ... reason=initial_replay_test
[NLM-PY-CONSOLE] model request finished ... status=ok
[NLM-LLM-INJECT] chain ... applied_actions=...
Solution found!
```

### Live model

Set the trained checkpoint and launch the complete workflow:

```bash
export NLM_VLLM_MODEL_PATH=/path/to/Qwen3.5-9B/checkpoint
bash scripts/run_hybrid_live_wsl.sh
```

Inside a Linux GPU container, use the equivalent formal-search entry point:

```bash
bash scripts/run_hybrid_live_linux.sh /models/depots-checkpoint
```

Both formal entry points leave the classical search active (`pending=normal`),
disable the special initial-state request by default, and enable the three
reserved online triggers. Trigger thresholds remain configurable through the
existing `NLM_LLM_*` environment variables.

The online main line is satisficing rather than optimal. It defaults to eager
greedy best-first search with numeric LM-cut:

```text
eager_greedy([lmcutnumeric(use_second_order_simple=true,
                           bound_iterations=10,
                           ceiling_less_than_one=true)])
```

The Open List is ordered by remaining `h`, and the first legally reached goal
is accepted; the planner does not continue to prove minimum fuel cost. Real
action costs are still accumulated in `g` and reported with the returned plan.
This makes a legal LLM prefix useful even when it is not cost-optimal. For an
explicit search-policy ablation, pass `--search ...` to the Python console or
set `SEARCH=...` when using `scripts/run_nlm_search_wsl.sh`.

The controller performs these steps:

1. Starts `vllm serve` with the configured checkpoint.
2. Waits until `/v1/models` exposes `Qwen3.5-9B`.
3. Starts one shared asynchronous `aiohttp` connection pool.
4. Starts the local C++ bridge and then the planner process.
5. Builds one prompt for each requested state and launches three independent
   generations concurrently by default.
6. Parses each sample's `action_Xxx(...)` calls and validates its longest legal
   prefix independently.
7. Returns every resulting PDDL-style legal action chain to the C++ bridge.
8. The search thread applies each chain from the requested state, evaluates all
   new states, and inserts them into the normal Open List. Existing closed or
   duplicate states are handled by the normal search registry.

A successful or partially successful response uses this compact contract:

```json
{
  "type": "llm_response",
  "request_id": "123-7",
  "state_id": 123,
  "status": "partial",
  "sample_count": 3,
  "usable_sample_count": 2,
  "action_chains": [
    [
      "(lift hoist0 crate1 pallet0 depot0)",
      "(load hoist0 crate1 truck0 depot0)"
    ],
    [
      "(lift hoist0 crate2 pallet1 depot0)"
    ],
    []
  ],
  "actions": [
    "(lift hoist0 crate1 pallet0 depot0)",
    "(load hoist0 crate1 truck0 depot0)"
  ]
}
```

`action_chains` is the authoritative multi-sample field. The legacy `actions`
field mirrors the first usable sample for compatibility. A failed or unusable
sample contributes an empty chain and does not prevent other samples from being
inserted. In the formal `normal` pending mode, the source state is never held
back while waiting: a legal LLM prefix arrives later as an additional branch,
while the original source and its ordinary successors remain available to
classical search. This keeps behavior independent of response latency and
preserves the classical fallback path.

To connect to a vLLM process managed elsewhere:

```bash
python3 -m hybrid_planner.console \
    --llm-mode live \
    --external-vllm \
    --vllm-base-url http://127.0.0.1:8091/v1 \
    --llm-model Qwen3.5-9B
```

Important runtime controls include `--llm-max-concurrency` (default `100`),
`--llm-samples-per-state` (default `3`),
`--llm-timeout` (a total per-state budget, default `300` seconds),
`--llm-max-tokens`,
`--vllm-gpus`, and `--vllm-max-model-len`. Model concurrency is independent
from the CPU-side `--prompt-workers` and `--validation-workers` limits, both
defaulting to `4`. The controller also defaults
`NLM_LLM_MAX_PENDING` to the smaller of the HTTP worker count and the model
concurrency divided by the sample count (33 states for the default 100/3),
preventing stale state requests from accumulating behind the shared pool. Run
`python3 -m hybrid_planner.console --help` for the complete list.

The bundled domain reference matches the depots model. A model trained for a
different domain must provide its matching manual with
`--prompt-domain-code`.

### Deterministic replay

After rebuilding the C++ solver, verify the complete bridge and Open List
injection path without starting vLLM:

```bash
bash scripts/run_hybrid_replay_wsl.sh
```

Replay mode still builds the real prompt, parses model-style
`action_Xxx(...)` calls, and runs Unified Planning validation. The bundled
depots fixture requests the initial state and supplies a deterministic
`Lift -> Load` prefix three times. Expected C++ logs include
`reason=initial_replay_test`, `[NLM-LLM-INJECT] action=`, and
`[NLM-LLM-INJECT] chain`.

For another saved model response:

```bash
python3 -m hybrid_planner.console \
    --llm-mode replay \
    --replay-model-output /path/to/model_output.txt
```

### Mock and diagnostics

Mock mode builds prompts without requiring a model:

```bash
python3 -m hybrid_planner.console --llm-mode mock
```

Use `--prompt-debug-dir prompt_debug` to save the incoming `:init`, prompts,
raw model output, and legal-prefix result for each request.

The final lightweight aggregates use the `[NLM-LLM-TRIGGER-STATS]` and
`[NLM-LLM-TRIGGER-REASON-STATS]` prefixes, including trigger-specific attempt,
submission, and response counts. Sparse `[NLM-LLM-MONITOR]` heartbeats include
the number of tracked/growing/ready h buckets, the highest-pressure bucket's
recent arrival/removal counts, best-h stall age, request count, pending count,
and peak resident memory. They are flushed every 100,000 expansions by default,
so a killed search normally leaves enough information to distinguish trigger
behavior from memory pressure. Set
`NLM_LLM_HEARTBEAT_INTERVAL=0` to disable them or increase the interval to
reduce log volume further. `NLM_LLM_LOG_RESPONSE_BODY=1`,
`NLM_LLM_EMIT_STATE=1`, `--print-prompts`, and `--prompt-debug-dir` provide
progressively heavier diagnostics and should remain off in timing runs.

### Trigger calibration without an LLM

Tune trigger frequency on complete original problems, not on the segmented
intermediate-state training tasks. The latter remain useful for isolated model
and prefix-validation evaluation, while only a complete problem reproduces the
frontier and stagnation history seen by end-to-end search.

The current `nlm-container-transfer/depots-numeric-segmented` bundle contains
the segmented tasks. It can be used for those diagnostic experiments, but it is
not a substitute for a complete-problem end-to-end benchmark. Use the matching
original problems under `generated-pddl/depots-numeric-new`; create a separate
preprocessed bundle of those originals later if batch startup cost matters.

Run ordinary search with log-only trigger submission:

```bash
bash scripts/run_trigger_calibration.sh \
    /data/depots-numeric-new/domain.pddl \
    /data/depots-numeric-new/problems/problem_scale_40_id_1.pddl
```

No Python controller or model is started. Read `submitted=<N>` from the final
`[NLM-LLM-TRIGGER-STATS]` line as the number of distinct state-level LLM
opportunities. `request_attempts` may include duplicate or rejected candidates,
and plateau/stall event counts describe condition checks rather than requests.
The calibration runner forces `NLM_LLM_MAX_REQUESTS=0` so this count is not
truncated.

Also inspect `first_request_expansion`, `last_request_expansion`, and
`avg_request_gap_expansions`. A configuration that produces roughly 20 requests
but emits most of them in one early batch is not a good ten-opportunity online
policy. Every request log includes its submission expansion for finer analysis.

For the first validation calibration pass, reproducibly sample two original
problems at scale 30 and two at scale 40 from the segmented dataset's
validation split, then run those four complete originals under WSL:

```bash
python3 scripts/run_validation_trigger_calibration.py
```

The default data root is `/mnt/e/Python Projects/PyPACE/data`; override it with
`--data-root` or `PYPACE_DATA_ROOT` when necessary. The selection is written to
`selected_problems.csv`. `summary.csv` records the exact search configuration,
first-solution status/time, plan length and cost, solver evaluated/expanded
counts, trigger totals and their early/middle/late distribution. `requests.csv`
records every request's expansion
number, normalized search position, trigger reason, and phase. Use
`--list-only` to inspect the reproducible selection without starting the solver.
The defaults are `--scales 30 40 --samples-per-scale 2 --jobs 1`.
Problem logs are line-buffered, and the two CSV files are atomically refreshed
after every completed problem. If final trigger statistics are absent because a
solver was killed, the summary falls back to the last flushed heartbeat.

`--jobs 4` is supported but deliberately not the default. A single scale-30/40
search can retain tens of millions of native search states and use many GB of
RAM; four concurrent processes multiply that base solver memory even though the
trigger monitor itself is bounded. Measure `peak_memory_kb` for one problem and
compare four times that value with WSL's configured RAM plus safety margin
before enabling four-way parallelism.

```bash
python3 scripts/run_validation_trigger_calibration.py --jobs 4
```

The calibration runner forces `NLM_LLM_MAX_REQUESTS=0`, even if a live-run value
is still present in the shell environment. Its conservative first-pass baseline
updates a fixed 16-entry table of h buckets and analyzes it every 8,192
expansions. Four rolling windows are retained, so interleaved h values do not
reset each other's evidence. A bucket needs at least 4,096 recent expansions,
absolute net growth of 1,024, ratio `opened / expanded >= 1.05`, and two
consecutive positive analyses before becoming eligible. Four calm windows end
an episode. All triggers share a 100,000-expansion request gap; the same h has a
500,000-expansion cooldown and must accumulate at least 8,192 new expansions
and 2,048 net growth after its previous request. There is no per-layer lifetime
or per-episode request cap: a persistently congested h can request again after
every cooldown, while the problem-wide live-run budget still bounds total model
use. The three most active eligible buckets are ranked, but the default submits
only one state per shared slot. Global stall waits for
500,000 expansions without meaningful best-h improvement. Ancestor stagnation
is sampled every 100,000 expansions, compares at most ten ancestors, and yields
to a mature frontier bucket. These are starting values for uncapped
measurement, not final tuned constants.

Calibration uses the same eager-greedy satisficing search policy as formal
online runs. Old A* calibration logs remain useful for diagnosing the former
trigger implementation, but their expansion rates and request frequencies are
not directly comparable with the new main line.

Override calibration values from the WSL command line when iterating; these
environment-only changes do not require rebuilding the planner:

```bash
NLM_LLM_ANALYSIS_INTERVAL=16384 \
NLM_LLM_ACTIVITY_WINDOWS=4 \
NLM_LLM_GROWTH_CONFIRM_WINDOWS=2 \
NLM_LLM_LAYER_MIN_RECENT_EXPANDED=8192 \
NLM_LLM_LAYER_MIN_RECENT_NET_GROWTH=2048 \
NLM_LLM_LAYER_MIN_SINCE_REQUEST_EXPANDED=16384 \
NLM_LLM_LAYER_MIN_SINCE_REQUEST_NET_GROWTH=4096 \
NLM_LLM_PLATEAU_GROWTH_RATIO=1.10 \
NLM_LLM_MIN_REQUEST_GAP_EXPANSIONS=200000 \
NLM_LLM_PER_LAYER_REQUEST_GAP_EXPANSIONS=1000000 \
NLM_LLM_ANCESTOR_CHECK_INTERVAL=200000 \
NLM_LLM_STALL_EXPANSIONS=1000000 \
python3 scripts/run_validation_trigger_calibration.py
```

Edit `scripts/run_trigger_calibration.sh` only when changing the persistent
calibration defaults. Numeric overrides in that shell script or in
`hybrid_planner/console.py` take effect on the next process start; only changes
to `eager_search.cc` or other C++ trigger logic require recompilation.

Formal live runs default to `NLM_LLM_MAX_REQUESTS=10`. This is a state-request
budget: with the default three samples per state it permits at most 30 model
generations for one planning problem. Set it explicitly to change or disable
the cap (`0` means unlimited). A shared trigger slot submits one state by
default; `NLM_LLM_REQUESTS_PER_SLOT` can explicitly enable a small batch from
the ranked candidate layers. All trigger types obey the common request gap.

The three rules can be isolated with
`NLM_LLM_ENABLE_ANCESTOR_STAGNATION`,
`NLM_LLM_ENABLE_FRONTIER_PLATEAU`, and
`NLM_LLM_ENABLE_GLOBAL_STALL`. Each defaults to `1`; set a variable to `0` to
disable that rule during calibration or ablation.

### WSL helpers

Build and direct search helpers live in `scripts/`:

```bash
bash scripts/compile_windows_source_wsl.sh
bash scripts/run_hybrid_replay_wsl.sh
bash scripts/run_nlm_search_wsl.sh DOMAIN.pddl PROBLEM.pddl [PLAN]
```

Pure Linux helpers are:

```bash
bash scripts/compile_linux.sh
bash scripts/run_hybrid_live_linux.sh /absolute/path/to/model
bash scripts/run_hybrid_initial_live_linux.sh /absolute/path/to/model
```

## Compile

Install CPLEX Optimization Studio 22.1.1 in `/opt/ibm/ILOG/CPLEX_Studio2211`.
Then, run the script with the root privilege.

```bash
./compile
```

## Run

### Optimal LNP

```bash
./run-opt-lnp-1 domain.pddl problem.pddl plan
```

```bash
./run-opt-lnp-2 domain.pddl problem.pddl plan
```

### Optimal SNP

```bash
./run-opt-snp-1 domain.pddl problem.pddl plan
```

```bash
./run-opt-snp-2 domain.pddl problem.pddl plan
```

### Satisficing/Agile LNP

```bash
./run-sat-lnp-1 domain.pddl problem.pddl plan
```

### Satisficing/Agile SNP

```bash
./run-sat-snp-1 domain.pddl problem.pddl plan
```

```bash
./run-sat-snp-2 domain.pddl problem.pddl plan
```
