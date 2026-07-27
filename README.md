# Team 1 -- Numeric Fast Downward

## Hybrid LLM integration

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

The cloud-container smoke test starts a local vLLM server from a model
directory and deliberately sends only the planner's initial state. This is the
appropriate entry point for the current checkpoint, which was not trained on
predicates that appear only in intermediate search states.

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
bash scripts/run_hybrid_live_linux.sh /models/depots-checkpoint
```

The default repository layout expects `domain.pddl` and
`problem_scale_10_id_1.pddl` in the sibling `../pddl` directory. Override them
when the container uses another layout:

```bash
export NLM_DOMAIN_PATH=/workspace/pddl/domain.pddl
export NLM_PROBLEM_PATH=/workspace/pddl/problem_scale_10_id_1.pddl
export NLM_PLAN_PATH=/workspace/results/depots.plan
bash scripts/run_hybrid_live_linux.sh /models/depots-checkpoint
```

By default the owned vLLM process inherits the container's existing
`CUDA_VISIBLE_DEVICES`; this avoids replacing device assignments made by the
cloud platform. For a two-GPU model shard:

```bash
export NLM_VLLM_TENSOR_PARALLEL_SIZE=2
bash scripts/run_hybrid_live_linux.sh /models/depots-checkpoint
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

The controller performs these steps:

1. Starts `vllm serve` with the configured checkpoint.
2. Waits until `/v1/models` exposes `Qwen3.5-9B`.
3. Starts one shared asynchronous `aiohttp` connection pool.
4. Starts the local C++ bridge and then the planner process.
5. Builds prompts for requested states, generates plans concurrently, parses
   `action_Xxx(...)` calls, and validates the longest legal prefix.
6. Returns only PDDL-style legal actions to the C++ bridge.
7. The search thread applies that action chain from the requested state,
   evaluates every new state, and inserts the chain into the normal Open List.

A successful or partially successful response uses this compact contract:

```json
{
  "type": "llm_response",
  "request_id": "123-7",
  "state_id": 123,
  "status": "partial",
  "actions": [
    "(lift hoist0 crate1 pallet0 depot0)",
    "(load hoist0 crate1 truck0 depot0)"
  ],
  "generated_action_count": 4,
  "legal_action_count": 2,
  "invalid_action_index": 2
}
```

If transport, parsing, inference, or validation fails, `actions` is empty. In
`skip` pending mode, a source state that was actually popped while waiting is
resumed after any response. A legal LLM prefix is inserted as an additional
branch, while the original source remains available to classical search. This
keeps behavior independent of response latency and preserves the classical
fallback path.

To connect to a vLLM process managed elsewhere:

```bash
python3 -m hybrid_planner.console \
    --llm-mode live \
    --external-vllm \
    --vllm-base-url http://127.0.0.1:8091/v1 \
    --llm-model Qwen3.5-9B
```

Important runtime controls include `--llm-max-concurrency` (default `100`),
`--llm-timeout` (a total per-state budget, default `300` seconds),
`--llm-max-tokens`,
`--vllm-gpus`, and `--vllm-max-model-len`. Model concurrency is independent
from the CPU-side `--prompt-workers` and `--validation-workers` limits, both
defaulting to `4`. The controller also defaults
`NLM_LLM_MAX_PENDING` to the HTTP worker count, preventing requests from
accumulating beyond the configured online concurrency. Run
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
`Lift -> Load` prefix. Expected C++ logs include
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

### WSL helpers

Build and direct probe helpers live in `scripts/`:

```bash
bash scripts/compile_windows_source_wsl.sh
bash scripts/run_hybrid_replay_wsl.sh
bash scripts/run_probe_test_wsl.sh
```

Pure Linux helpers are:

```bash
bash scripts/compile_linux.sh
bash scripts/run_hybrid_live_linux.sh /absolute/path/to/model
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
