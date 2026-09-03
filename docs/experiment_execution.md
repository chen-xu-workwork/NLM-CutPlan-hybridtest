# Anytime experiment execution

The experiment runner writes the same core artifact layout as the completed
Count-Downward pilot runs. Every job has an independent plan prefix, planner
log, structured anytime tables, and Fast Downward working directory. This is
required because the legacy driver always names its intermediate files
`output.sas` and `output`.

## Modes

- `off` is the native baseline. It does not construct prompts, initialize a live
  model runtime, start an HTTP bridge, or execute trigger analysis. The master
  trigger, all three trigger switches, and the initial-state request switch are
  explicitly set to zero in the child environment.
- `live` enables the existing online hybrid path and all three trigger
  switches. Existing trigger thresholds are not changed by the batch runner.

For timing comparisons, run baseline and live as separate homogeneous batches.
Parallel jobs compete for CPU and memory, so use the same parallelism and
machine allocation when comparing batches. Do not run the paired baseline and
live job for one problem at the same time if the objective is isolated
single-process performance.

## Manifest

Paths are resolved relative to the manifest file. A manifest may still mix
modes for smoke tests:

```json
{
  "domain": "../pddl/domain.pddl",
  "problem_dir": "../pddl/problems",
  "default_mode": "off",
  "time_limit_seconds": 7200,
  "jobs": [
    {"id": "001_problem_1_off", "problem": "problem_1.pddl"},
    {"id": "002_problem_2_off", "problem": "problem_2.pddl"}
  ]
}
```

Baseline example:

```bash
python3 -m hybrid_planner.batch_console \
  --manifest experiments/baseline.json \
  --output-dir experiments_results/baseline \
  --parallelism 4 \
  --overall-memory-limit 16G \
  --resume
```

Live example using one already-running vLLM service shared by all planner jobs:

```bash
python3 -m hybrid_planner.batch_console \
  --manifest experiments/live.json \
  --output-dir experiments_results/live \
  --parallelism 4 \
  --overall-memory-limit 16G \
  --external-vllm \
  --vllm-base-url http://127.0.0.1:8091/v1 \
  --llm-model Qwen3.5-9B \
  --llm-seed 0 \
  --resume
```

Omit `--external-vllm` and provide `--vllm-model-path` when the batch runner
should own the service. It starts the model once for the whole batch rather than
once per problem.

## Job artifacts

Each `<output-dir>/<job-id>/` contains:

- `job.json`: immutable job identity and exact child command.
- `job_result.json`: atomic completion marker used by `--resume`.
- `console.log`: the complete Python controller and planner output.
- `sas_plan.N`: anytime incumbent plans.
- `planner-work/output.sas` and `planner-work/output`: private translator and
  preprocessor outputs, never shared by concurrent jobs.
- `anytime/run.json`: input hashes, source revision, effective environment,
  memory/time limits, model generation settings, seed, timing, and totals.
- `anytime/phases.csv`: phase-level search, trigger, response, injection,
  memory, and cumulative counters.
- `anytime/incumbents.csv`: the anytime cost/length curve and cumulative work at
  every saved plan.
- `anytime/llm_requests.csv`: one row per requested search state.
- `anytime/llm_samples.csv`: one row per independently generated and validated
  sample, including its derived seed and latency.
- `anytime/llm_trigger_reasons.csv`: per-phase attempts, submissions, and
  responses grouped by trigger reason.
- `anytime/planner.log`: raw planner output retained for later parsing.

The JSON and CSV snapshots are written through atomic replacement while a run
is active. The raw planner log remains line-buffered, so interrupted jobs still
retain the last emitted diagnostic lines even if no final completion marker was
written.

## AutoDL pilot launcher

The repository includes a container-specific tmux launcher with the current
model and dataset paths already configured:

```bash
cd /root/autodl-tmp/NLM-CutPlan-hybridtest
bash scripts/run_autodl_anytime_batch.sh
```

Its default is the scale-30 live pilot: the first two deterministically sorted
problems, two parallel planners, and a 600-second anytime limit per problem.
The command prints the tmux session and output directory before returning.

The same entry point covers later experiment variants:

```bash
# Near-native baseline for the same two problems.
MODE=off bash scripts/run_autodl_anytime_batch.sh

# All scale-30 problems.
LIMIT=0 bash scripts/run_autodl_anytime_batch.sh

# Every problem scale, running in the foreground.
SCALE=all LIMIT=0 DETACH=0 bash scripts/run_autodl_anytime_batch.sh
```

Set `OUTPUT_DIR` to an existing batch directory to use the launcher's built-in
`--resume` behavior. `MEMORY_LIMIT=16G` optionally applies a per-planner Fast
Downward memory limit; it is unset in the pilot by default.
