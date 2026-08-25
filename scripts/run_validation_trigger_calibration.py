#!/usr/bin/env python3
"""Run uncapped trigger calibration on sampled validation originals."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


REQUEST_PREFIX = "[NLM-LLM-TRIGGER] request "
STATS_PREFIX = "[NLM-LLM-TRIGGER-STATS]"
REASON_STATS_PREFIX = "[NLM-LLM-TRIGGER-REASON-STATS]"
HEARTBEAT_PREFIX = "[NLM-LLM-MONITOR]"
KEY_VALUE_RE = re.compile(r'(\w+)=("[^"]*"|\S+)')
PLAN_LENGTH_RE = re.compile(r"^Plan length:\s+(\d+)\s+step")
PLAN_COST_RE = re.compile(r"^Plan cost:\s+([^\s]+)")
SOLVER_EXPANDED_RE = re.compile(r"^Expanded\s+(\d+)\s+state")
SOLVER_REOPENED_RE = re.compile(r"^Reopened\s+(\d+)\s+state")
SOLVER_EVALUATED_RE = re.compile(r"^Evaluated\s+(\d+)\s+state")
SEARCH_TIME_RE = re.compile(r"^Search time:\s+([0-9.eE+-]+)s")
OUTPUT_LOCK = threading.Lock()


def parse_key_values(line: str) -> dict[str, str]:
    values = {}
    for key, value in KEY_VALUE_RE.findall(line):
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        values[key] = value
    return values


def int_value(values: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(values.get(key, default))
    except (TypeError, ValueError):
        return default


def phase_for(expansion: int, total_expansions: int) -> tuple[str, float]:
    if total_expansions <= 0:
        return "unknown", 0.0
    fraction = max(0.0, min(1.0, expansion / total_expansions))
    if fraction < 1.0 / 3.0:
        return "early", fraction
    if fraction < 2.0 / 3.0:
        return "middle", fraction
    return "late", fraction


def load_validation_rows(manifest: Path) -> dict[int, list[dict]]:
    by_scale: dict[int, list[dict]] = defaultdict(list)
    with manifest.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {manifest}:{line_number}: {exc}"
                ) from exc
            if row.get("split") == "validation":
                by_scale[int(row["scale"])].append(row)
    for rows in by_scale.values():
        rows.sort(key=lambda row: int(row["source_problem_id"]))
    return dict(by_scale)


def select_rows(
    by_scale: dict[int, list[dict]],
    scales: list[int],
    samples_per_scale: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    selected = []
    for scale in sorted(set(scales)):
        rows = by_scale.get(scale, [])
        if len(rows) < samples_per_scale:
            raise ValueError(
                f"Scale {scale} has only {len(rows)} validation originals; "
                f"cannot select {samples_per_scale}."
            )
        chosen = rng.sample(rows, samples_per_scale)
        selected.extend(
            sorted(chosen, key=lambda row: int(row["source_problem_id"]))
        )
    return selected


def write_selection(path: Path, selected: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["scale", "source_problem_id", "source_problem"],
        )
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "scale": row["scale"],
                    "source_problem_id": row["source_problem_id"],
                    "source_problem": row["source_problem"],
                }
            )


def run_problem(
    runner: Path,
    domain: Path,
    problem: Path,
    plan: Path,
    log_path: Path,
    stream_output: bool,
) -> int:
    command = ["bash", str(runner), str(domain), str(problem), str(plan)]
    with log_path.open(
        "w", encoding="utf-8", newline="", buffering=1
    ) as log_stream:
        process = subprocess.Popen(
            command,
            cwd=runner.parent.parent,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_stream.write(line)
            log_stream.flush()
            important = line.startswith(
                (
                    REQUEST_PREFIX,
                    STATS_PREFIX,
                    REASON_STATS_PREFIX,
                    HEARTBEAT_PREFIX,
                )
            ) or "Terminated" in line
            if stream_output or important:
                with OUTPUT_LOCK:
                    sys.stdout.write(f"[{problem.stem}] {line}")
                    sys.stdout.flush()
        return process.wait()


def analyze_run(
    row: dict,
    problem: Path,
    log_path: Path,
    return_code: int,
) -> tuple[dict, list[dict]]:
    requests = []
    final_stats: dict[str, str] = {}
    reason_stats: dict[str, int] = {}
    last_heartbeat: dict[str, str] = {}
    solution_found = False
    plan_length = ""
    plan_cost = ""
    solver_expanded = ""
    solver_reopened = ""
    solver_evaluated = ""
    search_time_seconds = ""
    search_config = ""
    with log_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith(REQUEST_PREFIX):
                requests.append(parse_key_values(line))
            elif line.startswith(STATS_PREFIX):
                final_stats = parse_key_values(line)
            elif line.startswith(REASON_STATS_PREFIX):
                values = parse_key_values(line)
                reason_stats[values.get("reason", "unknown")] = int_value(
                    values, "submitted"
                )
            elif line.startswith(HEARTBEAT_PREFIX):
                last_heartbeat = parse_key_values(line)
            elif line.startswith("Solution found"):
                solution_found = True
            elif line.startswith("Search: "):
                search_config = line[len("Search: ") :].strip()
            else:
                match = PLAN_LENGTH_RE.match(line)
                if match:
                    plan_length = match.group(1)
                    continue
                match = PLAN_COST_RE.match(line)
                if match:
                    plan_cost = match.group(1)
                    continue
                match = SOLVER_EXPANDED_RE.match(line)
                if match:
                    solver_expanded = match.group(1)
                    continue
                match = SOLVER_REOPENED_RE.match(line)
                if match:
                    solver_reopened = match.group(1)
                    continue
                match = SOLVER_EVALUATED_RE.match(line)
                if match:
                    solver_evaluated = match.group(1)
                    continue
                match = SEARCH_TIME_RE.match(line)
                if match:
                    search_time_seconds = match.group(1)

    total_expansions = int_value(
        final_stats,
        "expansions",
        int_value(last_heartbeat, "expansions"),
    )
    if total_expansions <= 0 and requests:
        total_expansions = max(
            int_value(request, "expansions") for request in requests
        )
    if not reason_stats:
        for request in requests:
            reason = request.get("reason", "unknown")
            reason_stats[reason] = reason_stats.get(reason, 0) + 1
    request_expansions = [
        int_value(request, "expansions") for request in requests
    ]
    fallback_first = request_expansions[0] if request_expansions else ""
    fallback_last = request_expansions[-1] if request_expansions else ""
    fallback_average_gap = ""
    if len(request_expansions) > 1:
        fallback_average_gap = (
            request_expansions[-1] - request_expansions[0]
        ) / (len(request_expansions) - 1)
    phase_counts = {"early": 0, "middle": 0, "late": 0, "unknown": 0}
    request_rows = []
    for ordinal, request in enumerate(requests, start=1):
        expansion = int_value(request, "expansions")
        phase, fraction = phase_for(expansion, total_expansions)
        phase_counts[phase] += 1
        request_rows.append(
            {
                "scale": row["scale"],
                "source_problem_id": row["source_problem_id"],
                "problem": problem.name,
                "request_ordinal": ordinal,
                "request_id": request.get("request_id", ""),
                "state": request.get("state", ""),
                "reason": request.get("reason", ""),
                "g": request.get("g", ""),
                "h": request.get("h", ""),
                "expansion": expansion,
                "expansion_fraction": f"{fraction:.6f}",
                "phase": phase,
            }
        )

    summary = {
        "scale": row["scale"],
        "source_problem_id": row["source_problem_id"],
        "problem": problem.name,
        "return_code": return_code,
        "stats_complete": int(bool(final_stats)),
        "search_config": search_config,
        "solution_found": int(solution_found),
        "search_time_seconds": search_time_seconds,
        "plan_length": plan_length,
        "plan_cost": plan_cost,
        "solver_expanded": solver_expanded,
        "solver_reopened": solver_reopened,
        "solver_evaluated": solver_evaluated,
        "expansions": total_expansions,
        "submitted": int_value(final_stats, "submitted", len(requests)),
        "analysis_checks": final_stats.get("analysis_checks", ""),
        "plateau_events": final_stats.get("plateau_events", ""),
        "layer_episode_resets": final_stats.get(
            "layer_episode_resets", ""
        ),
        "layer_table_evictions": final_stats.get(
            "layer_table_evictions",
            last_heartbeat.get("layer_table_evictions", ""),
        ),
        "layer_requests": final_stats.get("layer_requests", ""),
        "global_stall_events": final_stats.get(
            "global_stall_events", ""
        ),
        "ancestor_checks": final_stats.get("ancestor_checks", ""),
        "ancestor_deferrals": final_stats.get("ancestor_deferrals", ""),
        "ancestor_events": final_stats.get("ancestor_events", ""),
        "early_requests": phase_counts["early"],
        "middle_requests": phase_counts["middle"],
        "late_requests": phase_counts["late"],
        "unknown_phase_requests": phase_counts["unknown"],
        "first_request_expansion": final_stats.get(
            "first_request_expansion", fallback_first
        ),
        "last_request_expansion": final_stats.get(
            "last_request_expansion", fallback_last
        ),
        "avg_request_gap_expansions": final_stats.get(
            "avg_request_gap_expansions", fallback_average_gap
        ),
        "submitted_by_reason": json.dumps(
            reason_stats, ensure_ascii=False, sort_keys=True
        ),
        "last_heartbeat_expansion": last_heartbeat.get("expansions", ""),
        "last_heartbeat_state": last_heartbeat.get("state", ""),
        "peak_memory_kb": final_stats.get(
            "peak_memory_kb", last_heartbeat.get("peak_memory_kb", "")
        ),
        "log": str(log_path),
    }
    return summary, request_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    default_data_root = os.environ.get(
        "PYPACE_DATA_ROOT", "/mnt/e/Python Projects/PyPACE/data"
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(
        description=(
            "Select validation originals per scale and run uncapped, log-only "
            "trigger calibration under WSL."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path(default_data_root))
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[30, 40],
        help="problem scales to sample (default: 30 40)",
    )
    parser.add_argument("--samples-per-scale", type=int, default=2)
    parser.add_argument("--selection-seed", type=int, default=523)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "number of solver processes; default 1 because scale-30/40 "
            "searches can each consume many GB"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "logs" / f"trigger-calibration-{timestamp}",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="write and display the selection without running the solver",
    )
    parser.add_argument(
        "--stream-output",
        action="store_true",
        help="mirror every solver line to the terminal (all lines are logged regardless)",
    )
    return parser


def main() -> int:
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    args = build_parser(project_root).parse_args()
    if args.samples_per_scale < 1:
        raise SystemExit("--samples-per-scale must be at least 1")
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    if not args.scales:
        raise SystemExit("--scales must contain at least one scale")
    if args.jobs > 1:
        print(
            "WARNING: concurrent scale-30/40 searches multiply solver RAM "
            "usage. Measure one run first; --jobs=4 is not a safe default.",
            file=sys.stderr,
            flush=True,
        )

    generated_root = args.data_root / "generated-pddl"
    original_root = generated_root / "depots-numeric-new"
    manifest = (
        generated_root
        / "depots-numeric-segmented"
        / "source_split_manifest.jsonl"
    )
    domain = original_root / "domain.pddl"
    runner = script_path.parent / "run_trigger_calibration.sh"
    for required in (manifest, domain, runner):
        if not required.is_file():
            raise SystemExit(f"Missing required file: {required}")

    selected = select_rows(
        load_validation_rows(manifest),
        args.scales,
        args.samples_per_scale,
        args.selection_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.output_dir / "selected_problems.csv"
    write_selection(selection_path, selected)

    print(f"Selection seed: {args.selection_seed}")
    print(f"Selected problems: {selection_path}")
    for row in selected:
        print(
            f"  scale={row['scale']} id={row['source_problem_id']} "
            f"problem={row['source_problem']}"
        )
    if args.list_only:
        return 0

    logs_dir = args.output_dir / "problem-logs"
    plans_dir = args.output_dir / "plans"
    logs_dir.mkdir(exist_ok=True)
    plans_dir.mkdir(exist_ok=True)
    summaries = []
    all_requests = []
    summary_fields = [
        "scale",
        "source_problem_id",
        "problem",
        "return_code",
        "stats_complete",
        "search_config",
        "solution_found",
        "search_time_seconds",
        "plan_length",
        "plan_cost",
        "solver_expanded",
        "solver_reopened",
        "solver_evaluated",
        "expansions",
        "submitted",
        "analysis_checks",
        "plateau_events",
        "layer_episode_resets",
        "layer_table_evictions",
        "layer_requests",
        "global_stall_events",
        "ancestor_checks",
        "ancestor_deferrals",
        "ancestor_events",
        "early_requests",
        "middle_requests",
        "late_requests",
        "unknown_phase_requests",
        "first_request_expansion",
        "last_request_expansion",
        "avg_request_gap_expansions",
        "submitted_by_reason",
        "last_heartbeat_expansion",
        "last_heartbeat_state",
        "peak_memory_kb",
        "log",
    ]
    request_fields = [
        "scale",
        "source_problem_id",
        "problem",
        "request_ordinal",
        "request_id",
        "state",
        "reason",
        "g",
        "h",
        "expansion",
        "expansion_fraction",
        "phase",
    ]
    summary_path = args.output_dir / "summary.csv"
    requests_path = args.output_dir / "requests.csv"

    work_items = []
    for index, row in enumerate(selected, start=1):
        problem = original_root / row["source_problem"]
        if not problem.is_file():
            raise SystemExit(f"Missing original problem: {problem}")
        stem = problem.stem
        log_path = logs_dir / f"{stem}.log"
        plan_path = plans_dir / f"{stem}.plan"
        work_items.append(
            (index, row, problem, plan_path, log_path)
        )

    def execute(item):
        index, row, problem, plan_path, log_path = item
        with OUTPUT_LOCK:
            print(
                f"\n[{index}/{len(selected)}] Calibrating "
                f"scale={row['scale']} id={row['source_problem_id']}",
                flush=True,
            )
        return_code = run_problem(
            runner,
            domain,
            problem,
            plan_path,
            log_path,
            args.stream_output,
        )
        summary, request_rows = analyze_run(
            row, problem, log_path, return_code
        )
        return summary, request_rows

    def persist_result(summary: dict, request_rows: list[dict]) -> None:
        summaries.append(summary)
        all_requests.extend(request_rows)
        summaries.sort(
            key=lambda value: (
                int(value["scale"]), int(value["source_problem_id"])
            )
        )
        all_requests.sort(
            key=lambda value: (
                int(value["scale"]),
                int(value["source_problem_id"]),
                int(value["request_ordinal"]),
            )
        )
        # Persist after every completed problem. If a later solver is killed,
        # completed summaries and all line-flushed problem logs remain usable.
        write_csv(summary_path, summaries, summary_fields)
        write_csv(requests_path, all_requests, request_fields)

    if args.jobs == 1:
        for item in work_items:
            persist_result(*execute(item))
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(execute, item) for item in work_items]
            for future in as_completed(futures):
                persist_result(*future.result())

    print(f"\nSummary: {summary_path}")
    print(f"Per-request timing: {requests_path}")
    return 0 if all(row["return_code"] == 0 for row in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
