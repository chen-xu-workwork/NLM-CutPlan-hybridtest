"""Anytime phase lifecycle and experiment records for the hybrid planner."""

import csv
import json
import pathlib
import shlex
import threading
import time


def _parse_structured_line(line):
    """Return ``(marker, key_values)`` for one structured planner line."""

    marker_start = line.find("[NLM-")
    if marker_start < 0:
        return None, {}
    marker_end = line.find("]", marker_start)
    if marker_end < 0:
        return None, {}
    marker = line[marker_start + 1 : marker_end]
    values = {}
    try:
        tokens = shlex.split(line[marker_end + 1 :].strip())
    except ValueError:
        return marker, values
    for token in tokens:
        if "=" in token:
            key, value = token.split("=", 1)
            values[key] = value
    return marker, values


class ActiveIterationRegistry:
    """Tracks the one phase whose LLM responses may still affect search."""

    def __init__(self, run_id):
        self.run_id = str(run_id)
        self._lock = threading.Lock()
        self._active_iteration = None
        self._lifecycle_seen = False
        self._futures = {}

    def start_iteration(self, iteration):
        iteration = int(iteration)
        with self._lock:
            self._lifecycle_seen = True
            self._active_iteration = iteration

    def accept_request(self, run_id, iteration):
        """Accept the active phase, including a direct non-iterated search."""

        with self._lock:
            if str(run_id) != self.run_id:
                return False
            if self._active_iteration is None and not self._lifecycle_seen:
                self._active_iteration = int(iteration)
            return self._active_iteration == int(iteration)

    def is_active(self, run_id, iteration):
        with self._lock:
            return (
                str(run_id) == self.run_id
                and self._active_iteration == int(iteration)
            )

    def register_future(self, run_id, iteration, request_id, future):
        with self._lock:
            if (
                str(run_id) != self.run_id
                or self._active_iteration != int(iteration)
            ):
                return False
            self._futures[str(request_id)] = (int(iteration), future)
            return True

    def unregister_future(self, request_id):
        with self._lock:
            self._futures.pop(str(request_id), None)

    def end_iteration(self, iteration):
        """Invalidate a phase and request cancellation of all of its futures."""

        iteration = int(iteration)
        cancelled = []
        with self._lock:
            if self._active_iteration == iteration:
                self._active_iteration = None
            for request_id, (future_iteration, future) in list(
                self._futures.items()
            ):
                if future_iteration != iteration:
                    continue
                cancel_accepted = future.cancel()
                cancelled.append((request_id, cancel_accepted))
                self._futures.pop(request_id, None)
        return cancelled

    def close(self):
        with self._lock:
            futures = list(self._futures.items())
            self._futures.clear()
            active_iteration = self._active_iteration
            self._active_iteration = None
        cancelled = []
        for request_id, (iteration, future) in futures:
            cancelled.append((request_id, future.cancel()))
        return active_iteration, cancelled


class AnytimeRunRecorder:
    """Collect planner/LLM events and write graph-ready CSV artifacts."""

    PHASE_FIELDS = [
        "run_id", "iteration", "bound", "result", "elapsed_seconds",
        "phase_seconds", "plan_cost", "plan_length", "phase_expanded",
        "phase_evaluated", "phase_generated", "phase_reopened",
        "cumulative_expanded", "cumulative_evaluated",
        "cumulative_generated", "cumulative_reopened",
        "peak_memory_kb", "submitted", "responses", "usable_responses",
        "injected_chains", "injected_actions", "injected_states",
        "discarded_phase_end", "completed_unconsumed", "discarded_queued",
        "cancelled_inflight",
        "python_cancel_requested", "python_cancel_accepted",
        "model_generations", "completed_samples", "usable_samples",
        "stale_requests",
        "cumulative_submitted", "cumulative_model_generations",
        "cumulative_usable_samples", "cumulative_injected_states",
    ]
    INCUMBENT_FIELDS = [
        "run_id", "mode", "iteration", "incumbent", "elapsed_seconds",
        "plan_cost", "plan_length", "cumulative_expanded",
        "cumulative_evaluated", "cumulative_generated",
        "cumulative_reopened", "phase_state_requests",
        "cumulative_state_requests", "phase_model_generations",
        "cumulative_model_generations", "phase_usable_samples",
        "cumulative_usable_samples", "phase_injected_states",
        "cumulative_injected_states", "plan_file",
    ]
    REQUEST_FIELDS = [
        "run_id", "iteration", "request_id", "state_id", "state_label",
        "reason", "g", "h", "search_expansions", "received_seconds",
        "finished_seconds", "status", "sample_count",
        "usable_sample_count", "model_generations_started",
        "model_wall_seconds", "applied_actions", "inserted_states",
        "seen_previous_iteration", "error",
    ]

    def __init__(
        self,
        output_dir,
        run_id,
        mode,
        registry,
        metadata=None,
        plan_file="",
    ):
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self.mode = str(mode)
        self.registry = registry
        self.plan_file = str(plan_file)
        self.started_at = time.monotonic()
        self._lock = threading.Lock()
        self._phases = {}
        self._incumbents = []
        self._requests = {}
        self._state_first_iteration = {}
        self._planner_log = (self.output_dir / "planner.log").open(
            "w", encoding="utf-8", buffering=1
        )
        self._metadata = dict(metadata or {})
        self._metadata.update(
            {
                "run_id": self.run_id,
                "mode": self.mode,
                "status": "running",
                "output_dir": str(self.output_dir),
            }
        )
        self._write_run_json()

    def _write_run_json(self):
        (self.output_dir / "run.json").write_text(
            json.dumps(self._metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def elapsed(self):
        return time.monotonic() - self.started_at

    def request_received(self, request):
        request_id = str(request.get("request_id", ""))
        iteration = int(request.get("iteration", 1))
        state_label = str(
            request.get("state_label", request.get("state_id", ""))
        )
        with self._lock:
            first_iteration = self._state_first_iteration.get(state_label)
            if first_iteration is None:
                self._state_first_iteration[state_label] = iteration
            row = self._requests.setdefault(request_id, {})
            row.update(
                {
                    "run_id": self.run_id,
                    "iteration": iteration,
                    "request_id": request_id,
                    "state_id": request.get("state_id", ""),
                    "state_label": state_label,
                    "reason": request.get("reason", ""),
                    "g": request.get("g", ""),
                    "h": request.get("h", ""),
                    "search_expansions": request.get(
                        "search_expansions", ""
                    ),
                    "received_seconds": self.elapsed(),
                    "status": "received",
                    "applied_actions": 0,
                    "inserted_states": 0,
                    "seen_previous_iteration": int(
                        first_iteration is not None
                        and first_iteration < iteration
                    ),
                }
            )

    def request_finished(self, request_id, status, **values):
        with self._lock:
            row = self._requests.setdefault(str(request_id), {})
            # Once a phase is stale, a late handler must never turn it back
            # into an apparently usable completion.
            if row.get("status") in {
                "stale_iteration", "discarded_phase_end"
            }:
                status = row["status"]
            row["status"] = status
            row["finished_seconds"] = self.elapsed()
            row.update(values)

    def model_started(self, request_id, generation_count):
        with self._lock:
            row = self._requests.setdefault(str(request_id), {})
            row["model_generations_started"] = int(generation_count)

    def _mark_iteration_stale(self, iteration, cancellations):
        cancellation_map = dict(cancellations)
        for row in self._requests.values():
            if int(row.get("iteration", -1)) != int(iteration):
                continue
            if row.get("status") in {
                "ok", "partial", "mock", "prompt_error", "llm_error",
                "internal_error",
            }:
                continue
            row["status"] = "stale_iteration"
            row["finished_seconds"] = self.elapsed()
            request_id = str(row.get("request_id", ""))
            if request_id in cancellation_map:
                row["error"] = "phase ended; model future cancellation=%s" % (
                    "accepted" if cancellation_map[request_id] else "late"
                )

    def handle_planner_line(self, line):
        self._planner_log.write(line)
        marker, values = _parse_structured_line(line)
        if not marker:
            return
        iteration_text = values.get("iteration")
        iteration = int(iteration_text) if iteration_text else None
        if marker == "NLM-ANYTIME-PHASE-START" and iteration is not None:
            self.registry.start_iteration(iteration)
            with self._lock:
                phase = self._phases.setdefault(iteration, {})
                phase.update(values)
                phase["run_id"] = self.run_id
        elif marker == "NLM-ANYTIME-PHASE-END" and iteration is not None:
            cancellations = self.registry.end_iteration(iteration)
            with self._lock:
                phase = self._phases.setdefault(iteration, {})
                phase.update(values)
                phase["run_id"] = self.run_id
                phase["python_cancel_requested"] = len(cancellations)
                phase["python_cancel_accepted"] = sum(
                    int(accepted) for _, accepted in cancellations
                )
                self._mark_iteration_stale(iteration, cancellations)
        elif marker == "NLM-LLM-TRIGGER-STATS" and iteration is not None:
            with self._lock:
                phase = self._phases.setdefault(iteration, {})
                phase.update(values)
                phase["run_id"] = self.run_id
        elif marker == "NLM-ANYTIME-INCUMBENT" and iteration is not None:
            with self._lock:
                row = dict(values)
                plan_number = values.get("plan_number", "")
                actual_plan_file = (
                    "%s.%s" % (self.plan_file, plan_number)
                    if plan_number else self.plan_file
                )
                row.update(
                    {
                        "run_id": self.run_id,
                        "mode": self.mode,
                        "plan_file": actual_plan_file,
                    }
                )
                self._incumbents.append(row)
        elif marker == "NLM-LLM-INJECT" and values.get("request_id"):
            if "applied_actions" in values:
                with self._lock:
                    row = self._requests.setdefault(
                        values["request_id"], {}
                    )
                    row["applied_actions"] = int(
                        row.get("applied_actions", 0)
                    ) + int(values.get("applied_actions", 0))
                    row["inserted_states"] = int(
                        row.get("inserted_states", 0)
                    ) + int(values.get("inserted_states", 0))
        elif (
            marker == "NLM-LLM-BRIDGE"
            and values.get("reason") == "phase_end"
            and values.get("request_id")
        ):
            with self._lock:
                row = self._requests.setdefault(values["request_id"], {})
                if row.get("status") != "stale_iteration":
                    row["status"] = "discarded_phase_end"
                    row["finished_seconds"] = self.elapsed()
        elif marker == "NLM-ANYTIME-RUN-TIMEOUT":
            with self._lock:
                self._metadata["termination_reason"] = (
                    "search_wall_time_limit"
                )
                self._metadata["timeout"] = dict(values)
        elif marker == "NLM-SEARCH-TIMEOUT":
            with self._lock:
                self._metadata.setdefault(
                    "termination_reason", "search_time_limit"
                )
                self._metadata.setdefault("timeout", dict(values))

    @staticmethod
    def _write_csv(path, fields, rows):
        with pathlib.Path(path).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})

    def planner_stopped(self):
        """Close an unfinished phase when the planner exits or is terminated."""

        iteration, cancellations = self.registry.close()
        if iteration is None:
            return
        with self._lock:
            phase = self._phases.setdefault(iteration, {})
            phase.setdefault("run_id", self.run_id)
            phase.setdefault("iteration", iteration)
            phase.setdefault("result", "process_ended")
            phase.setdefault("elapsed_seconds", self.elapsed())
            phase["python_cancel_requested"] = int(
                phase.get("python_cancel_requested", 0) or 0
            ) + len(cancellations)
            phase["python_cancel_accepted"] = int(
                phase.get("python_cancel_accepted", 0) or 0
            ) + sum(int(accepted) for _, accepted in cancellations)
            self._mark_iteration_stale(iteration, cancellations)

    def close(self, return_code=None):
        self.planner_stopped()
        with self._lock:
            cumulative = {
                "expanded": 0,
                "evaluated": 0,
                "generated": 0,
                "reopened": 0,
            }
            for iteration in sorted(self._phases):
                phase = self._phases[iteration]
                for name in cumulative:
                    try:
                        increment = int(phase.get("phase_%s" % name, 0) or 0)
                    except (TypeError, ValueError):
                        increment = 0
                    cumulative[name] += increment
                    phase["cumulative_%s" % name] = cumulative[name]
            total_model_generations = 0
            total_completed_samples = 0
            total_usable_samples = 0
            for row in self._requests.values():
                try:
                    iteration = int(row.get("iteration", -1))
                    started_count = int(
                        row.get("model_generations_started", 0) or 0
                    )
                    sample_count = int(row.get("sample_count", 0) or 0)
                    usable_count = int(
                        row.get("usable_sample_count", 0) or 0
                    )
                except (TypeError, ValueError):
                    continue
                total_model_generations += started_count
                total_completed_samples += sample_count
                total_usable_samples += usable_count
                phase = self._phases.setdefault(iteration, {})
                phase.setdefault("run_id", self.run_id)
                phase["model_generations"] = int(
                    phase.get("model_generations", 0)
                ) + started_count
                phase["completed_samples"] = int(
                    phase.get("completed_samples", 0)
                ) + sample_count
                phase["usable_samples"] = int(
                    phase.get("usable_samples", 0)
                ) + usable_count
                if row.get("status") in {
                    "stale_iteration", "discarded_phase_end"
                }:
                    phase["stale_requests"] = int(
                        phase.get("stale_requests", 0)
                    ) + 1

            cumulative_llm = {
                "submitted": 0,
                "model_generations": 0,
                "usable_samples": 0,
                "injected_states": 0,
            }
            for iteration in sorted(self._phases):
                phase = self._phases[iteration]
                for name in cumulative_llm:
                    try:
                        increment = int(float(phase.get(name, 0) or 0))
                    except (TypeError, ValueError):
                        increment = 0
                    cumulative_llm[name] += increment
                    phase["cumulative_%s" % name] = cumulative_llm[name]

            for incumbent in self._incumbents:
                try:
                    iteration = int(incumbent.get("iteration", -1))
                except (TypeError, ValueError):
                    continue
                phase = self._phases.get(iteration, {})
                incumbent.update(
                    {
                        "phase_state_requests": phase.get("submitted", 0),
                        "cumulative_state_requests": phase.get(
                            "cumulative_submitted", 0
                        ),
                        "phase_model_generations": phase.get(
                            "model_generations", 0
                        ),
                        "cumulative_model_generations": phase.get(
                            "cumulative_model_generations", 0
                        ),
                        "phase_usable_samples": phase.get(
                            "usable_samples", 0
                        ),
                        "cumulative_usable_samples": phase.get(
                            "cumulative_usable_samples", 0
                        ),
                        "phase_injected_states": phase.get(
                            "injected_states", 0
                        ),
                        "cumulative_injected_states": phase.get(
                            "cumulative_injected_states", 0
                        ),
                    }
                )
            self._write_csv(
                self.output_dir / "phases.csv",
                self.PHASE_FIELDS,
                [self._phases[key] for key in sorted(self._phases)],
            )
            self._write_csv(
                self.output_dir / "incumbents.csv",
                self.INCUMBENT_FIELDS,
                self._incumbents,
            )
            self._write_csv(
                self.output_dir / "llm_requests.csv",
                self.REQUEST_FIELDS,
                list(self._requests.values()),
            )
            self._metadata.update(
                {
                    "status": "finished",
                    "return_code": return_code,
                    "elapsed_seconds": self.elapsed(),
                    "phase_count": len(self._phases),
                    "incumbent_count": len(self._incumbents),
                    "state_request_count": len(self._requests),
                    "model_generation_count": total_model_generations,
                    "completed_sample_count": total_completed_samples,
                    "usable_sample_count": total_usable_samples,
                }
            )
            self._write_run_json()
            self._planner_log.close()
