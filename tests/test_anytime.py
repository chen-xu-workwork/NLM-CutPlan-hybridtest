import csv
import json
import pathlib
import tempfile
import types
import unittest

from hybrid_planner.anytime import ActiveIterationRegistry, AnytimeRunRecorder


class FakeFuture:
    def __init__(self):
        self.cancel_calls = 0

    def cancel(self):
        self.cancel_calls += 1
        return True


class AnytimeLifecycleTests(unittest.TestCase):
    def test_budget_scope_uses_independent_phase_lifecycles(self):
        registry = ActiveIterationRegistry("run-1")
        registry.start_iteration(1)
        first = FakeFuture()
        self.assertTrue(
            registry.register_future("run-1", 1, "request-1", first)
        )
        cancelled = registry.end_iteration(1)
        self.assertEqual(cancelled, [("request-1", True)])
        self.assertEqual(first.cancel_calls, 1)

        registry.start_iteration(2)
        self.assertTrue(registry.is_active("run-1", 2))
        self.assertFalse(registry.is_active("run-1", 1))

    def test_phase_end_writes_curve_and_stale_request_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ActiveIterationRegistry("run-2")
            recorder = AnytimeRunRecorder(
                temp_dir,
                "run-2",
                "live",
                registry,
                plan_file="sas_plan",
            )
            recorder.handle_planner_line(
                "[NLM-ANYTIME-PHASE-START] iteration=1 bound=999\n"
            )
            recorder.request_received(
                {
                    "run_id": "run-2",
                    "iteration": 1,
                    "request_id": "run-2-p1-1",
                    "state_id": 1,
                    "state_label": "#1",
                    "reason": "global_stall",
                }
            )
            future = FakeFuture()
            registry.register_future(
                "run-2", 1, "run-2-p1-1", future
            )
            recorder.model_started("run-2-p1-1", 3)
            recorder.samples_finished(
                "run-2-p1-1",
                1,
                [
                    {
                        "sample_index": 0,
                        "status": "partial",
                        "llm_attempts": 1,
                        "llm_seconds": 0.25,
                        "generated_action_count": 2,
                        "legal_action_count": 1,
                        "actions": ["(move a b)"],
                    }
                ],
                [types.SimpleNamespace(seed=1234)],
            )
            self.assertTrue(
                (pathlib.Path(temp_dir) / "llm_samples.csv").is_file()
            )
            recorder.handle_planner_line("Actual search time: 1.25s\n")
            recorder.handle_planner_line(
                "[NLM-ANYTIME-PHASE-END] iteration=1 result=solved "
                "elapsed_seconds=12 phase_seconds=12 plan_cost=80 "
                "plan_length=30 phase_expanded=1000\n"
            )
            recorder.handle_planner_line(
                "[NLM-LLM-TRIGGER-STATS] run_id=run-2 iteration=1 "
                "submitted=10 responses=8 discarded_phase_end=1 "
                "discarded_queued=1 cancelled_inflight=1\n"
            )
            recorder.handle_planner_line(
                "[NLM-LLM-TRIGGER-REASON-STATS] "
                'reason="global_stall" attempts=2 submitted=1 responses=1\n'
            )
            recorder.handle_planner_line(
                "[NLM-ANYTIME-INCUMBENT] iteration=1 incumbent=1 "
                "elapsed_seconds=12 plan_cost=80 plan_length=30 "
                "plan_number=1 cumulative_expanded=1000\n"
            )
            recorder.handle_planner_line(
                "[NLM-ANYTIME-RUN-TIMEOUT] elapsed_seconds=7200 "
                "completed_iterations=1 incumbent_count=1 best_bound=80\n"
            )
            recorder.close(return_code=0)

            with (pathlib.Path(temp_dir) / "llm_requests.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                requests = list(csv.DictReader(stream))
            self.assertEqual(requests[0]["status"], "stale_iteration")

            with (pathlib.Path(temp_dir) / "llm_samples.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                samples = list(csv.DictReader(stream))
            self.assertEqual(samples[0]["seed"], "1234")
            self.assertEqual(samples[0]["legal_action_count"], "1")

            with (pathlib.Path(temp_dir) / "phases.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                phases = list(csv.DictReader(stream))
            self.assertEqual(phases[0]["reported_search_seconds"], "1.25")

            with (
                pathlib.Path(temp_dir) / "llm_trigger_reasons.csv"
            ).open(encoding="utf-8", newline="") as stream:
                reasons = list(csv.DictReader(stream))
            self.assertEqual(reasons[0]["reason"], "global_stall")
            self.assertEqual(reasons[0]["iteration"], "1")

            with (pathlib.Path(temp_dir) / "incumbents.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                incumbents = list(csv.DictReader(stream))
            self.assertEqual(incumbents[0]["plan_file"], "sas_plan.1")
            self.assertEqual(incumbents[0]["phase_state_requests"], "10")
            self.assertEqual(
                incumbents[0]["cumulative_state_requests"], "10"
            )
            self.assertEqual(
                incumbents[0]["phase_model_generations"], "3"
            )

            metadata = json.loads(
                (pathlib.Path(temp_dir) / "run.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["incumbent_count"], 1)
            self.assertEqual(metadata["return_code"], 0)
            self.assertEqual(
                metadata["termination_reason"], "search_wall_time_limit"
            )


if __name__ == "__main__":
    unittest.main()
