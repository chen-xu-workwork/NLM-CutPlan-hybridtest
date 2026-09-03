import json
import pathlib
import tempfile
import threading
import time
import types
import unittest

from hybrid_planner.batch_console import (
    JobResult,
    JobSpec,
    build_child_command,
    build_job_environment,
    classify_return_code,
    load_jobs,
    partition_resumable_jobs,
    run_parallel_jobs,
    write_job_result,
)


def make_args(**overrides):
    values = {
        "manifest": "",
        "domain": "",
        "problems": [],
        "default_mode": "live",
        "search_time_limit": 7200.0,
        "overall_memory_limit": "16G",
        "build": "release64",
        "planner_python": "python3",
        "prompt_domain_code": "",
        "pending_behavior": "normal",
        "http_workers": 0,
        "prompt_workers": 4,
        "validation_workers": 4,
        "llm_model": "Qwen3.5-9B",
        "llm_seed": 17,
        "llm_max_concurrency": 6,
        "llm_samples_per_state": 3,
        "llm_max_qps": 0.0,
        "llm_max_retries": 3,
        "llm_timeout": 300.0,
        "llm_temperature": 0.7,
        "llm_top_p": 0.9,
        "llm_max_tokens": 16384,
        "llm_extra_params": "",
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


class BatchConsoleTests(unittest.TestCase):
    def test_manifest_matches_reference_layout_and_can_mix_modes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            domain = root / "domain.pddl"
            first = root / "p1.pddl"
            second = root / "p2.pddl"
            for path in (domain, first, second):
                path.write_text("(define)", encoding="utf-8")
            manifest = root / "batch.json"
            manifest.write_text(
                json.dumps(
                    {
                        "domain": domain.name,
                        "jobs": [
                            {"problem": first.name, "mode": "off"},
                            {"problem": second.name, "mode": "live"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            loaded_domain, jobs = load_jobs(
                make_args(manifest=str(manifest))
            )

        self.assertEqual(loaded_domain, domain.resolve())
        self.assertEqual([job.mode for job in jobs], ["off", "live"])

    def test_each_child_has_private_work_and_anytime_directories(self):
        args = make_args()
        job = JobSpec(1, "baseline", pathlib.Path("problem.pddl"), "off", 60)
        job_dir = pathlib.Path("results") / job.job_id
        command = build_child_command(
            job,
            pathlib.Path("domain.pddl"),
            job_dir,
            args,
            "",
        )

        self.assertEqual(command[command.index("--llm-mode") + 1], "off")
        self.assertEqual(
            command[command.index("--planner-work-dir") + 1],
            str(job_dir / "planner-work"),
        )
        self.assertEqual(
            command[command.index("--anytime-log-dir") + 1],
            str(job_dir / "anytime"),
        )
        self.assertNotIn("--external-vllm", command)

    def test_off_environment_overrides_inherited_live_switches(self):
        job = JobSpec(1, "baseline", pathlib.Path("p.pddl"), "off", 60)
        environment = build_job_environment(
            job,
            {"NLM_LLM_TRIGGER": "1", "NLM_LLM_REQUEST_INITIAL": "1"},
        )
        self.assertEqual(environment["NLM_LLM_TRIGGER"], "0")
        self.assertEqual(environment["NLM_LLM_REQUEST_INITIAL"], "0")
        self.assertEqual(environment["NLM_LLM_MAX_REQUESTS"], "0")

    def test_parallel_scheduler_really_overlaps_jobs(self):
        jobs = [
            JobSpec(index, "job-%d" % index, pathlib.Path("p"), "off", 1)
            for index in range(1, 4)
        ]
        active = 0
        peak = 0
        lock = threading.Lock()

        def run_job(job):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return types.SimpleNamespace(index=job.index)

        run_parallel_jobs(jobs, 3, run_job)
        self.assertEqual(peak, 3)

    def test_resume_accepts_normal_timeout_and_retries_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = pathlib.Path(temp_dir)
            problem = output / "problem.pddl"
            problem.write_text("(define)", encoding="utf-8")
            job = JobSpec(1, "job", problem.resolve(), "off", 60)
            job_dir = output / job.job_id
            job_dir.mkdir()
            normal = JobResult(
                1, "job", str(job.problem), "off", 60,
                "timeout", 7, 60.1, str(job_dir), "",
            )
            write_job_result(job_dir, normal)
            completed, pending = partition_resumable_jobs([job], output)
            self.assertEqual([item.status for item in completed], ["timeout"])
            self.assertEqual(pending, [])

            failed = JobResult(
                1, "job", str(job.problem), "off", 60,
                "failed", 1, 0.1, str(job_dir), "",
            )
            write_job_result(job_dir, failed)
            completed, pending = partition_resumable_jobs([job], output)
            self.assertEqual(completed, [])
            self.assertEqual(pending, [job])

    def test_return_codes_match_fast_downward_contract(self):
        self.assertEqual(classify_return_code(0), "plan_found")
        self.assertEqual(classify_return_code(7), "timeout")
        self.assertEqual(classify_return_code(1), "failed")


if __name__ == "__main__":
    unittest.main()
