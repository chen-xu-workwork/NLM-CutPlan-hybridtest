#!/usr/bin/env python3
"""Parallel experiment runner with one isolated directory per planner job.

The output layout intentionally follows the completed Count-Downward runs:
each job owns ``job.json``, ``job_result.json``, ``console.log``, plan files,
an ``anytime`` directory, and a private Fast Downward working directory.
One persistent vLLM service is shared by all live jobs in a batch.
"""

import argparse
import concurrent.futures
import csv
import json
import math
import os
import pathlib
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass

from .llm.vllm_service import VLLMService, VLLMServiceConfig


@dataclass(frozen=True)
class JobSpec:
    index: int
    job_id: str
    problem: pathlib.Path
    mode: str
    time_limit_seconds: float


@dataclass(frozen=True)
class JobResult:
    index: int
    job_id: str
    problem: str
    mode: str
    time_limit_seconds: float
    status: str
    return_code: int
    elapsed_seconds: float
    output_dir: str
    error: str = ""


EXPECTED_PLANNER_STATUSES = {
    0: "plan_found",
    4: "unsolvable",
    5: "incomplete",
    7: "timeout",
}
RESUMABLE_STATUSES = frozenset(EXPECTED_PLANNER_STATUSES.values())


def _safe_component(value):
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return normalized[:160] or "job"


def _resolve_path(value, base_dir):
    path = pathlib.Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _atomic_write_json(path, payload):
    path = pathlib.Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_write_results(path, results):
    path = pathlib.Path(path)
    temporary = path.with_name(path.name + ".tmp")
    fields = list(JobResult.__dataclass_fields__)
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results, key=lambda item: item.index):
            writer.writerow(asdict(result))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def classify_return_code(return_code, error=""):
    if error:
        return "failed"
    return EXPECTED_PLANNER_STATUSES.get(return_code, "failed")


def load_jobs(args):
    """Load positional problems or a Count-compatible JSON manifest."""

    manifest = {}
    manifest_dir = pathlib.Path.cwd()
    if args.manifest:
        manifest_path = pathlib.Path(args.manifest).expanduser().resolve()
        manifest_dir = manifest_path.parent
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("failed to read batch manifest: %s" % exc) from exc
        if not isinstance(manifest, dict):
            raise ValueError("batch manifest must contain one JSON object")

    domain_value = args.domain or manifest.get("domain", "")
    if not domain_value:
        raise ValueError("domain is required")
    domain = _resolve_path(domain_value, manifest_dir)
    if not domain.is_file():
        raise ValueError("domain file does not exist: %s" % domain)

    problem_base = manifest_dir
    if manifest.get("problem_dir"):
        problem_base = _resolve_path(manifest["problem_dir"], manifest_dir)
    raw_jobs = list(manifest.get("jobs", [])) if args.manifest else []
    raw_jobs.extend(args.problems)
    if not raw_jobs:
        raise ValueError("at least one problem is required")

    default_mode = str(manifest.get("default_mode", args.default_mode))
    default_limit = float(
        manifest.get("time_limit_seconds", args.search_time_limit)
    )
    jobs = []
    for index, raw_job in enumerate(raw_jobs, start=1):
        values = {"problem": raw_job} if isinstance(raw_job, str) else raw_job
        if not isinstance(values, dict) or not values.get("problem"):
            raise ValueError("job %d must provide a problem path" % index)
        problem = _resolve_path(values["problem"], problem_base)
        if not problem.is_file():
            raise ValueError("problem file does not exist: %s" % problem)
        mode = str(values.get("mode", default_mode)).lower()
        if mode not in ("off", "live"):
            raise ValueError("job %d mode must be 'off' or 'live'" % index)
        time_limit = float(values.get("time_limit_seconds", default_limit))
        if not math.isfinite(time_limit) or time_limit <= 0:
            raise ValueError("job %d time limit must be positive" % index)
        requested_id = values.get(
            "id", "%03d_%s_%s" % (index, problem.stem, mode)
        )
        jobs.append(
            JobSpec(
                index=index,
                job_id=_safe_component(requested_id),
                problem=problem,
                mode=mode,
                time_limit_seconds=time_limit,
            )
        )

    identifiers = [job.job_id for job in jobs]
    duplicates = sorted(
        item for item in set(identifiers) if identifiers.count(item) > 1
    )
    if duplicates:
        raise ValueError("duplicate job IDs: %s" % ", ".join(duplicates))
    return domain, jobs


def build_job_environment(job, base_environment=None):
    """Return a child environment with mode switches made unambiguous."""

    environment = dict(
        os.environ if base_environment is None else base_environment
    )
    enabled = job.mode == "live"
    environment["NLM_LLM_TRIGGER"] = "1" if enabled else "0"
    environment["NLM_LLM_REQUEST_INITIAL"] = "0"
    for name in (
        "ENABLE_ANCESTOR_STAGNATION",
        "ENABLE_FRONTIER_PLATEAU",
        "ENABLE_GLOBAL_STALL",
    ):
        environment["NLM_LLM_" + name] = "1" if enabled else "0"
    if not enabled:
        environment["NLM_LLM_MAX_REQUESTS"] = "0"
    return environment


def build_child_command(job, domain, job_dir, args, vllm_base_url):
    plan_path = job_dir / "sas_plan"
    command = [
        sys.executable,
        "-m",
        "hybrid_planner.console",
        str(domain),
        str(job.problem),
        str(plan_path),
        "--problem-id",
        job.problem.stem,
        "--run-id",
        job.job_id,
        "--anytime-log-dir",
        str(job_dir / "anytime"),
        "--planner-work-dir",
        str(job_dir / "planner-work"),
        "--build",
        args.build,
        "--python2",
        args.planner_python,
        "--search-time-limit",
        "%.12g" % job.time_limit_seconds,
        "--llm-mode",
        job.mode,
        "--llm-seed",
        str(args.llm_seed),
    ]
    if args.overall_memory_limit:
        command.extend(
            ["--overall-memory-limit", args.overall_memory_limit]
        )
    if job.mode == "live":
        command.extend(
            [
                "--prompt-domain",
                str(domain),
                "--prompt-problem-dir",
                str(job.problem.parent),
                "--external-vllm",
                "--vllm-base-url",
                vllm_base_url,
                "--llm-model",
                args.llm_model,
                "--llm-max-concurrency",
                str(args.llm_max_concurrency),
                "--llm-samples-per-state",
                str(args.llm_samples_per_state),
                "--llm-max-qps",
                str(args.llm_max_qps),
                "--llm-max-retries",
                str(args.llm_max_retries),
                "--llm-timeout",
                str(args.llm_timeout),
                "--llm-temperature",
                str(args.llm_temperature),
                "--llm-top-p",
                str(args.llm_top_p),
                "--llm-max-tokens",
                str(args.llm_max_tokens),
                "--http-workers",
                str(args.http_workers),
                "--prompt-workers",
                str(args.prompt_workers),
                "--validation-workers",
                str(args.validation_workers),
                "--pending-behavior",
                args.pending_behavior,
            ]
        )
        if args.prompt_domain_code:
            command.extend(
                ["--prompt-domain-code", args.prompt_domain_code]
            )
        if args.llm_extra_params:
            command.extend(["--llm-extra-params", args.llm_extra_params])
    return command


def write_job_result(job_dir, result):
    _atomic_write_json(pathlib.Path(job_dir) / "job_result.json", asdict(result))


def partition_resumable_jobs(jobs, output_dir):
    completed = []
    pending = []
    for job in jobs:
        marker = pathlib.Path(output_dir) / job.job_id / "job_result.json"
        try:
            stored = JobResult(**json.loads(marker.read_text(encoding="utf-8")))
            valid = (
                stored.job_id == job.job_id
                and pathlib.Path(stored.problem).resolve() == job.problem
                and stored.mode == job.mode
                and float(stored.time_limit_seconds) == job.time_limit_seconds
                and stored.status in RESUMABLE_STATUSES
                and EXPECTED_PLANNER_STATUSES.get(stored.return_code)
                == stored.status
                and not stored.error
            )
            if not valid:
                raise ValueError("mismatched result marker")
        except (OSError, ValueError, TypeError, KeyError):
            pending.append(job)
            continue
        completed.append(stored)
    return completed, pending


class BatchJobRunner:
    def __init__(self, domain, output_dir, args, base_url):
        self.domain = domain
        self.output_dir = output_dir
        self.args = args
        self.base_url = base_url
        self._active = set()
        self._process_lock = threading.Lock()
        self._print_lock = threading.Lock()

    def stop_all(self):
        with self._process_lock:
            processes = list(self._active)
        for process in processes:
            if process.poll() is None:
                process.terminate()

    def _print(self, message):
        with self._print_lock:
            print(message, flush=True)

    def __call__(self, job):
        job_dir = self.output_dir / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        command = build_child_command(
            job, self.domain, job_dir, self.args, self.base_url
        )
        environment = build_job_environment(job)
        _atomic_write_json(
            job_dir / "job.json",
            {
                **asdict(job),
                "problem": str(job.problem),
                "domain": str(self.domain),
                "command": command,
            },
        )

        started = time.monotonic()
        process = None
        return_code = -1
        error = ""
        self._print(
            "[NLM-BATCH] start job=%s mode=%s limit=%ss"
            % (job.job_id, job.mode, "%.12g" % job.time_limit_seconds)
        )
        try:
            with (job_dir / "console.log").open(
                "w", encoding="utf-8", buffering=1
            ) as log_file:
                process = subprocess.Popen(
                    command,
                    cwd=str(pathlib.Path(__file__).resolve().parent.parent),
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                )
                with self._process_lock:
                    self._active.add(process)
                for line in process.stdout:
                    log_file.write(line)
                    with self._print_lock:
                        print("[%s] %s" % (job.job_id, line), end="", flush=True)
                return_code = process.wait()
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
        finally:
            if process is not None:
                with self._process_lock:
                    self._active.discard(process)

        result = JobResult(
            index=job.index,
            job_id=job.job_id,
            problem=str(job.problem),
            mode=job.mode,
            time_limit_seconds=job.time_limit_seconds,
            status=classify_return_code(return_code, error),
            return_code=return_code,
            elapsed_seconds=time.monotonic() - started,
            output_dir=str(job_dir),
            error=error,
        )
        write_job_result(job_dir, result)
        self._print(
            "[NLM-BATCH] end job=%s status=%s code=%d seconds=%.3f"
            % (
                job.job_id,
                result.status,
                return_code,
                result.elapsed_seconds,
            )
        )
        return result


def run_parallel_jobs(jobs, parallelism, run_job):
    results = []
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, int(parallelism)),
        thread_name_prefix="nlm-experiment",
    )
    completed_normally = False
    try:
        futures = [executor.submit(run_job, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
        completed_normally = True
    finally:
        executor.shutdown(
            wait=completed_normally,
            cancel_futures=True,
        )
    return sorted(results, key=lambda item: item.index)


def _build_service(args, output_dir):
    return VLLMService(
        VLLMServiceConfig(
            model_path=args.vllm_model_path,
            served_model_name=args.llm_model,
            host=args.vllm_host,
            port=args.vllm_port,
            api_base_url=args.vllm_base_url,
            gpus=args.vllm_gpus,
            executable=args.vllm_executable,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            dtype=args.vllm_dtype,
            trust_remote_code=args.vllm_trust_remote_code,
            omp_num_threads=args.vllm_omp_threads,
            startup_timeout=args.vllm_startup_timeout,
            poll_interval=args.vllm_poll_interval,
            log_path=args.vllm_log or str(output_dir / "vllm.log"),
            extra_args=tuple(args.vllm_extra_arg),
        )
    )


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Run isolated NLM-CutPlan experiments in parallel."
    )
    parser.add_argument("domain", nargs="?", help="Shared domain PDDL")
    parser.add_argument("problems", nargs="*", help="Problem PDDL files")
    parser.add_argument("--manifest", default="", help="JSON batch manifest")
    parser.add_argument("--default-mode", choices=("off", "live"), default="live")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--search-time-limit", type=float, default=7200.0)
    parser.add_argument(
        "--overall-memory-limit",
        default=os.environ.get("NLM_OVERALL_MEMORY_LIMIT", ""),
    )
    parser.add_argument("--build", default="release64")
    parser.add_argument("--planner-python", default="python2")

    parser.add_argument("--prompt-domain-code", default="")
    parser.add_argument("--pending-behavior", default="normal")
    parser.add_argument("--http-workers", type=int, default=0)
    parser.add_argument("--prompt-workers", type=int, default=4)
    parser.add_argument("--validation-workers", type=int, default=4)
    parser.add_argument("--llm-model", default="Qwen3.5-9B")
    parser.add_argument(
        "--llm-seed",
        type=int,
        default=int(os.environ.get("NLM_LLM_SEED", "0")),
    )
    parser.add_argument("--llm-max-concurrency", type=int, default=6)
    parser.add_argument("--llm-samples-per-state", type=int, default=3)
    parser.add_argument("--llm-max-qps", type=float, default=0.0)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--llm-timeout", type=float, default=300.0)
    parser.add_argument("--llm-temperature", type=float, default=0.7)
    parser.add_argument("--llm-top-p", type=float, default=0.9)
    parser.add_argument("--llm-max-tokens", type=int, default=16384)
    parser.add_argument("--llm-extra-params", default="")

    parser.add_argument("--vllm-model-path", default=os.environ.get("NLM_VLLM_MODEL_PATH", ""))
    parser.add_argument("--vllm-base-url", default=os.environ.get("NLM_VLLM_BASE_URL", ""))
    parser.add_argument("--vllm-host", default=os.environ.get("NLM_VLLM_HOST", "127.0.0.1"))
    parser.add_argument("--vllm-port", type=int, default=int(os.environ.get("NLM_VLLM_PORT", "8091")))
    parser.add_argument("--vllm-gpus", default=os.environ.get("NLM_VLLM_GPUS", ""))
    parser.add_argument("--vllm-executable", default=os.environ.get("NLM_VLLM_EXECUTABLE", "vllm"))
    parser.add_argument("--vllm-command", default="")
    parser.add_argument("--external-vllm", action="store_true")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--vllm-max-model-len", type=int, default=32768)
    parser.add_argument("--vllm-dtype", default="bfloat16")
    parser.add_argument(
        "--vllm-trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--vllm-omp-threads", type=int, default=2)
    parser.add_argument("--vllm-startup-timeout", type=float, default=1200.0)
    parser.add_argument("--vllm-poll-interval", type=float, default=2.0)
    parser.add_argument("--vllm-log", default="")
    parser.add_argument("--vllm-extra-arg", action="append", default=[])
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    if args.parallelism < 1:
        parser.error("--parallelism must be at least 1")
    if not math.isfinite(args.search_time_limit) or args.search_time_limit <= 0:
        parser.error("--search-time-limit must be positive")
    if args.llm_samples_per_state < 1 or args.llm_max_concurrency < 1:
        parser.error("LLM samples and concurrency must be positive")
    try:
        domain, jobs = load_jobs(args)
        if args.llm_extra_params:
            parsed = json.loads(args.llm_extra_params)
            if not isinstance(parsed, dict):
                raise ValueError("--llm-extra-params must be a JSON object")
    except (ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    output_dir = (
        pathlib.Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else pathlib.Path.cwd() / "logs" / "experiments" / time.strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    pending = list(jobs)
    if args.resume:
        completed, pending = partition_resumable_jobs(jobs, output_dir)

    live_jobs = [job for job in pending if job.mode == "live"]
    if live_jobs and not args.external_vllm:
        if not args.vllm_model_path and not args.vllm_command:
            parser.error(
                "live jobs require --vllm-model-path, --vllm-command, "
                "or --external-vllm"
            )

    _atomic_write_json(
        output_dir / "batch_config.json",
        {
            "domain": str(domain),
            "output_dir": str(output_dir),
            "parallelism": args.parallelism,
            "overall_memory_limit": args.overall_memory_limit or None,
            "llm_seed": args.llm_seed,
            "llm_generation": {
                "model": args.llm_model,
                "samples_per_state": args.llm_samples_per_state,
                "max_concurrency": args.llm_max_concurrency,
                "max_qps": args.llm_max_qps,
                "max_retries": args.llm_max_retries,
                "timeout_seconds": args.llm_timeout,
                "temperature": args.llm_temperature,
                "top_p": args.llm_top_p,
                "max_tokens": args.llm_max_tokens,
                "extra_params": args.llm_extra_params,
            },
            "vllm_service": {
                "external": args.external_vllm,
                "base_url": args.vllm_base_url,
                "model_path": args.vllm_model_path,
                "host": args.vllm_host,
                "port": args.vllm_port,
                "gpus": args.vllm_gpus,
                "tensor_parallel_size": args.vllm_tensor_parallel_size,
                "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                "max_model_len": args.vllm_max_model_len,
                "dtype": args.vllm_dtype,
            },
            "jobs": [
                {**asdict(job), "problem": str(job.problem)} for job in jobs
            ],
        },
    )
    _atomic_write_results(output_dir / "batch_results.csv", completed)

    service = None
    runner = None
    results = list(completed)
    base_url = ""
    try:
        if live_jobs:
            service = _build_service(args, output_dir)
            if not args.external_vllm:
                override = shlex.split(args.vllm_command) if args.vllm_command else None
                service.start(command_override=override)
            service.wait_until_ready()
            base_url = service.config.base_url
        if pending:
            runner = BatchJobRunner(domain, output_dir, args, base_url)
            results.extend(
                run_parallel_jobs(pending, args.parallelism, runner)
            )
    except KeyboardInterrupt:
        if runner is not None:
            runner.stop_all()
    finally:
        if service is not None:
            service.stop()
        _atomic_write_results(output_dir / "batch_results.csv", results)

    failures = [result for result in results if result.status == "failed"]
    print(
        "[NLM-BATCH] complete jobs=%d failures=%d results=%s"
        % (len(results), len(failures), output_dir / "batch_results.csv"),
        flush=True,
    )
    return 1 if failures or len(results) != len(jobs) else 0


if __name__ == "__main__":
    sys.exit(main())
