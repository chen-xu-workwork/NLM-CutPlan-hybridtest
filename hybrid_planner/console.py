#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""混合规划器的 Python 主控台与本地 HTTP 桥接服务。

live 模式下，主控台依次启动 vLLM、异步模型客户端、本地 HTTP 服务和 C++
搜索器；replay 模式使用保存的模型文本替代 vLLM，以便确定性测试同一条链路；
off 模式跳过 prompt、模型和 HTTP 桥，用作关闭全部触发器的原生基线。
搜索器触发 LLM 介入时，会把 ``problem_id`` 与当前完整 ``:init`` 发送到本服务；
服务构造训练格式一致的 prompt，取得模型输出，解析 ``action_Xxx(...)`` 调用，
并用 Unified Planning 只保留最长合法动作前缀。

HTTP 服务使用 :class:`ThreadingHTTPServer`，因此通信线程可以并发接收多个状态
请求，不会阻塞 C++ 搜索主线程等待其他请求完成。
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import pathlib
import platform
import re
import shlex
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .prompting.builder import (
    DEFAULT_DOMAIN_CODE,
    HybridPromptBuilder,
    PromptBuildError,
    PromptBuilderConfig,
)
from .anytime import ActiveIterationRegistry, AnytimeRunRecorder
from .llm.client import (
    BackgroundLLMRuntime,
    LLMClientConfig,
    ReplayLLMRuntime,
)
from .llm.vllm_service import VLLMService, VLLMServiceConfig
from .validation.response_processor import (
    PlanResponseProcessor,
    PlanValidationError,
    UnifiedPlanningPrefixValidator,
)


# Each anytime iteration is a fresh satisficing search under the incumbent's
# strict cost bound. Numeric LM-cut remains the primary key and goal count
# breaks broad h ties. Repeating exactly the same phase is the conservative
# baseline: curve changes come from a tighter bound and optional LLM advice,
# not from an unrelated search-policy schedule.
DEFAULT_SATISFICING_HEURISTIC = (
    "nlm_h=lmcutnumeric(use_second_order_simple=true, "
    "bound_iterations=10, ceiling_less_than_one=true)"
)
DEFAULT_SEARCH_TIME_LIMIT_SECONDS = 7200.0
DEFAULT_SINGLE_PASS_SEARCH = (
    "eager(tiebreaking([nlm_h, goalcount()]), reopen_closed=false, "
    "llm_h=nlm_h, llm_h_open_list_key_index=0)"
)


def _format_search_time_limit(seconds):
    """Format a positive finite planner wall-clock budget for its DSL."""

    return "%.12g" % float(seconds)


def build_single_pass_search(max_time_seconds):
    """Build the smoke-test search with an internal wall-clock cutoff."""

    return "%s, max_time=%s)" % (
        DEFAULT_SINGLE_PASS_SEARCH[:-1],
        _format_search_time_limit(max_time_seconds),
    )


def build_satisficing_search(max_time_seconds):
    """Build repeated-last anytime search with one shared total deadline."""

    return (
        "iterated([%s], "
        "pass_bound=true, repeat_last=true, continue_on_solve=true, "
        "continue_on_fail=false, max_time=%s)"
    ) % (
        DEFAULT_SINGLE_PASS_SEARCH,
        _format_search_time_limit(max_time_seconds),
    )


DEFAULT_SATISFICING_SEARCH = build_satisficing_search(
    DEFAULT_SEARCH_TIME_LIMIT_SECONDS
)


class StaleIterationError(RuntimeError):
    """Raised when a model result belongs to a finished anytime phase."""


def _safe_filename_component(value):
    """把请求标识转换成可安全用作文件名的短字符串。"""

    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return normalized[:120] or "unknown"


def save_prompt_debug_record(debug_dir, request, built):
    """把请求 init 和最终 prompt 持久化为一个 UTF-8 JSON 文件。

    文件在 HTTP 回包前写入，因此即使搜索器已经退出，调试记录也不会依赖响应
    是否成功送达。

    Args:
        debug_dir: 调试记录输出目录。
        request: C++ 搜索器发来的原始 JSON 对象。
        built: :class:`hybrid_planner.prompting.builder.BuiltPrompts`。

    Returns:
        已写入的 :class:`pathlib.Path`。
    """

    output_dir = pathlib.Path(debug_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    request_id = _safe_filename_component(request.get("request_id", ""))
    state_label = _safe_filename_component(
        request.get("state_label", request.get("state_id", ""))
    )
    output_path = output_dir / (
        "request_%s_state_%s.json" % (request_id, state_label)
    )
    record = {
        "request_id": request.get("request_id"),
        "run_id": request.get("run_id"),
        "iteration": request.get("iteration", 1),
        "state_id": request.get("state_id"),
        "state_label": request.get("state_label"),
        "problem_id": request.get("problem_id"),
        "reason": request.get("reason"),
        "g": request.get("g"),
        "h": request.get("h"),
        "init": request.get("init", ""),
        "system": built.system,
        "user": built.user,
        "problem_description": built.problem_description,
    }
    output_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def update_prompt_debug_record(debug_path, generation, processed=None):
    """Append model output and prefix-validation metadata to a debug record."""

    if debug_path is None:
        return
    record = json.loads(pathlib.Path(debug_path).read_text(encoding="utf-8"))
    record["model_output"] = generation.content
    record["llm"] = {
        "error": generation.error,
        "attempts": generation.attempts,
        "elapsed_seconds": generation.elapsed_seconds,
    }
    if processed is not None:
        record["processed_response"] = processed.as_dict()
    pathlib.Path(debug_path).write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def update_prompt_debug_samples(debug_path, generations, processed_results):
    """Persist every independently generated sample for one state request."""

    if debug_path is None:
        return
    record = json.loads(pathlib.Path(debug_path).read_text(encoding="utf-8"))
    samples = []
    for index, (generation, processed) in enumerate(
        zip(generations, processed_results)
    ):
        sample = {
            "sample_index": index,
            "model_output": generation.content,
            "llm": {
                "error": generation.error,
                "attempts": generation.attempts,
                "elapsed_seconds": generation.elapsed_seconds,
                "seed": generation.seed,
            },
        }
        if processed is not None:
            sample["processed_response"] = processed.as_dict()
        samples.append(sample)
    record["samples"] = samples
    pathlib.Path(debug_path).write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_prompt_debug_record(request, built):
    """用带边界标记的格式把 init、system 和 user prompt 打印到控制台。"""

    request_id = request.get("request_id", "")
    state_label = request.get("state_label", request.get("state_id", ""))
    sections = [
        ("INIT", request.get("init", "")),
        ("SYSTEM", built.system),
        ("USER", built.user),
    ]
    for section_name, content in sections:
        print(
            "[NLM-PY-PROMPT-%s] begin request_id=%s state=%s"
            % (section_name, request_id, state_label),
            flush=True,
        )
        print(content, flush=True)
        print(
            "[NLM-PY-PROMPT-%s] end request_id=%s state=%s"
            % (section_name, request_id, state_label),
            flush=True,
        )


def make_handler(
    path,
    prompt_builder,
    llm_runtime=None,
    response_processor=None,
    prompt_semaphore=None,
    echo_prompts=False,
    echo_model_output=False,
    print_prompts=False,
    prompt_debug_dir=None,
    samples_per_state=3,
    anytime_registry=None,
    anytime_recorder=None,
):
    """创建绑定了 endpoint 和 prompt 构造器的 HTTP handler 类。

    请求体至少应包含 ``problem_id`` 和 ``init``；``request_id``、``state_id``、
    ``state_label`` 与 ``reason`` 用于关联请求和打印诊断日志。

    Args:
        path: 接收搜索器 POST 请求的 URL path。
        prompt_builder: 已配置的 :class:`HybridPromptBuilder` 实例。
        llm_runtime: live 模式下共享的异步模型请求运行时；``None`` 表示 mock。
        response_processor: 模型回复解析及合法前缀验证器。
        prompt_semaphore: 限制本地 PDDL 翻译并发，避免与搜索线程争抢 CPU。
        echo_prompts: 是否把完整 prompt 放入 mock 响应。默认关闭，避免在 C++
            日志和通信链路中复制大段文本。
        echo_model_output: 是否把模型原始输出放入响应，仅用于调试。
        print_prompts: 是否把 init/system/user 全文打印到 Python 控制台。
        prompt_debug_dir: 可选持久化目录；设置后每个请求写入一个 JSON 文件。
        samples_per_state: 每个状态并行提交给模型的独立采样数。

    Returns:
        可交给 :class:`ThreadingHTTPServer` 的 handler 类。
    """

    class LLMRequestHandler(BaseHTTPRequestHandler):
        """处理单个搜索器 HTTP 请求；实例生命周期由 HTTP server 管理。"""

        server_version = "NLMHybridConsole/0.1"

        def log_message(self, fmt, *args):
            """将标准 HTTP 访问日志统一加上控制台前缀。"""

            print("[NLM-PY-CONSOLE] " + fmt % args, flush=True)

        def do_POST(self):
            """构造 prompt、请求模型并返回经过验证的动作前缀。"""

            if self.path != path:
                self.send_error(404, "unknown endpoint")
                return

            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length)
            try:
                request = json.loads(raw_body.decode("utf-8"))
            except Exception as exc:
                self.send_error(400, "bad json: %s" % exc)
                return

            request_id = request.get("request_id", "")
            run_id = request.get(
                "run_id",
                anytime_registry.run_id if anytime_registry else "standalone",
            )
            try:
                iteration = int(request.get("iteration", 1))
            except (TypeError, ValueError):
                iteration = 1
            state_label = request.get("state_label", request.get("state_id", ""))
            reason = request.get("reason", "")
            init_text = request.get("init", "")
            if anytime_recorder is not None:
                anytime_recorder.request_received(request)

            def ensure_active_iteration(first_check=False):
                if anytime_registry is None:
                    return
                active = (
                    anytime_registry.accept_request(run_id, iteration)
                    if first_check
                    else anytime_registry.is_active(run_id, iteration)
                )
                if not active:
                    raise StaleIterationError(
                        "iteration %s is no longer active" % iteration
                    )

            print(
                "[NLM-PY-CONSOLE] received run_id=%s iteration=%d "
                "request_id=%s state=%s reason=%s init_bytes=%d"
                % (
                    run_id,
                    iteration,
                    request_id,
                    state_label,
                    reason,
                    len(init_text.encode("utf-8")),
                ),
                flush=True,
            )

            try:
                ensure_active_iteration(first_check=True)
                if prompt_semaphore is None:
                    built = prompt_builder.build(
                        request.get("problem_id", ""),
                        init_text,
                    )
                else:
                    with prompt_semaphore:
                        built = prompt_builder.build(
                            request.get("problem_id", ""),
                            init_text,
                        )
                ensure_active_iteration()
                print(
                    "[NLM-PY-CONSOLE] prompt ready request_id=%s problem=%s "
                    "system_bytes=%d user_bytes=%d"
                    % (
                        request_id,
                        built.problem_path.name,
                        len(built.system.encode("utf-8")),
                        len(built.user.encode("utf-8")),
                    ),
                    flush=True,
                )
                if print_prompts:
                    print_prompt_debug_record(request, built)

                debug_path = None
                if prompt_debug_dir:
                    debug_path = save_prompt_debug_record(
                        prompt_debug_dir,
                        request,
                        built,
                    )
                    print(
                        "[NLM-PY-CONSOLE] prompt debug saved request_id=%s path=%s"
                        % (request_id, debug_path),
                        flush=True,
                    )

                response = {
                    "type": "llm_response",
                    "request_id": request_id,
                    "run_id": run_id,
                    "iteration": iteration,
                    "state_id": request.get("state_id"),
                    "state_label": state_label,
                    "prompt_ready": True,
                    "system_bytes": len(built.system.encode("utf-8")),
                    "user_bytes": len(built.user.encode("utf-8")),
                }
                if llm_runtime is None:
                    response.update(
                        {
                            "status": "mock",
                            "actions": [],
                            "action_chains": [],
                            "sample_count": 0,
                            "note": "prompts built; live LLM mode is disabled",
                        }
                    )
                else:
                    print(
                        "[NLM-PY-CONSOLE] model request started request_id=%s "
                        "state=%s samples=%d"
                        % (request_id, state_label, samples_per_state),
                        flush=True,
                    )
                    if anytime_recorder is not None:
                        anytime_recorder.model_started(
                            request_id, samples_per_state
                        )
                    if (
                        anytime_registry is not None
                        and hasattr(llm_runtime, "submit_many")
                    ):
                        future = llm_runtime.submit_many(
                            built.as_messages(),
                            samples_per_state,
                            request_id=request_id,
                        )
                        if not anytime_registry.register_future(
                            run_id, iteration, request_id, future
                        ):
                            future.cancel()
                            raise StaleIterationError(
                                "iteration ended before model submission"
                            )
                        try:
                            generations = future.result()
                        finally:
                            anytime_registry.unregister_future(request_id)
                    else:
                        generations = llm_runtime.generate_many(
                            built.as_messages(),
                            samples_per_state,
                            request_id=request_id,
                        )
                    ensure_active_iteration()
                    processed_results = []
                    sample_results = []
                    action_chains = []
                    for sample_index, generation in enumerate(generations):
                        ensure_active_iteration()
                        if generation.ok:
                            processed = response_processor.process(
                                generation.content,
                                built.runtime_problem,
                            )
                            sample = processed.as_dict()
                        else:
                            processed = None
                            sample = {
                                "status": "llm_error",
                                "actions": [],
                                "error": generation.error,
                            }
                        processed_results.append(processed)
                        sample["sample_index"] = sample_index
                        sample["llm_attempts"] = generation.attempts
                        sample["llm_seconds"] = generation.elapsed_seconds
                        sample["llm_seed"] = generation.seed
                        sample_results.append(sample)
                        action_chains.append(list(sample.get("actions", [])))
                        print(
                            "[NLM-PY-CONSOLE] model sample finished "
                            "request_id=%s sample=%d status=%s legal=%d "
                            "seconds=%.3f"
                            % (
                                request_id,
                                sample_index,
                                sample["status"],
                                len(action_chains[-1]),
                                generation.elapsed_seconds,
                            ),
                            flush=True,
                        )

                    if anytime_recorder is not None:
                        anytime_recorder.samples_finished(
                            request_id,
                            iteration,
                            sample_results,
                            generations,
                        )

                    usable_indices = [
                        index
                        for index, sample in enumerate(sample_results)
                        if sample["status"] in ("ok", "partial")
                        and action_chains[index]
                    ]
                    primary_index = usable_indices[0] if usable_indices else 0
                    primary = dict(sample_results[primary_index])
                    primary.pop("sample_index", None)
                    response.update(primary)
                    response["actions"] = action_chains[primary_index]
                    response["action_chains"] = action_chains
                    response["sample_count"] = len(sample_results)
                    response["usable_sample_count"] = len(usable_indices)
                    response["samples"] = sample_results
                    update_prompt_debug_samples(
                        debug_path,
                        generations,
                        processed_results,
                    )
                    if echo_model_output:
                        response["model_outputs"] = [
                            generation.content for generation in generations
                        ]
                    print(
                        "[NLM-PY-CONSOLE] model request finished request_id=%s "
                        "samples=%d usable=%d wall_proxy_seconds=%.3f"
                        % (
                            request_id,
                            len(sample_results),
                            len(usable_indices),
                            max(
                                generation.elapsed_seconds
                                for generation in generations
                            ),
                        ),
                        flush=True,
                    )
                ensure_active_iteration()
                if debug_path is not None:
                    response["prompt_debug_file"] = str(debug_path)
                if echo_prompts:
                    response["system"] = built.system
                    response["user"] = built.user
            except (
                StaleIterationError,
                concurrent.futures.CancelledError,
            ) as exc:
                print(
                    "[NLM-PY-CONSOLE] stale iteration discarded "
                    "run_id=%s iteration=%d request_id=%s: %s"
                    % (run_id, iteration, request_id, exc),
                    flush=True,
                )
                response = {
                    "type": "llm_response",
                    "request_id": request_id,
                    "run_id": run_id,
                    "iteration": iteration,
                    "state_id": request.get("state_id"),
                    "state_label": state_label,
                    "status": "stale_iteration",
                    "prompt_ready": False,
                    "actions": [],
                    "action_chains": [],
                    "sample_count": 0,
                    "usable_sample_count": 0,
                    "error": str(exc),
                }
            except PromptBuildError as exc:
                print(
                    "[NLM-PY-CONSOLE] prompt error request_id=%s: %s"
                    % (request_id, exc),
                    flush=True,
                )
                response = {
                    "type": "llm_response",
                    "request_id": request_id,
                    "run_id": run_id,
                    "iteration": iteration,
                    "state_id": request.get("state_id"),
                    "state_label": state_label,
                    "status": "prompt_error",
                    "prompt_ready": False,
                    "actions": [],
                    "error": str(exc),
                }
            except Exception as exc:
                print(
                    "[NLM-PY-CONSOLE] request error request_id=%s: %s"
                    % (request_id, exc),
                    flush=True,
                )
                response = {
                    "type": "llm_response",
                    "request_id": request_id,
                    "run_id": run_id,
                    "iteration": iteration,
                    "state_id": request.get("state_id"),
                    "state_label": state_label,
                    "status": "internal_error",
                    "prompt_ready": False,
                    "actions": [],
                    "error": "%s: %s" % (type(exc).__name__, exc),
                }

            if (
                anytime_registry is not None
                and response.get("status") != "stale_iteration"
                and not anytime_registry.is_active(run_id, iteration)
            ):
                response.update(
                    {
                        "status": "stale_iteration",
                        "actions": [],
                        "action_chains": [],
                        "usable_sample_count": 0,
                        "error": "iteration ended before HTTP response",
                    }
                )

            if anytime_recorder is not None:
                samples = response.get("samples", [])
                anytime_recorder.request_finished(
                    request_id,
                    response.get("status", "unknown"),
                    sample_count=response.get("sample_count", len(samples)),
                    usable_sample_count=response.get(
                        "usable_sample_count", 0
                    ),
                    model_wall_seconds=max(
                        [sample.get("llm_seconds", 0.0) for sample in samples]
                        or [0.0]
                    ),
                    error=response.get("error", ""),
                )

            # HTTP 200 表示桥接通信本身成功；应用层错误由 status 字段表达，便于
            # C++ 端始终解析同一种 JSON 响应结构。
            encoded = json.dumps(response, ensure_ascii=False).encode("utf-8")
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            except (
                BrokenPipeError,
                ConnectionResetError,
                ConnectionAbortedError,
            ) as exc:
                print(
                    "[NLM-PY-CONSOLE] response abandoned request_id=%s state=%s: %s"
                    % (request_id, state_label, exc),
                    flush=True,
                )

    return LLMRequestHandler


def build_planner_command(args, project_root):
    """根据命令行参数构造 Fast Downward 子进程参数列表。

    Args:
        args: ``argparse`` 解析结果。
        project_root: 包含 ``fast-downward.py`` 的 NLM-CutPlan 目录。

    Returns:
        可直接传给 :class:`subprocess.Popen` 的参数列表。
    """

    command = [
        args.python2,
        str(project_root / "fast-downward.py"),
        "--build",
        args.build,
        "--plan-file",
        str(pathlib.Path(args.plan).expanduser().resolve()),
    ]
    if getattr(args, "overall_memory_limit", ""):
        command.extend(
            ["--overall-memory-limit", args.overall_memory_limit]
        )
    command.extend([
        str(pathlib.Path(args.domain).expanduser().resolve()),
        str(pathlib.Path(args.problem).expanduser().resolve()),
        "--heuristic",
        getattr(args, "heuristic", DEFAULT_SATISFICING_HEURISTIC),
        "--search",
        args.search,
    ])
    return command


def prepend_ld_library_path(env, entries):
    """把依赖库目录放到 ``LD_LIBRARY_PATH`` 前部。

    Args:
        env: 即将传给搜索器子进程的环境变量字典。
        entries: 需要优先搜索的库目录序列，空字符串会被忽略。

    Returns:
        合并后的 ``LD_LIBRARY_PATH`` 字符串；本函数不直接修改 ``env``。
    """

    existing = env.get("LD_LIBRARY_PATH", "")
    joined = ":".join(entry for entry in entries if entry)
    if existing:
        return joined + ":" + existing
    return joined


def configure_planner_environment(args, problem_id):
    """构造 C++ 搜索器所需的运行环境和 LLM 桥接配置。

    除 CPLEX/COIN 动态库路径外，本函数还把 HTTP 地址、problem_id、挂起策略和
    触发器默认参数传给搜索器。所有实验参数均通过 ``setdefault`` 设置，因此
    调用者预先声明的环境变量拥有更高优先级。

    Args:
        args: 控制台命令行参数，其中 ``actual_port`` 已由 HTTP server 回填。
        problem_id: 本次规划问题的唯一编号。

    Returns:
        可传给搜索器 :class:`subprocess.Popen` 的独立环境变量字典。
    """

    env = os.environ.copy()

    cplex_home = env.get("CPLEX_HOME", "/opt/ibm/ILOG/CPLEX_Studio_Community222")
    cplex_root = env.setdefault("DOWNWARD_CPLEX_ROOT", cplex_home + "/cplex")
    concert_root = env.setdefault("DOWNWARD_CONCERT_ROOT", cplex_home + "/concert")
    coin_root = env.setdefault("DOWNWARD_COIN_ROOT", "/opt/osi")
    env["LD_LIBRARY_PATH"] = prepend_ld_library_path(
        env,
        [
            coin_root + "/lib",
            cplex_root + "/lib/x86-64_linux/static_pic",
            concert_root + "/lib/x86-64_linux/static_pic",
        ],
    )

    if args.llm_mode == "off":
        # Override inherited values so live and baseline children may safely
        # share one batch parent process.
        env["NLM_LLM_TRIGGER"] = "0"
        env["NLM_LLM_COMM_MODE"] = "log"
        env["NLM_LLM_PROBLEM_ID"] = problem_id
        env["NLM_LLM_RUN_ID"] = getattr(args, "run_id", problem_id)
        env["NLM_LLM_EMIT_STATE"] = "0"
        env["NLM_LLM_MAX_REQUESTS"] = "0"
        env["NLM_LLM_REQUEST_INITIAL"] = "0"
        env["NLM_LLM_ENABLE_ANCESTOR_STAGNATION"] = "0"
        env["NLM_LLM_ENABLE_FRONTIER_PLATEAU"] = "0"
        env["NLM_LLM_ENABLE_GLOBAL_STALL"] = "0"
        return env

    env["NLM_LLM_TRIGGER"] = "1"
    env["NLM_LLM_COMM_MODE"] = "http"
    env["NLM_LLM_HTTP_HOST"] = args.host
    env["NLM_LLM_HTTP_PORT"] = str(args.actual_port)
    env["NLM_LLM_HTTP_PATH"] = args.path
    env["NLM_LLM_PROBLEM_ID"] = problem_id
    env["NLM_LLM_RUN_ID"] = getattr(args, "run_id", problem_id)
    env["NLM_LLM_PENDING_BEHAVIOR"] = args.pending_behavior
    env["NLM_LLM_EMIT_STATE"] = args.emit_state
    env["NLM_LLM_ENABLE_ANCESTOR_STAGNATION"] = "1"
    env["NLM_LLM_ENABLE_FRONTIER_PLATEAU"] = "1"
    env["NLM_LLM_ENABLE_GLOBAL_STALL"] = "1"
    env.setdefault(
        "NLM_LLM_HTTP_TIMEOUT_MS",
        str(int(max(30.0, args.llm_timeout + 60.0) * 1000)),
    )

    # HTTP worker 数量限制同时在途的状态请求。每个状态会占用多个模型
    # generation slot，所以默认 pending 上限按 samples_per_state 折算，
    # 避免在 shared pool 后方积压数轮“已经过时”的状态请求。
    if args.http_workers > 0:
        env["NLM_LLM_HTTP_WORKERS"] = str(args.http_workers)
    else:
        default_workers = (
            args.llm_max_concurrency if args.llm_mode == "live" else 8
        )
        env.setdefault("NLM_LLM_HTTP_WORKERS", str(default_workers))
    if args.llm_mode == "live":
        samples_per_state = max(
            1, int(getattr(args, "llm_samples_per_state", 3))
        )
        parallel_state_capacity = max(
            1, args.llm_max_concurrency // samples_per_state
        )
        default_max_pending = min(
            int(env["NLM_LLM_HTTP_WORKERS"]),
            parallel_state_capacity,
        )
    else:
        default_max_pending = int(env["NLM_LLM_HTTP_WORKERS"])
    env.setdefault("NLM_LLM_MAX_PENDING", str(default_max_pending))
    # 每个状态会进一步产生 samples_per_state 次模型推理。状态级硬预算与
    # pending 并发上限分离，防止长时间困难题持续消耗模型机会。
    env.setdefault("NLM_LLM_MAX_REQUESTS", "10")
    env.setdefault("NLM_LLM_ANALYSIS_INTERVAL", "8192")
    env.setdefault("NLM_LLM_ACTIVITY_WINDOWS", "4")
    env.setdefault("NLM_LLM_GROWTH_CONFIRM_WINDOWS", "2")
    env.setdefault("NLM_LLM_LAYER_RESET_WINDOWS", "4")
    env.setdefault("NLM_LLM_LAYER_MIN_RECENT_EXPANDED", "4096")
    env.setdefault("NLM_LLM_LAYER_MIN_RECENT_NET_GROWTH", "1024")
    env.setdefault("NLM_LLM_LAYER_MIN_SINCE_REQUEST_EXPANDED", "8192")
    env.setdefault("NLM_LLM_LAYER_MIN_SINCE_REQUEST_NET_GROWTH", "2048")
    env.setdefault("NLM_LLM_PLATEAU_GROWTH_RATIO", "1.05")
    env.setdefault("NLM_LLM_STALL_EXPANSIONS", "500000")
    env.setdefault("NLM_LLM_ANCESTOR_CHECK_INTERVAL", "100000")
    env.setdefault("NLM_LLM_ANCESTOR_DEPTH", "10")
    env.setdefault("NLM_LLM_MIN_DEPTH", "20")
    env.setdefault("NLM_LLM_MIN_REQUEST_GAP_EXPANSIONS", "100000")
    env.setdefault("NLM_LLM_PER_LAYER_REQUEST_GAP_EXPANSIONS", "500000")
    env.setdefault("NLM_LLM_CANDIDATE_LAYERS", "3")
    env.setdefault("NLM_LLM_REQUESTS_PER_SLOT", "1")
    env.setdefault("NLM_LLM_HEARTBEAT_INTERVAL", "100000")
    env.setdefault("NLM_LLM_H_RELATIVE_EPSILON", "0.005")
    return env


def stop_process(process, label):
    """停止由控制台启动的子进程，并在超时后强制结束。

    Args:
        process: :class:`subprocess.Popen` 或 ``None``。
        label: 用于日志显示的进程名称。
    """

    if process is None or process.poll() is not None:
        return
    print("[NLM-PY-CONSOLE] stopping %s" % label, flush=True)
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print("[NLM-PY-CONSOLE] killing %s" % label, flush=True)
        process.kill()
        process.wait()


def _parse_json_object(parser, value, option_name):
    """Parse one CLI JSON object and report errors through argparse."""

    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        parser.error("%s is not valid JSON: %s" % (option_name, exc))
    if not isinstance(parsed, dict):
        parser.error("%s must be a JSON object" % option_name)
    return parsed


def build_vllm_service_config(args):
    """Translate command-line settings into the owned service configuration."""

    return VLLMServiceConfig(
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
        log_path=args.vllm_log,
        extra_args=tuple(args.vllm_extra_arg),
    )


def build_llm_client_config(args):
    """Build the shared online inference client configuration."""

    base_url = args.vllm_base_url or (
        "http://%s:%d/v1" % (args.vllm_host, args.vllm_port)
    )
    return LLMClientConfig(
        base_url=base_url,
        api_key=args.llm_api_key,
        model=args.llm_model,
        max_concurrency=args.llm_max_concurrency,
        max_qps=args.llm_max_qps,
        max_retries=args.llm_max_retries,
        request_timeout=args.llm_timeout,
        temperature=args.llm_temperature,
        top_p=args.llm_top_p,
        max_tokens=args.llm_max_tokens,
        seed=args.llm_seed,
        extra_params=args.llm_extra_params_object,
    )


def _file_fingerprint(path):
    """Return a compact, content-addressed input record."""

    resolved = pathlib.Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _source_revision(project_root):
    """Record the checked-out source revision without making Git required."""

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=str(project_root),
                stderr=subprocess.DEVNULL,
                universal_newlines=True,
            ).strip()
        )
        return {"commit": commit, "tracked_changes": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "", "tracked_changes": None}


def _recorded_planner_environment(environment):
    """Keep experiment settings while excluding credentials and library paths."""

    keys = [
        key
        for key in environment
        if key.startswith("NLM_LLM_") or key.startswith("NLM_SEARCH_")
    ]
    keys.extend(
        key
        for key in ("OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES")
        if key in environment
    )
    return {
        key: environment[key]
        for key in sorted(set(keys))
        if key != "NLM_LLM_API_KEY"
    }


def main():
    """运行控制台完整生命周期并返回搜索器退出码。

    生命周期为：解析配置 -> 校验 prompt/验证依赖 -> 启动并等待 vLLM ->
    启动异步请求池与 HTTP bridge -> 启动搜索器 -> 依次清理后台资源。
    """

    project_root = pathlib.Path(__file__).resolve().parent.parent
    default_domain = str(project_root / "../pddl/domain.pddl")
    default_problem = str(project_root / "../pddl/problem_scale_10_id_1.pddl")
    default_plan = str(project_root / "../pddl/nlm_hybrid_console.plan")
    parser = argparse.ArgumentParser(
        description="Start the Python control plane for the hybrid LLM planner."
    )
    parser.add_argument("domain", nargs="?", default=default_domain)
    parser.add_argument("problem", nargs="?", default=default_problem)
    parser.add_argument("plan", nargs="?", default=default_plan)
    parser.add_argument("--problem-id", default="")
    parser.add_argument(
        "--run-id",
        default="",
        help="Stable identifier written into every phase and LLM request.",
    )
    parser.add_argument(
        "--anytime-log-dir",
        default="",
        help=(
            "Directory for run.json, planner.log, phases.csv, "
            "incumbents.csv and llm_requests.csv."
        ),
    )
    parser.add_argument(
        "--planner-work-dir",
        default="",
        help=(
            "Private directory for Fast Downward's fixed output.sas/output "
            "intermediates; defaults to the plan file's directory."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--path", default="/llm/request")
    parser.add_argument("--build", default="release64")
    parser.add_argument("--python2", default="python2")
    parser.add_argument(
        "--overall-memory-limit",
        default=os.environ.get("NLM_OVERALL_MEMORY_LIMIT", ""),
        help="Optional Fast Downward memory limit such as 16G or 32768M.",
    )
    parser.add_argument(
        "--heuristic", default=DEFAULT_SATISFICING_HEURISTIC
    )
    parser.add_argument(
        "--search",
        default=None,
        help=(
            "Custom search DSL. Custom configurations must include their "
            "own outer max_time; --search-time-limit configures the built-in "
            "anytime/single-pass searches."
        ),
    )
    parser.add_argument(
        "--search-time-limit",
        type=float,
        default=os.environ.get(
            "NLM_SEARCH_TIME_LIMIT_SECONDS",
            _format_search_time_limit(DEFAULT_SEARCH_TIME_LIMIT_SECONDS),
        ),
        help=(
            "Total planner search wall time in seconds for the built-in "
            "search (default: 7200). The deadline is shared by all anytime "
            "iterations."
        ),
    )
    parser.add_argument(
        "--single-pass",
        action="store_true",
        help=(
            "Run one satisficing phase; intended for initial-state/replay "
            "bridge smoke tests rather than anytime experiments."
        ),
    )
    parser.add_argument("--pending-behavior", default="normal")
    parser.add_argument("--emit-state", default="0")
    parser.add_argument(
        "--http-workers",
        type=int,
        default=0,
        help=(
            "Maximum concurrent C++ HTTP requests; "
            "0 uses NLM_LLM_HTTP_WORKERS, live LLM concurrency, or mock default 8."
        ),
    )
    parser.add_argument(
        "--prompt-domain",
        default="",
        help="Domain PDDL used for validation; defaults to the solver domain.",
    )
    parser.add_argument(
        "--prompt-problem-dir",
        default="",
        help="Problem lookup directory; defaults to the solver problem directory.",
    )
    parser.add_argument("--prompt-domain-code", default=str(DEFAULT_DOMAIN_CODE))
    parser.add_argument(
        "--echo-prompts",
        action="store_true",
        help="Include full system/user prompts in the HTTP response.",
    )
    parser.add_argument(
        "--echo-model-output",
        action="store_true",
        help="Include raw model output in the HTTP response for debugging.",
    )
    parser.add_argument(
        "--print-prompts",
        action="store_true",
        help="Print each request init and its full system/user prompts.",
    )
    parser.add_argument(
        "--prompt-debug-dir",
        default="",
        help="Write each request init and full prompts to a UTF-8 JSON file.",
    )
    parser.add_argument(
        "--llm-mode",
        choices=("off", "mock", "replay", "live"),
        default=os.environ.get("NLM_LLM_MODE", "mock"),
        help=(
            "off disables all trigger/bridge work; mock only builds prompts; "
            "replay validates saved output; live starts/connects to vLLM."
        ),
    )
    parser.add_argument(
        "--replay-model-output",
        default=os.environ.get("NLM_LLM_REPLAY_OUTPUT", ""),
        help=(
            "UTF-8 file containing deterministic model text for replay mode."
        ),
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("NLM_LLM_MODEL", "Qwen3.5-9B"),
        help="Model name sent to /v1/chat/completions and exposed by vLLM.",
    )
    parser.add_argument(
        "--llm-api-key",
        default=os.environ.get("NLM_LLM_API_KEY", "EMPTY"),
    )
    parser.add_argument("--llm-max-concurrency", type=int, default=100)
    parser.add_argument(
        "--llm-samples-per-state",
        type=int,
        default=int(os.environ.get("NLM_LLM_SAMPLES_PER_STATE", "3")),
        help="Independent parallel model samples generated for each state.",
    )
    parser.add_argument(
        "--llm-max-qps",
        type=float,
        default=0.0,
        help="Maximum request starts per second; 0 disables QPS limiting.",
    )
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--llm-timeout", type=float, default=300.0)
    parser.add_argument("--llm-temperature", type=float, default=0.7)
    parser.add_argument("--llm-top-p", type=float, default=0.9)
    parser.add_argument("--llm-max-tokens", type=int, default=16384)
    parser.add_argument(
        "--llm-seed",
        type=int,
        default=int(os.environ.get("NLM_LLM_SEED", "0")),
        help=(
            "Base seed for reproducible sampling; each request/sample gets a "
            "stable derived seed."
        ),
    )
    parser.add_argument(
        "--prompt-workers",
        type=int,
        default=4,
        help="Maximum concurrent local PDDL-to-prompt translations.",
    )
    parser.add_argument(
        "--validation-workers",
        type=int,
        default=4,
        help="Maximum concurrent Unified Planning prefix simulations.",
    )
    parser.add_argument(
        "--llm-extra-params",
        default="",
        help="Additional chat-completion parameters as one JSON object.",
    )
    parser.add_argument(
        "--vllm-model-path",
        default=os.environ.get("NLM_VLLM_MODEL_PATH", ""),
        help="Trained model/checkpoint path used by `vllm serve`.",
    )
    parser.add_argument(
        "--vllm-base-url",
        default=os.environ.get("NLM_VLLM_BASE_URL", ""),
        help="OpenAI-compatible base URL; defaults to vLLM host/port plus /v1.",
    )
    parser.add_argument(
        "--vllm-host",
        default=os.environ.get("NLM_VLLM_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=int(os.environ.get("NLM_VLLM_PORT", "8091")),
    )
    parser.add_argument(
        "--vllm-gpus",
        default=os.environ.get("NLM_VLLM_GPUS", ""),
        help=(
            "CUDA device list for the owned vLLM process. Empty preserves the "
            "container's inherited CUDA_VISIBLE_DEVICES."
        ),
    )
    parser.add_argument(
        "--vllm-executable",
        default=os.environ.get("NLM_VLLM_EXECUTABLE", "vllm"),
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=int(os.environ.get("NLM_VLLM_TENSOR_PARALLEL_SIZE", "1")),
        help="Number of visible GPUs used to shard one model replica.",
    )
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--vllm-max-model-len", type=int, default=32768)
    parser.add_argument("--vllm-dtype", default="bfloat16")
    parser.add_argument(
        "--vllm-trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--vllm-omp-threads", type=int, default=2)
    parser.add_argument("--vllm-startup-timeout", type=float, default=600.0)
    parser.add_argument("--vllm-poll-interval", type=float, default=2.0)
    parser.add_argument(
        "--vllm-log",
        default=str(project_root / "logs/vllm.log"),
    )
    parser.add_argument(
        "--external-vllm",
        action="store_true",
        help="Do not launch vLLM; wait for an already running compatible server.",
    )
    parser.add_argument(
        "--vllm-command",
        default="",
        help="Override the generated vLLM launch command.",
    )
    parser.add_argument(
        "--vllm-extra-arg",
        action="append",
        default=[],
        help="Append one argument to the generated `vllm serve` command.",
    )
    args = parser.parse_args()
    if (
        not math.isfinite(args.search_time_limit)
        or args.search_time_limit <= 0
    ):
        parser.error("--search-time-limit must be a positive finite number")
    if args.single_pass:
        args.search = build_single_pass_search(args.search_time_limit)
    elif args.search is None:
        args.search = build_satisficing_search(args.search_time_limit)
    if not args.prompt_domain:
        args.prompt_domain = args.domain
    if not args.prompt_problem_dir:
        args.prompt_problem_dir = str(
            pathlib.Path(args.problem).expanduser().resolve().parent
        )
    args.llm_extra_params_object = _parse_json_object(
        parser,
        args.llm_extra_params,
        "--llm-extra-params",
    )

    if args.prompt_workers < 1:
        parser.error("--prompt-workers must be at least 1")
    if args.validation_workers < 1:
        parser.error("--validation-workers must be at least 1")
    if args.llm_samples_per_state < 1:
        parser.error("--llm-samples-per-state must be at least 1")
    if args.llm_mode == "replay":
        replay_path = pathlib.Path(args.replay_model_output).expanduser()
        if not args.replay_model_output or not replay_path.is_file():
            parser.error(
                "replay mode requires an existing --replay-model-output file"
            )
        try:
            args.replay_model_output_text = replay_path.read_text(
                encoding="utf-8"
            )
        except OSError as exc:
            parser.error("failed to read replay model output: %s" % exc)
        if not args.replay_model_output_text.strip():
            parser.error("--replay-model-output file is empty")
    elif args.replay_model_output:
        parser.error("--replay-model-output requires --llm-mode replay")

    if args.llm_mode == "live":
        if args.llm_max_concurrency < 1:
            parser.error("--llm-max-concurrency must be at least 1")
        if args.llm_max_retries < 0:
            parser.error("--llm-max-retries must not be negative")
        if args.llm_timeout <= 0:
            parser.error("--llm-timeout must be positive")
        if args.vllm_tensor_parallel_size < 1:
            parser.error("--vllm-tensor-parallel-size must be at least 1")
        if args.vllm_max_model_len < 1:
            parser.error("--vllm-max-model-len must be at least 1")
        if not 0.0 < args.vllm_gpu_memory_utilization <= 1.0:
            parser.error(
                "--vllm-gpu-memory-utilization must be in the interval (0, 1]"
            )
        if (
            not args.external_vllm
            and not args.vllm_command
            and not args.vllm_model_path
        ):
            parser.error(
                "live mode requires --vllm-model-path, --vllm-command, "
                "or --external-vllm"
            )

    problem_id = args.problem_id or pathlib.Path(args.problem).stem
    if not args.run_id:
        args.run_id = _safe_filename_component(
            "%s-%s-%d"
            % (problem_id, time.strftime("%Y%m%d-%H%M%S"), os.getpid())
        )
    else:
        args.run_id = _safe_filename_component(args.run_id)
    anytime_log_dir = (
        pathlib.Path(args.anytime_log_dir).expanduser().resolve()
        if args.anytime_log_dir
        else project_root / "logs" / "anytime" / args.run_id
    )
    planner_work_dir = (
        pathlib.Path(args.planner_work_dir).expanduser().resolve()
        if args.planner_work_dir
        else pathlib.Path(args.plan).expanduser().resolve().parent
    )
    planner_work_dir.mkdir(parents=True, exist_ok=True)
    prompt_builder = None
    if args.llm_mode != "off":
        prompt_builder = HybridPromptBuilder(
            PromptBuilderConfig(
                domain_pddl=pathlib.Path(args.prompt_domain),
                problem_dir=pathlib.Path(args.prompt_problem_dir),
                domain_code=pathlib.Path(args.prompt_domain_code),
            )
        )
        try:
            prompt_builder.validate()
        except PromptBuildError as exc:
            parser.error(str(exc))

    prompt_debug_dir = (
        pathlib.Path(args.prompt_debug_dir).resolve()
        if args.prompt_debug_dir
        else None
    )
    server = None
    server_thread = None
    planner_process = None
    vllm_service = None
    llm_runtime = None
    response_processor = None
    anytime_registry = ActiveIterationRegistry(args.run_id)
    anytime_recorder = AnytimeRunRecorder(
        anytime_log_dir,
        args.run_id,
        args.llm_mode,
        anytime_registry,
        metadata={
            "problem_id": problem_id,
            "domain": str(pathlib.Path(args.domain).resolve()),
            "problem": str(pathlib.Path(args.problem).resolve()),
            "plan": str(pathlib.Path(args.plan).resolve()),
            "heuristic": args.heuristic,
            "search": args.search,
            "build": args.build,
            "planner_python": args.python2,
            "search_time_limit_seconds": args.search_time_limit,
            "overall_memory_limit": args.overall_memory_limit or None,
            "llm_samples_per_state": args.llm_samples_per_state,
            "llm_max_requests_per_iteration": (
                "0"
                if args.llm_mode == "off"
                else os.environ.get("NLM_LLM_MAX_REQUESTS", "10")
            ),
            "inputs": {
                "domain": _file_fingerprint(args.domain),
                "problem": _file_fingerprint(args.problem),
            },
            "source": _source_revision(project_root),
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
            },
        },
        plan_file=str(pathlib.Path(args.plan).resolve()),
    )
    return_code = 1
    try:
        if args.llm_mode in ("replay", "live"):
            validator = UnifiedPlanningPrefixValidator(
                pathlib.Path(args.prompt_domain)
            )
            try:
                validator.validate_environment()
            except PlanValidationError as exc:
                parser.error(str(exc))
            response_processor = PlanResponseProcessor(
                validator,
                max_validation_concurrency=args.validation_workers,
            )

        if args.llm_mode == "replay":
            llm_runtime = ReplayLLMRuntime(
                args.replay_model_output_text
            )
            print(
                "[NLM-PY-CONSOLE] replay model output loaded path=%s bytes=%d"
                % (
                    args.replay_model_output,
                    len(args.replay_model_output_text.encode("utf-8")),
                ),
                flush=True,
            )

        if args.llm_mode == "live":
            vllm_service = VLLMService(build_vllm_service_config(args))
            if not args.external_vllm:
                command_override = (
                    shlex.split(args.vllm_command)
                    if args.vllm_command
                    else None
                )
                launch_command = (
                    command_override
                    if command_override
                    else vllm_service.build_command()
                )
                print(
                    "[NLM-PY-CONSOLE] launching vLLM: %s"
                    % " ".join(launch_command),
                    flush=True,
                )
                vllm_service.start(command_override=command_override)
            print(
                "[NLM-PY-CONSOLE] waiting for vLLM at %s"
                % vllm_service.config.base_url,
                flush=True,
            )
            available_models = vllm_service.wait_until_ready()
            print(
                "[NLM-PY-CONSOLE] vLLM ready models=%s"
                % ",".join(available_models),
                flush=True,
            )
            anytime_recorder.update_metadata(
                vllm_ready_seconds=anytime_recorder.elapsed(),
                vllm_available_models=available_models,
            )

            llm_runtime = BackgroundLLMRuntime(build_llm_client_config(args))
            llm_runtime.start()

        if args.llm_mode != "off":
            server = ThreadingHTTPServer(
                (args.host, args.port),
                make_handler(
                    args.path,
                    prompt_builder,
                    llm_runtime=llm_runtime,
                    response_processor=response_processor,
                    prompt_semaphore=threading.BoundedSemaphore(
                        args.prompt_workers
                    ),
                    echo_prompts=args.echo_prompts,
                    echo_model_output=args.echo_model_output,
                    print_prompts=args.print_prompts,
                    prompt_debug_dir=prompt_debug_dir,
                    samples_per_state=args.llm_samples_per_state,
                    anytime_registry=anytime_registry,
                    anytime_recorder=anytime_recorder,
                ),
            )
            # Phase-end cancellation makes model waits bounded. Join request
            # threads before writing the final CSVs so no late handler can mutate
            # records after they have been flushed.
            server.daemon_threads = False
            server.block_on_close = True
            actual_port = server.server_address[1]
            args.actual_port = actual_port
            server_thread = threading.Thread(
                target=server.serve_forever, daemon=True
            )
            server_thread.start()
            print(
                "[NLM-PY-CONSOLE] listening on http://%s:%d%s mode=%s"
                % (args.host, actual_port, args.path, args.llm_mode),
                flush=True,
            )
        else:
            print(
                "[NLM-PY-CONSOLE] LLM disabled; launching native baseline",
                flush=True,
            )

        env = configure_planner_environment(args, problem_id)
        command = build_planner_command(args, project_root)
        anytime_recorder.update_metadata(
            planner_command=command,
            planner_work_dir=str(planner_work_dir),
            effective_planner_environment=_recorded_planner_environment(env),
            llm_generation={
                "model": args.llm_model if args.llm_mode == "live" else None,
                "model_path": (
                    str(pathlib.Path(args.vllm_model_path).expanduser())
                    if args.llm_mode == "live" and args.vllm_model_path
                    else None
                ),
                "base_url": (
                    build_llm_client_config(args).base_url
                    if args.llm_mode == "live"
                    else None
                ),
                "samples_per_state": args.llm_samples_per_state,
                "temperature": args.llm_temperature,
                "top_p": args.llm_top_p,
                "max_tokens": args.llm_max_tokens,
                "max_concurrency": args.llm_max_concurrency,
                "max_qps": args.llm_max_qps,
                "max_retries": args.llm_max_retries,
                "request_timeout_seconds": args.llm_timeout,
                "base_seed": args.llm_seed,
                "extra_params": args.llm_extra_params_object,
            },
            vllm_service={
                "external": args.external_vllm,
                "host": args.vllm_host,
                "port": args.vllm_port,
                "gpus": args.vllm_gpus,
                "tensor_parallel_size": args.vllm_tensor_parallel_size,
                "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                "max_model_len": args.vllm_max_model_len,
                "dtype": args.vllm_dtype,
                "trust_remote_code": args.vllm_trust_remote_code,
            } if args.llm_mode == "live" else None,
            planner_started_seconds=anytime_recorder.elapsed(),
        )
        print(
            "[NLM-PY-CONSOLE] launching planner: %s" % " ".join(command),
            flush=True,
        )
        print(
            "[NLM-PY-CONSOLE] planner work directory: %s"
            % planner_work_dir,
            flush=True,
        )
        planner_started_at = time.monotonic()
        planner_process = subprocess.Popen(
            command,
            cwd=str(planner_work_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        for planner_line in planner_process.stdout:
            print(planner_line, end="", flush=True)
            anytime_recorder.handle_planner_line(planner_line)
        return_code = planner_process.wait()
        anytime_recorder.update_metadata(
            planner_elapsed_seconds=time.monotonic() - planner_started_at,
            planner_finished_seconds=anytime_recorder.elapsed(),
        )
    finally:
        stop_process(planner_process, "planner")
        anytime_recorder.planner_stopped()
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(timeout=2)
        if llm_runtime is not None:
            llm_runtime.close()
        if vllm_service is not None:
            vllm_service.stop()
        anytime_recorder.close(return_code=return_code)
        print(
            "[NLM-PY-CONSOLE] anytime records: %s" % anytime_log_dir,
            flush=True,
        )
        print("[NLM-PY-CONSOLE] runtime stopped", flush=True)

    return return_code


if __name__ == "__main__":
    sys.exit(main())
