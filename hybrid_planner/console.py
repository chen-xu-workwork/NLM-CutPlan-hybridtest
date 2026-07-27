#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""混合规划器的 Python 主控台与本地 HTTP 桥接服务。

live 模式下，主控台依次启动 vLLM、异步模型客户端、本地 HTTP 服务和 C++
搜索器；replay 模式使用保存的模型文本替代 vLLM，以便确定性测试同一条链路。
搜索器触发 LLM 介入时，会把 ``problem_id`` 与当前完整 ``:init`` 发送到本服务；
服务构造训练格式一致的 prompt，取得模型输出，解析 ``action_Xxx(...)`` 调用，
并用 Unified Planning 只保留最长合法动作前缀。

HTTP 服务使用 :class:`ThreadingHTTPServer`，因此通信线程可以并发接收多个状态
请求，不会阻塞 C++ 搜索主线程等待其他请求完成。
"""

import argparse
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .prompting.builder import (
    DEFAULT_DOMAIN_CODE,
    HybridPromptBuilder,
    PromptBuildError,
    PromptBuilderConfig,
)
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
            state_label = request.get("state_label", request.get("state_id", ""))
            reason = request.get("reason", "")
            init_text = request.get("init", "")
            print(
                "[NLM-PY-CONSOLE] received request_id=%s state=%s reason=%s init_bytes=%d"
                % (request_id, state_label, reason, len(init_text.encode("utf-8"))),
                flush=True,
            )

            try:
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
                            "note": "prompts built; live LLM mode is disabled",
                        }
                    )
                else:
                    print(
                        "[NLM-PY-CONSOLE] model request started request_id=%s state=%s"
                        % (request_id, state_label),
                        flush=True,
                    )
                    generation = llm_runtime.generate(
                        built.as_messages(),
                        request_id=request_id,
                    )
                    if not generation.ok:
                        response.update(
                            {
                                "status": "llm_error",
                                "actions": [],
                                "error": generation.error,
                                "llm_attempts": generation.attempts,
                                "llm_seconds": generation.elapsed_seconds,
                            }
                        )
                        update_prompt_debug_record(debug_path, generation)
                    else:
                        processed = response_processor.process(
                            generation.content,
                            built.runtime_problem,
                        )
                        update_prompt_debug_record(
                            debug_path,
                            generation,
                            processed,
                        )
                        response.update(processed.as_dict())
                        response["llm_attempts"] = generation.attempts
                        response["llm_seconds"] = generation.elapsed_seconds
                        if echo_model_output:
                            response["model_output"] = generation.content
                        print(
                            "[NLM-PY-CONSOLE] model request finished request_id=%s "
                            "status=%s generated=%d legal=%d seconds=%.3f"
                            % (
                                request_id,
                                processed.status,
                                processed.generated_action_count,
                                processed.legal_action_count,
                                generation.elapsed_seconds,
                            ),
                            flush=True,
                        )
                if debug_path is not None:
                    response["prompt_debug_file"] = str(debug_path)
                if echo_prompts:
                    response["system"] = built.system
                    response["user"] = built.user
            except PromptBuildError as exc:
                print(
                    "[NLM-PY-CONSOLE] prompt error request_id=%s: %s"
                    % (request_id, exc),
                    flush=True,
                )
                response = {
                    "type": "llm_response",
                    "request_id": request_id,
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
                    "state_id": request.get("state_id"),
                    "state_label": state_label,
                    "status": "internal_error",
                    "prompt_ready": False,
                    "actions": [],
                    "error": "%s: %s" % (type(exc).__name__, exc),
                }

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

    return [
        args.python2,
        str(project_root / "fast-downward.py"),
        "--build",
        args.build,
        "--plan-file",
        args.plan,
        args.domain,
        args.problem,
        "--search",
        args.search,
    ]


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

    env["NLM_LLM_TRIGGER"] = env.get("NLM_LLM_TRIGGER", "1")
    env["NLM_LLM_COMM_MODE"] = "http"
    env["NLM_LLM_HTTP_HOST"] = args.host
    env["NLM_LLM_HTTP_PORT"] = str(args.actual_port)
    env["NLM_LLM_HTTP_PATH"] = args.path
    env["NLM_LLM_PROBLEM_ID"] = problem_id
    env["NLM_LLM_PENDING_BEHAVIOR"] = args.pending_behavior
    env["NLM_LLM_EMIT_STATE"] = args.emit_state
    env.setdefault(
        "NLM_LLM_HTTP_TIMEOUT_MS",
        str(int(max(30.0, args.llm_timeout + 60.0) * 1000)),
    )

    # HTTP worker 数量决定同时在途的 LLM 请求上限。live 模式默认与
    # 模型并发能力对齐；触发器使用保守实验值，专门的 probe 脚本另行覆盖。
    if args.http_workers > 0:
        env["NLM_LLM_HTTP_WORKERS"] = str(args.http_workers)
    else:
        default_workers = (
            args.llm_max_concurrency if args.llm_mode == "live" else 8
        )
        env.setdefault("NLM_LLM_HTTP_WORKERS", str(default_workers))
    env.setdefault(
        "NLM_LLM_MAX_PENDING",
        env["NLM_LLM_HTTP_WORKERS"],
    )
    env.setdefault("NLM_LLM_FRONTIER_K", "64")
    env.setdefault("NLM_LLM_BATCH_SIZE", "8")
    env.setdefault("NLM_LLM_CHECK_INTERVAL", "50")
    env.setdefault("NLM_LLM_STALL_EXPANSIONS", "500")
    env.setdefault("NLM_LLM_ANCESTOR_DEPTH", "4")
    env.setdefault("NLM_LLM_MIN_DEPTH", "4")
    env.setdefault("NLM_LLM_H_RELATIVE_EPSILON", "0.01")
    env.setdefault("NLM_LLM_PLATEAU_H_CV", "0.05")
    env.setdefault("NLM_LLM_PLATEAU_F_CV", "0.05")
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
        extra_params=args.llm_extra_params_object,
    )


def main():
    """运行控制台完整生命周期并返回搜索器退出码。

    生命周期为：解析配置 -> 校验 prompt/验证依赖 -> 启动并等待 vLLM ->
    启动异步请求池与 HTTP bridge -> 启动搜索器 -> 依次清理后台资源。
    """

    project_root = pathlib.Path(__file__).resolve().parent.parent
    default_domain = str(project_root / "../pddl/domain.pddl")
    default_problem = str(project_root / "../pddl/problem_scale_10_id_1.pddl")
    default_plan = str(project_root / "../pddl/nlm_hybrid_console.plan")
    default_search = (
        "astar(lmcutnumeric(use_second_order_simple=true, "
        "bound_iterations=10, ceiling_less_than_one=true))"
    )

    parser = argparse.ArgumentParser(
        description="Start the Python control plane for the hybrid LLM planner."
    )
    parser.add_argument("domain", nargs="?", default=default_domain)
    parser.add_argument("problem", nargs="?", default=default_problem)
    parser.add_argument("plan", nargs="?", default=default_plan)
    parser.add_argument("--problem-id", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--path", default="/llm/request")
    parser.add_argument("--build", default="release64")
    parser.add_argument("--python2", default="python2")
    parser.add_argument("--search", default=default_search)
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
        choices=("mock", "replay", "live"),
        default=os.environ.get("NLM_LLM_MODE", "mock"),
        help=(
            "mock only builds prompts; replay validates a saved model output; "
            "live starts/connects to vLLM."
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

            llm_runtime = BackgroundLLMRuntime(build_llm_client_config(args))
            llm_runtime.start()

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
            ),
        )
        # 调试模式必须等待正在构造/保存 prompt 的请求结束；否则搜索器先退出时，
        # daemon request thread 可能随控制台进程一起结束，来不及保存调试文件。
        if args.print_prompts or prompt_debug_dir is not None:
            server.daemon_threads = False
            server.block_on_close = True
        actual_port = server.server_address[1]
        args.actual_port = actual_port
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        print(
            "[NLM-PY-CONSOLE] listening on http://%s:%d%s mode=%s"
            % (args.host, actual_port, args.path, args.llm_mode),
            flush=True,
        )

        env = configure_planner_environment(args, problem_id)
        command = build_planner_command(args, project_root)
        print(
            "[NLM-PY-CONSOLE] launching planner: %s" % " ".join(command),
            flush=True,
        )
        planner_process = subprocess.Popen(
            command,
            cwd=str(project_root),
            env=env,
        )
        return_code = planner_process.wait()
    finally:
        stop_process(planner_process, "planner")
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(timeout=2)
        if llm_runtime is not None:
            llm_runtime.close()
        if vllm_service is not None:
            vllm_service.stop()
        print("[NLM-PY-CONSOLE] runtime stopped", flush=True)

    return return_code


if __name__ == "__main__":
    sys.exit(main())
