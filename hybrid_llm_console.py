#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""混合规划器的 Python 主控台与本地 HTTP 桥接服务。

主控台负责按顺序启动 HTTP 服务、可选 vLLM 服务和 C++ 搜索器。搜索器触发
LLM 介入时，会把 ``problem_id`` 与当前完整 ``:init`` 发送到本服务；当前版本
已完成 prompt 构造，但模型推理、动作合法性验证和合法前缀回注仍以 mock 响应
占位。

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

from hybrid_prompt_builder import (
    DEFAULT_DOMAIN_CODE,
    DEFAULT_DOMAIN_PDDL,
    DEFAULT_PROBLEM_DIR,
    DEFAULT_PYPACE_SRC,
    HybridPromptBuilder,
    PromptBuildError,
    PromptBuilderConfig,
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
        built: :class:`hybrid_prompt_builder.BuiltPrompts`。

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
    echo_prompts=False,
    print_prompts=False,
    prompt_debug_dir=None,
):
    """创建绑定了 endpoint 和 prompt 构造器的 HTTP handler 类。

    请求体至少应包含 ``problem_id`` 和 ``init``；``request_id``、``state_id``、
    ``state_label`` 与 ``reason`` 用于关联请求和打印诊断日志。

    Args:
        path: 接收搜索器 POST 请求的 URL path。
        prompt_builder: 已配置的 :class:`HybridPromptBuilder` 实例。
        echo_prompts: 是否把完整 prompt 放入 mock 响应。默认关闭，避免在 C++
            日志和通信链路中复制大段文本。
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
            """解析状态请求、构造 prompt，并返回当前阶段的 mock LLM 响应。"""

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
                built = prompt_builder.build(request.get("problem_id", ""), init_text)
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
                    "status": "mock",
                    "prompt_ready": True,
                    "system_bytes": len(built.system.encode("utf-8")),
                    "user_bytes": len(built.user.encode("utf-8")),
                    "actions": [],
                    "note": "prompts built; vLLM and validation are not wired yet",
                }
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


def build_planner_command(args, script_dir):
    """根据命令行参数构造 Fast Downward 子进程参数列表。

    Args:
        args: ``argparse`` 解析结果。
        script_dir: 包含 ``fast-downward.py`` 的 NLM-CutPlan 目录。

    Returns:
        可直接传给 :class:`subprocess.Popen` 的参数列表。
    """

    return [
        args.python2,
        str(script_dir / "fast-downward.py"),
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

    # 默认使用小问题 smoke-test 参数；正式实验时可在外部环境变量中覆盖。
    # HTTP worker 数量决定同时在途的 LLM 请求上限；默认与 batch size 对齐。
    if args.http_workers > 0:
        env["NLM_LLM_HTTP_WORKERS"] = str(args.http_workers)
    else:
        env.setdefault("NLM_LLM_HTTP_WORKERS", "8")
    env.setdefault("NLM_LLM_FRONTIER_K", "16")
    env.setdefault("NLM_LLM_BATCH_SIZE", "8")
    env.setdefault("NLM_LLM_CHECK_INTERVAL", "1")
    env.setdefault("NLM_LLM_STALL_EXPANSIONS", "20")
    env.setdefault("NLM_LLM_ANCESTOR_DEPTH", "2")
    env.setdefault("NLM_LLM_MIN_DEPTH", "2")
    env.setdefault("NLM_LLM_H_RELATIVE_EPSILON", "0.5")
    env.setdefault("NLM_LLM_PLATEAU_H_CV", "10")
    env.setdefault("NLM_LLM_PLATEAU_F_CV", "10")
    return env


def start_optional_vllm(args):
    """按 ``--vllm-command`` 启动可选模型服务。

    Returns:
        已启动的 :class:`subprocess.Popen`；未配置命令时返回 ``None``。
    """

    if not args.vllm_command:
        return None
    command = shlex.split(args.vllm_command)
    print("[NLM-PY-CONSOLE] launching vLLM: %s" % " ".join(command), flush=True)
    return subprocess.Popen(command)


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


def main():
    """运行控制台完整生命周期并返回搜索器退出码。

    生命周期为：解析配置 -> 校验 prompt 依赖 -> 启动 HTTP server -> 配置并
    启动可选 vLLM -> 启动搜索器 -> 搜索结束后依次清理所有后台资源。
    """

    script_dir = pathlib.Path(__file__).resolve().parent
    default_domain = str(script_dir / "../pddl/domain.pddl")
    default_problem = str(script_dir / "../pddl/problem_scale_10_id_1.pddl")
    default_plan = str(script_dir / "../pddl/nlm_hybrid_console.plan")
    default_search = (
        "astar(lmcutnumeric(use_second_order_simple=true, "
        "bound_iterations=10, ceiling_less_than_one=true))"
    )

    parser = argparse.ArgumentParser(
        description="Start a mock Python controller for the hybrid LLM planner."
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
            "0 uses NLM_LLM_HTTP_WORKERS or the default 8."
        ),
    )
    parser.add_argument("--pypace-src", default=str(DEFAULT_PYPACE_SRC))
    parser.add_argument("--prompt-domain", default=str(DEFAULT_DOMAIN_PDDL))
    parser.add_argument("--prompt-problem-dir", default=str(DEFAULT_PROBLEM_DIR))
    parser.add_argument("--prompt-domain-code", default=str(DEFAULT_DOMAIN_CODE))
    parser.add_argument(
        "--echo-prompts",
        action="store_true",
        help="Include full system/user prompts in the mock HTTP response.",
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
        "--vllm-command",
        default="",
        help="Optional command to start vLLM before launching the planner.",
    )
    args = parser.parse_args()

    problem_id = args.problem_id or pathlib.Path(args.problem).stem
    prompt_builder = HybridPromptBuilder(
        PromptBuilderConfig(
            pypace_src=pathlib.Path(args.pypace_src),
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
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(
            args.path,
            prompt_builder,
            args.echo_prompts,
            args.print_prompts,
            prompt_debug_dir,
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
        "[NLM-PY-CONSOLE] listening on http://%s:%d%s"
        % (args.host, actual_port, args.path),
        flush=True,
    )

    env = configure_planner_environment(args, problem_id)

    command = build_planner_command(args, script_dir)
    print("[NLM-PY-CONSOLE] launching planner: %s" % " ".join(command), flush=True)
    vllm_process = start_optional_vllm(args)

    try:
        process = subprocess.Popen(command, cwd=str(script_dir), env=env)
        return_code = process.wait()
    finally:
        stop_process(vllm_process, "vLLM")
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)
        print("[NLM-PY-CONSOLE] server stopped", flush=True)

    return return_code


if __name__ == "__main__":
    sys.exit(main())
