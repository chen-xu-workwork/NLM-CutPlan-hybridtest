#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""为混合规划器中的中间状态构造 LLM prompt。

模块接收求解器发送的 ``problem_id`` 和完整 ``(:init ...)`` 文本，读取原始
problem PDDL，并用运行时 init 替换原始初态。替换后的临时 PDDL 继续交给
迁入本项目的同版翻译器生成 ``problem_description``，从而保持训练和推理格式一致。

对外优先使用 :func:`build_llm_prompts`；需要问题路径、翻译结果等调试信息时，
可以直接使用 :class:`HybridPromptBuilder`。
"""

import argparse
import pathlib
import sys
import tempfile
from dataclasses import dataclass


PROMPTING_ROOT = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = PROMPTING_ROOT.parents[1]
DEFAULT_DOMAIN_PDDL = PROJECT_ROOT / "../pddl/domain.pddl"
DEFAULT_PROBLEM_DIR = PROJECT_ROOT / "../pddl"
DEFAULT_DOMAIN_CODE = PROMPTING_ROOT / "resources/depots/domain_NOassert.txt"

INITIAL_GENERATION_PREFIX = (
    "你是一个自动规划智能体。你的当前任务模式是【首次生成 (Initial Generation)】。"
    "任务是阅读下文提供的【物流世界运转法则】，并根据用户后续给出的"
    "【具体任务场景（初始状态与目标）】，推演出一套从初始状态到达目标状态的"
    "完美解决方案。\n\n"
)


class PromptBuildError(RuntimeError):
    """prompt 输入、路径或 PDDL 翻译过程不合法时抛出的统一异常。"""

    pass


@dataclass(frozen=True)
class PromptBuilderConfig:
    """构造 prompt 所需的外部资源路径。

    在线推理所需的翻译器、模板和 depots 领域说明已经迁入本项目，不再依赖
    PyPACE 训练仓库。调用方仍可传入配置覆盖 PDDL 和领域说明路径。

    Attributes:
        domain_pddl: 原始 domain PDDL。当前用于启动前完整性检查，并保留为以后
            多 domain 路由的依据。
        problem_dir: 按 ``<problem_id>.pddl`` 查找问题文件的目录。
        domain_code: 训练时使用的 ``domain_NOassert.txt``。
    """

    domain_pddl: pathlib.Path = DEFAULT_DOMAIN_PDDL
    problem_dir: pathlib.Path = DEFAULT_PROBLEM_DIR
    domain_code: pathlib.Path = DEFAULT_DOMAIN_CODE


@dataclass(frozen=True)
class BuiltPrompts:
    """一次 prompt 构造的完整结果。

    Attributes:
        problem_id: 调用方传入的问题编号。
        problem_path: 实际读取的原始 problem PDDL 路径。
        system: 可直接发送给模型的 system prompt。
        user: 可直接发送给模型的 user prompt。
        problem_description: 翻译器生成的对象、运行时初态和原始目标描述。
        runtime_problem: 已用中间状态覆盖 ``:init`` 的完整 PDDL problem；
            用于验证模型动作前缀。
    """

    problem_id: str
    problem_path: pathlib.Path
    system: str
    user: str
    problem_description: str
    runtime_problem: str

    def as_messages(self):
        """按 OpenAI/vLLM 常用 chat messages 结构返回两个 prompt。"""

        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]


def _find_balanced_section(text, section_name):
    """定位 PDDL 中一个完整的顶层 section。

    这里通过括号深度寻找 section 末尾，而不是使用正则直接匹配内容，因为
    ``:init``、``:goal`` 等 section 内部本身包含任意层嵌套括号。扫描时忽略
    从 ``;`` 到行尾的 PDDL 注释。

    Args:
        text: 完整 PDDL 文本，或只包含一个 section 的文本。
        section_name: 不带冒号的 section 名，例如 ``"init"``。

    Returns:
        ``(start, end)``，其中 ``end`` 是 section 右括号后的下标。

    Raises:
        PromptBuildError: section 不存在或括号不配平。
    """

    lowered = text.lower()
    marker = "(:" + section_name.lower()
    search_from = 0

    while True:
        start = lowered.find(marker, search_from)
        if start < 0:
            raise PromptBuildError("PDDL section (:%s ...) was not found" % section_name)

        marker_end = start + len(marker)
        if marker_end == len(text) or text[marker_end].isspace() or text[marker_end] == ")":
            break
        search_from = marker_end

    depth = 0
    in_comment = False
    for index in range(start, len(text)):
        char = text[index]
        if in_comment:
            if char in "\r\n":
                in_comment = False
            continue
        if char == ";":
            in_comment = True
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return start, index + 1
            if depth < 0:
                break

    raise PromptBuildError("PDDL section (:%s ...) is not balanced" % section_name)


def _normalize_init(init_text):
    """校验并规范化求解器提供的运行时初态。

    调用方既可以传完整的 ``(:init ...)``，也可以只传其中的事实列表；后一种
    形式会被自动包装。最终结果必须只包含一个完整 init block。

    Args:
        init_text: 求解器导出的完整世界状态。

    Returns:
        一个去除首尾空白、带 ``(:init ...)`` 外壳的字符串。

    Raises:
        PromptBuildError: 文本为空、括号不配平或混入 init 以外内容。
    """

    normalized = init_text.strip()
    if not normalized:
        raise PromptBuildError("init text is empty")

    if not normalized.lower().startswith("(:init"):
        normalized = "(:init\n%s\n)" % normalized

    start, end = _find_balanced_section(normalized, "init")
    if start != 0 or normalized[end:].strip():
        raise PromptBuildError("init text must contain exactly one (:init ...) block")
    return normalized


def replace_problem_init(problem_text, init_text):
    """用运行时 init 替换原 problem PDDL 中的初始状态。

    对象声明、goal 和 metric 均原样保留；多行 init 会沿用原 section 的缩进。

    Args:
        problem_text: 原始 problem PDDL 全文。
        init_text: 完整 ``(:init ...)`` 或 init 内部的事实列表。

    Returns:
        只替换了 ``:init`` section 的新 problem PDDL 文本。

    Raises:
        PromptBuildError: 原问题或新 init 缺失、格式错误或括号不配平。
    """

    init_block = _normalize_init(init_text)
    start, end = _find_balanced_section(problem_text, "init")

    line_start = problem_text.rfind("\n", 0, start) + 1
    indentation = problem_text[line_start:start]
    init_lines = init_block.splitlines()
    indented_init = ("\n" + indentation).join(init_lines)
    return problem_text[:start] + indented_init + problem_text[end:]


class HybridPromptBuilder:
    """可复用的混合规划 prompt 构造器。

    PDDL 翻译模块采用延迟导入，并在同一构造器实例中缓存导入结果。HTTP 控制台
    可以共享一个实例处理多个状态请求；每次 build 使用独立临时目录，避免并发
    请求互相覆盖中间文件。
    """

    def __init__(self, config=None):
        """创建构造器。

        Args:
            config: 可选路径配置；省略时使用模块顶部的当前本机默认路径。
        """

        self.config = config or PromptBuilderConfig()
        self._translate_problem = None
        self._system_template = None
        self._user_template = None

    def _validate_paths(self):
        """确认所有外部输入存在，缺失时抛出 :class:`PromptBuildError`。"""

        required = {
            "domain PDDL": self.config.domain_pddl,
            "problem directory": self.config.problem_dir,
            "domain code": self.config.domain_code,
        }
        missing = [
            "%s: %s" % (label, path)
            for label, path in required.items()
            if not pathlib.Path(path).exists()
        ]
        if missing:
            raise PromptBuildError("required path does not exist: " + "; ".join(missing))

    def _load_runtime_components(self):
        """延迟加载迁入本项目的翻译器及无 assert prompt 模板。"""

        if self._translate_problem is not None:
            return

        try:
            from .pddl_translation.translate_problem import translate_problem
            from .templates import SYSTEM_PROMPT_WITH_DOMAIN, USER_PROMPT_WO_DOMAIN
        except Exception as exc:
            raise PromptBuildError(
                "failed to import runtime PDDL prompt tools: %s. "
                "Install requirements/hybrid.txt in the planner environment."
                % exc
            ) from exc

        self._translate_problem = translate_problem
        self._system_template = SYSTEM_PROMPT_WITH_DOMAIN
        self._user_template = USER_PROMPT_WO_DOMAIN

    def validate(self):
        """执行启动前检查，不生成 prompt。

        控制台应在启动搜索器前调用本方法，以便尽早发现路径错误或当前 Python
        环境中缺少 ``pddl`` 等运行时依赖。

        Raises:
            PromptBuildError: 外部路径或 PDDL 翻译模块不可用。
        """

        self._validate_paths()
        self._load_runtime_components()

    def resolve_problem_path(self, problem_id):
        """将问题编号解析成实际 PDDL 文件路径。

        常规形式直接解析为 ``problem_dir/<problem_id>.pddl``。为方便手工调试，
        纯数字编号在直接查找失败后还会尝试
        ``problem_scale_10_id_<id>.pddl``。

        Args:
            problem_id: 问题文件名主体，也可以包含 ``.pddl`` 后缀。

        Returns:
            已确认存在的 :class:`pathlib.Path`。

        Raises:
            PromptBuildError: 编号为空或对应问题文件不存在。
        """

        normalized_id = str(problem_id).strip()
        if not normalized_id:
            raise PromptBuildError("problem_id is empty")

        filename = normalized_id
        if not filename.lower().endswith(".pddl"):
            filename += ".pddl"

        problem_path = self.config.problem_dir / filename
        if problem_path.exists():
            return problem_path

        if normalized_id.isdigit():
            fallback = (
                self.config.problem_dir
                / ("problem_scale_10_id_%s.pddl" % normalized_id)
            )
            if fallback.exists():
                return fallback

        raise PromptBuildError(
            "problem file does not exist for problem_id %r: %s"
            % (normalized_id, problem_path)
        )

    def build(self, problem_id, init_text):
        """构造一个中间状态对应的完整 prompt。

        实现顺序为：读取原 problem -> 覆盖 ``:init`` -> 写入独立临时 PDDL ->
        调用运行时 ``translate_problem`` -> 将返回的描述填入无 assert 模板。
        训练阶段使用的 Python 环境脚本不会在在线请求中落盘。

        Args:
            problem_id: 用于定位原始 problem PDDL 的唯一编号。
            init_text: C++ 搜索器导出的当前完整世界状态。

        Returns:
            包含 system/user prompt 及调试元数据的 :class:`BuiltPrompts`。

        Raises:
            PromptBuildError: 路径、PDDL 依赖或翻译过程出现错误。
        """

        self._validate_paths()
        self._load_runtime_components()

        problem_path = self.resolve_problem_path(problem_id)
        try:
            original_problem = problem_path.read_text(encoding="utf-8")
            domain_code = self.config.domain_code.read_text(encoding="utf-8")
        except OSError as exc:
            raise PromptBuildError("failed to read prompt input: %s" % exc) from exc

        overlaid_problem = replace_problem_init(original_problem, init_text)

        try:
            with tempfile.TemporaryDirectory(prefix="nlm_prompt_") as temp_dir:
                temp_root = pathlib.Path(temp_dir)
                temp_problem = temp_root / problem_path.name
                temp_problem.write_text(overlaid_problem, encoding="utf-8")
                _, problem_description = self._translate_problem(
                    str(temp_problem)
                )
        except PromptBuildError:
            raise
        except Exception as exc:
            raise PromptBuildError(
                "failed to translate problem %r with the runtime init: %s"
                % (str(problem_path), exc)
            ) from exc

        base_system = self._system_template.format(domain_code=domain_code).strip()
        system_prompt = INITIAL_GENERATION_PREFIX + base_system
        user_prompt = self._user_template.format(
            problem_description=problem_description
        ).strip()

        return BuiltPrompts(
            problem_id=str(problem_id),
            problem_path=problem_path,
            system=system_prompt,
            user=user_prompt,
            problem_description=problem_description,
            runtime_problem=overlaid_problem,
        )


def build_llm_prompts(problem_id, init_text, config=None):
    """以最简接口返回可直接送入模型的 system 和 user prompt。

    Args:
        problem_id: 问题文件名主体，例如 ``"problem_scale_10_id_21"``。
        init_text: 求解器当前状态的完整 ``(:init ...)`` 文本。
        config: 可选 :class:`PromptBuilderConfig`。

    Returns:
        ``(system_prompt, user_prompt)`` 二元组。

    Raises:
        PromptBuildError: prompt 无法构造；具体原因包含在异常消息中。
    """

    built = HybridPromptBuilder(config).build(problem_id, init_text)
    return built.system, built.user


def main():
    """命令行调试入口：从 UTF-8 init 文件构造并打印两个 prompt。"""

    parser = argparse.ArgumentParser(
        description="Build the system and user prompts for one hybrid-planner state."
    )
    parser.add_argument("problem_id")
    parser.add_argument("init_file", help="UTF-8 file containing one (:init ...) block")
    parser.add_argument("--domain-pddl", default=str(DEFAULT_DOMAIN_PDDL))
    parser.add_argument("--problem-dir", default=str(DEFAULT_PROBLEM_DIR))
    parser.add_argument("--domain-code", default=str(DEFAULT_DOMAIN_CODE))
    args = parser.parse_args()

    config = PromptBuilderConfig(
        domain_pddl=pathlib.Path(args.domain_pddl),
        problem_dir=pathlib.Path(args.problem_dir),
        domain_code=pathlib.Path(args.domain_code),
    )
    init_text = pathlib.Path(args.init_file).read_text(encoding="utf-8")
    built = HybridPromptBuilder(config).build(args.problem_id, init_text)
    print("===== SYSTEM =====")
    print(built.system)
    print("===== USER =====")
    print(built.user)
    return 0


if __name__ == "__main__":
    sys.exit(main())
