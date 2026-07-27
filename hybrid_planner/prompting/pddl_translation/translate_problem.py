"""Problem-description translator adapted from PyPACE for online inference."""

import os
from pddl import parse_problem
from pddl.logic.functions import EqualTo, Metric         
from collections import defaultdict
from .translate_domain import logic_tree_handler, _visit_numeric


def _to_pascal_case(pddl_name: str) -> str:
    """
    转换 PDDL 名称为 PascalCase (大驼峰)。
    用于: 类名 (Types)
    """
    # 1. 移除 ? (如果是变量)
    clean = pddl_name.replace("?", "")
    # 2. 将 - 和 _ 替换为空格，然后分割
    # PDDL 不区分大小写，先转小写保证标准化
    words = clean.replace("-", " ").replace("_", " ").lower().split()
    # 3. 每个单词首字母大写，拼接
    return "".join(word.capitalize() for word in words)

def _to_snake_case(pddl_name: str) -> str:
    """
    转换 PDDL 名称为 snake_case (蛇形)。
    用于: 谓词名, 函数名, 动作名
    """
    clean = pddl_name.replace("?", "")
    # 将 - 替换为 _
    return clean.replace("-", "_").lower()


def get_python_var_name(pddl_name: str, is_function: bool = False) -> str:
    """
    将 PDDL 名称转换为 Python 变量名。
    必须与 domain 生成器中的命名逻辑保持严格一致！
    
    Example: 
    - "fuel-cost" (func) -> "func_fuel_cost"
    - "at" (pred) -> "pred_at"
    """
    # 1. 替换非法字符 (连字符变下划线)
    clean_name = pddl_name.replace("-", "_")
    
    # 2. 添加前缀 (区分谓词和函数)
    prefix = "func" if is_function else "pred"
    
    return f"{prefix}_{clean_name}"


def translate_problem(
    problem_file_path,
    python_save_path=None,
    txt_save_path=None,
    verbose=False,
):
    """
    Translate one PDDL problem into the training-time Python and prompt text.

    Online inference only needs the returned prompt text, so output paths are
    optional. This avoids two unnecessary disk writes for every requested
    search state while preserving the original file-generating interface.

    Returns:
        ``(python_source, prompt_text)``.
    """
    if python_save_path is not None and txt_save_path is None:
        # Preserve the original CLI behavior when a Python output is requested.
        txt_save_path = python_save_path.rsplit('.', 1)[0] + ".txt"

    lines = []           # 用于保存 Python 代码
    txt_lines = []       # 用于保存给 LLM 的自然语言 Prompt

    problem = parse_problem(problem_file_path)
    sorted_objs = sorted(problem.objects, key=lambda x: x.name)

    # ==========================================
    # 准备 TXT 头部
    # ==========================================
    txt_lines.append("=========================================")
    txt_lines.append("【任务背景：问题实例初始化】")
    txt_lines.append("以下是当前物流世界中存在的对象以及精确的初始状态。请基于此状态推演你的规划方案。")
    txt_lines.append("")
    txt_lines.append("【1. 存在的对象 (Entities)】")

    # ==========================================
    # 1. 遍历对象 (Objects)
    # ==========================================
    lines.append("")
    lines.append("# The following content is an explanation of the objects that appear in this specific problem.")
    lines.append("")

    entities_by_type = defaultdict(list)

    for obj in sorted_objs:
        pddl_name = obj.name
        py_var_name = _to_snake_case(pddl_name)
        type_tags = list(obj.type_tags)
        
        if not type_tags:
            class_name = "Object"
        else:
            raw_type = type_tags[0]
            class_name = _to_pascal_case(raw_type)
            # 收集供 TXT 使用
            entities_by_type[class_name].append(py_var_name)

        lines.append(f'{py_var_name} = {class_name}(name="{pddl_name}")')
        lines.append("")

    # 将对象写入 TXT
    for obj_type, obj_names in entities_by_type.items():
        txt_lines.append(f"{obj_type}s: {', '.join(obj_names)}")
    txt_lines.append("")

    # ==========================================
    # 2. 遍历初始状态 (Init State)
    # ==========================================
    lines.append("")
    lines.append("# The following content is the initialization of the problem state.")
    lines.append("# We use 'init_effects' to set the initial values of predicates and functions.")
    lines.append("")

    init_code_pieces = []
    
    txt_numeric_states = []
    txt_predicates = defaultdict(list)

    sorted_init = sorted(list(problem.init), key=lambda x: str(x))
    for atom in sorted_init:
        if isinstance(atom, EqualTo):
            # 处理数值函数
            func_node = atom.operands[0]
            value_node = atom.operands[1]

            py_func_name = get_python_var_name(func_node.name, is_function=True)

            args = [t.name.replace("-", "_") for t in func_node.terms]
            if not args:
                args_str = "(GLOBAL_WORLD,)"
            else:
                args_str = f"({', '.join(args)},)" if len(args) == 1 else f"({', '.join(args)})"
 
            val_str = _visit_numeric(value_node)

            # Python 代码片段
            snippet = (f"NumericEffect("
                       f"field_name={py_func_name}, "
                       f"objects_involved={args_str}, "
                       f"value_expression=lambda: {val_str}, "
                       f"operation='assign', "
                       f"description='init {py_func_name}')")
            init_code_pieces.append(snippet)

            # TXT 自然语言片段 (增加 get_ 前缀)
            raw_func_name = _to_snake_case(func_node.name)
            safe_func = raw_func_name if raw_func_name.startswith("get_") else f"get_{raw_func_name}"
            txt_numeric_states.append(f"{safe_func}({', '.join(args)}) = {val_str}")

        else:
            # 处理逻辑谓词
            snippet = logic_tree_handler(atom)
            init_code_pieces.append(snippet)

            # TXT 自然语言片段 (增加 is_ 前缀)
            raw_pred_name = _to_snake_case(atom.name)
            safe_pred = raw_pred_name if raw_pred_name.startswith("is_") else f"is_{raw_pred_name}"
            args = [_to_snake_case(t.name) for t in atom.terms]
            txt_predicates[raw_pred_name].append(f"{safe_pred}({', '.join(args)})")

    # 写入 Python 的 Init 代码
    if init_code_pieces:
        joined_content = ",\n        ".join(init_code_pieces)
        lines.append(f"init_effects = And(sub_goals=[\n        {joined_content}\n    ])")
        lines.append("apply_effects(init_effects)")
    else:
        lines.append("# Empty init state")
        lines.append("pass")

    # 写入 TXT 的 Init 状态
    txt_lines.append("【2. 初始数值状态 (Initial Numeric Functions)】")
    for num_stat in txt_numeric_states:
        txt_lines.append(num_stat)
    txt_lines.append("")

    txt_lines.append("【3. 初始逻辑状态 (Initial Predicates)】")
    
    # 动态按谓词名称分组并按字母顺序排序，实现真正的领域泛化
    for raw_pred_name in sorted(txt_predicates.keys()):
        # 还原出带前缀的安全谓词名，用作小标题
        safe_pred = raw_pred_name if raw_pred_name.startswith("is_") else f"is_{raw_pred_name}"
        
        txt_lines.append(f"# {safe_pred}")
        txt_lines.extend(txt_predicates[raw_pred_name])
        txt_lines.append("") # 每组状态后加一个空行，视觉更清晰


    # ==========================================
    # 3. 遍历目标状态 (Goal State)
    # ==========================================
    goal_code = logic_tree_handler(problem.goal)
    lines.append("")
    lines.append("# The following content is the definition of the goal state.")
    lines.append("# 'goal_tree' describes the conditions that must be met at the end.")
    lines.append(f"goal_tree = {goal_code}")
    lines.append("")

    # 写入 TXT 的 Goal 状态
    txt_lines.append("【4. 最终任务目标 (Goal State)】")
    txt_lines.append("你必须通过调用动作函数，使得世界最终满足以下所有条件：")
    
    # 简单的递归提取目标谓词供 TXT 使用
    goal_counter = 1
    def extract_goal_txt(node):
        nonlocal goal_counter
        node_type = type(node).__name__
        if node_type == "And":
            for child in node.operands:
                extract_goal_txt(child)
        elif node_type == "Predicate":
            raw_pred_name = _to_snake_case(node.name)
            safe_pred = raw_pred_name if raw_pred_name.startswith("is_") else f"is_{raw_pred_name}"
            args = [_to_snake_case(t.name) for t in node.terms]
            txt_lines.append(f"{goal_counter}. {safe_pred}({', '.join(args)})")
            goal_counter += 1

    if problem.goal:
        extract_goal_txt(problem.goal)
    txt_lines.append("=========================================")


    # ==========================================
    # 4. 优化目标 (Metric) - Python 专属
    # ==========================================
    if problem.metric:
        opt_dir = str(problem.metric.optimization)
        opt_name = _to_snake_case(str(problem.metric.expression)) 

        lines.append("")
        lines.append(f"# The optimization objective of this planning task is to {opt_dir} the value of function {opt_name}.")

        metric_code = _visit_numeric(problem.metric.expression)
        lines.append("def get_metric():")
        lines.append(f"    return {metric_code}")
    else:
        lines.append("")
        lines.append(f"# The optimization objective of this planning task is to minimize the number of actions.")
        lines.append("def get_metric():")
        lines.append("    global global_step_counter")
        lines.append("    return global_step_counter")

    lines.append("global global_step_counter")
    lines.append("global_step_counter = 0")


    # ==========================================
    # 保存文件
    # ==========================================
    full_py_content = "\n".join(lines)
    full_txt_content = "\n".join(txt_lines)

    try:
        if python_save_path is not None:
            with open(python_save_path, "w", encoding="utf-8") as f:
                f.write(full_py_content)
            if verbose:
                print(f"Successfully generated Python env script: {python_save_path}")

        if txt_save_path is not None:
            with open(txt_save_path, "w", encoding="utf-8") as f:
                f.write(full_txt_content)
            if verbose:
                print(f"Successfully generated TXT prompt: {txt_save_path}")
    except IOError:
        raise

    return full_py_content, full_txt_content


if __name__ == "__main__":
    pddl_path = r"E:\Python Projects\PlanPy\data\raw-pddl\depots-numeric-automatic\instances\instance-1.pddl"
    save_py_path = r"E:\Python Projects\PlanPy\problem-sample.py"
    # save_txt_path 会自动推断为 problem-sample.txt

    translate_problem(pddl_path, save_py_path, verbose=True)
