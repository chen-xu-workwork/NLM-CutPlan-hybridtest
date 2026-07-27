"""PDDL expression translation adapted from PyPACE's training toolchain."""

import os
from pddl import parse_domain
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict
from pddl.logic.functions import (
    # --- 1. 比较类 (生成 Expression) ---
    EqualTo,            # =
    GreaterEqualThan,   # >=
    GreaterThan,        # >
    LesserEqualThan,    # <=
    LesserThan,         # <
    
    # --- 2. 算术类 (递归计算数值) ---
    Plus,               # +
    Minus,              # -
    Times,              # *
    Divide,             # /
    
    # --- 3. 函数类 (获取具体数值) ---
    NumericFunction,     # 例如 (fuel ?t)
    NumericValue,        # 立即数

    # 数值效果类 (Effect 用)
    Assign, Increase, Decrease, ScaleUp, ScaleDown
)

# 使用 'as' 关键字重命名，避免冲突
from pddl.logic.base import And as PddlAnd, Or as PddlOr, Not as PddlNot
from pddl.logic.predicates import Predicate



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


def _build_type_tree(pddl_types_dict: dict) -> dict:
    """
    构建继承树，并在存入树之前就将名字转换为 PascalCase。

    输入：
    pddl_types_dict：pddl库解析出的types

    返回：
    一个树，包含了继承关系。树本身是邻接表表示的，用字典存储，key为父节点，对应的value为包含其所有子节点的列表。根必然是Object。
    """
    tree = defaultdict(list)
    
    for child_raw, parent_raw in pddl_types_dict.items():
        
        # 1. 处理子类名
        child_name = _to_pascal_case(child_raw)
        
        # 2. 处理父类名
        # 如果 parent_raw 存在(不是 None) 且 不是 "object"
        if parent_raw and parent_raw.lower() != "object":
            parent_name = _to_pascal_case(parent_raw)
        else:
            # 如果是 None 或者 "object"，都归为我们的基类 Object
            parent_name = "Object"
            
        # 3. 构建树
        tree[parent_name].append(child_name)
    
    if "WorldState" not in tree["Object"]:
        tree["Object"].append("WorldState")
        
    return tree


@dataclass
class ClassBlueprint:
    """
    用于在生成代码前存储类的元数据。
    """
    class_name: str
    parent_name: str
    attributes: List[str] = field(default_factory=list)

    def __repr__(self):
        return f"<Class: {self.class_name}({self.parent_name}) | {len(self.attributes)} attrs>"
    
from typing import List, Dict


def get_definition_order(type_tree: Dict[str, List[str]], root_name: str = "Object") -> List[str]:
    """
    遍历继承树，返回一个严格符合“父类在前，子类在后”顺序的类名列表。
    
    输入:
        type_tree: 之前构建的邻接表 {Parent: [Child1, Child2...]}
        root_name: 遍历的起始节点，默认为 "Object"
        
    输出:
        order_list: 包含所有类名的列表，顺序即为生成的顺序
    """
    
    # 用于记录访问顺序的列表
    order_list = []
    
    def _traverse_recursive(current_node):
        """
        辅助递归函数：先序遍历 (Pre-order DFS)
        """
        # 1. 【访问】首先记录当前节点
        # 这确保了父类一定比它的子类先进入列表
        order_list.append(current_node)
        
        # 2. 【获取子节点】
        # 使用 sorted() 确保同级子类的顺序是确定的 (字母序)，避免生成结果随机跳动
        children = sorted(type_tree.get(current_node, []))
        
        # 3. 【递归】依次深入每一个子分支
        for child_name in children:
            _traverse_recursive(child_name)
            
    # 启动递归
    # 只要 WorldState 已经被挂在 Object 下面，它也会在这里被自动遍历到
    _traverse_recursive(root_name)
    
    return order_list



def logic_tree_handler(node) -> str:
    """
    【入口函数】将pddl库解析出的逻辑树结构转换为 Python 代码的结构化内容。 可用于action.precondition（动作的前提条件）和action.effect（动作的效果） 
    
    Args:
        node: PDDL 逻辑节点 (And, Predicate, Comparison...) 可以直接把action.precondition、action.effect传进来（相当于传根节点）
        
    Returns:
        str: Python 代码字符串
    """
    if node is None:
        return None
    
    return _visit_node(node)
    
def _visit_node(node) -> str:
    """
    【内部调度器】根据节点类型分流到具体的处理函数。
    """
    # 1. 数值效果分支 (Effect 特有)
    # 如果遇到 (increase ...) 或 (assign ...)
    if isinstance(node, (Assign, Increase, Decrease, ScaleUp, ScaleDown)):
        return _visit_numeric_effect(node)

    # 2. 逻辑/谓词分支 (Precondition & Effect 通用)
    # 如果遇到 And, Or, Not, Predicate, Comparison
    # (注：这里保留你习惯的 _visit_logic 函数名，但它内部的递归必须改，详见下文)
    return _visit_logic(node)


def _visit_logic(node) -> str:
    """递归处理逻辑节点"""
    
    # --- 递归结构 ---
    if isinstance(node, PddlAnd):
        children = [_visit_node(c) for c in node.operands]
        return f"And(sub_goals=[{', '.join(filter(None, children))}])"

    elif isinstance(node, PddlOr):
        children = [_visit_node(c) for c in node.operands]
        return f"Or(sub_goals=[{', '.join(filter(None, children))}])"

    elif isinstance(node, PddlNot):
        return f"Not(sub_goal={_visit_logic(node.argument)})"

    # --- 原子谓词 ---
    elif isinstance(node, Predicate):
        # 【修改点】直接转换名字: (at ...) -> pred_at
        var_name = _get_var_name(node.name, prefix="pred")
        
        args = [t.name.replace("?", "") for t in node.terms]
        args_tuple = _fmt_tuple(args)
        
        return f"Predicate(field_name={var_name}, objects_involved={args_tuple})"

    # --- 数值比较 ---
    elif isinstance(node, (EqualTo, GreaterEqualThan, GreaterThan, LesserEqualThan, LesserThan)):
        return _visit_comparison(node)
    
    else:
        return f"# Unknown node: {type(node)}"


def _visit_comparison(node) -> str:
    """处理数值比较"""
    # 【注意】Key 必须完全匹配 type(node).__name__
    op_map = {
        "GreaterThan": ">", 
        "LesserThan": "<", 
        "EqualTo": "==",           
        "GreaterEqualThan": ">=", 
        "LesserEqualThan": "<="
    }
    
    py_op = op_map.get(type(node).__name__, "==")
    
    left_code = _visit_numeric(node.operands[0])
    right_code = _visit_numeric(node.operands[1])
    
    raw_expr = f"{left_code} {py_op} {right_code}"
    
    return (
        f"Expression(\n"
        f"            expression=lambda: {raw_expr},\n"
        f"            description='{raw_expr.replace(chr(39), '')}'\n"
        f"        )"
    )


def _visit_numeric(node) -> str:
    """递归生成数值计算表达式"""
    
    # 1. 算术运算节点
    if isinstance(node, (Plus, Minus, Times, Divide)):
        sub_exprs = [_visit_numeric(child) for child in node.operands]
        
        # 确定符号
        if isinstance(node, Plus): op = "+"
        elif isinstance(node, Minus): op = "-"
        elif isinstance(node, Times): op = "*"
        elif isinstance(node, Divide): op = "/"
        else: op = "+" 
        
        sep = f" {op} "
        return f"({sep.join(sub_exprs)})"

    # 2. 具体数值函数
    elif isinstance(node, NumericFunction):
        # 你的变量名转换逻辑
        var_name = _get_var_name(node.name, prefix="func")
        args = [t.name.replace("?", "") for t in node.terms]

        if not args:
            # Case A: 0元函数 (全局流)，如 (fuel-cost)
            # 强制挂载到 GLOBAL_WORLD 对象上，注意元组逗号
            arg_str = "(GLOBAL_WORLD,)"
        else:
            # Case B: 普通函数，如 (weight ?c)
            # 使用原本的 _fmt_tuple 处理 (确保它能处理单元素元组如 "(c,)")
            arg_str = _fmt_tuple(args)

        return f"get_value({var_name}, {arg_str})"
        
    # 3. 纯数字
    elif isinstance(node, (int, float, NumericValue)):
        return str(node)
    
        
    return str(node)


def _get_var_name(pddl_name: str, prefix: str) -> str:
    """
    【核心工具】将 PDDL 名字转换为 Python 变量名。
    例如: "fuel-cost" (prefix="func") -> "func_fuel_cost"
    """
    # 1. 替换连字符为下划线 (Python 变量命名规范)
    clean_name = pddl_name.replace("-", "_")
    # 2. 加上前缀
    return f"{prefix}_{clean_name}"


def _fmt_tuple(args: list) -> str:
    """辅助：格式化元组字符串"""
    if not args: return "()"
    if len(args) == 1: return f"({args[0]},)"
    return f"({', '.join(args)})"




def _visit_numeric_effect(node) -> str:
    """处理 (increase (fuel ?t) 10) 这类节点"""
    
    # 1. 映射操作符
    op_map = {
        "Assign": "assign", "Increase": "increase", "Decrease": "decrease",
        "ScaleUp": "scale_up", "ScaleDown": "scale_down"
    }
    op_str = op_map.get(type(node).__name__, "assign")

    # ==========================================================
    # 2. 解析 operands (根据你的发现)
    # ==========================================================
    # operands[0] 是左值 (目标函数)
    # operands[1] 是右值 (数值或表达式)
    
    target_node = node.operands[0]
    value_node = node.operands[1]

    # --- 处理左值 (Function) ---
    # target_node 应该是一个 NumericFunction 对象
    var_name = _get_var_name(target_node.name, prefix="func")
    
    # 解析参数: (t,) 或 (world,)
    if not hasattr(target_node, "terms") or not target_node.terms:
        args_str = "(world,)" # 0元函数
    else:
        args = [t.name.replace("?", "") for t in target_node.terms]
        args_str = _fmt_tuple(args)

    # --- 处理右值 (Value) ---
    # 这里的 value_node 可能是 NumericValue，也可能是 Plus/Minus 表达式
    # 直接扔给 _visit_numeric 递归处理即可
    val_expr_code = _visit_numeric(value_node)

    # 3. 生成代码
    return (
        f"NumericEffect(\n"
        f"            field_name={var_name},\n"
        f"            objects_involved={args_str},\n"
        f"            value_expression=lambda: {val_expr_code},\n"
        f"            operation='{op_str}',\n"
        f"            description='{op_str} {var_name}'\n"
        f"        )"
    )





def translate_domain(domain_file_path, python_save_path):
    """
    功能：
    读取一个pddl语言表示的domain文件，并将其转换成本项目使用的python风格，然后保存到指定路径。
    """

    # 读取并解析原始PDDL文件
    if not os.path.exists(domain_file_path):
        print(f"[Error] File not found: {domain_file_path}")
        return None
    
    try:
        domain = parse_domain(domain_file_path)
        print(f"Successfully parsed domain: '{domain.name}'")
    except Exception as e:
        print(f"PDDL Parsing failed: {e}")
        raise e
    
    # 这个字典用于收集所有谓词和函数对应的在类内的变量名，用于动作的转换
    global_constants = dict()

    # 【新增】用于收集所有的全局查询函数（API层）代码
    query_functions_code = []
    query_functions_code.append("")
    query_functions_code.append("# ==========================================")
    query_functions_code.append("# State Query API (Read-Only) for LLM Asserts")
    query_functions_code.append("# ==========================================")
    query_functions_code.append("")


    # 准备python文件内容，首先硬写入头部部分，包括依赖包和基础类Object的定义。
    lines = []

    lines.append("from dataclasses import dataclass, field")
    lines.append("from typing import List, Tuple, Union")
    lines.append("from pddl_lib.update_helper import apply_effects, ActionPreconditionError")
    lines.append("from pddl_lib.fact_checker import Expression, Predicate, And, Or, Not, NumericEffect, get_value, check_facts")
    lines.append("")

    lines.append("# The following content is the definition of objects in this problem.")
    
    lines.append("@dataclass")
    lines.append("class Object:")
    lines.append('    """ Base class for all PDDL objects. """')
    lines.append("    name: str")
    lines.append("")
    lines.append("    # Critical: Only return name to prevent recursion loops in holographic prints")
    lines.append("    def __repr__(self): return self.name")
    lines.append("")
    lines.append("    # Value-based equality (two objects with same name and class type are considered same)")
    lines.append("    def __eq__(self, other):")
    lines.append("        return type(self) is type(other) and self.name == other.name")
    lines.append("")

    # 先准备好一个字典用于存储元数据，之后根据这个列表构建完整的类定义
    meta_datas_dict: Dict[str, ClassBlueprint] = {}

    # 手动添加处理0元函数的WorldState
    if "WorldState" not in meta_datas_dict:
        meta_datas_dict["WorldState"] = ClassBlueprint(
            class_name="WorldState",
            parent_name="Object", 
            attributes=[]
        )

    # 根据解析结果构建继承关系树
    type_tree = _build_type_tree(domain.types)

    def _register_recursive(current_node):
        children = sorted(type_tree.get(current_node, []))
        for child_name in children:
            blueprint = ClassBlueprint(
                class_name=child_name,
                parent_name=current_node,
                attributes=[] 
            )
            meta_datas_dict[child_name] = blueprint
            _register_recursive(child_name)
    
    _register_recursive("Object")

    # ========================================================
    # 处理谓词 (Predicates)
    # ========================================================
    for pred in domain.predicates:
        pred_name = _to_snake_case(pred.name)
        arg_types = [_to_pascal_case(list(t.type_tags)[0]) for t in pred.terms]
        
        joined_args = "_".join(arg_types)
        attr_name = f"predicate__{pred_name}__{joined_args}"
        global_constants[f"pred_{pred_name}"] = attr_name

        # 【修复】安全处理 0 元谓词
        if not arg_types:
            involved_class_blueprint = meta_datas_dict['WorldState']
            if attr_name not in involved_class_blueprint.attributes:
                involved_class_blueprint.attributes.append(attr_name)
        else:
            for involed_class_name in arg_types:
                involed_class_blueprint = meta_datas_dict[involed_class_name]
                if attr_name not in involed_class_blueprint.attributes:
                    involed_class_blueprint.attributes.append(attr_name)

        # 【新增】自动生成返回 bool 的查询函数
        # 规避 python 关键字
        safe_func_name = pred_name if pred_name.startswith("is_") else f"is_{pred_name}"
        
        # 构造参数列表 "arg0: Truck, arg1: Place"
        params_def = [f"arg{i}: {arg_type}" for i, arg_type in enumerate(arg_types)]
        params_str = ", ".join(params_def)
        
        # 构造元组 "(arg0, arg1)"
        tuple_args = [f"arg{i}" for i in range(len(arg_types))]
        if not tuple_args:
            tuple_str = "()"
        elif len(tuple_args) == 1:
            tuple_str = f"({tuple_args[0]},)"
        else:
            tuple_str = f"({', '.join(tuple_args)})"
            
        host_obj = "arg0" if arg_types else "GLOBAL_WORLD"
        
        query_functions_code.append(f"def {safe_func_name}({params_str}) -> bool:")
        query_functions_code.append(f'    """ Auto-generated query for {pred_name} """')
        query_functions_code.append(f"    return {tuple_str} in {host_obj}.{attr_name}")
        query_functions_code.append("")


    # ========================================================
    # 处理数值函数 (Functions)
    # ========================================================
    if hasattr(domain, "functions") and domain.functions:
        for func in domain.functions:
            func_name = _to_snake_case(func.name)
            arg_types = [_to_pascal_case(list(t.type_tags)[0]) for t in func.terms]

            if not arg_types:
                joined_args = "WorldState"
            else:
                joined_args = "_".join(arg_types)

            attr_name = f"function__{func_name}__{joined_args}"
            global_constants[f"func_{func_name}"] = attr_name

            if not arg_types:
                involved_class_blueprint = meta_datas_dict['WorldState']
                if attr_name not in involved_class_blueprint.attributes:
                    involved_class_blueprint.attributes.append(attr_name)
            else:
                for involved_class_name in arg_types:
                    if involved_class_name in meta_datas_dict:
                        involved_class_blueprint = meta_datas_dict[involved_class_name]
                        if attr_name not in involved_class_blueprint.attributes:
                            involved_class_blueprint.attributes.append(attr_name)

            # 【新增】自动生成返回 float 的查询函数
            safe_func_name = func_name if func_name.startswith("get_") else f"get_{func_name}"
            params_def = [f"arg{i}: {arg_type}" for i, arg_type in enumerate(arg_types)]
            params_str = ", ".join(params_def)
            
            host_obj = "arg0" if arg_types else "GLOBAL_WORLD"
            
            query_functions_code.append(f"def {safe_func_name}({params_str}) -> float:")
            query_functions_code.append(f'    """ Auto-generated query for {func_name} """')
            query_functions_code.append(f"    if {host_obj}.{attr_name}:")
            query_functions_code.append(f"        return float({host_obj}.{attr_name}[-1][-1])")
            query_functions_code.append(f"    return 0.0")
            query_functions_code.append("")


    # 提取蓝图信息并构建完整的类定义
    access_sequence = get_definition_order(type_tree)

    for _class_name in access_sequence:
        if _class_name == "Object":
            continue
        
        class_blueprint = meta_datas_dict[_class_name]
        class_name = class_blueprint.class_name
        parent_name = class_blueprint.parent_name
        attributes = class_blueprint.attributes

        lines.append("@dataclass")
        lines.append(f"class {class_name}({parent_name}):")

        if class_name == "WorldState":
            lines.append('    """ Global state container (0-arity predicates & functions) """')

        if not attributes:
            lines.append("    pass")
        else:
            for attr_name in attributes:
                category, name, classes_str = attr_name.split("__")
                involved_classes = classes_str.split("_")
                tuple_elements = [f"'{cls}'" for cls in involved_classes]

                if category == "function":
                    tuple_elements.append("float")

                tuple_content = ", ".join(tuple_elements)
                line = f"    {attr_name}: List[Tuple[{tuple_content}]] = field(default_factory=list)"
                lines.append(line)

        lines.append('')

    lines.append('GLOBAL_WORLD = WorldState(name="GlobalWorld")')

    lines.append("# The following content is an explanation of the variable names corresponding to predicates and functions in this domain.")
    
    sorted_keys = sorted(global_constants.keys())
    for key in sorted_keys:
        value = global_constants[key]
        lines.append(f'{key} = "{value}"')

    lines.append("")
    lines.append("# The following content is the definition of actions in this domain.")
    
    # 处理动作
    for action in domain.actions:
        raw_action_name = action.name
        action_name = f"action_{_to_pascal_case(raw_action_name)}"

        parameter_info = []
        for parameter in action.parameters:
            if parameter.name in ["from", "class", "global", "return", "pass", "import"]:
                p_name = f"{parameter.name}_"
                parameter_info.append([p_name,  _to_pascal_case(list(parameter.type_tags)[0])])
            else:
                parameter_info.append([parameter.name,  _to_pascal_case(list(parameter.type_tags)[0])])

        param_definitions = [f"{p_name}: {p_type}" for p_name, p_type in parameter_info]
        param_definitions.append("world: WorldState = GLOBAL_WORLD")
        params_str = ", ".join(param_definitions)

        lines.append("")
        function_def_line = f"def {action_name}({params_str}) -> bool:"
        lines.append(function_def_line)

        precondition_content = logic_tree_handler(action.precondition)
        prompt_content1 = f"    precondition = {precondition_content}"
        lines.append(prompt_content1)
        prompt_content2 = "    errors = check_facts(precondition)"
        lines.append(prompt_content2)
        lines.append("    if errors:")
        lines.append(f'        error_summary = "\\n".join([f"    - {{msg}}" for msg in errors])')
        args_str = ",".join([f"{{{p[0]}.name}}" for p in parameter_info])
        lines.append(f'        raise ActionPreconditionError(f"Precondition unmet for action {action.name}({args_str}):\\n{{error_summary}}")')
        lines.append("")

        lines.append("    global global_step_counter")
        lines.append("    global_step_counter += 1")

        effect_content = logic_tree_handler(action.effect)
        prompt_content3 = f"    effects = {effect_content}"
        lines.append(prompt_content3)
        lines.append("    try:")
        lines.append("        apply_effects(effects)")
        lines.append("    except Exception as e:")
        lines.append('        print(f"!!! [System Panic] Execution crashed: {e}")')
        lines.append("        return False")
        lines.append(f'    print(f"    Action {action.name}({args_str}) executed successfully. State updated.")')
        lines.append("    return True")

    # 【终极拼接】将前面收集好的查询函数追加到文件末尾
    lines.extend(query_functions_code)

    full_content = "\n".join(lines)

    try:
        with open(python_save_path, "w", encoding="utf-8") as f:
            f.write(full_content)
    except IOError as e:
        print(f"Error writing file to {python_save_path}: {e}")



if __name__ == "__main__":
    pddl_path = r"E:\Python Projects\PlanPy\data\raw-pddl\depots-numeric-automatic\domain.pddl"
    save_path = r"E:\Python Projects\PlanPy\domain.py"

    translate_domain(pddl_path, save_path)












