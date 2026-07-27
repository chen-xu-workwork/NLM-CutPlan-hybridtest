"""Inference prompt templates matching the PyPACE no-assert training data."""

SYSTEM_PROMPT_WITH_DOMAIN = """
你的最终输出必须严格由两部分组成：
1. 纯粹推演过程：用 `<think>` 和 `</think>` 包裹。你必须以第一人称视角在此进行深度的逻辑推理、路径规划、数值验算（如卡车是否超载）以及状态冲突排查。
2. 执行代码块：用 ```python 和 ``` 包裹。这是一套流式的动作序列（Plan），目标是让世界从初始状态转移到目标状态。

在这个任务中最重要最基本的就是要保证每个动作在执行前其所有的前提条件都满足。每个动作的具体要求详见下面的手册。任务的首要目标是达成目标状态；此外可以尝试优化使得总体的油耗最低。
# 物流世界运转法则 (Domain Reference)
{domain_code}
"""


USER_PROMPT_WO_DOMAIN = """
请仔细阅读以下具体的物流任务场景。

{problem_description}

"""
