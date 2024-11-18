from dsl import *
from dsl2 import *
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Callable, Optional
from collections import defaultdict
import logging
import traceback
from config import *
from dslIsDo import safe_execute



def update_objects_proper_flags(input_grid, output_grid, flags):
    # 提取输入和输出的对象集合
    input_objects = objects(input_grid, True, True, True)
    output_objects = objects(output_grid, True, True, True)

    # 获取第一个对象
    input_first_obj = first(input_objects)
    output_first_obj = first(output_objects)

    # 更新 flags 中的对象信息，确保不重复添加
    if "input_first_obj" not in flags:
        flags["input_first_obj"] = input_first_obj
    if "output_first_obj" not in flags:
        flags["output_first_obj"] = output_first_obj
    if "all_obj" not in flags:
        flags["all_obj"] = {"input": input_objects, "output": output_objects}
    else:
        flags["all_obj"]["input"] = input_objects
        flags["all_obj"]["output"] = output_objects



# 初始化 flags 的函数


def update_proper_in_out_flags(input_grid: Grid, output_grid: Grid, flagK: Dict[str, List[bool]]) -> None:
    """
    更新 proper_functions 中函数的正向及反向操作的标志。

    参数:
    input_grid: Grid - 输入网格。
    output_grid: Grid - 输出网格。
    flags: Dict[str, List[bool]] - 标志字典。
    """
    # from config import proper_functions
    I = input_grid
    O = output_grid
    height_i, width_i = height(I), width(I)    # 输入对象的高度和宽度
    height_o, width_o = height(O), width(O)    # 输出对象的高度和宽度

    flagK["height_ratio"].append(height_o / height_i)
    flagK["width_ratio"].append( width_o / width_i)

    for fun in proper_functions:
        # 正向操作
        transformed = safe_execute(fun, input_grid)
        if transformed == output_grid:
            flagK["in_out_fun"].append(fun)
            # continue

        # 反向操作
        transformed = safe_execute(fun, output_grid)
        if transformed == input_grid:
            flagK["out_in_fun"].append(fun)
