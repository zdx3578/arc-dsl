from dsl import *
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Callable, Optional
from collections import defaultdict
import logging
import traceback
from config import *


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

# 初始化 flags 的函数






def update_proper_in_out_flags(input_grid, output_grid, flags):

    height_i, width_i = height(input_grid), width(input_grid)
    height_o, width_o = height(output_grid), width(output_grid)
    flags["height_ratio"] = height_o/height_i
    flags["width_ratio"] = width_o/width_i

    # 正向操作
    if vmirror(input_grid) == output_grid and vmirror not in flags["in_out_fun"]:
        flags["in_out_fun"].append(vmirror)
    if hmirror(input_grid) == output_grid and hmirror not in flags["in_out_fun"]:
        flags["in_out_fun"].append(hmirror)
    if cmirror(input_grid) == output_grid and cmirror not in flags["in_out_fun"]:
        flags["in_out_fun"].append(cmirror)
    if dmirror(input_grid) == output_grid and dmirror not in flags["in_out_fun"]:
        flags["in_out_fun"].append(dmirror)
    if rot90(input_grid) == output_grid and rot90 not in flags["in_out_fun"]:
        flags["in_out_fun"].append(rot90)
    if rot180(input_grid) == output_grid and rot180 not in flags["in_out_fun"]:
        flags["in_out_fun"].append(rot180)
    if rot270(input_grid) == output_grid and rot270 not in flags["in_out_fun"]:
        flags["in_out_fun"].append(rot270)
    if upper_third(input_grid) == output_grid and upper_third not in flags["in_out_fun"]:
        flags["in_out_fun"].append(upper_third)
    if middle_third(input_grid) == output_grid and middle_third not in flags["in_out_fun"]:
        flags["in_out_fun"].append(middle_third)
    if lower_third(input_grid) == output_grid and lower_third not in flags["in_out_fun"]:
        flags["in_out_fun"].append(lower_third)
    if left_third(input_grid) == output_grid and left_third not in flags["in_out_fun"]:
        flags["in_out_fun"].append(left_third)
    if center_third(input_grid) == output_grid and center_third not in flags["in_out_fun"]:
        flags["in_out_fun"].append(center_third)
    if right_third(input_grid) == output_grid and right_third not in flags["in_out_fun"]:
        flags["in_out_fun"].append(right_third)
    if bottomhalf(input_grid) == output_grid and bottomhalf not in flags["in_out_fun"]:
        flags["in_out_fun"].append(bottomhalf)
    if lefthalf(input_grid) == output_grid and lefthalf not in flags["in_out_fun"]:
        flags["in_out_fun"].append(lefthalf)
    if tophalf(input_grid) == output_grid and tophalf not in flags["in_out_fun"]:
        flags["in_out_fun"].append(tophalf)
    if righthalf(input_grid) == output_grid and righthalf not in flags["in_out_fun"]:
        flags["in_out_fun"].append(righthalf)

    # 反向操作
    if vmirror(output_grid) == input_grid and vmirror not in flags["out_in_fun"]:
        flags["out_in_fun"].append(vmirror)
    if hmirror(output_grid) == input_grid and hmirror not in flags["out_in_fun"]:
        flags["out_in_fun"].append(hmirror)
    if cmirror(output_grid) == input_grid and cmirror not in flags["out_in_fun"]:
        flags["out_in_fun"].append(cmirror)
    if dmirror(output_grid) == input_grid and dmirror not in flags["out_in_fun"]:
        flags["out_in_fun"].append(dmirror)
    if rot90(output_grid) == input_grid and rot90 not in flags["out_in_fun"]:
        flags["out_in_fun"].append(rot90)
    if rot180(output_grid) == input_grid and rot180 not in flags["out_in_fun"]:
        flags["out_in_fun"].append(rot180)
    if rot270(output_grid) == input_grid and rot270 not in flags["out_in_fun"]:
        flags["out_in_fun"].append(rot270)
    if upper_third(output_grid) == input_grid and upper_third not in flags["out_in_fun"]:
        flags["out_in_fun"].append(upper_third)
    if middle_third(output_grid) == input_grid and middle_third not in flags["out_in_fun"]:
        flags["out_in_fun"].append(middle_third)
    if lower_third(output_grid) == input_grid and lower_third not in flags["out_in_fun"]:
        flags["out_in_fun"].append(lower_third)
    if left_third(output_grid) == input_grid and left_third not in flags["out_in_fun"]:
        flags["out_in_fun"].append(left_third)
    if center_third(output_grid) == input_grid and center_third not in flags["out_in_fun"]:
        flags["out_in_fun"].append(center_third)
    if right_third(output_grid) == input_grid and right_third not in flags["out_in_fun"]:
        flags["out_in_fun"].append(right_third)
    if bottomhalf(output_grid) == input_grid and bottomhalf not in flags["out_in_fun"]:
        flags["out_in_fun"].append(bottomhalf)
    if lefthalf(output_grid) == input_grid and lefthalf not in flags["out_in_fun"]:
        flags["out_in_fun"].append(lefthalf)
    if tophalf(output_grid) == input_grid and tophalf not in flags["out_in_fun"]:
        flags["out_in_fun"].append(tophalf)
    if righthalf(output_grid) == input_grid and righthalf not in flags["out_in_fun"]:
        flags["out_in_fun"].append(righthalf)
