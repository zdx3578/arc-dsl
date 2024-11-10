from dsl import *
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Callable, Optional
from collections import defaultdict
# from solvers2 import *
# from solvers2 import do_output_most_input_color
from dslIsDo import *

# part_functions = [
#     # 长方形
#     righthalf,
#     lefthalf,
#     bottomhalf,
#     tophalf
# ]

proper_functions = [
    # out is what input 正方形
    vmirror,
    hmirror,
    cmirror,
    dmirror,
    rot90,
    rot180,
    rot270,

    bottomhalf,
    lefthalf,
    tophalf,
    righthalf,
    # do_output_most_input_color,
    # out is what output

    # in is what output
]
# 先验证  replace switch
proper_1arg_functions = [upscale,
                         hupscale,
                         vupscale,
                         downscale,
                         replace,
                         switch,
                         crop,

                         ]

# proper_concat_functions = [hconcat,
#                            vconcat]

# proper_Complex_functions = [is_output_most_input_color]

is_do_mapping = {
    # out is what input
    vmirror: vmirror,
    hmirror: hmirror,
    cmirror: cmirror,
    dmirror: dmirror,
    rot90: rot90,
    rot180: rot180,
    rot270: rot270,

    upscale: upscale,
    hupscale: hupscale,
    vupscale: vupscale,
    downscale: downscale,

    hconcat: hconcat,
    vconcat: vconcat,
    # replace:replace,

    bottomhalf: vconcat, lefthalf: hconcat, tophalf: vconcat, righthalf: hconcat,
    is_output_most_input_color: do_output_most_input_color,
    # in is what output
}

# 初始化标志变量的函数


def initialize_flags() -> Dict[str, List[bool]]:
    """
    初始化一组标志变量。

    返回:
    - Dict[str, List[bool]]: 标志变量的字典，默认为 False。
    """
    return {
        "is_mirror": [],
        "is_fun_ok": [],
        "is_scale": [],
        "is_diff_same_posit": [],
        "is_position_swap": [],
        "is_rotation": [],
        "is_translation": [],
        "is_color_transform": [],
        "is_output_one_color": [],
        "is_output_most_input_color": [],
        # # template
        # "in_is_?": [],
        # "out_is_??": [],
        # "out_of_in_is_??": [],
        # "in_of_out_is_?": [],
        # #
        # "all_in_is_?": [],
        # "all_out_is_??": [],
        # "all_out_of_in_is_??": [],
        # "all_in_of_out_is_?": [],

        #
        "in_out": [],
        "out_in": [],
        "out_out": [],
        "in_in": [],
        "in_out_fun": [],
        "out_in": [],
        "out_out_fun": [],
        "in_in": [],
        #
        "all_in_is_?": [],
        "all_out_is_??": [],
        "all_out_of_in_is_??": [],
        "all_in_of_out_is_?": [],

        'ok_fun': [],

        # 控制每个函数是否执行
        "use_fun1": [True],
        "use_fun2": [False],
        "use_fun3": [False],  # 默认不执行
        "use_fun4": [False],
        # 执行顺序 (可以根据条件动态修改)
        "order": [1, 2, 4],

        "in_out_fun": [],
        "out_in_fun": [],
        "out_out_fun": [],
        "in_in_fun": []

    }


# def is_judg_result_fun_flags() -> Dict[str, List[bool]]:
#     #in out is input process to output
#     return {
#         "in_out_fun": [],
#         "out_in_fun": [],
#         "out_out_fun": [],
#         "in_in_fun": []
# }
