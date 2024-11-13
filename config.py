from dsl import *
from dsl2 import *
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
    # ! ! 每个任务都可能是一个 几个 独立的 高级 抽象 的is proper判断
    # 模式判断  patter，高级 复杂 抽象 patter；语言中的单词
    vmirror,
    hmirror,
    cmirror,
    dmirror,
    rot90,
    rot180,
    rot270,
    upper_third, middle_third, lower_third, left_third, center_third, right_third,

    bottomhalf,
    lefthalf,
    tophalf,
    righthalf,
    do_output_most_input_color,
    get_first_object,
    get_partition_min_subgrid,
    get_max_object,
    get_min_object,
    move_down,

    # out is what output
    # 扣洞
    # 叠加
    # 溢水
    # 直线框
    # 沿线，连线，
    # 墙柱
    # 单词含义

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
                         #  upper_third, middle_third, lower_third, left_third, center_third, right_third,

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
    几乎每个任务都是一个独立的操作单元的标志
    """
    return {
        "is_mirror": [],
        "in_is_out_mirror": [],
        "out_is_in_mirror": [],
        "is_fun_ok": [],
        "is_scale": [],
        "is_diff_same_posit": [],
        "is_position_swap": [],
        "is_rotation": [],
        "is_translation": [],
        "is_color_transform": [],
        "is_output_one_color": [],
        "is_output_most_input_color": [],

        'is_subgrid': [],
        "out_is_in_subgrid": [False],
        "in_is_out_subgrid": [False],

        "out_is_in_third": [],
        "is_in_third_left": [],

        'is_a_object_of': [],

        'height_ratio': [],
        'width_ratio': [],
        'output_allone_color': [],
        "position_same_contained": [],
        "diff_position_same": [],
        "diff_in_same_contained": False,
        "diff_in_same_fg": [],
        "is_in_box_change_color": [],
        "is_in_box_change_shape": [],
        "is_in_box_change_size": [],
        "is_in_box_change_position": [],



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

        #  property
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
        # process in to out
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
