from dsl import *
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Callable, Optional
import logging
import traceback
from config import *
from dsl2 import *


def is_underfill_corners(task, flags) -> bool:
    train_data = task['train']
    for data_pair in train_data:
        I = data_pair['input']
        O = data_pair['output']
        _, _, colorset, _ = getIO_diff(I, O, flags)
        # merged_diffs = {
        #     "diff2": defaultdict(list)
        # }
        # # 将 diff1_unique 中的数据按第一个值分组
        # for value, coord in diff2:
        #     merged_diffs["diff2"][value].append(coord)
        # colorset = set(merged_diffs["diff2"])
        with safe_context():
            if len(colorset) == 1:
                color = next(iter(colorset))
                # flags["out_train_i_diff_color_is"].append(color)
                if underfill_corners(I, color) == O:
                    flags["is_underfill_corners_color"].append(color)

    all_values_equal = len(set(flags["is_underfill_corners_color"])) == 1
    if all_values_equal:
        color = next(iter(flags["is_underfill_corners_color"]))
        # flags["out_train_all_diff_color_is"] = color
        I = task['test'][0]['input']
        O = underfill_corners(I, color)
        return O
    return False


def is_input_firstobjsame_outallobject():
    return


# # 如何判断是get_first_object
# def is_a_object_of(I,O,flags):
#     x1 = objects(I, T, T, T)
#     # O is  partof x1
#     flags["is_a_object_of"] = True
#     return
# def is_move():
#     return


def do_portrait_half(I: Grid) -> Grid:
    x1 = portrait(I)
    x2 = branch(x1, tophalf, lefthalf)
    O = x2(I)
    return O


def get_most_colors_part(grid: Grid) -> Grid:
    x1 = vsplit(grid, TWO)
    x2 = rbind(hsplit, TWO)
    x3 = mapply(x2, x1)

    subgrids = split_grid_by_indices(grid, include_lines=False)
    O = argmax(subgrids, numcolors)
    return O

def do_not_mirror_part(I: Grid, O: Grid) -> bool:
    if is_half_mirror(I):
        return False
    # x1 = vsplit(I, THREE)
    # x2 = fork(equality, dmirror, identity)
    # x3 = compose(flip, x2)
    # O = extract(x1, x3)
    # return O
def is_not_mirror_part(I: Grid, O: Grid) -> bool:
    if O == half_not_mirror(I):
        return True
    if O == third_not_mirror(I):
        return True
    return False

# def is_output_most_input_color(I, O) -> bool:
#     """
#     判断 output 是否完全由 input 中出现最多的颜色组成。

#     参数:
#     - task (Dict[str, Any]): 包含 'input' 和 'output' 的任务字典，分别为二维列表。

#     返回:
#     - bool: 如果 output 由 input 中的最多颜色组成，返回 True；否则返回 False。
#     """
#     input_grid = I
#     output_grid = O

#     # 统计每种颜色的出现次数
#     color_counts = {}
#     for row in input_grid:
#         for color in row:
#             if color in color_counts:
#                 color_counts[color] += 1
#             else:
#                 color_counts[color] = 1

#     # 找到出现次数最多的颜色
#     most_common_color = max(color_counts, key=color_counts.get)

#     # 检查 output 中是否完全由该颜色组成
#     for row in output_grid:
#         if any(cell != most_common_color for cell in row):
#             return False

#     return True


# def is_mirror_hole_get_args(task: Dict, flags: Dict[str, List[bool]]) -> List[Any]:
#     """    判断是否是镜像扣洞任务，并获取参数。
#     参数:
#     - task (Dict[str, Any]): 包含 'input' 和 'output' 的任务字典，分别为二维列表。
#     - flags (Dict[str, List[bool]]): 用于控制任务执行的标志字典。
#     返回:
#     - List[Any]: 镜像扣洞任务的参数列表，包括扣洞的位置和大小。    """

#     # need judge is !! mirror,half is mirrir otherhalf not mirror
#     # color is size big obj obj(I,T,T,F)  default zero,hole in the not mirror part
#     # todo complete
#     train_data = task['train']
#     test_data = task['test']

#     # 遍历所有训练样本
#     for data_pair in train_data:
#         input_grid = data_pair['input']
#         output_grid = data_pair['output']

#         result = is_half_mirror(input_grid)
#         if result:
#             if len(result) == 2:
#                 # where is the hole   how to mirror                         # 获取镜像扣洞的参数
#                 is_mirror_hole_get_args

#     return get_hole(input_grid)


# def get_hole(input_grid):
#     # or  get  Space-time portal
#     assert False


def is_half_mirror(grid: Grid) -> bool:
    """
    判断网格是否满足一半是镜像的，另一半不是镜像的。

    参数:
    grid: Grid - 输入的网格。

    返回:
    bool - 如果网格满足条件，返回 True；否则返回 False。
    """
    # 获取网格的不同部分
    left_half = lefthalf(grid)
    right_half = righthalf(grid)
    top_half = tophalf(grid)
    bottom_half = bottomhalf(grid)

    # 检查左右两半是否自身是镜像的
    is_left_mirror = left_half == vmirror(left_half)
    is_right_mirror = right_half == vmirror(right_half)

    # 检查上下两半是否自身是镜像的
    is_top_mirror = top_half == hmirror(top_half)
    is_bottom_mirror = bottom_half == hmirror(bottom_half)

    # 检查是否满足一半是镜像的，另一半不是镜像的
    true_conditions = []
    if is_left_mirror:
        true_conditions.append("is_left_mirror")
    if is_right_mirror:
        true_conditions.append("is_right_mirror")
    if is_top_mirror:
        true_conditions.append("is_top_mirror")
    if is_bottom_mirror:
        true_conditions.append("is_bottom_mirror")

    return true_conditions


def process_value(value: bool) -> Any:
    return


def preprocess_noise(task):
    """
    now just for #18, 5614dbcf
    """
    # 遍历任务中的所有训练和测试样本
    for sample in task['train'] + task['test']:
        input2dgrid = sample['input']
        # 找到所有噪声位置
        noise = ofcolor(input2dgrid, 5)
        replaced_grid = input2dgrid

        # 遍历每个噪声位置，替换为其邻居的主要颜色
        for n in noise:
            # 获取噪声位置的邻居
            neighbors_list = neighbors(n)
            neighbor_colors = [index(input2dgrid, pos) for pos in neighbors_list if index(
                input2dgrid, pos) is not None]
            # 计算邻居颜色的频率
            most_color = mostcolor2(neighbor_colors)
            # 将噪声位置的值替换为邻居中最多的颜色
            replaced_grid = replace2(replaced_grid, n, most_color)
        # 更新 sample 中的 input 为替换后的网格
        sample['input'] = replaced_grid
    return task


def safe_execute(fun, *args):
    try:
        # 调用传入的函数并传递参数
        result = fun(*args)
        return result
    except Exception as e:
        # 捕获异常并打印错误信息
        logging.error("捕获到异常：%s", e)
        # logging.error("详细错误信息：\n%s", traceback.format_exc())
        pass


def do_check_inputOutput_proper_1_arg_functions(proper_1arg_functions, task: Dict, flags: Dict[str, List[bool]]):
    train_data = task['train']
    test_data = task['test']
    print('do_check_inputOutput_proper_1___arg___functions')

    flags.setdefault("ok_fun", [])

    I = train_data[0]['input']
    O = train_data[0]['output']

    height_i, width_i = height(I), width(I)    # 输入对象的高度和宽度
    height_o, width_o = height(O), width(O)    # 输出对象的高度和宽度

    if height_o > height_i and width_o > width_i:
        height_ratio = int(height_o / height_i)
    elif height_o < height_i and width_o < width_i:
        height_ratio = int(height_i / height_o)
    else:
        height_ratio = 0

    # get proper and  args

    for fun in proper_1arg_functions:
        # if "half" in fun.__name__:
        #     flags["out_in"] = True
        # else:
        #     ##!!!!!! set false after use
        #     flags["out_in"] = False

        if fun == switch or fun == replace:
            args = []
            funarg = prepare_diff(task, flags)
            if funarg:
                if len(funarg) == 3:
                    funget, arg1, arg2 = funarg
                    if funget == fun:
                        fun = funget
                        args = [arg1, arg2]
        elif fun == crop:
            args = []
            funarg = is_subgrid(task, flags)
            if funarg:
                if len(funarg) == 3:
                    funget, arg1, arg2 = funarg
                    args = [(arg1, arg2), (height_o, width_o)]
                    fun = funget
                    # if funget == fun:
                    # fun = funget
                    # args = [arg1, arg2]
        elif fun == do_neighbour_color:
            # is_underfill_corners(task, flags)  had exe this
            for data_pair in train_data:
                I = data_pair['input']
                O = data_pair['output']
                getIO_diff(I, O, flags)

            all_values_equal = len(
                set(flags["out_train_i_diff_color_is"])) == 1
            if all_values_equal:
                color = next(iter(flags["out_train_i_diff_color_is"]))
                flags["out_train_all_diff_color_is"] = color
                args = [color]
        else:
            args = [height_ratio]

        success = True
        for data_pair in train_data:
            input_grid = data_pair['input']
            output_grid = data_pair['output']

            # if fun == crop:
            #     args = []
            #     height_o, width_o = height(output_grid), width(output_grid)
            #     funarg = is_subgrid(task, flags)
            #     if funarg:
            #         if len(funarg) == 3:
            #             funget, arg1, arg2 = funarg
            #             args = [(arg1, arg2), (height_o, width_o)]
            #             fun = funget
            # fun(output_grid)
            if flags.get("out_in") == True:
                transformed = safe_execute(fun, output_grid, *args)
                if transformed == input_grid:
                    # out-input-proper_flags
                    continue

            # fun(input_grid)
            transformed = safe_execute(fun, input_grid, *args)
            if transformed == output_grid:
                # out-input-proper_flags
                continue

            # else:
            # print(f"failed : {fun.__name__}")
            success = False
            break
        if success:
            print(f"ok____ : {fun.__name__}")
            flags["ok_fun"].append([fun, *args])
            # height_ratio is args to exe
            # return fun, height_ratio
        else:
            # print(f"failed : {fun.__name__}")
            pass

    print('do_check_inputOutput_proper_1___arg___functions')
    return flags["ok_fun"] if flags["ok_fun"] else [False]


def do_check_inputOutput_proper_1functions(proper_functions, task: Dict, flags: Dict[str, List[bool]]):
    train_data = task['train']
    # test_data = task['test']
    print('do_check_inputOutput___proper___functions')
    flags.setdefault("ok_fun", [])
    for fun in proper_functions:

        # if "concat" in fun.__name__:
        #     flags["out_in"] = True
        # else:
        #     ##!!!!!! set false after use
        #     flags["out_in"] = False

        success = True
        for data_pair in train_data:
            input_grid = data_pair['input']
            output_grid = data_pair['output']

            # # fun(output_grid)
            # transformed = fun(output_grid)
            # if transformed == output_grid:
            #     # flags["out_out"].append(True)
            #     continue

            # if transformed == input_grid:
            #     # flags["out_in"].append(True)
            #     continue

            # # fun(input_grid)
            # transformed = fun(input_grid)
            # if transformed == output_grid:
            #     # flags["in_out"].append(True)
            #     continue

            # if transformed == input_grid:
            #     # flags["in_in"].append(True)
            #     continue

            # fun(output_grid)
            # if flags["out_in"] == True:
            #     transformed = safe_execute(fun, output_grid)
            #     if transformed == input_grid:
            #         # out-input-proper_flags
            #         continue
            if flags.get("out_in") == True:
                transformed = safe_execute(fun, output_grid)
                if transformed == input_grid:
                    # out-input-proper_flags
                    continue

            # fun(input_grid)
            transformed = safe_execute(fun, input_grid)
            if transformed == output_grid:
                # out-input-proper_flags
                continue

            # else:
            # print(f"failed : {fun.__name__}")
            success = False
            break
        if success:
            # all_out_in_ok = all(flags["out_in"])
            print(f"ok____ : {fun.__name__}")
            flags["ok_fun"].append(fun)            # return fun
            # if all_out_in_ok:
            #     return fun , 'out_in'
            # else:
            #     return fun
        else:
            pass
            # print(f"failed : {fun.__name__}")
    # 验证成功几个函数
    print('do_check_inputOutput___proper___functions')
    return flags["ok_fun"] if flags["ok_fun"] else [False]


# 扩展后的多函数任务处理函数
def do_4fun_task(
        input_grid: list,
        flags: Dict[str, List[Any]],
        fun1: Callable[[Any], Any], args1: List[Any],
        fun2: Callable[[Any], Any], args2: List[Any],
        fun3: Optional[Callable[[Any], Any]] = None, args3: Optional[List[Any]] = None,
        fun4: Optional[Callable[[Any], Any]] = None, args4: Optional[List[Any]] = None) -> Any:
    # 将函数和参数绑定到列表中，方便按顺序调用
    functions = [(fun1, args1), (fun2, args2), (fun3, args3), (fun4, args4)]

    # 获取顺序
    order = flags.get("order", [1, 2, 3, 4])
    oldinput_grid = input_grid
    # 根据顺序调用函数
    for idx in order:
        fun, args = functions[idx - 1]  # idx-1是因为order是从1开始的
        if flags.get(f"use_fun{idx}", [True])[0]:  # 检查是否需要调用当前函数

            # oldinput_grid = input_grid

            if "concat_first_obj" in fun.__name__:
                input_grid = fun(
                    input_grid, *args) if args else fun(input_grid)
            elif "concat" in fun.__name__:
                # 执行包含 "concat" 的函数
                if args == ['pin', 'in']:
                    input_grid = fun(input_grid, oldinput_grid)
                    oldinput_grid = input_grid
                if args == ['in', 'pin']:
                    input_grid = fun(oldinput_grid, input_grid)
                    oldinput_grid = input_grid
                if args == ['in', 'in']:
                    input_grid = fun(oldinput_grid, oldinput_grid)
                    oldinput_grid = input_grid
                # else:
                #     input_grid = fun(input_grid, *args) if args else fun(input_grid)
            else:
                input_grid = fun(
                    input_grid, *args) if args else fun(input_grid)

    out = input_grid
    return out


def do_check_train_get_test(
    do_4fun_task: Callable,
    task: List[Dict],
    flags: Dict[str, List[bool]],
    fun1: Callable[[Any], Any], args1:  Optional[List[Any]] = None,
    fun2: Optional[Callable[[Any], Any]] = None,  args2: Optional[List[Any]] = None,
    fun3: Optional[Callable[[Any], Any]] = None, args3: Optional[List[Any]] = None,
    fun4: Optional[Callable[[Any], Any]] = None, args4: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """
    依次执行多批任务，每一批任务都调用 do_4fun_task 函数，返回每批任务的执行结果。

    参数:
    - do_4fun_task (Callable): 执行函数，接收每批任务的具体逻辑。
    - tasks (List[Dict]): 包含多批任务的列表，每个任务包含 train 和 test 数据。
    - flags (Dict[str, List[bool]]): 用于控制任务执行的标志字典。
    - fun1, fun2, fun3, fun4 (Callable): 处理任务的函数，可选传递。
    - args1, args2, args3, args4 (List[Any]): 对应函数的参数列表。

    返回:
    - Dict[str, Any]: 每批任务的执行结果字典。
    """
    all_results = {}  # 存储每批任务的执行结果

    train_data = task['train']
    test_data = task['test']

    for data_pair in train_data:
        input_grid = data_pair['input']
        output_grid = data_pair['output']

        # 使用传入的函数 fun 来检查是否满足条件
        transformed = do_4fun_task(
            input_grid, flags, fun1, args1, fun2, args2, fun3, args3, fun4, args4)
        # display_diff_color_ofa_matrices(transformed)
        # display_diff_color_ofa_matrices(output_grid)
        if transformed == output_grid:
            # flags["is_fun_ok"].append(True)
            continue  # 结束本轮循环，直接进行下一个 data_pair
        else:
            print(f"failed : {do_4fun_task.__name__}")
            # return f'failed {fun.__name__}'
            return False
    print(f"ok__________ : {do_4fun_task.__name__}")
    print(f"ok___ fun1 _ : {fun1.__name__}")
    input_grid = test_data[0]['input']
    testin = do_4fun_task(input_grid, flags, fun1, args1,
                          fun2, args2, fun3, args3, fun4, args4)

    assert testin == test_data[0]['output']
    print(f"ok_________- : test - ok ")
    return testin


class BidirectionalMap:
    def __init__(self, is_do_mapping):
        self.forward = mapping  # 正向映射
        # 构建反向映射，将每个函数映射到对应的键（可能有多个键映射到相同函数）
        self.reverse = {}
        for k, v in mapping.items():
            if v in self.reverse:
                self.reverse[v].append(k)
            else:
                self.reverse[v] = [k]

    def get(self, key):
        print('! convert a function ')
        return self.forward.get(key) or self.reverse.get(key)
