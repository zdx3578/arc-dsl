from dsl import *
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Callable, Optional
from collections import defaultdict
import logging
import traceback
from config import *


def is_output_most_input_color(I, O) -> bool:
    """
    判断 output 是否完全由 input 中出现最多的颜色组成。

    参数:
    - task (Dict[str, Any]): 包含 'input' 和 'output' 的任务字典，分别为二维列表。

    返回:
    - bool: 如果 output 由 input 中的最多颜色组成，返回 True；否则返回 False。
    """
    input_grid = I
    output_grid = O

    # 统计每种颜色的出现次数
    color_counts = {}
    for row in input_grid:
        for color in row:
            if color in color_counts:
                color_counts[color] += 1
            else:
                color_counts[color] = 1

    # 找到出现次数最多的颜色
    most_common_color = max(color_counts, key=color_counts.get)

    # 检查 output 中是否完全由该颜色组成
    for row in output_grid:
        if any(cell != most_common_color for cell in row):
            return False

    return True


def do_output_most_input_color(color, h, w):

    return canvas(color, (h, w))


def safe_execute(fun, *args):
    try:
        # 调用传入的函数并传递参数
        result = fun(*args)
        return result
    except Exception as e:
        # 捕获异常并打印错误信息
        logging.error("捕获到异常：%s", e)
        logging.error("详细错误信息：\n%s", traceback.format_exc())
        return None


def do_check_inputOutput_proper_1functions(proper_functions, task: Dict, flags: Dict[str, List[bool]]):
    train_data = task['train']
    # test_data = task['test']
    flags.get("ok_fun", [])
    for fun in proper_functions:
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
            if flags["out_in"] == True:
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
            print(f"failed : {fun.__name__}")
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
            print(f"failed : {fun.__name__}")
    # 验证成功几个函数
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

    # 根据顺序调用函数
    for idx in order:
        fun, args = functions[idx - 1]  # idx-1是因为order是从1开始的
        if flags.get(f"use_fun{idx}", [True])[0]:  # 检查是否需要调用当前函数
            if args == 'sameinput':  # or args == [['righthalf', 'lefthalf']]:
                input_grid = fun(input_grid, input_grid)
            # elif args == '' :
            #     input_grid = fun(input_grid, input_grid)
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
        if transformed == output_grid:
            # flags["is_fun_ok"].append(True)
            continue  # 结束本轮循环，直接进行下一个 data_pair
        else:
            print(f"failed : {do_4fun_task.__name__}")
            # return f'failed {fun.__name__}'
            return False
    print(f"ok____ : {do_4fun_task.__name__}")
    input_grid = test_data[0]['input']
    testin = do_4fun_task(input_grid, flags, fun1, args1,
                          fun2, args2, fun3, args3, fun4, args4)

    assert testin == test_data[0]['output']
    print(f"ok____ - test - ok ")
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


def prepare_diff(task, flags: Dict[str, bool]):
    train_data = task['train']
    test_data = task['test']

    flags["is_diff_same_posit"] = []

    for data_pair in train_data:
        I = data_pair['input']
        O = data_pair['output']

        # 调用 objects 函数两次
        oi = objects(I, False, True, False)
        oo = objects(O, False, True, False)

        same_objects = oi.intersection(oo)
        # 获取对称差集
        # diff_objects = oi.symmetric_difference(oo)

        # # 检查是否恰好有两个不同部分
        # # if len(diff_objects) == 2:
        #     # 解包不同部分为两个 frozenset
        # diff1, diff2 = diff_objects

        oi_unique = oi - oo  # 获取在 oi 中但不在 oo 中的元素
        oo_unique = oo - oi  # 获取在 oo 中但不在 oi 中的元素

        # 将它们分别赋给 diff1 和 diff2
        diff1, diff2 = next(iter(oi_unique)), next(iter(oo_unique))

        # 将两个 frozenset 转换为有序列表
        sorted_diff1 = sorted(diff1, key=lambda x: (x[0], x[1]))  # 按值和坐标排序
        sorted_diff2 = sorted(diff2, key=lambda x: (x[0], x[1]))  # 按值和坐标排序

        # 输出排序后的比较结果
        # print("第一个 frozenset 排序后的元素:", sorted_diff1)
        # print("第二个 frozenset 排序后的元素:", sorted_diff2)
        # 比较差异
        diff1_unique = sorted(set(sorted_diff1) - set(sorted_diff2))
        diff2_unique = sorted(set(sorted_diff2) - set(sorted_diff1))

        # print("第一个 frozenset 特有的元素（排序后）:", diff1_unique)
        # print("第二个 frozenset 特有的元素（排序后）:", diff2_unique)

        merged_diffs = {
            "diff1": defaultdict(list),
            "diff2": defaultdict(list)
        }

        # 将 diff1_unique 中的数据按第一个值分组
        for value, coord in diff1_unique:
            merged_diffs["diff1"][value].append(coord)

        # 将 diff2_unique 中的数据按第一个值分组
        for value, coord in diff2_unique:
            merged_diffs["diff2"][value].append(coord)

        # # 输出合并后的差异
        # for key in merged_diffs:
        #     for value, positions in merged_diffs[key].items():
        #         print(f"{key} - 值 {value} 的特有坐标:", positions)

        # display_diff_matrices(diff1_unique,diff2_unique)

        if compare_positions(merged_diffs):
            flags["is_diff_same_posit"].append(True)
        else:
            flags["is_diff_same_posit"].append(False)

        if is_position_swapped(merged_diffs["diff1"], merged_diffs["diff2"]):
            flags["is_position_swap"].append(True)
        else:
            flags["is_position_swap"].append(False)

    is_diff_same_posit = all(flags["is_diff_same_posit"])
    all_is_position_swap_ok = all(flags["is_position_swap"])

    if len(list(merged_diffs['diff1'].keys())) >= 2:
        if all_is_position_swap_ok:
            keys_diff1 = list(merged_diffs['diff1'].keys())[0]  # 获取 diff1 中的键
            keys_diff2 = list(merged_diffs['diff1'].keys())[1]  # 获取  中的键
            print('switch', keys_diff1, keys_diff2)
            return switch, keys_diff1, keys_diff2

    elif is_diff_same_posit:
        keys_diff1 = list(merged_diffs['diff1'].keys())[0]  # 获取 diff1 中的键
        keys_diff2 = list(merged_diffs['diff2'].keys())[0]  # 获取 diff2 中的键
        print('replace', keys_diff1, keys_diff2)
        return replace, keys_diff1, keys_diff2
    return False

    # print("todo ！ 执行 ！  不同部分不止两个 frozenset 或无差异。")
    # return 0


def compare_positions(merged_diffs: Dict[str, defaultdict]) -> str:
    """
    比较 'diff1' 和 'diff2' 字典中的坐标列表是否完全一致。
    忽略字典的键，仅比较坐标部分。
    """
    # 提取 diff1 和 diff2 中的坐标列表，忽略键
    coords1 = [coord for coords in merged_diffs['diff1'].values()
               for coord in coords]
    coords2 = [coord for coords in merged_diffs['diff2'].values()
               for coord in coords]

    # 比较坐标列表是否一致
    if sorted(coords1) == sorted(coords2):
        # return "Identical positions"
        return True
    else:
        # return "Different positions"
        return False


def is_position_swapped(diff1: defaultdict, diff2: defaultdict) -> bool:
    for value1, positions1 in diff1.items():
        found_swap = False
        for value2, positions2 in diff2.items():
            # 跳过相同 value 的情况，只检查不同 value 的互换
            if value1 == value2:
                continue
            # 检查 positions 是否一致
            if sorted(positions1) == sorted(positions2):
                found_swap = True
                break
        # 如果当前 value1 没有找到对应的交换关系，返回 False
        if not found_swap:
            return False
    return True


def display_diff_matrices(diff1: List[Tuple[int, Tuple[int, int]]],
                          diff2: List[Tuple[int, Tuple[int, int]]],
                          diff3: Optional[List[Tuple[int, Tuple[int, int]]]] = None):
    """
    展示不同元素位置的二维矩阵。

    参数:
    - diff1, diff2: 必填，每个包含不同元素及其位置的集合。
    - diff3: 可选，额外的不同元素及其位置集合。
    """
    combined_diff = {}

    # 合并所有不同元素的位置
    for value, pos in diff1 + diff2 + (diff3 if diff3 else []):
        if value not in combined_diff:
            combined_diff[value] = []
        combined_diff[value].append(pos)

    # 展示每个数值在二维矩阵中的位置
    for key, positions in sorted(combined_diff.items()):
        print(f"数值 {key} 的不同元素位置：")

        # 确定矩阵的大小
        max_row = max(pos[0] for pos in positions) + 1
        max_col = max(pos[1] for pos in positions) + 1
        matrix = [[' ' for _ in range(max_col)] for _ in range(max_row)]

        # 填充矩阵
        for row, col in positions:
            matrix[row][col] = str(key)

        # 打印矩阵
        for row in matrix:
            print(" ".join(row))
        print("\n" + "-"*20 + "\n")


def is_subgrid(task, flags):
    """判断较小的网格是否是较大网格的一个部分"""
    train_data = task['train']
    test_data = task['test']

    for data_pair in train_data:
        grid1 = data_pair['input']
        grid2 = data_pair['output']

        # 获取两个矩阵的大小
        rows1, cols1 = len(grid1), len(grid1[0])
        rows2, cols2 = len(grid2), len(grid2[0])

        # 确定较大的矩阵和较小的矩阵
        if rows1 >= rows2 and cols1 >= cols2:
            big_grid, small_grid = grid1, grid2
            big_rows, big_cols, small_rows, small_cols = rows1, cols1, rows2, cols2
        elif rows2 >= rows1 and cols2 >= cols1:
            big_grid, small_grid = grid2, grid1
            big_rows, big_cols, small_rows, small_cols = rows2, cols2, rows1, cols1
        else:
            return False  # 两个矩阵形状不兼容，无法嵌套

        # 遍历大矩阵，检查是否存在小矩阵匹配的位置
        for i in range(big_rows - small_rows + 1):
            for j in range(big_cols - small_cols + 1):
                match = True
                # 检查大矩阵的当前位置是否与小矩阵完全匹配
                for x in range(small_rows):
                    for y in range(small_cols):
                        if big_grid[i + x][j + y] != small_grid[x][y]:
                            match = False
                            break
                    if not match:
                        break
                if match:
                    return crop, i, j  # 找到匹配位置，返回 True

    return False  # 未找到匹配位置，返回 False
