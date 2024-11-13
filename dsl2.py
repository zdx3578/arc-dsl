from collections import Counter, defaultdict
from dsl import *
from typing import Dict, Any, List, Tuple, Callable, Optional
import logging
import traceback
# from config import *
# from config import *
# from dslIsDo import *
# from oldfun import *


def contains_object(obj: Object, objects: List[Object]) -> bool:
    """检查对象是否包含在对象列表中"""
    obj_set = set(obj)
    objects_sorted = [sorted(o) for o in objects]

    for other_obj in objects_sorted:
        if obj_set.issubset(set(other_obj)):
            return other_obj
    return False


def complementofobject(obj: Object) -> Object:
    """获取对象在矩阵中的补集"""
    obj_indices = toindices(obj)
    ul = ulcorner(obj_indices)
    lr = lrcorner(obj_indices)
    h, w = lr[0] - ul[0] + 1, lr[1] - ul[1] + 1

    # 创建一个全 1 矩阵
    full_matrix = [[1] * w for _ in range(h)]

    # 将对象矩阵中的值从全 1 矩阵中减去
    for _, (i, j) in obj:
        full_matrix[i - ul[0]][j - ul[1]] = 0

    # 将补集矩阵转换为对象类型
    complement_obj = set()
    for i in range(h):
        for j in range(w):
            if full_matrix[i][j] == 1:
                complement_obj.add((1, (ul[0] + i, ul[1] + j)))

    return complement_obj


def move2(obj: Object, direction: Tuple[int, int]) -> Object:
    """
    移动对象到指定方向。

    参数:
    obj: Object - 要移动的对象。
    direction: Tuple[int, int] - 移动的方向，格式为 (dx, dy)。

    返回:
    Object - 移动后的对象。
    """
    dx, dy = direction
    moved_obj = frozenset({(value, (i + dx, j + dy)) for value, (i, j) in obj})
    return moved_obj


def object_to_rectangle(obj: Object) -> Grid:
    """ 将对象扩展为包含对象的一个长方形矩阵 """
    indices = toindices(obj)
    if not indices:
        return tuple()

    ul = ulcorner(indices)
    lr = lrcorner(indices)
    h, w = lr[0] - ul[0] + 1, lr[1] - ul[1] + 1

    # 创建一个空的矩阵
    rectangle = [[0] * w for _ in range(h)]

    # 填充矩阵
    for value, (i, j) in obj:
        rectangle[i - ul[0]][j - ul[1]] = value

    return tuple(tuple(row) for row in rectangle)


def mostcolor2(colors: list) -> int:
    """ 返回列表中出现次数最多的颜色 """
    if not colors:  # 如果列表为空，返回 None 或其默认值
        return None
    count = Counter(colors)  # 统计颜色出现的次数
    most_common_color, _ = count.most_common(1)[0]  # 获取出现次数最多的颜色
    return most_common_color


def replace2(grid, position, new_value):
    """替换网格中特定位置的值"""
    i, j = position

    # 将 grid 转换为列表，以便进行修改
    new_grid = [list(row) for row in grid]  # 深复制并将每一行转换为列表
    new_grid[i][j] = new_value  # 修改指定位置的值

    # 将新网格转换回不可变的元组结构
    return tuple(tuple(row) for row in new_grid)


def upper_third(grid: Grid) -> Grid:
    """ Upper third of grid """
    third = len(grid) // 3
    return grid[:third]


def middle_third(grid: Grid) -> Grid:
    """ Middle third of grid """
    third = len(grid) // 3
    return grid[third:2 * third]


def lower_third(grid: Grid) -> Grid:
    """ Lower third of grid """
    third = len(grid) // 3
    return grid[2 * third + (len(grid) % 3 != 0):]


def left_third(grid: Grid) -> Grid:
    """ Left third of grid """
    return rot270(upper_third(rot90(grid)))


def center_third(grid: Grid) -> Grid:
    """ Center third of grid """
    return rot270(middle_third(rot90(grid)))


def right_third(grid: Grid) -> Grid:
    """ Right third of grid """
    return rot270(lower_third(rot90(grid)))


def getIO_same(I, O):
    oi = objects(I, False, True, False)
    oo = objects(O, False, True, False)
    same_objects = oi.intersection(oo)

    # 将 same_objects 转换为适当的格式
    same_objects_list = [(value, coord) for obj in same_objects for value, coord in obj]

    display_diff_matrices(same_objects_list)
    return same_objects


def getIO_diff(I, O, flags: Dict[str, bool]):
    # 调用 objects 函数两次
    oi = objects(I, False, True, False)
    oo = objects(O, False, True, False)

    # same_objects = oi.intersection(oo)

    oi_unique = oi - oo  # 获取在 oi 中但不在 oo 中的元素
    oo_unique = oo - oi  # 获取在 oo 中但不在 oi 中的元素

    # 将它们分别赋给 diff1 和 diff2
    diff1, diff2 = next(iter(oi_unique)), next(iter(oo_unique))

    assert oi_unique == {diff1} and oo_unique == {diff2}

    # 将两个 frozenset 转换为有序列表
    sorted_diff1 = sorted(diff1, key=lambda x: (x[0], x[1]))  # 按值和坐标排序
    sorted_diff2 = sorted(diff2, key=lambda x: (x[0], x[1]))  # 按值和坐标排序

    # 输出排序后的比较结果
    # # # print("第一个diff frozenset 排序后的元素:", sorted_diff1)
    # # # print("第二个diff frozenset 排序后的元素:", sorted_diff2)
    # 比较差异
    diff1_unique = sorted(set(sorted_diff1) - set(sorted_diff2))
    diff2_unique = sorted(set(sorted_diff2) - set(sorted_diff1))

    print("第一个 frozenset 特有的元素（排序后）:", diff1_unique)
    print("第二个 frozenset 特有的元素（排序后）:", diff2_unique)

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

    # 输出合并后的差异
    for key in merged_diffs:
        for value, positions in merged_diffs[key].items():
            print(f"{key} - 值 {value} 的特有坐标:", positions)

    display_diff_matrices(diff1_unique, diff2_unique)
    return diff1_unique, diff2_unique


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

        display_diff_matrices(diff1_unique, diff2_unique)

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

# def position(grid):
#     coords1 = [coord for coords in merged_diffs['diff1'].values()
#                for coord in coords]


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
                          diff2: Optional[List[Tuple[int,
                                                     Tuple[int, int]]]] = None,
                          diff3: Optional[List[Tuple[int, Tuple[int, int]]]] = None):
    """
    展示不同元素位置的二维矩阵。

    参数:
    - diff1: 必填，包含不同元素及其位置的集合。
    - diff2, diff3: 可选，额外的不同元素及其位置集合。
    """
    combined_diff = {}

    # 合并所有不同元素的位置
    for value, pos in diff1 + (diff2 if diff2 else []) + (diff3 if diff3 else []):
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

        # 打印矩阵并添加边界框
        print("+" + "-" * (max_col * 2 - 1) + "+")
        for row in matrix:
            print("|" + " ".join(row) + "|")
        print("+" + "-" * (max_col * 2 - 1) + "+")
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
                    flags['is_subgrid'] = [True]
                    # if i == small_rows and j == small_rows:
                    return crop, i, j  # 找到匹配位置，返回 True

    return False  # 未找到匹配位置，返回 False
