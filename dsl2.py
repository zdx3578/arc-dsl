from collections import Counter, defaultdict
from dsl import *
from typing import Dict, Any, List, Tuple, Callable, Optional
# from config import *
# from dslIsDo import *
# from oldfun import *


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
    if not colors:  # 如果列表为空，返回 None 或其他默认值
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
