from dsl import *
from constants import *
import dsl2
import inspect
from arc_types import *
import logging
import traceback
from contextlib import contextmanager


#! 特殊点是什么？

#! weight 权重 先验权重 大小

box
snow
repeat
param
recolor
position
mirror
subpart
upscale
colorcount
downscale
rotate
shrink
move
draw   move

frontiers
line
fill
param_repeat
button
recover
denoise

logic
scaling
diff
shape
flow
lenght

filter
colorClassify

bighole
arithmetic
concatinout
denoise






@contextmanager
def safe_context():
    """安全上下文管理器，用于异常捕获和日志记录"""
    try:
        yield
    except Exception as e:
        print(f"___________safe_context___An error occurred: {e}")
        logging.error("捕获到异常：%s", e)
        logging.error("详细错误信息：\n%s", traceback.format_exc())



def search(I: Grid, O: Grid) -> Optional[Callable]:
    for date in [input-input,output-output,input-outputpaire]:
        for fun in get_all_isfunctions():
            try:
                if fun(I, O):
                    return fun
            except Exception:
                continue
    hwratio
    hratio
    is_out_is_in_subgrid
    is_in_is_out_subgrid
    objects

# def common_proper(I: Grid, O: Grid) -> bool:
#     return True

# def grid_property(I, O):

def diff_property(I: Grid, O: Grid) -> dict:
    diff = advanced_difference(I, O)
    obj_property(diff)
    # 归纳从diff 归纳  ilp

    # 函数 输入数据  是推理  演绎


from typing import Union

def obj_property(input_data: Union[Object, Grid]) -> dict:
    # 类型检查和转换
    if isinstance(input_data, frozenset) and all(
        isinstance(item, tuple) and len(item) == 2
        and isinstance(item[1], tuple) and len(item[1]) == 2
        for item in input_data
    ):
        # 输入是 Object 类型
        obj = input_data
    elif isinstance(input_data, (tuple, list)) and all(
        isinstance(row, (tuple, list)) for row in input_data
    ):
        # 输入是 Grid 类型，转换为 Object
        obj = asobject(input_data)
    else:
        raise ValueError("输入必须是 Object 或 Grid 类型")

    #     # 类型检查和转换
    # obj = (asobject(input_data) if isinstance(input_data, list)
    #        else input_data if isinstance(input_data, frozenset)
    #        else None)

    # if obj is None:
    #     raise ValueError("输入必须是Object或Grid类型")

    # 计算对象属性

    obj_I = obj
    properties_I = {
        'size': len(obj_I),
        'shape': shape(obj_I),
        'asindices': asindices(obj_I),
        'position': center(obj_I),
        'move': move(obj_I, (1, 1)),  # 示例：移动 (1, 1)
        'shift': shift(obj_I, (1, 1)),  # 示例：平移 (1, 1)
        'mirror': hmirror(obj_I),  # 示例：水平镜像
        'rotate': rot90(obj_I),  # 示例：旋转 90 度
        'colorcount': colorcount(obj_I, 1),  # 示例：颜色 1 的数量
        'palette': palette(obj_I),
        'numcolors': numcolors(obj_I),
        'numcolors_nozero': len(palette(obj_I) - {0}),  # 排除 0 颜色
        'all_colorcount': {color: colorcount(obj_I, color) for color in palette(obj_I)},
        'leastcolor': leastcolor(obj_I),
        'mostcolor': mostcolor(obj_I)
        backdrop(obj_I)

        ######.....................
    }
    return properties_I




def process_objs_property(I: Grid) -> List[Set[Object]]:
    """处理输入网格中的对象分组"""
    objs = get_all_obj(I)  # 假设objects函数返回网格中的所有对象
    group_adjacent_obj = group_adjacent_objects(objs, diagonal=True)  # 支持对角线相邻

    group_same_shape_obj = group_same_shape_objects(objs)

    is_same_shape_move_shift_parameters


    shape

    all_colorcount
    palette
    numcolors
    numcolors_nozero = numcolors - 1
    all_colorcount
    leastcolor #！！！！！！

    size
    is_mirror




def is_partition_obj(I: Grid, O: Grid) -> bool:
    obj = partition(I)
    return objproperty(I, O)


def is_upsacle_proper_colorcount(I: Grid, O: Grid) -> bool:
    return


#37 3aa6fb7a
def is_diff_corner_color(I: Grid, O: Grid) -> bool:
    advanced_difference(I, O)
    is_part_of_obj(I, O)


def is_diff_is_neighibor(I: Grid, O: Grid) -> bool:
    advanced_difference(I, O)

# 33 c909285e ??

# 42


def is_sizefilter(I: Grid, O: Grid) -> bool:
    size proper?
    return ?





















def is_part_of_obj(obj1: Object, obj2: Object) -> bool:
    """
    判断 obj1 是否是 obj2 的一部分

    Args:
        obj1: 可能是部分的对象
        obj2: 完整的对象

    Returns:
        bool: True 如果 obj1 是 obj2 的一部分
    """
    # 空对象特殊处理
    if not obj1:
        return True
    if not obj2:
        return False

    # 检查 obj1 是否是 obj2 的子集
    # 注意：这里直接使用集合操作，因为Object类型是frozenset
    # 由于Object的元素是(value, (i, j))格式，所以自动会匹配值和位置
    return obj1.issubset(obj2) or obj2.issubset(obj1)





def is_valid_empty_box(obj: Object, grid: Grid) -> bool:
    """
    判断对象是否是一个空心矩阵框，并且高度和宽度大于 2，并且小于输入网格的高度和宽度。

    参数:
    obj: Object - 输入的对象。
    grid: Grid - 输入的网格。

    返回:
    bool - 如果对象是一个空心矩阵框，并且高度和宽度大于 2，并且小于输入网格的高度和宽度，返回 True；否则返回 False。
    """
    # 确保 obj 的格式正确
    if not (isinstance(obj, frozenset) and all(isinstance(item, tuple) and len(item) == 2 for item in obj)):
        return False

    # 获取对象的高度和宽度
    obj_height, obj_width = get_object_dimensions(obj)
    grid_height, grid_width = len(grid), len(grid[0])

    # 检查对象的高度和宽度是否大于 2，并且小于输入网格的高度和宽度
    if not (obj_height > 2 and obj_width > 2 and obj_height < grid_height and obj_width < grid_width):
        return False

    # 获取对象的边框
    obj_box = box(obj)

    # 获取对象的内部
    obj_interior = toindices(obj) - obj_box

    # 检查对象的内部是否为空，并且对象的边框与 obj_box 相同
    return len(obj_interior) == 0 and obj_box == toindices(obj)


def is_box(obj: Object) -> bool:
    """
    判断对象是否是一个矩阵框。

    参数:
    obj: Object - 输入的对象。

    返回:
    bool - 如果对象是一个矩阵框，返回 True；否则返回 False。
    """
    # 获取对象的边框
    obj_box = box(obj)

    # 检查对象是否与其边框相同
    return obj == obj_box

def is_positive_diagonal(
    patch: Patch
) -> Boolean:
    """判断 patch 是否形成从左上到右下的正对角线"""
    if len(patch) == 0:
        return False

    # 检查高度和宽度是否等于 patch 的长度
    if height(patch) != len(patch) or width(patch) != len(patch):
        return False

    # 将 patch 中的点按行排序
    sorted_points = sorted(patch)

    # 获取起始点的坐标
    start_i, start_j = sorted_points[0]

    # 检查是否所有点都在从左上到右下的对角线上
    for i, j in sorted_points:
        if j != start_j + (i - start_i):
            return False

    return True


def is_negative_diagonal(
    patch: Patch
) -> Boolean:
    """判断 patch 是否形成从右上到左下的负对角线"""
    if len(patch) == 0:
        return False

    # 检查高度和宽度是否等于 patch 的长度
    if height(patch) != len(patch) or width(patch) != len(patch):
        return False

    # 将 patch 中的点按行排序
    sorted_points = sorted(patch)

    # 获取起始点的坐标
    start_i, start_j = sorted_points[0]

    # 检查是否所有点都在从右上到左下的对角线上
    for i, j in sorted_points:
        if j != start_j - (i - start_i):
            return False

    return True




from itertools import product
def get_all_obj(grid: Grid) -> list[Objects]:
    # 枚举所有参数组合
    results = []
    for univalued, diagonal, without_bg in product([True, False], repeat=3):
        result = objects(grid, univalued, diagonal, without_bg)
        results.append({
            "univalued": univalued,
            "diagonal": diagonal,
            "without_bg": without_bg,
            "result": result
        })
    return results


def is_has_frontier(I, O):
    return  frontiers(I)

def is_adjacent(p1,p2):
    return adjacent(p1,p2)

def is_upscale_numcolors(I, O):
    return O == upscale(I, numcolors(I))


def getall_isfunctions():
    # 获取当前模块所有函数
    current_module = inspect.currentframe().f_globals
    is_functions = [
        func for name, func in current_module.items()
        if callable(func) and
           name.startswith('is_') and
           not name.startswith('is_is_')
    ]
    return is_functions

def is_is_part_oflarge_property(I: Grid, O: Grid) -> bool:  #9ecd008a
    # movevec = is_same_shape_move_shift_parameters(shape(ofcolor(I, ZERO)), shape(O))
    movevec = is_same_shape_move_shift_parameters((ofcolor(I, ZERO)), (O))
    x1 = move(I,O,movevec)
    #x1isproperofthisfileotherproper #x1 is proper of this file other is fun proper
    # 获取当前模块所有函数
    current_module = inspect.currentframe().f_globals
    is_functions = get_allisfunctions()

    # 遍历所有符合条件的函数
    for fun in is_functions:
        if fun != is_is_part_oflargeproperty:  # 避免自身调用
            try:
                if fun(x1, x1):
                    return True
            except Exception:
                continue

    return False

from typing import List, Set
from collections import defaultdict

def group_adjacent_objects(objects: Objects, diagonal: bool = True) -> List[Set[Object]]:
    """
    将相邻的对象归为一组

    Args:
        objects: 所有对象集合
        diagonal: 是否考虑对角线相邻

    Returns:
        分组后的对象列表，每组是一个集合
    """
    # 构建邻接图
    graph = defaultdict(set)
    objects_list = list(objects)

    # 构建邻接关系
    for i, obj1 in enumerate(objects_list):
        for j, obj2 in enumerate(objects_list[i+1:], i+1):
            if adjacent(obj1, obj2, diagonal=diagonal):
                graph[i].add(j)
                graph[j].add(i)

    # DFS找出连通分量
    def dfs(node: int, visited: Set[int], component: Set[int]):
        visited.add(node)
        component.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, visited, component)

    # 获取所有连通分量
    visited = set()
    components = []

    for node in range(len(objects_list)):
        if node not in visited:
            current_component = set()
            dfs(node, visited, current_component)
            # 将索引转换回实际对象
            components.append({objects_list[i] for i in current_component})

    return components




def is_part_ofzero_of_bigpicture(I: Grid, O: Grid) -> bool:
    return shape(O) == shape(ofcolor(I, ZERO))


def is_mirror(grid1: Grid, grid2: Grid) -> bool:
    # 检查两个网格是否是镜像
    return grid1 == dmirror(grid2) or grid1 == hmirror(grid2) or grid1 == vmirror(grid2) or grid1 == cmirror(grid2)

def is_rote(grid1: Grid, grid2: Grid) -> bool:
    # 检查两个网格是否是旋转
    return grid1 == rot90(grid2) or grid1 == rot180(grid2) or grid1 == rot270(grid2)

def is_sclae(grid1: Grid, grid2: Grid) -> bool:
    # 检查两个网格是否是缩放
    scalesh = height(grid1) / height(grid2)
    scales = width(grid1) / width(grid2)
    return grid1 == upscale(grid2, scales) or grid1 == downscale(grid2, scales) or grid1 == hupscale(grid2, scalesh) or grid1 == vupscale(grid2, scales)

def is_concat_mirror(I: Grid, O: Grid) -> bool:
    # 遍历镜像和拼接操作的组合
    for mirror in [hmirror, vmirror, cmirror, dmirror]:
        mirrored_matrix = mirror(I)  # 应用镜像操作
        for concat in [hconcat, vconcat]:
            try:
                # 检查两种拼接顺序
                if concat(I, mirrored_matrix) == O or concat(mirrored_matrix, I) == O:
                    return (True)
            except Exception:
                # 防止操作不匹配导致错误
                continue
    # 如果没有匹配的组合，返回 False
    return (False)
def is_concat_rot(I: Grid, O: Grid) -> bool:
    # 遍历旋转和拼接操作的组合
    for rotate in [rot90, rot180, rot270]:
        rotated_matrix = rotate(I)
        for concat in [hconcat, vconcat]:
            try:
                # 检查两种拼接顺序
                if concat(I, rotated_matrix) == O or concat(rotated_matrix, I) == O:
                    return (True)
            except Exception:
                # 防止操作不匹配导致错误
                continue
    # 如果没有匹配的组合，返回 False
    return (False)
def is_concat_mirror_rot(I: Grid, O: Grid) -> bool:
    # 遍历镜像、旋转和拼接操作的组合
    for mirror in [hmirror, vmirror, cmirror, dmirror]:
        mirrored_matrix = mirror(I)
        for rotate in [rot90, rot180, rot270]:
            rotated_mirrored_matrix = rotate(mirrored_matrix)
            for concat in [hconcat, vconcat]:
                try:
                    # 检查两种拼接顺序
                    if concat(I, rotated_mirrored_matrix) == O or concat(rotated_mirrored_matrix, I) == O:
                        return (True)
                except Exception:
                    # 防止操作不匹配导致错误
                    continue
    # 如果没有匹配的组合，返回 False
    return (False)
def is_concat(I: Grid, O: Grid) -> bool:
    # 检查水平拼接
    if hconcat(I, I) == O:
        return (True)
    # 检查垂直拼接
    if vconcat(I, I) == O:
        return (True)
    # 如果没有匹配的组合，返回 False
    return (False)


def is_split_one_n1(I: Grid, O: Grid) -> Tuple[bool, Optional[int]]:
    split = [hsplit, vsplit]
    for s in split:
        try:
            # 尝试分割输入
            split_result = s(I)
            # 检查分割后的每一部分是否等于输出
            for index, split_matrix in enumerate(split_result):
                if split_matrix == O:
                    return (True, index)
        except Exception:
            # 防止操作不匹配导致错误
            continue
    # 如果没有匹配的组合，返回 False
    return (False)


def is_get_first_object(I: Grid) -> Grid:
    x1 = objects(I, T, T, T)
    x2 = first(x1)
    O = subgrid(x2, I)
    return O

def is_replace(I: Grid, O: Grid) -> bool:
    # if is_diff_positon_color(I, O):
    # 检查替换操作
    for color1 in range(10):
        for color2 in range(10):
            if replace(I, color1, color2) == O:
                return (True)
    # 如果没有匹配的组合，返回 False
    return (False)

def is_switch(I: Grid, O: Grid) -> bool:
    # 检查交换操作
    for color1 in range(10):
        for color2 in range(10):
            if switch(I, color1, color2) == O:
                return (True)
    # 如果没有匹配的组合，返回 False
    return (False)

# 第 1 个函数  d10ecb37


# def is_upscale(grid1: Grid, grid2: Grid) -> bool:
#     x1 = hratio(grid1,grid2)
#     x2 = upscale(grid1,x1)
#     return grid2 == x2

def is_fill_I_box_color(I: Grid, O: Grid, color: int = 8) -> bool:
    # x1 = asindices(I)
    # # x0 = outbox(x1)
    # x2 = box(x1)
    return (O == fill(I, color, box(asindices(I))))



from typing import List, Tuple, Union, Set, Optional, Dict, Any
from arc_types import *


def is_same_shape_move_shift_parameters(patch1: Patch, patch2: Patch,  check_values: bool = False ) -> Tuple[Optional[IntegerTuple], str, Dict[str, Any]]:
    """判断两个patch形状和值是否相同，并计算移动参数"""
    transformations = [
        ("original", lambda p: p),
        ("rot90", rot90),
        ("rot180", rot180),
        ("rot270", rot270),
        ("hmirror", hmirror),
        ("vmirror", vmirror),
        ("cmirror", cmirror),
        ("dmirror", dmirror),
        # hmirror组合
        ("hmirror_rot90", lambda p: rot90(hmirror(p))),
        ("hmirror_rot180", lambda p: rot180(hmirror(p))),
        ("hmirror_rot270", lambda p: rot270(hmirror(p))),
        # vmirror组合
        ("vmirror_rot90", lambda p: rot90(vmirror(p))),
        ("vmirror_rot180", lambda p: rot180(vmirror(p))),
        ("vmirror_rot270", lambda p: rot270(vmirror(p))),
        # cmirror组合
        ("cmirror_rot90", lambda p: rot90(cmirror(p))),
        ("cmirror_rot180", lambda p: rot180(cmirror(p))),
        ("cmirror_rot270", lambda p: rot270(cmirror(p))),
        # dmirror组合
        ("dmirror_rot90", lambda p: rot90(dmirror(p))),
        ("dmirror_rot180", lambda p: rot180(dmirror(p))),
        ("dmirror_rot270", lambda p: rot270(dmirror(p)))
    ]

    coords2: Indices = asindices_patch(patch2)

    for transform_name, transform_func in transformations:
        transformed_patch1 = transform_func(patch1)
        coords1: Indices = asindices_patch(transformed_patch1)

        if len(coords1) != len(coords2):
            continue

        # 如果 patch1 或 patch2 为空
        if not coords1 or not coords2:
            if coords1 == coords2:
                return ((0, 0), "patches are identical (both empty)", {
                    "transformation": transform_name,
                    "match_type": "full_match"
                })
            continue

        try:
            ref1 = next(iter(coords1))
            ref2 = next(iter(coords2))
            di = ref1[0] - ref2[0]
            dj = ref1[1] - ref2[1]
            shift_distance = (di, dj)

            # 检查形状匹配
            shape_match = all(
                (i1 - di, j1 - dj) in coords2
                for (i1, j1) in coords1
            )

            if not shape_match:
                continue

            # 如果不需要检查值，直接返回形状匹配结果
            if not check_values:
                return (shift_distance, "shapes are the same", {
                    "transformation": transform_name,
                    "match_type": "shape_only",
                    "transformed_patch": transformed_patch1  })

            # 检查值匹配
            value_map1 = {}
            value_map2 = {}

            for elem in transformed_patch1:
                if isinstance(elem, tuple) and isinstance(elem[1], tuple):
                    value, (i, j) = elem
                    value_map1[(i, j)] = value

            for elem in patch2:
                if isinstance(elem, tuple) and isinstance(elem[1], tuple):
                    value, (i, j) = elem
                    value_map2[(i, j)] = value

            # 如果有值映射，检查值是否匹配
            if value_map1 and value_map2:
                value_match = all(
                    value_map1.get((i1, j1)) == value_map2.get((i1 - di, j1 - dj))
                    for (i1, j1) in coords1
                )

                if value_match:
                    return (shift_distance, "objects are identical", {
                        "transformation": transform_name,
                        "match_type": "full_match",
                        "transformed_patch": transformed_patch1
                    })
                else:
                    return (shift_distance, "same shape but different values", {
                        "transformation": transform_name,
                        "match_type": "shape_only",
                        "transformed_patch": transformed_patch1
                    })
            else:
                # 没有值映射时，只返回形状匹配结果
                return (shift_distance, "shapes are the same", {
                    "transformation": transform_name,
                    "match_type": "shape_only",
                    "transformed_patch": transformed_patch1
                })

        except StopIteration:
            continue

    return (None, "shapes are different", {
        "transformation": None,
        "match_type": "no_match"
    })



def group_same_shape_objects(objects: Objects) -> List[Set[Object]]:
    """
    将形状相同的对象分到同一组

    Args:
        objects: 所有对象的集合

    Returns:
        List[Set[Object]]: 形状相同的对象分组列表
    """
    if not objects:
        return []

    # 初始化分组
    groups = []
    processed = set()
    objects_list = list(objects)

    # 遍历每个对象
    for i, obj1 in enumerate(objects_list):
        if obj1 in processed:
            continue

        # 创建新组，包含当前对象
        current_group = {obj1}
        processed.add(obj1)

        # 与其他未处理对象比较
        for j, obj2 in enumerate(objects_list[i+1:], i+1):
            if obj2 in processed:
                continue

            # 使用is_same_shape_move_shift_parameters判断形状是否相同
            shift_params, message, transform_info = is_same_shape_move_shift_parameters(obj1, obj2)

            if shift_params is not None and message == "shapes are the same":
                current_group.add(obj2)
                processed.add(obj2)

        groups.append(current_group)

    return groups


def is_complete_change_color(grid1: Grid, grid2: Grid) -> bool:
    I = grid1
    for color in range(10):
        x1 = ofcolor(I, color)
        x2 = delta(x1)
        for i in range(10):
            O = fill(I, i, x2)
            if O == grid2:
                return True
    return False



def is_in_is_out_subgrid(grid1: Grid, grid2: Grid) -> bool:
    return is_out_is_in_subgrid(grid2, grid1)



def solv00_is_subgrid_grid(grid1: Grid, grid2: Grid) -> bool:
    """
    检查 grid1 是否是 grid2 的子网格。

    参数:
    - grid1: Grid - 第一个矩形网格。
    - grid2: Grid - 第二个矩形网格。

    返回:
    - bool: 如果 grid1 是 grid2 的子网格，返回 True；否则返回 False。
    """
    h1, w1 = len(grid1), len(grid1[0])
    h2, w2 = len(grid2), len(grid2[0])

    # 检查 grid1 的尺寸是否小于或等于 grid2
    # if h1 > h2 or w1 > w2:
    #     return False
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
        return (False)  # 两个矩阵形状不兼容，无法嵌套

    # 遍历 grid2，检查是否存在与 grid1 匹配的子网格
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
                return (True, 'crop', (i, j),(small_rows, small_cols))

    return (False)

# 第 2 个函数  74dd1130

