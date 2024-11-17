from typing import List, Tuple, Set, FrozenSet
from collections import Counter, defaultdict
from dsl import *
from typing import Dict, Any, List, Tuple, Callable, Optional
import logging
import traceback


def get_object_dimensions(obj: Object) -> Tuple[int, int]:
    """
    获取对象的高度和宽度。

    参数:
    obj: Object - 输入的对象。

    返回:
    Tuple[int, int] - 对象的高度和宽度。
    """
    indices = toindices(obj)
    if not indices:
        return 0, 0

    ul = ulcorner(indices)
    lr = lrcorner(indices)
    height = lr[0] - ul[0] + 1
    width = lr[1] - ul[1] + 1

    return height, width


def find_largest_objects(grid: Grid) -> Tuple[Object, Object, bool]:
    """
    找到网格中最高和最宽的对象，并检查它们是否是同一个对象。

    参数:
    grid: Grid - 输入的网格。

    返回:
    Tuple[Object, Object, bool] - 最高和最宽的对象，以及它们是否是同一个对象。
    """
    objs = objects(grid, univalued=True, diagonal=True, without_bg=False)
    max_height = 0
    max_width = 0
    tallest_obj = None
    widest_obj = None

    for obj in objs:
        height, width = get_object_dimensions(obj)
        if height > max_height:
            max_height = height
            tallest_obj = obj
        if width > max_width:
            max_width = width
            widest_obj = obj

    same_object = tallest_obj == widest_obj
    return tallest_obj, widest_obj, same_object


def check_largest_objects_dimensions(grid: Grid) -> bool:
    """
    检查网格中最高和最宽的对象的高度和宽度是否等于输入网格的高度和宽度，并确认它们是否是同一个对象。

    参数:
    grid: Grid - 输入的网格。

    返回:
    bool - 如果最高和最宽的对象的高度和宽度等于输入网格的高度和宽度，并且它们是同一个对象，返回 True；否则返回 False。
    """
    height, width = len(grid), len(grid[0])
    tallest_obj, widest_obj, same_object = find_largest_objects(grid)

    tallest_height, _ = get_object_dimensions(tallest_obj)
    _, widest_width = get_object_dimensions(widest_obj)

    return tallest_height == height and widest_width == width and same_object


def preprocess_cut_background(task: Dict[str, Any]) -> None:
    """
    处理任务中的所有训练和测试样本，去掉矩阵中成行或成列的 0。

    参数:
    task: Dict[str, Any] - 包含 'train' 和 'test' 的任务字典，分别为二维列表。
    """
    # 遍历任务中的所有训练和测试样本
    for sample in task['train'] + task['test']:
        input_grid = sample['input']
        output_grid = sample['output']

        # 处理输入和输出网格
        sample['input'] = cut_background(input_grid)
        # sample['output'] = cut_background(output_grid)
    return task


def underfill_corners(I: Grid, color: int) -> Grid:
    x1 = objects(I, T, F, T)
    x2 = mapply(corners, x1)
    O = underfill(I, color, x2)
    return O


def cut_background(grid: Grid) -> Grid:
    """
    去掉矩阵中成行或成列的 0，但保留包含非 0 元素的行和列，以及这些行和列之间的所有行和列。

    参数:
    grid: Grid - 输入的矩阵。

    返回:
    Grid - 去掉成行或成列的 0 后的矩阵。
    """
    # 找到包含非 0 元素的行和列的索引
    non_zero_rows = {i for i, row in enumerate(
        grid) if any(cell != 0 for cell in row)}
    non_zero_cols = {j for j in range(
        len(grid[0])) if any(row[j] != 0 for row in grid)}

    # 找到需要保留的行和列的范围
    if non_zero_rows and non_zero_cols:
        min_row, max_row = min(non_zero_rows), max(non_zero_rows)
        min_col, max_col = min(non_zero_cols), max(non_zero_cols)

        # 保留这些行和列之间的所有行和列
        grido = [row[min_col:max_col + 1] for row in grid[min_row:max_row + 1]]

    return tuple(tuple(row) for row in grido)


def get_mirror_hole(I, color=0):
    # need judge is !! mirror,half is mirrir otherhalf not mirror
    # color is size big obj obj(I,T,T,F)  default zero,hole in the not mirror part
    x1 = vmirror(I)
    x2 = ofcolor(I, color)
    O = subgrid(x2, x1)
    return O


def get_partition_min_subgrid(I):
    x1 = partition(I)
    x2 = argmin(x1, size)
    O = subgrid(x2, I)
    return O


def frontiers2(
    grid: Grid
) -> Objects:
    """ set of frontiers """
    h, w = len(grid), len(grid[0])
    row_indices = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
    column_indices = tuple(j for j, c in enumerate(
        dmirror(grid)) if len(set(c)) == 1)
    hfrontiers = frozenset(
        {frozenset({(grid[i][j], (i, j)) for j in range(w)}) for i in row_indices})
    vfrontiers = frozenset(
        {frozenset({(grid[i][j], (i, j)) for i in range(h)}) for j in column_indices})
    return hfrontiers, vfrontiers

##############################################################


def extract_line_indices(frontier_lines):
    """
    从 hfrontiers 或 vfrontiers 中提取行或列的索引。

    参数:
    frontier_lines: frozenset，包含多个 frozenset，每个内部 frozenset 代表一条线。

    返回:
    line_indices: 列表，包含每条线的行索引或列索引。
    """
    line_indices = []
    for line in frontier_lines:
        # 提取该线上的所有坐标
        coords = [coord for value, coord in line]
        if coords:
            # 如果是水平线，所有坐标的行索引相同
            # 如果是垂直线，所有坐标的列索引相同
            i_indices = {coord[0] for coord in coords}
            j_indices = {coord[1] for coord in coords}
            if len(i_indices) == 1:
                # 水平线，提取行索引
                line_indices.append(list(i_indices)[0])
            elif len(j_indices) == 1:
                # 垂直线，提取列索引
                line_indices.append(list(j_indices)[0])
    return line_indices


def split_grid_by_indices(grid,  include_lines=False):
    """
    使用行索引和列索引来分割矩阵。

    参数:
    grid: 二维列表，表示矩阵
    h_indices: 行索引列表，用于水平分割
    v_indices: 列索引列表，用于垂直分割
    include_lines: 布尔值，是否包含分割线，默认 False

    返回:
    sub_grids: 字典，包含分割后的子矩阵
    """
    hline, vline = frontiers2(grid)
    h_indices = extract_line_indices(hline)
    v_indices = extract_line_indices(vline)

    h_indices = sorted(h_indices)
    v_indices = sorted(v_indices)
    h_splits = [0] + h_indices + [len(grid)]
    v_splits = [0] + v_indices + [len(grid[0])]

    sub_grids = {}
    index = 0  # 子网格编号
    for i in range(len(h_splits) - 1):
        for j in range(len(v_splits) - 1):
            if include_lines:
                start_row = h_splits[i]
                end_row = h_splits[i+1]
                start_col = v_splits[j]
                end_col = v_splits[j+1]
            else:
                start_row = h_splits[i] + (0 if i == 0 else 1)
                end_row = h_splits[i+1]
                start_col = v_splits[j] + (0 if j == 0 else 1)
                end_col = v_splits[j+1]

            # 检查子网格是否为空
            if start_row >= end_row or start_col >= end_col:
                continue

            sub_grid = [row[start_col:end_col]
                        for row in grid[start_row:end_row]]
            key = f'grid_{index}'
            sub_grids[key] = sub_grid
            index += 1
    return sub_grids


def do_numb_color_upscale(I):
    x1 = numcolors(I)
    x2 = decrement(x1)
    O = upscale(I, x2)
    return O


def box_cut(I):
    x2 = get_inbox_position(I)
    O = subgrid(x2, I)
    return O


def get_inbox_position(I):
    box = get_empty_box(I)

    return toindices(box)


def get_empty_box(I):
    x1 = objects(I, T, T, T)
    try:
        for obj in x1:
            # if isinstance(obj, frozenset) and all(isinstance(item, tuple) and len(item) == 2 for item in obj):
            #     diff1 = [(value, pos) for value, pos in obj]  # 创建 diff1 列表
            #     display_diff_matrices(diff1)
            # else:
            #     logging.error("对象格式不正确：%s", obj)
            if is_valid_empty_box(obj, I):
                ##
                return obj
    except Exception as e:
        logging.error("捕获到异常：%s", e)
        logging.error("详细错误信息：\n%s", traceback.format_exc())
        pass
    return False


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


def get_max_object(I):
    x1 = objects(I, T, T, T)
    x2 = argmax(x1, size)
    O = subgrid(x2, I)
    return O


def get_min_object(I):
    x1 = objects(I, T, T, T)
    x2 = argmin(x1, size)
    O = subgrid(x2, I)
    return O


def get_max_object_color(I):
    x1 = objects(I, T, T, T)
    x2 = argmax(x1, color)
    O = subgrid(x2, I)
    return O


# def get_max_object_size(I):
#     x1 = objects(I, T, T, T)
#     x2 = argmax(x1, size)
#     O = subgrid(x2, I)
#     return O


def is_objectComplete_change_color(task, flags, done=False):
    train_data = task['train']
    test_data = task['test']
    for i, data_pair in enumerate(train_data):
        # data_pair = train_data[1]
        # flags = initialize_flags()

        I = data_pair['input']
        O = data_pair['output']

        diff1, diff2, _, _ = getIO_diff(I, O, flags)
        # same_obj = getIO_same_obj(I, O)
        same_fg = getIO_same_fg(I, O)

        if toindices(diff1) == toindices(diff2):
            flags["diff_position_same"] = True
            # contain_object = contains_object(diff1, same_obj)
            fg_outof_diff = contains_object(diff1, same_fg)
            if fg_outof_diff:
                flags["diff_in_same_fg"] = True
                fg_outof_diff_complete = complementofobject(fg_outof_diff)
                if toindices(fg_outof_diff_complete) == toindices(diff2):
                    same_fg_color = list(next(iter(same_fg)))[0][0]
                    tocolor = list(diff2)[0][0]
                    flags["diff_in_same_contained"] = [tocolor, same_fg_color]
                    if fill(I, tocolor, delta(ofcolor(I, same_fg_color))) == O:
                        print("input  same output: ", {i})
                        if done:
                            pass
                        else:
                            return True
                    else:
                        print("input no same output")
                        return False
                else:
                    print("fg_box_diff_complete not same diff ")
                    return False
            else:
                print("not contain object")
                return False
        else:
            print("toindices not same")
            flags["diff_position_same"] = False
            return False
    # change what ；change where

    if done:
        I = test_data[0]['input']

        if flags["diff_in_same_contained"]:
            tocolor, same_fg_color = flags["diff_in_same_contained"]
            x1 = ofcolor(I, same_fg_color)
            posit = delta(x1)
            result = fill(I, tocolor, posit)
            assert result == test_data[0]['output']
            return result


def contains_object(obj: Object, objects: List[Object]) -> bool:
    """检查对象是否包含在对象列表中"""
    obj_set = set(obj)
    objects_sorted = [set(o) for o in objects]  # 将每个对象转换为集合
    objtang = object_to_rectangle(obj_set)

    for other_obj in objects_sorted:
        othertang = object_to_rectangle(other_obj)
        # and len(objtang[0]) == len(othertang[0]):
        if len(objtang) <= len(othertang):
            if is_subgrid_grid(objtang, othertang):
                return other_obj
    return False


def getIO_same_fg(I, O):
    fg1 = fgpartition(I)
    fg2 = fgpartition(O)
    same_fg = fg1.intersection(fg2)
    # display_diff_matrices(same_fg)
    same_objects_list = [(value, coord)
                         for obj in same_fg for value, coord in obj]
    print("fgpartition 相同对象的值和坐标:")
    # display_diff_matrices(same_objects_list)
    return same_fg


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


# def move(obj: Object, direction: Tuple[int, int]) -> Object:
#     """
#     移动对象到指定方向。

#     参数:
#     obj: Object - 要移动的对象。
#     direction: Tuple[int, int] - 移动的方向，格式为 (dx, dy)。

#     返回:
#     Object - 移动后的对象。
#     """
#     dx, dy = direction
#     moved_obj = frozenset({(value, (i + dx, j + dy)) for value, (i, j) in obj})
#     return moved_obj


def move_down(I, obj: Object) -> Object:
    """
    向下移动对象。

    参数:
    obj: Object - 要移动的对象。

    返回:
    Object - 移动后的对象。
    """
    return move(I, obj, (1, 0))


def move_down_1obj(input):
    obj = first(objects(input, T, T, T))
    return move_down(input, obj)


def move_right(I, obj: Object) -> Object:
    """
    向右移动对象。

    参数:
    obj: Object - 要移动的对象。

    返回:
    Object - 移动后的对象。
    """
    return move(I, obj, (0, 1))


def move_up(I, obj: Object) -> Object:
    """
    向上移动对象。

    参数:
    obj: Object - 要移动的对象。

    返回:
    Object - 移动后的对象。
    """
    return move(I, obj, (-1, 0))


def move_left(I, obj: Object) -> Object:
    """
    向左移动对象。

    参数:
    obj: Object - 要移动的对象。

    返回:
    Object - 移动后的对象。
    """
    return move(I, obj, (0, -1))


def object_to_rectangle(obj: Object) -> Grid:
    """ 将对象扩展为包含对象的一个长方形矩阵 """
    # same as   backdrop

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
        rectangle[i - ul[0]][j - ul[1]] = 0

    return tuple(tuple(row) for row in rectangle)


def get_first_object(I):
    x1 = objects(I, T, T, T)
    x2 = first(x1)
    O = subgrid(x2, I)
    return O


def do_output_most_input_color(I):
    x1 = mostcolor(I)
    return canvas(x1, (height(I), width(I)))


def mostcolor2(colors: list) -> int:
    """ 返回列表中出现次数最多的颜色 """
    if not colors:  # 如果列表为空，返回 None 或其��认值
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


def getIO_same_obj(I, O):
    oi = objects(I, True, True, True)
    oo = objects(O, True, True, True)
    same_objects = oi.intersection(oo)

    # if same_objects:
    #     pass
    # else:
    #     oi = objects(I, True, True, False)
    #     oo = objects(O, True, True, False)
    #     same_objects = oi.intersection(oo)

    # 将 same_objects 转换为适当的格式
    same_objects_list = [(value, coord)
                         for obj in same_objects for value, coord in obj]
    print("相同对象的值和坐标:")
    # display_diff_matrices(same_objects_list)

    # same_objects = [(value, coord)
    #                 for obj in same_objects for value, coord in obj]
    # print("单个对象 :")
    # for obj in same_objects:
    #     display_diff_matrices([obj])
    return same_objects


def getIO_diff(I: Grid, O: Grid, flags: Optional[Dict[str, bool]] = None):
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

    diff_output_colorset = set(merged_diffs["diff2"])
    if len(diff_output_colorset) == 1:
        diff_output_color = next(iter(diff_output_colorset))
        flags["out_train_i_diff_color_is"].append(diff_output_color)
    else:
        diff_output_color = None

    # 输出合并后的差异
    # for key in merged_diffs:
    #     for value, positions in merged_diffs[key].items():
    #         print(f"{key} - 值 {value} 的特有坐标:", positions)

    # print("比较结果:不同 diff 集合的坐标是否一致:")
    # display_diff_matrices(diff1_unique, diff2_unique)
    return diff1_unique, diff2_unique, diff_output_colorset, diff_output_color


def do_neighbour_color(I, color):
    x1 = objects(I, T, T, T)
    # x3 = ofcolor(I, FIVE)
    positions = frozenset(
        coord for inner_set in x1 for _, coord in inner_set)
    x2 = mapply(neighbors, positions)
    O = fill(I, color, x2)
    return O


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

        # display_diff_matrices(diff1_unique, diff2_unique)

        if compare_positions(merged_diffs):
            flags["is_diff_same_posit"].append(True)
            if is_frontier(diff1_unique, I):
                flags["same_diff_is_frontier"].append(True)
                colorset = set(merged_diffs["diff2"])
                color = next(iter(colorset))
                flags["fill_frontier_color"] = color
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


def do_frontier(I, color):
    x1 = frontiers(I)
    x2 = merge(x1)
    O = fill(I, color, x2)
    return O


def is_frontier(diff, grid: Grid) -> bool:
    """检查网格是否有边界元素"""
    x1 = frontiers(grid)
    x2 = merge(x1)
    x3 = sorted(list(x2))
    # assert diff == x3
    return diff == x3


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


def is_subgrid_grid(grid1: Grid, grid2: Grid) -> bool:
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
    if h1 > h2 or w1 > w2:
        return False

    # 遍历 grid2，检查是否存在与 grid1 匹配的子网格
    for i in range(h2 - h1 + 1):
        for j in range(w2 - w1 + 1):
            match = True
            for x in range(h1):
                for y in range(w1):
                    if grid1[x][y] != grid2[i + x][j + y]:
                        match = False
                        break
                if not match:
                    break
            if match:
                return True

    return False


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
