from dsl import *
from collections import defaultdict








def is_vmirror_equal(obj) -> bool:
    """判断对象在进行垂直镜像操作后是否与原对象相等"""
    return obj == vmirror(obj)

def isHorizontalMirror(piece) -> bool:
    """判断是否是上下镜像对称"""
    return piece == hmirror(piece)

def isVerticalMirror(piece) -> bool:
    """判断是否是左右镜像对称"""
    return piece == vmirror(piece)

def isDiagonalMirror(piece) -> bool:
    """判断是否是对角线镜像对称"""
    return piece == dmirror(piece)

def iscounterdiagonalMirror(piece: Piece) -> bool:
    """判断对象是否是逆对角线镜像对称"""
    return piece == cmirror(piece)

def isrot90(grid: Grid) -> bool:
    """判断对象是否是90度旋转对称"""
    return grid == rot90(grid)

def isrot180(grid: Grid) -> bool:
    """判断对象是否是180度旋转对称"""
    return grid == rot180(grid)

def isrot270(grid: Grid) -> bool:
    """判断对象是否是270度旋转对称"""
    return grid == rot270(grid)









def is_subgrid(small_grid, big_grid):
    return

def has_same_obj(I, O):
    return

def composeObject(I):
    # remove bg 
    return

def unique_objects(I, O):
    # 使用 set 存储唯一的结果
    unique_results = {
        objects(I, F, T, T),
        objects(I, F, F, T),
        objects(I, T, F, T),
        objects(I, T, T, T),
        objects(O, F, T, T),
        objects(O, F, F, T),
        objects(O, T, F, T),
        objects(O, T, T, T),
    }
    return unique_results






def prepare00(I,O):
    
    # 调用 objects 函数两次
    oi = objects(I, False, True, True)
    oo = objects(O, False, True, True)
    
    # 假设 height_i 和 width_i 是输入的高度和宽度，height_o 和 width_o 是输出的高度和宽度
    height_i, width_i = height(I), width(I)    # 输入对象的高度和宽度
    height_o, width_o = height(O), width(O)    # 输出对象的高度和宽度
    
    # hi = height(I)
    # wi = width(I)
    # ho = height(O)
    # wo = width(O)

    print("输入对象的高度:", height_i)
    print("输出对象的高度:", height_o)
    print("输入对象的宽度:", width_i)
    print("输出对象的宽度:", width_o)


    # 获取对称差集
    diff_objects = oi.symmetric_difference(oo)

    # 检查是否恰好有两个不同部分
    if len(diff_objects) == 2:
        # 解包不同部分为两个 frozenset
        diff1, diff2 = diff_objects

        # 将两个 frozenset 转换为有序列表
        sorted_diff1 = sorted(diff1, key=lambda x: (x[0], x[1]))  # 按值和坐标排序
        sorted_diff2 = sorted(diff2, key=lambda x: (x[0], x[1]))  # 按值和坐标排序

        # 输出排序后的比较结果
        # print("第一个 frozenset 排序后的元素:", sorted_diff1)
        # print("第二个 frozenset 排序后的元素:", sorted_diff2)
        # 比较差异
        diff1_unique = sorted(set(sorted_diff1) - set(sorted_diff2))
        diff2_unique = sorted(set(sorted_diff2) - set(sorted_diff1))

        print("第一个 frozenset 特有的元素（排序后）:", diff1_unique)
        print("第二个 frozenset 特有的元素（排序后）:", diff2_unique)
        
        # # 假设 diff1_unique 和 diff2_unique 是已经得到的排序后差异列表
        # 创建字典按第一个值分组
        merged_diffs = defaultdict(lambda: {"diff1": [], "diff2": []})
        # 将 diff1_unique 中的数据按第一个值分组
        for value, coord in diff1_unique:
            merged_diffs[value]["diff1"].append(coord)
        # 将 diff2_unique 中的数据按第一个值分组
        for value, coord in diff2_unique:
            merged_diffs[value]["diff2"].append(coord)
        # 输出合并后的差异
        for value in merged_diffs:
            print(f"值 {value} 的差异:")
            print("  第一个 frozenset 特有的坐标:", merged_diffs[value]["diff1"])
            print("  第二个 frozenset 特有的坐标:", merged_diffs[value]["diff2"])
            
        
        combined_diff = {}
        for value, pos in diff1_unique + diff2_unique:
            if value not in combined_diff:
                combined_diff[value] = []
            combined_diff[value].append(pos)

        # 展示为二维矩阵的格式
        for key, positions in sorted(combined_diff.items()):
            print(f"数值 {key} 的不同元素位置：")
            
            # 获取最大行和列用于确定矩阵的大小
            max_row = max(pos[0] for pos in positions) + 1
            max_col = max(pos[1] for pos in positions) + 1
            matrix = [[' ' for _ in range(max_col)] for _ in range(max_row)]
            
            # 填充矩阵中的位置
            for row, col in positions:
                matrix[row][col] = str(key)
            
            # 打印二维矩阵
            for row in matrix:
                print(" ".join(row))
            print("\n" + "-"*20 + "\n")    
        
        


        # 定义特有的坐标
        # diff1_coords = [(1, 0), (2, 2), (3, 1), (5, 3)]  # 第一个 frozenset 特有的坐标
        # diff2_coords = [(1, 5), (2, 3), (3, 4), (5, 2)]  # 第二个 frozenset 特有的坐标

        # 计算差异的维度和差值
        def calc_diffs(coords1, coords2):
            x_diffs, y_diffs = [], []
            x_sums, y_sums = [], []
            for (x1, y1), (x2, y2) in zip(coords1, coords2):
                x_diffs.append(abs(x1 - x2))
                y_diffs.append(abs(y1 - y2))
                x_sums.append(x1 + x2)
                y_sums.append(y1 + y2)
            return x_diffs, y_diffs, x_sums, y_sums
        
        def calculate_diff_sum(coords):
            x_diff = max(x for x, y in coords) - min(x for x, y in coords)
            y_diff = max(y for x, y in coords) - min(y for x, y in coords)
            x_sum = sum(x for x, y in coords)
            y_sum = sum(y for x, y in coords)
            return x_diff, y_diff, x_sum, y_sum

        # # 计算坐标差异和和
        # x_diffs, y_diffs, x_sums, y_sums = calc_diffs(merged_diffs[value]["diff1"], merged_diffs[value]["diff2"])
        
        # 计算坐标差异和和
        x_diffs, y_diffs, x_sums, y_sums = calc_diffs(merged_diffs[value]["diff1"], merged_diffs[value]["diff2"])

        # 分别打印每个变量的内容
        print("X 维度的差值列表:", x_diffs)
        print("Y 维度的差值列表:", y_diffs)
        print("X 维度的和列表:", x_sums)
        print("Y 维度的和列表:", y_sums)


        # 计算与输入和输出的高度和宽度的比较
        # 计算与输入和输出的高度和宽度的比较（索引从0开始，需减一）
        x_diff_matches_height = any(diff == (height_i - 1) or diff == (height_o - 1) for diff in x_diffs)
        y_diff_matches_width = any(diff == (width_i - 1) or diff == (width_o - 1) for diff in y_diffs)

        x_sum_matches_height = any(s == (height_i - 1) or s == (height_o - 1) for s in x_sums)
        y_sum_matches_width = any(s == (width_i - 1) or s == (width_o - 1) for s in y_sums)

        # 输出结果
        print("X 维度差值是否匹配输入/输出高度:", x_diff_matches_height)
        print("Y 维度差值是否匹配输入/输出宽度:", y_diff_matches_width)
        print("X 维度和是否匹配输入/输出高度:", x_sum_matches_height)
        print("Y 维度和是否匹配输入/输出宽度:", y_sum_matches_width)




        
        
    else:
        
        print("todo ！ 执行 ！  不同部分不止两个 frozenset 或无差异。")


    
    return 0

