from dsl import *
from collections import defaultdict









from typing import Dict, Any, List, Tuple

def initialize_flags() -> Dict[str, bool]:
    """
    初始化一组标志变量。
    
    返回:
    - Dict[str, bool]: 标志变量的字典，默认为 False。
    """
    return {
        "is_mirror": [],
        "is_fun_ok": [],
        "is_scale": [],
        "is_diff_same_posit": [],
        "is_rotation": [],
        "is_translation": [],
        "is_color_transform": [],
        # 可以添加更多标志变量
    }

def solve_arc_task(task):
    """
    解决ARC任务的主框架，使用标志变量管理不同的特征。
    
    参数:
    - task: 包含多个输入输出对的ARC任务数据，格式为
      {'train': [(input1, output1), (input2, output2), ...], 'test': [test_input]}
    
    返回:
    - List: 解决任务的方案或结果。
    """
    # 初始化标志变量
    flags = initialize_flags()
    
    # train_data = task['train']
    # test_data = task['test']

    # 尝试逐个处理每个输入输出对
    solutions = []
    

    solution = solve_individual(task, flags)
    if solution:
        return solution
        solutions.append(solution)
    else:
        print("单独处理失败，需进一步尝试联合处理。")

    # 如果单独处理失败或无法找到方案，尝试整体处理
    if len(solutions) != len(train_data):
        combined_solution = solve_combined([pair[0] for pair in train_data], [pair[1] for pair in train_data], flags)
        if combined_solution:
            solutions = combined_solution

    # 用解决方案应用于测试数据
    results = [apply_solution(test_input, solutions) for test_input in test_data]
    return results

def solve_individual(task, flags: Dict[str, bool]):
    """
    尝试单独处理每个输入输出对，根据标志变量确定操作。
    """
    train_data = task['train']
    
    I = train_data[0]['input']
    O = train_data[0]['output']
    
    height_i, width_i = height(I), width(I)    # 输入对象的高度和宽度
    height_o, width_o = height(O), width(O)    # 输出对象的高度和宽度
    
    height_ratio = height_o / height_i
    width_ratio = width_o / width_i
    
    if height_ratio == 1 and width_ratio == 1:
        print("输入和输出的高度和宽度都保持不变")
        # 处理无缩放的情况
        functions = [
            vmirror,
            hmirror,
            cmirror,
            dmirror,
            rot90,
            rot180,
            rot270
        ]
        
        for fun in functions:
            result = do_fun_task(fun, task, flags)  # 执行 do_fun_task
            
            if  result:
                return result  
        
        
        
        fun,arg1,arg2 =prepare_diff(task,flags)
        

            # functions = [
            #     replace
            # ]

        result = do_fun_arg_task(fun, task, flags,arg1,arg2) # 执行 do_fun_task
        
        if result :
            return result  
        
        
        
        
    elif height_ratio == 2 and width_ratio == 2:
        print("输入和输出的高度和宽度均为 2 倍")
        # 处理高度和宽度均为 2 倍的情况
        functions = [
            upscale,
            hupscale,
            vupscale,
            downscale
        ]
        
        factor = 2
        
        for fun in functions:
            result = do_fun_arg_task(fun, task, flags, factor)  # 执行 do_fun_task
            
            if  result:
                return result  

    elif height_ratio == 3 and width_ratio == 3:
        print("输入和输出的高度和宽度均为 3 倍")
        # 处理高度和宽度均为 3 倍的情况
        functions = [
            upscale,
            hupscale,
            vupscale,
            downscale
        ]
        
        factor = 3
        
        for fun in functions:
            result = do_fun_arg_task(fun, task, flags, factor)  # 执行 do_fun_task
            
            if  result:
                return result  
        
        
        
        
        
        
        
    elif height_ratio == 2 and width_ratio == 1:
        print("高度为 2 倍，宽度保持不变")
        # 处理高度为 2 倍，宽度不变的情况

    elif height_ratio == 1 and width_ratio == 2:
        print("高度保持不变，宽度为 2 倍")
        # 处理高度不变，宽度为 2 倍的情况
        functions = [
            hconcat
        ]
        for fun in functions:
            result = do_fun_arg_task(fun, task, flags, 'input')  # 执行 do_fun_task
            
            if  result:
                return result  
        
        
        

    elif height_ratio == 3 and width_ratio == 1:
        print("高度为 3 倍，宽度保持不变")
        # 处理高度为 3 倍，宽度不变的情况

    elif height_ratio == 1 and width_ratio == 3:
        print("高度保持不变，宽度为 3 倍")
        # 处理高度不变，宽度为 3 倍的情况

    elif height_ratio == 2 and width_ratio == 3:
        print("高度为 2 倍，宽度为 3 倍")
        # 处理高度为 2 倍，宽度为 3 倍的情况

    elif height_ratio == 3 and width_ratio == 2:
        print("高度为 3 倍，宽度为 2 倍")
        # 处理高度为 3 倍，宽度为 2 倍的情况

    else:
        print("高度和宽度的比率不在预期范围内")
        # 处理其他情况
    
    do_fun_task(vmirror,task,flags)
    
    
    return None


def do_fun_task(fun: Callable, task: Dict, flags: Dict[str, List[bool]]) -> str:
    """
    尝试单独处理每个输入输出对，根据传入的函数 fun 检查每对输入输出是否满足条件。
    """
    
    train_data = task['train']
    test_data = task['test']
    
    for data_pair in train_data:
        input_grid = data_pair['input']
        output_grid = data_pair['output']
        
        # 使用传入的函数 fun 来检查是否满足条件
        transformed = fun(input_grid)
        if transformed == output_grid:
            # flags["is_fun_ok"].append(True)
            continue  # 结束本轮循环，直接进行下一个 data_pair
        else:
            print(f"failed : {fun.__name__}")
            # return f'failed {fun.__name__}'
            return False
    print(f"Do fun all ok : {fun.__name__}")
    testin = fun(test_data[0]['input'])
    assert testin == test_data[0]['output']
    return testin


def do_fun_arg_task(fun: Callable, task: Dict, flags: Dict[str, List[bool]], *args: Any) -> str:
    """
    尝试单独处理每个输入输出对，根据传入的函数 fun 和额外的参数 args 检查每对输入输出是否满足条件。
    """
    
    train_data = task['train']
    test_data = task['test']
    
    for data_pair in train_data:
        input_grid = data_pair['input']
        output_grid = data_pair['output']
        
        if 'input' in args :
            transformed = fun(input_grid, input_grid)
        else:
            # 使用传入的函数 fun 和额外的参数 args 来检查是否满足条件
            transformed = fun(input_grid, *args)  # 将额外参数传递给函数
        if transformed == output_grid:
            continue  # 结束本轮循环，直接进行下一个 data_pair
        else:
            print(f"failed : {fun.__name__}")
            # return f'failed {fun.__name__}'
            return False
    print(f"Do fun all ok : {fun.__name__}")
    
    if 'input' in args :
        testin = fun(test_data[0]['input'], test_data[0]['input'])
    else:
        testin = fun(test_data[0]['input'], *args)
    assert testin == test_data[0]['output']
    return testin

def solve_combined(input_grids, output_grids, flags: Dict[str, bool]):
    """
    尝试将多个输入和多个输出整体处理，使用标志变量辅助判断。
    """
    # 示例：根据标志条件判断
    if flags["is_mirror"] and flags["is_scale"]:
        return handle_combined_mirror_scale(input_grids, output_grids)
    elif flags["is_rotation"]:
        return handle_combined_rotation(input_grids, output_grids)
    # 可以添加更多组合操作

    return None

def apply_solution(test_input, solution):
    """
    将解决方案应用于测试输入。
    """
    return solution(test_input)


def handle_combined_mirror_scale(input_grids, output_grids):
    """处理组合镜像和缩放的逻辑"""
    return lambda x: x  # 示例处理函数

def handle_combined_rotation(input_grids, output_grids):
    """处理组合旋转的逻辑"""
    return lambda x: x  # 示例处理函数


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

def prepare_diff_insec(I,O):
    
    # 调用 objects 函数两次
    oi = objects(I, False, True, True)
    oo = objects(O, False, True, True)
    


    same_objects = oi.intersection(oo) 
    # 获取对称差集
    diff_objects = oi.symmetric_difference(oo)
    
    



def prepare_diff(task,flags: Dict[str, bool]):
    train_data = task['train']
    test_data = task['test']
    
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
    all_is_fun_ok = all(flags["is_diff_same_posit"])
    if all_is_fun_ok:
        keys_diff1 = list(merged_diffs['diff1'].keys())[0]  # 获取 diff1 中的键
        keys_diff2 = list(merged_diffs['diff2'].keys())[0]  # 获取 diff2 中的键
        # 如果所有数据对均满足条件，对 test_data 应用该函数并返回结果
        return replace, keys_diff1,keys_diff2
    return False

        
    
        
        # print("todo ！ 执行 ！  不同部分不止两个 frozenset 或无差异。")
    return 0        
        
        

        # # 定义特有的坐标
        # # diff1_coords = [(1, 0), (2, 2), (3, 1), (5, 3)]  # 第一个 frozenset 特有的坐标
        # # diff2_coords = [(1, 5), (2, 3), (3, 4), (5, 2)]  # 第二个 frozenset 特有的坐标

        # # 计算差异的维度和差值
        # def calc_diffs(coords1, coords2):
        #     x_diffs, y_diffs = [], []
        #     x_sums, y_sums = [], []
        #     for (x1, y1), (x2, y2) in zip(coords1, coords2):
        #         x_diffs.append(abs(x1 - x2))
        #         y_diffs.append(abs(y1 - y2))
        #         x_sums.append(x1 + x2)
        #         y_sums.append(y1 + y2)
        #     return x_diffs, y_diffs, x_sums, y_sums
        
        # def calculate_diff_sum(coords):
        #     x_diff = max(x for x, y in coords) - min(x for x, y in coords)
        #     y_diff = max(y for x, y in coords) - min(y for x, y in coords)
        #     x_sum = sum(x for x, y in coords)
        #     y_sum = sum(y for x, y in coords)
        #     return x_diff, y_diff, x_sum, y_sum

        # # # 计算坐标差异和和
        # # x_diffs, y_diffs, x_sums, y_sums = calc_diffs(merged_diffs[value]["diff1"], merged_diffs[value]["diff2"])
        
        # # 计算坐标差异和和
        # x_diffs, y_diffs, x_sums, y_sums = calc_diffs(merged_diffs[value]["diff1"], merged_diffs[value]["diff2"])

        # # 分别打印每个变量的内容
        # print("X 维度的差值列表:", x_diffs)
        # print("Y 维度的差值列表:", y_diffs)
        # print("X 维度的和列表:", x_sums)
        # print("Y 维度的和列表:", y_sums)


        # # 计算与输入和输出的高度和宽度的比较
        # # 计算与输入和输出的高度和宽度的比较（索引从0开始，需减一）
        # x_diff_matches_height = any(diff == (height_i - 1) or diff == (height_o - 1) for diff in x_diffs)
        # y_diff_matches_width = any(diff == (width_i - 1) or diff == (width_o - 1) for diff in y_diffs)

        # x_sum_matches_height = any(s == (height_i - 1) or s == (height_o - 1) for s in x_sums)
        # y_sum_matches_width = any(s == (width_i - 1) or s == (width_o - 1) for s in y_sums)

        # # 输出结果
        # print("X 维度差值是否匹配输入/输出高度:", x_diff_matches_height)
        # print("Y 维度差值是否匹配输入/输出宽度:", y_diff_matches_width)
        # print("X 维度和是否匹配输入/输出高度:", x_sum_matches_height)
        # print("Y 维度和是否匹配输入/输出宽度:", y_sum_matches_width)

 

from typing import Dict, List, Tuple
from collections import defaultdict
def compare_positions(merged_diffs: Dict[str, defaultdict]) -> str:
    """
    比较 'diff1' 和 'diff2' 字典中的坐标列表是否完全一致。
    忽略字典的键，仅比较坐标部分。
    """
    # 提取 diff1 和 diff2 中的坐标列表，忽略键
    coords1 = [coord for coords in merged_diffs['diff1'].values() for coord in coords]
    coords2 = [coord for coords in merged_diffs['diff2'].values() for coord in coords]
    
    # 比较坐标列表是否一致
    if sorted(coords1) == sorted(coords2):
        # return "Identical positions"
        return True
    else:
        # return "Different positions"
        return False






from typing import List, Tuple, Optional
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




def is_subgrid(grid1, grid2):
    """判断较小的网格是否是较大网格的一个部分"""
    
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
                return True  # 找到匹配位置，返回 True

    return False  # 未找到匹配位置，返回 False



# print(is_subgrid(big_grid, small_grid))  # 输出 True 或 False
