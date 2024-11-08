from dsl import *
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Callable, Optional
from collections import defaultdict
from config import *
from dslIsDo import *


def solve_arc_task(task):

    # 初始化标志变量
    # flags = initialize_flags()

    # train_data = task['train']
    # test_data = task['test']

    # 尝试逐个处理每个输入输出对
    solutions = []

    solution = solve_individual2(task)
    if solution:
        return solution
        solutions.append(solution)
    else:
        print("单独处理失败，需进一步尝试联合处理。")

    # 如果单独处理失败或无法找到方案，尝试整体处理
    if len(solutions) != len(train_data):
        combined_solution = solve_combined([pair[0] for pair in train_data], [
                                           pair[1] for pair in train_data], flags)
        if combined_solution:
            solutions = combined_solution

    # 用解决方案应用于测试数据
    results = [apply_solution(test_input, solutions)
               for test_input in test_data]
    return results


def solve_individual2(task):
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

    flags_data = initialize_flags()
    flags_data["use_fun1"] = [True]
    flags_data["use_fun2"] = [False]
    flags_data["use_fun3"] = [False]
    flags_data["use_fun4"] = [False]  # 设置 use_fun2 为 False，不执行 fun2
    flags_data["order"] = [1]

    # flags = initialize_flags()

    # if height_ratio == 1 and width_ratio == 1:
        # for fun in proper_functions:
    [fun] = do_check_inputOutput_proper_1functions(
        proper_functions, task, flags_data)
    # proper_fun = fun
    # args = []
    # if fun list order  = [1,2,3] and usefun2 ture
    args_for_fun1 = []
    if fun:
        result = do_check_train_get_test(
            do_4fun_task,
            task,
            flags_data,
            fun, args_for_fun1)
        if result:
            return result

    fun, arg = do_check_inputOutput_proper_1_arg_functions(
        proper_1arg_functions, task, flags_data)
    # args = []
    # if fun list order  = [1,2,3] and usefun2 ture
    args_for_fun1 = [arg]
    if fun:
        result = do_check_train_get_test(
            do_4fun_task,
            task,
            flags_data,
            fun, args_for_fun1)
        if result:
            return result

    # if (height_o == 2 * height_i and width_o == width_i) or (width_o == 2 * width_i and height_o == height_i):
    fun = do_check_inputOutput_proper_1functions(
        part_functions, task, flags_data)

    # fun, arg = do_check_inputOutput_proper_concat_functions(proper_concat_functions
    #     task, flags_data)
    ok_fun_names = [fun.__name__ for fun in flags_data["ok_fun"]]
    if ok_fun_names == ['righthalf', 'lefthalf']:
        fun = hconcat
        args_for_fun1 = 'sameinput'

    if fun:
        result = do_check_train_get_test(
            do_4fun_task,
            task,
            flags_data,
            fun, args_for_fun1)
        if result:
            return result

        # proper_fun = do_check_inputOutput_proper_1functions(proper_functions, task, flags_data)
        # proper_fun = fun
        # partfun = do_check_inputOutput_proper_1functions(
        #     part_functions, task, flags_data)

        # fun, arg = do_check_inputOutput_proper_mirror_concat_functions(
        #     task, flags_data)

    # for fun in proper_functions:
    #     result = do_check_inputComplexOutput_proper_functions

    print("----------------------------------------------------------------------------")
    # return





def do_check_inputOutput_proper_1functions(proper_functions, task: Dict, flags: Dict[str, List[bool]]):
    train_data = task['train']
    # test_data = task['test']
    for fun in proper_functions:
        success = True
        for data_pair in train_data:
            input_grid = data_pair['input']
            output_grid = data_pair['output']

            # fun(output_grid)
            transformed = fun(output_grid)
            if transformed == output_grid:
                # flags["out_out"].append(True)
                continue

            if transformed == input_grid:
                # flags["out_in"].append(True)
                continue

            # fun(input_grid)
            transformed = fun(input_grid)
            if transformed == output_grid:
                # flags["in_out"].append(True)
                continue

            if transformed == input_grid:
                # flags["in_in"].append(True)
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
    #验证成功几个函数
    return flags["ok_fun"] if flags["ok_fun"] else [False]

def do_check_inputOutput_proper_1_arg_functions(proper_1arg_functions, task: Dict, flags: Dict[str, List[bool]]):
    train_data = task['train']
    test_data = task['test']

    I = train_data[0]['input']
    O = train_data[0]['output']

    height_i, width_i = height(I), width(I)    # 输入对象的高度和宽度
    height_o, width_o = height(O), width(O)    # 输出对象的高度和宽度

    if height_o > height_i and width_o > width_i:
        height_ratio = int(height_o / height_i)
    else:
        height_ratio = int(height_i / height_o)

    # if height_o == 2 * height_i and width_o == width_i:
    #     return "Vertical stack (上下堆叠)"
    # elif width_o == 2 * width_i and height_o == height_i:
    #     return "Horizontal stack (左右堆叠)"

    for fun in proper_1arg_functions:
        success = True
        for data_pair in train_data:
            input_grid = data_pair['input']
            output_grid = data_pair['output']

            # fun(output_grid)
            transformed = fun(output_grid, height_ratio)
            if transformed == output_grid:
                # out-out-proper
                continue

            if transformed == input_grid:
                # out-input-proper_flags
                continue

            # fun(input_grid)
            transformed = fun(input_grid, height_ratio)
            if transformed == output_grid:
                # out-out-proper
                continue

            if transformed == input_grid:
                # out-input-proper_flags
                continue

            # else:
            print(f"failed : {fun.__name__}")
            success = False
            break
        if success:
            print(f"ok____ : {fun.__name__}")
            return fun, height_ratio
        else:
            print(f"failed : {fun.__name__}")
    return False


def do_check_inputComplexOutput_proper_functions(properComplex_functions, task: Dict, flags: Dict[str, List[bool]]):
    train_data = task['train']
    for fun in properComplex_functions:
        result = fun(train_data)
        if result:
            # todo
            return fun


def check_train_fun(
    do_4fun_task: Callable,
    task: List[Dict],
    flags: Dict[str, List[bool]],
    fun1: Callable[[Any], Any], args1: List[Any],
    fun2: Callable[[Any], Any], args2: Optional[List[Any]] = None,
    fun3: Optional[Callable[[Any], Any]] = None, args3: Optional[List[Any]] = None,
    fun4: Optional[Callable[[Any], Any]] = None, args4: Optional[List[Any]] = None
) -> Dict[str, Any]:

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
            if args == 'sameinput' : #or args == [['righthalf', 'lefthalf']]:
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


def outintput_is_part_fun(fun: Callable, task: Dict, flags: Dict[str, List[bool]]) -> str:
    """
    尝试单独处理每个输入输出对，根据传入的函数 fun 检查每对输入输出是否满足条件。
    """

    train_data = task['train']
    test_data = task['test']

    for data_pair in train_data:
        input_grid = data_pair['input']
        output_grid = data_pair['output']

        # 使用传入的函数 fun 来检查是否满足条件
        transformed = fun(output_grid)
        if transformed == input_grid:
            # flags["is_fun_ok"].append(True)
            continue  # 结束本轮循环，直接进行下一个 data_pair
        else:
            print(f"failed : {fun.__name__}")
            # return f'failed {fun.__name__}'
            return False
    print(f"Do fun all ok : {fun.__name__}")
    return fun


def out_is_proper_fun(fun: Callable, task: Dict, flags: Dict[str, List[bool]]) -> str:
    """
    尝试单独处理每个输入输出对，根据传入的函数 fun 检查每对输入输出是否满足条件。
    """

    train_data = task['train']
    # test_data = task['test']

    for data_pair in train_data:
        # input_grid = data_pair['input']
        output_grid = data_pair['output']

        # 使用传入的函数 fun 来检查是否满足条件
        transformed = fun(output_grid)
        if transformed == output_grid:
            # flags["is_fun_ok"].append(True)
            continue  # 结束本轮循环，直接进行下一个 data_pair
        else:
            print(f"failed : {fun.__name__}")
            # return f'failed {fun.__name__}'
            return False
    print(f"Do fun all ok : {fun.__name__}")
    return fun
    # testin = fun(test_data[0]['input'])
    # assert testin == test_data[0]['output']
    # return testin


def do_2funswicharg_task(fun: Callable, fun22: Callable, task: Dict, flags: Dict[str, List[bool]]) -> str:
    """
    尝试单独处理每个输入输出对，根据传入的函数 fun 检查每对输入输出是否满足条件。
    """

    train_data = task['train']
    test_data = task['test']
    # bi_map = BidirectionalMap(mapping)

    # fun2 = bi_map.get(fun22)

    for data_pair in train_data:
        input_grid = data_pair['input']
        output_grid = data_pair['output']

        # 使用传入的函数 fun 来检查是否满足条件
        I2 = fun(input_grid)
        transformed = fun_swicharg_action(fun22, input_grid, I2)
        # transformed = fun2(transformed,input_grid)
        if transformed == output_grid:
            # flags["is_fun_ok"].append(True)
            continue  # 结束本轮循环，直接进行下一个 data_pair
        else:
            print(f"failed : {fun.__name__},{fun22.__name__}")
            # return f'failed {fun.__name__}'
            return False
    print(f"2 Do fun all ok : {fun.__name__},{fun22.__name__}")
    I = test_data[0]['input']
    I2 = fun(test_data[0]['input'])
    testin = fun_swicharg_action(fun22, I, I2)
    # testin = fun2(testin,test_data[0]['input'])
    assert testin == test_data[0]['output']
    print(f"2 Do fun all - test - ok ")
    return testin


def fun_swicharg_action(fun22: Callable, I, I2):
    # 获取函数名
    func_name = fun22.__name__

    bi_map = BidirectionalMap(mapping)
    fun2 = bi_map.get(fun22)
    # 根据函数名包含的关键字返回相应的值
    if "top" in func_name:
        return fun2(I, I2)
        # 也可以返回对应的处理结果或执行相应的功能
    elif "bottom" in func_name:
        return fun2(I2, I)
        # 执行 down 的代码
    elif "left" in func_name:
        return fun2(I, I2)
        # 执行 left 的代码
    elif "right" in func_name:
        return fun2(I, I2)
        # 执行 right 的代码
    else:
        return "No action matches."


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
    print(f"2 Do fun all - test - ok ")
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

        if 'input' in args:
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

    if 'input' in args:
        testin = fun(test_data[0]['input'], test_data[0]['input'])
    else:
        testin = fun(test_data[0]['input'], *args)
    assert testin == test_data[0]['output']
    print(f"2 Do fun all - test - ok ")
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


def prepare_diff_insec(I, O):

    # 调用 objects 函数两次
    oi = objects(I, False, True, True)
    oo = objects(O, False, True, True)

    same_objects = oi.intersection(oo)
    # 获取对称差集
    diff_objects = oi.symmetric_difference(oo)


def prepare_color_count(task, flags: Dict[str, bool]):

    return


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
                    return crop, (i, j)  # 找到匹配位置，返回 True

    return False  # 未找到匹配位置，返回 False


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

            if result:
                return result

        fun, arg1, arg2 = prepare_diff(task, flags)
        # functions = [
        #     replace
        # ]
        result = do_fun_arg_task(
            fun, task, flags, arg1, arg2)  # 执行 do_fun_task

        if result:
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
            result = do_fun_arg_task(
                fun, task, flags, factor)  # 执行 do_fun_task

            if result:
                return result

    elif height_ratio == 2 or width_ratio == 2:
        functions = [
            hconcat,
            vconcat

        ]
        for fun in functions:
            result = do_fun_arg_task(
                fun, task, flags, 'input')  # 执行 do_fun_task
            if result:
                return result

        exe_fun = [
            canvas
        ]
        for fun in proper_functions:
            # type: ignore # 执行 do_fun_task
            isproper = out_is_proper_fun(fun, task, flags)
            if isproper:
                for fun in part_functions:
                    part_fun = outintput_is_part_fun(
                        fun, task, flags)  # type: ignore
                    if part_fun:
                        result3 = do_2funswicharg_task(
                            isproper, part_fun, task, flags)
                        return result3
                for fun in exe_fun:
                    exe_fun = check_train_fun

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
            result = do_fun_arg_task(
                fun, task, flags, factor)  # 执行 do_fun_task

            if result:
                return result

    elif height_ratio <= 1 and width_ratio <= 1:

        ifsubgrid = is_subgrid(task, flags)
        if ifsubgrid:
            fun, (arg1, arg2) = ifsubgrid
            result = do_fun_arg_task(
                # 执行 do_fun_task
                fun, task, flags, (arg1, arg2), (height_o, width_o))
            if result:
                return result

    # elif height_ratio == 0.3333333333333333 and width_ratio == 0.3333333333333333:
        functions = [
            replace,
            downscale
        ]

        flags_data = initialize_flags()
        flags_data["use_fun1"] = [True]
        flags_data["use_fun2"] = [True]
        flags_data["use_fun3"] = [False]
        flags_data["use_fun4"] = [False]  # 设置 use_fun2 为 False，不执行 fun2
        flags_data["order"] = [1, 2]

        args_for_fun1 = [5, 0]
        args_for_fun2 = [3]

        # 调用函数
        result = do_check_train_get_test(
            do_4fun_task,
            task,
            flags_data,
            replace, args_for_fun1,
            downscale, args_for_fun2)
        if result:
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

    # do_fun_task(vmirror,task,flags)

    return None
