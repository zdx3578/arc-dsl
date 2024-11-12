from collections import Counter, defaultdict
from dsl import *
from dsl2 import *
from typing import Dict, Any, List, Tuple, Callable, Optional
from config import *
from dslIsDo import *
from oldfun import *
from dslProperIs import *


def solve_arc_task(task):

    solutions = []

    solution = solve_individual2(task)

    # flags = initialize_flags()
    # solution = solve_individual(task, flags)

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

    flags = initialize_flags()
    flags["use_fun1"] = [True]
    flags["use_fun2"] = [False]
    flags["use_fun3"] = [False]
    flags["use_fun4"] = [False]  # 设置 use_fun2 为 False，不执行 fun2
    flags["order"] = [1, 2, 3, 4]
    # flags = initialize_flags()
    # if height_ratio == 1 and width_ratio == 1:
    # for fun in proper_functions:

    for i in range(2):
        try:
            [fun] = do_check_inputOutput_proper_1functions(
                proper_functions, task, flags)
            # proper_fun = fun
            # args = []
            # if fun list order  = [1,2,3] and usefun2 ture
            args_for_fun1 = []

            if fun:
                result = do_check_train_get_test(
                    do_4fun_task,
                    task,
                    flags,
                    fun, args_for_fun1)
                if result:
                    return result
        except:
            pass

        ###############################
        try:
            [funarg] = do_check_inputOutput_proper_1_arg_functions(
                proper_1arg_functions, task, flags)
            if funarg:
                if len(funarg) == 3:
                    fun, arg1, arg2 = funarg
                    args_for_fun1 = [arg1, arg2]
                elif len(funarg) == 2:
                    fun, arg1 = funarg
                    args_for_fun1 = [arg1]

            if fun:
                result = do_check_train_get_test(
                    do_4fun_task,
                    task,
                    flags,
                    fun, args_for_fun1)
                if result:
                    return result
        except:
            pass

        # part_functions
        # flags["out_in"] = True
        # ##!!!!!! set false after use
        # flags["out_in"] = False

        # flags.get["out_in"] = [True]

        ##############################
        proper_all_functions = proper_functions
        is_fun_flag = do_check_inputComplexOutput_proper_functions(
            proper_all_functions, task, flags)

        fun_process_list = howtodo(is_fun_flag)

        if fun_process_list:
            result = prepare_and_call_do_test(
                fun_process_list,
                do_check_train_get_test,
                do_4fun_task,
                task,
                flags,
            )
            if result:
                return result

        # if all failed
        task = preprocess_noise(task)
        is_proper_finding(task, flags)


def is_proper_finding(task):
    train_data = task['train']
    findedflags = {}

    for i, data_pair in enumerate(train_data):
        flags = initialize_flags()

        input_grid = data_pair['input']
        output_grid = data_pair['output']

        # 获取输入和输出的高度和宽度



        # 提取输入对象特征
        update_objects_proper_flags(input_grid, output_grid, flags)

        # 处理信息更新 findflags[i]
        update_proper_in_out_flags(input_grid, output_grid, flags)





        # 更新 is_scale 标志项
        if (height_i != height_o or width_i != width_o):
            flags["is_scale"].append(True)
        else:
            flags["is_scale"].append(False)

        # 更新 is_color_transform 标志项
        if palette(input_grid) != palette(output_grid):
            flags["is_color_transform"].append(True)
        else:
            flags["is_color_transform"].append(False)

        # 更新 is_position_swap 标志项（示例：通过位置互换判断）
        if position_swap(input_grid, output_grid):
            flags["is_position_swap"].append(True)
        else:
            flags["is_position_swap"].append(False)

        # 更新 is_output_one_color 标志项
        if len(palette(output_grid)) == 1:
            flags["is_output_one_color"].append(True)
        else:
            flags["is_output_one_color"].append(False)

        # 更新 output_allone_color 标志项
        flags["output_allone_color"].append(all(cell == output_grid[0][0] for row in output_grid for cell in row))

        # 更新 out_is_in_subgrid 和 in_is_out_subgrid 标志项
        if is_subgrid(input_grid, output_grid):
            flags["in_is_out_subgrid"][0] = True
        if is_subgrid(output_grid, input_grid):
            flags["out_is_in_subgrid"][0] = True


        findedflags[i] = flags

    is_input_firstobjsame_outallobject()


    return findedflags


def prepare_and_call_do_test(fun_process_list: List[List[Any]],
                             do_check_train_get_test: Callable,
                             do_4fun_task: Callable,
                             task: List[Dict],
                             flags: Dict[str, List[bool]]):
    # 准备要传入 do_check_train_get_test 的参数
    fun_args = {}

    # 循环遍历 fun_process_list 并提取函数和参数
    for i, (fun, args) in enumerate(fun_process_list):
        fun_key = f"fun{i + 1}"
        args_key = f"args{i + 1}"
        fun_args[fun_key] = fun                    # 提取函数
        fun_args[args_key] = args if args else None  # 提取参数，若为空则设为 None

    # 为没有传回的函数和参数提供默认值 None
    for i in range(1, 5):  # 确保 fun1 到 fun4 和 args1 到 args4 都存在
        fun_args.setdefault(f"fun{i}", None)
        fun_args.setdefault(f"args{i}", None)

    # 调用 do_check_train_get_test 函数
    return do_check_train_get_test(
        do_4fun_task,
        task,
        flags,
        fun_args["fun1"], fun_args["args1"],
        fun_args["fun2"], fun_args["args2"],
        fun_args["fun3"], fun_args["args3"],
        fun_args["fun4"], fun_args["args4"]
    )


def howtodo(flags):
    processed_flags = {}
    flags = flags

    if flags["in_out_fun"]:
        if left_third in flags["in_out_fun"] and right_third in flags["in_out_fun"]:
            flags["use_fun2"] = [False]
            return [[left_third, []]]

    # 处理 "out_in_fun" 标签
    if flags["out_in_fun"]:
        if lefthalf in flags["out_in_fun"] and righthalf in flags["out_in_fun"]:
            flags["use_fun2"] = [False]
            return [[hconcat, ['in', 'in']]]
        if bottomhalf in flags["out_in_fun"] and tophalf in flags["out_in_fun"]:
            flags["use_fun2"] = [False]
            return [[vconcat, ['in', 'in']]]

        if lefthalf in flags["out_in_fun"]:
            # 处理 hmirror + lefthalf 的情况
            pass
        if righthalf in flags["out_in_fun"]:
            pass

        if bottomhalf in flags["out_in_fun"]:
            # 处理 vmirror + bottomhalf 的情况
            pass
        if tophalf in flags["out_in_fun"]:
            # 处理 vmirror + tophalf 的情况
            pass

    # 处理 "out_out_fun" 标签
    if flags["out_out_fun"]:

        if vmirror in flags["out_out_fun"]:
            if lefthalf in flags["out_in_fun"]:
                # 处理 hmirror + lefthalf 的情况
                flags["use_fun2"] = [True]
                return [
                    [vmirror, []],            # vmirror 函数，无参数
                    [hconcat, ['in', 'pin']]   # hconcat 函数，有参数 'pin' 和 'in'
                ]

            if righthalf in flags["out_in_fun"]:
                # 处理 hmirror + righthalf 的情况
                flags["use_fun2"] = [True]
                return [
                    [vmirror, []],            # vmirror 函数，无参数
                    [hconcat, ['pin', 'in']]   # hconcat 函数，有参数 'pin' 和 'in'
                ]

        if hmirror in flags["out_out_fun"]:
            if bottomhalf in flags["out_in_fun"]:
                # 处理 vmirror + bottomhalf 的情况
                flags["use_fun2"] = [True]
                return [
                    [hmirror, []],          # hmirror 函数，无参数
                    [vconcat, ['pin', 'in']]  # vconcat 函数，有参数 'pin' 和 'in'
                ]
            if tophalf in flags["out_in_fun"]:
                # 处理 vmirror + tophalf 的情况
                flags["use_fun2"] = [True]
                return [
                    [hmirror, []],          # hmirror 函数，无参数
                    [vconcat, ['in', 'pin']]  # vconcat 函数，有参数 'pin' 和 'in'
                ]

    # 处理 "in_in_fun" 标签
    if flags["in_in_fun"]:
        processed_values = []
        for value in flags["in_in_fun"]:
            # 在这里添加处理每个值的逻辑
            processed_value = process_value(value)
            processed_values.append(processed_value)
        processed_flags["in_in_fun"] = processed_values

    return False


def do_check_inputComplexOutput_proper_functions(proper_functions, task: Dict, flags: Dict[str, List[bool]]):

    print('do_check_input___ComplexOutput___proper_functions')
    train_data = task['train']
    test_data = task['test']

    # flags = flags

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

    for fun in proper_functions:

        if "half" in fun.__name__ or "mirror" in fun.__name__:
            flags["out_in"] = True
        # else:
        #     ##!!!!!! set false after use  half not concat
        #     flags["out_in"] = False

        args = []

        success = True
        for data_pair in train_data:
            input_grid = data_pair['input']
            output_grid = data_pair['output']

            # fun(output_grid)
            if flags["out_in"] == True:
                transformed = safe_execute(fun, output_grid, *args)
                if transformed == input_grid:
                    if fun not in flags["out_in_fun"]:
                        flags["out_in_fun"].append(fun)
                    continue

                if transformed == output_grid:
                    if fun not in flags["out_out_fun"]:
                        flags["out_out_fun"].append(fun)
                    continue

            # fun(input_grid)
            transformed = safe_execute(fun, input_grid, *args)
            if transformed == output_grid:
                if fun not in flags["in_out_fun"]:
                    flags["in_out_fun"].append(fun)
                continue

            if transformed == input_grid:
                if fun not in flags["in_in_fun"]:
                    flags["in_in_fun"].append(fun)
                continue

            # else:
            print(f"failed : {fun.__name__}")
            success = False
            break
        if success:
            print(f"ok____ : {fun.__name__}")
        else:
            print(f"failed : {fun.__name__}")
        flags["out_in"] = False
    print('do_check_input___ComplexOutput___proper_functions')
    return flags if flags else [False]


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

# change name to get fun args
