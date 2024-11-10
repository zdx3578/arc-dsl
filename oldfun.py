from dsl import *
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Callable, Optional
from collections import defaultdict
from config import *


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

        flags = initialize_flags()
        flags["use_fun1"] = [True]
        flags["use_fun2"] = [True]
        flags["use_fun3"] = [False]
        flags["use_fun4"] = [False]  # 设置 use_fun2 为 False，不执行 fun2
        flags["order"] = [1, 2]

        args_for_fun1 = [5, 0]
        args_for_fun2 = [3]

        # 调用函数
        result = do_check_train_get_test(
            do_4fun_task,
            task,
            flags,
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
