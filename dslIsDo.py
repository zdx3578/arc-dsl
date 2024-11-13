from dsl import *
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Callable, Optional
import logging
import traceback
from config import *


def is_input_firstobjsame_outallobject():

    return

def is_objectComplete_change_color(I,O):
    difference = prepare_diff
    contained = contains_object(difference)
    complete = complementofobject(contained)
    complete_ischange = is_change_color(complete)


    return



# # 如何判断是get_first_object
# def is_a_object_of(I,O,flags):
#     x1 = objects(I, T, T, T)
#     # O is  partof x1
#     flags["is_a_object_of"] = True
#     return
def is_move():
    return

def get_first_object(I):
    x1 = objects(I, T, T, T)
    x2 = first(x1)
    O = subgrid(x2, I)
    return O

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


def do_output_most_input_color(I):
    x1 = mostcolor(I)
    return canvas(x1, (height(I), width(I)))


def process_value(value: bool) -> Any:
    return


def preprocess_noise(task):
    """
    now just for #18, 5614dbcf
    """
    # 遍历任务中的所有训练和测试样本
    for sample in task['train'] + task['test']:
        input2dgrid = sample['input']
        # 找到所有噪声位置
        noise = ofcolor(input2dgrid, 5)
        replaced_grid = input2dgrid

        # 遍历每个噪声位置，替换为其邻居的主要颜色
        for n in noise:
            # 获取噪声位置的邻居
            neighbors_list = neighbors(n)
            neighbor_colors = [index(input2dgrid, pos) for pos in neighbors_list if index(
                input2dgrid, pos) is not None]
            # 计算邻居颜色的频率
            most_color = mostcolor2(neighbor_colors)
            # 将噪声位置的值替换为邻居中最多的颜色
            replaced_grid = replace2(replaced_grid, n, most_color)
        # 更新 sample 中的 input 为替换后的网格
        sample['input'] = replaced_grid
    return task


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


def do_check_inputOutput_proper_1_arg_functions(proper_1arg_functions, task: Dict, flags: Dict[str, List[bool]]):
    train_data = task['train']
    test_data = task['test']
    print('do_check_inputOutput_proper_1___arg___functions')

    flags.get("ok_fun", [])

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

    for fun in proper_1arg_functions:
        # if "half" in fun.__name__:
        #     flags["out_in"] = True
        # else:
        #     ##!!!!!! set false after use
        #     flags["out_in"] = False

        if fun == switch or fun == replace:
            args = []
            funarg = prepare_diff(task, flags)
            if funarg:
                if len(funarg) == 3:
                    funget, arg1, arg2 = funarg
                    if funget == fun:
                        fun = funget
                        args = [arg1, arg2]
        elif fun == crop:
            args = []
            funarg = is_subgrid(task, flags)
            if funarg:
                if len(funarg) == 3:
                    funget, arg1, arg2 = funarg
                    args = [(arg1, arg2), (height_o, width_o)]
                    fun = funget
                    # if funget == fun:
                        # fun = funget
                        # args = [arg1, arg2]
        else:
            args = [height_ratio]

        success = True
        for data_pair in train_data:
            input_grid = data_pair['input']
            output_grid = data_pair['output']

            # if fun == crop:
            #     args = []
            #     height_o, width_o = height(output_grid), width(output_grid)
            #     funarg = is_subgrid(task, flags)
            #     if funarg:
            #         if len(funarg) == 3:
            #             funget, arg1, arg2 = funarg
            #             args = [(arg1, arg2), (height_o, width_o)]
            #             fun = funget
            # fun(output_grid)
            if flags["out_in"] == True:
                transformed = safe_execute(fun, output_grid, *args)
                if transformed == input_grid:
                    # out-input-proper_flags
                    continue

            # fun(input_grid)
            transformed = safe_execute(fun, input_grid, *args)
            if transformed == output_grid:
                # out-input-proper_flags
                continue

            # else:
            print(f"failed : {fun.__name__}")
            success = False
            break
        if success:
            print(f"ok____ : {fun.__name__}")
            flags["ok_fun"].append([fun, *args])
            # height_ratio is args to exe
            # return fun, height_ratio
        else:
            print(f"failed : {fun.__name__}")
    print('do_check_inputOutput_proper_1___arg___functions')
    return flags["ok_fun"] if flags["ok_fun"] else [False]


def do_check_inputOutput_proper_1functions(proper_functions, task: Dict, flags: Dict[str, List[bool]]):
    train_data = task['train']
    # test_data = task['test']
    print('do_check_inputOutput___proper___functions')
    flags.get("ok_fun", [])
    for fun in proper_functions:

        # if "concat" in fun.__name__:
        #     flags["out_in"] = True
        # else:
        #     ##!!!!!! set false after use
        #     flags["out_in"] = False

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
    print('do_check_inputOutput___proper___functions')
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
    tmpinput_grid = input_grid
    # 根据顺序调用函数
    for idx in order:
        fun, args = functions[idx - 1]  # idx-1是因为order是从1开始的
        if flags.get(f"use_fun{idx}", [True])[0]:  # 检查是否需要调用当前函数
            if "concat" in fun.__name__:
                # 执行包含 "concat" 的函数
                if args == ['pin', 'in']:
                    input_grid = fun(input_grid, tmpinput_grid)
                if args == ['in', 'pin']:
                    input_grid = fun(tmpinput_grid, input_grid)
                if args == ['in', 'in']:
                    input_grid = fun(tmpinput_grid, tmpinput_grid)
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

