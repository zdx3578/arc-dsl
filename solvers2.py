from collections import Counter, defaultdict
from dsl import *
from dsl2 import *
from typing import Dict, Any, List, Tuple, Callable, Optional, Union
from config import *
from dslIsDo import *
from oldfun import *
from dslupdateProperflagsIs import *
from solvers import *
from arc_types import *
from dataclasses import dataclass

import sys
sys.path.append("/Users/zhangdexiang/github/VSAHDC/arc-dsl/forpopper2")
# from objutil import *





def solve_arc_task(task):

    solutions = []

    # solution = solve_individual3(task)

    solution = solve_individual2(task)

    # flags = initialize_flags()
    # solution = solve_individual(task, flags)
    # 测试解决方案是否正确test_data[0]['output']
    assert solution == task['test'][0]['output']

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

    # height_i, width_i = height(I), width(I)    # 输入对象的高度和宽度
    # height_o, width_o = height(O), width(O)    # 输出对象的高度和宽度

    # height_ratio = height_o / height_i
    # width_ratio = width_o / width_i

    # spm1 = split_matrix_by_frontiers(train_data[2]['input'], False)
    # spg2 = split_grid_by_indices(train_data[2]['input'], False)
    # for i, (key, sub_grid) in enumerate(spg2.items()):
    #     assert spm1[i] == spg2[key]

    #!! 两套 flags 一个是全局的，一个是每个训练数据对的
    flagsNtrain = [initialize_flags() for _ in range(len(train_data))]
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
            funs = do_check_inputOutput_proper_1functions(
                proper_functions, task, flags)
            # proper_fun = fun
            # args = []
            # if fun list order  = [1,2,3] and usefun2 ture
            args_for_fun1 = []

            for fun in funs:
                if fun:
                    result = do_check_train_get_test(
                        do_4fun_task,
                        task,
                        flags,
                        fun, args_for_fun1)
                    if result:
                        return result
        except Exception as e:
            # 捕获异常并打印错误信息
            logging.error("捕获到异常：%s", e)
            logging.error("详细错误信息：\n%s", traceback.format_exc())
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
        except Exception as e:
            logging.error("捕获到异常：%s", e)
            logging.error("详细错误信息：\n%s", traceback.format_exc())
            pass

        # part_functions
        # flags["out_in"] = True
        # ##!!!!!! set false after use
        # flags["out_in"] = False

        # flags.get["out_in"] = [True]

        ##############################

        try:

            proper_all_functions = proper_functions
            is_fun_flag = do_check_inputComplexOutput_proper_functions(
                proper_all_functions, task, flags, flagsNtrain)

            fun_process_list = howtodo(is_fun_flag)

            if fun_process_list:
                result = prepare_funlist_and_call_do_test(
                    fun_process_list,
                    do_check_train_get_test,
                    do_4fun_task,
                    task,
                    flags,
                )
                if result:
                    return result

            result = is_proper_finding(task, flagsNtrain)
            # ！！ add prepare_diff(task)
            if result:
                return result
        except Exception as e:
            logging.error("捕获到异常：%s", e)
            logging.error("详细错误信息：\n%s", traceback.format_exc())
            pass

        try:
            result = do_check_inputOutput_proper_flagsK_functions(proper_small_functions, task, flags)
            if result:
                return result

            # do_check_Part_InputOutput_proper_flagsK_functions(proper_all_functions, task, flags)
            print(
                "-----------------------------------------单独处理失败，需进一步尝试联合处理。---------------------------------")
            # if all failed
            task = preprocess_cut_background(task)
            task = preprocess_noise(task)

        except Exception as e:
            logging.error("捕获到异常：%s", e)
            logging.error("详细错误信息：\n%s", traceback.format_exc())
            pass

# def do_check_Part_InputOutput_proper_flagsK_functions(proper_functions, task: Dict, flags: Dict[str, List[bool]]):
#     print('do_check_Part_InputOutput_proper_flagsK_functions')
#     from copy import deepcopy

#     task0 = deepcopy(task)
#     train_data = task0['train']
#     test_data = task0['test']

#     # for test_pair in test_data:
#     train_data.append({'input': test_data[0]['input'], 'output': None})  # 增加合并时将 output 设置为 None
#     flagsNTtrain = [initialize_flags() for _ in range(len(train_data))]

#     for fun in proper_functions:
#         # for fun in [vmirror]:
#         flags["out_in"] = True
#         *args, = []
#         for i, data_pair in enumerate(train_data):
#             I = input_grid = data_pair['input']
#             O = output_grid = data_pair.get('output')  # 使用 get 方法获取 output，默认为 None

#             if output_grid is not None:
#                 height_i, width_i = height(I), width(I)    # 输入对象的高度和宽度
#                 height_o, width_o = height(O), width(O)    # 输出对象的高度和宽度

#                 height_ratio = height_o / height_i
#                 width_ratio = width_o / width_i

#                 flags["height_ratio"].append(height_ratio)
#                 flags["width_ratio"].append(width_ratio)
#                 flags["height_o"].append(height_o)
#                 flags["width_o"].append(width_o)

#             if height_ratio == 0.5 and width_ratio == 1 :
#                 #half_part
#                 pass
#             elif height_ratio == 0.3333333333333333 and width_ratio == 1:
#                 part_input = [upper_third(I), middle_third(I), lower_third(I)]
#                 part_names = ["upper_third", "middle_third", "lower_third"]

#                 for part, part_name in zip(part_input, part_names):
#                     transformed = safe_execute(fun, part, *args)
#                     if transformed == part:
#                         flagsNTtrain[i][f"{part_name}_in_in_fun"].append(fun)

#     return



def do_check_inputOutput_proper_flagsK_functions(proper_functions, task: Dict, flags: Dict[str, List[bool]]):

    print('do_check_input___FlagsK___proper_functions')
    from copy import deepcopy
    # prepare_diff(task, flags)
    task0 = deepcopy(task)
    train_data = task0['train']
    test_data = task0['test']


    # for test_pair in test_data:
    train_data.append({'input': test_data[0]['input'], 'output': None})  # 增加合并时将 output 设置为 None
    flagsNTtrain = [initialize_flags() for _ in range(len(train_data))]
    getIO_diff_task_flagslist(task, flagsNTtrain)

    for fun in proper_functions:
        # for fun in [vmirror]:

        # if "half" in fun.__name__ or "mirror" in fun.__name__:
        flags["out_in"] = True
        args = []

        for i, data_pair in enumerate(train_data):
            I = input_grid = data_pair['input']
            O = output_grid = data_pair.get('output')  # 使用 get 方法获取 output，默认为 None




            if output_grid is not None:
                height_i, width_i = height(I), width(I)    # 输入对象的高度和宽度
                height_o, width_o = height(O), width(O)    # 输出对象的高度和宽度

                height_ratio = height_o / height_i
                width_ratio = width_o / width_i

                flags["height_ratio"].append(height_ratio)
                flags["width_ratio"].append(width_ratio)
                flags["height_o"].append(height_o)
                flags["width_o"].append(width_o)

                numbcolor = palette(output_grid)
                if len(numbcolor) == 1:
                    # compare_flagK_dicts will error when numbcolor is 1 use []
                    flagsNTtrain[i]["output_allone_color"] = [next(iter(numbcolor))]

                #subgrid check
                result =  is_subgrid_grid(O, I)
                # if result:
                #     flagsNTtrain[i]["out_is_in_subgrid"].append(result)
                if result and result not in flagsNTtrain[i]["out_is_in_subgrid"]:
                    flagsNTtrain[i]["out_is_in_subgrid"].append(result)

                # fun(output_grid)
                if flags["out_in"] == True:
                    transformed = safe_execute(fun, output_grid, *args)
                    if transformed == input_grid:
                        flagsNTtrain[i]["out_in_fun"].append(fun)
                        # if fun not in flags["out_in_fun"]:
                        #     flags["out_in_fun"].append(fun)

                    if transformed == output_grid:
                        flagsNTtrain[i]["out_out_fun"].append(fun)
                        # if fun not in flags["out_out_fun"]:
                        #     flags["out_out_fun"].append(fun)

                # fun(input_grid)
                transformed = safe_execute(fun, input_grid, *args)
                if transformed == output_grid:
                    flagsNTtrain[i]["in_out_fun"].append(fun)
                    flagsNTtrain[i]["out_is_in_fun"].append(fun)
                    # if fun not in flags["in_out_fun"]:
                    #     flags["in_out_fun"].append(fun)

            transformed = safe_execute(fun, input_grid, *args)
            if transformed == input_grid:
                flagsNTtrain[i]["in_in_fun"].append(fun)
                # if fun not in flags["in_in_fun"]:
                #     flags["in_in_fun"].append(fun)


            if height_ratio == 0.5 and width_ratio == 1 :
                #half_part
                pass
            elif height_ratio == 0.3333333333333333 and width_ratio == 1:
                part_input = [upper_third(I), middle_third(I), lower_third(I)]
                part_names = ["upper_third", "middle_third", "lower_third"]
                for part, part_name in zip(part_input, part_names):
                    transformed = safe_execute(fun, part, *args)
                    if transformed == part:
                        flagsNTtrain[i][f"{part_name}_in_in_fun"].append(fun.__name__)
                        # flagsNTtrain[i]["third_in_in_fun"].append(fun.__name__)
                        flagsNTtrain[i]["all_third_in_in_fun"].append(f"{part_name}+{fun.__name__}")
                # flagsNTtrain[i]["spset_third_in_in_fun"] = list(    set(flagsNTtrain[i]["third_in_in_fun"]) - set(part_names))
                    elif transformed != part:
                        pass
                        # flagsNTtrain[i][f"{part_name}_in_in_fun"].append(fun.__name__)
                        # flagsNTtrain[i]["spset_third_in_in_fun"].append(fun.__name__)
                        # flagsNTtrain[i]["all_third_in_in_fun"].append(f"{part_name}+{fun.__name__}")



            # if flagsNTtrain[i]["height_ratio"] or flagsNTtrain[i]["width_ratio"] == 0.333:
            #     part_input

        flags["out_in"] = False
    print('do_check_input___FlagsK___proper_functions')
    # return flags if flags else [False]

    flagsNTtrain_part1 = flagsNTtrain[:-1]
    flagsNTtrain_part2 = flagsNTtrain[-1:]

    proper2, output_proper_result, output_proper_result_reversed , non_empty_values = compare_flagK_dicts(flagsNTtrain_part1)
    matches = match_test_flagK(flagsNTtrain_part2, output_proper_result)
    if matches and all_equal(match[0][0] for match in matches):
        if matches[0][0][0] == "output_allone_color":
            result  = canvas( matches[0][0][1][0],( all_equal(flags["height_o"]), all_equal(flags["width_o"])  ))
            if result:
                return result

    ## 47 some part proper select
    if flagsNTtrain_part2[0]['all_third_in_in_fun']:
        for key,value in flagsNTtrain_part2[0].items():
            # if value:
            if 'third_in_in_fun' in key:
                if value:
                    pass
                else:
                    new_str = key.replace('_in_in_fun', '')
                    if new_str :
                        # solver = getattr(solvers_module, f'solve_{key}')
                        result = globals()[new_str](test_data[0]['input'])
                        if result:
                            return result

    # 48 42a50994
    if all(flag['out_is_IOintersec_obj'] for flag in flagsNTtrain_part1):
        if all(flag['diff1_unique_all_is_single'] for flag in flagsNTtrain_part1):
            result = solve_42a50994(test_data[0]['input'])
            if result:
                return result



    return




def match_test_flagK(test_flagK_list, output_proper_result):
    """
    根据 output_proper_result 循环匹配 test 的 flagK。

    参数:
    test_flagK_list: 测试数据的 flagK 列表。
    output_proper_result: 存储匹配结果的字典。

    返回:
    匹配结果的列表。
    """
    matches = []
    for test_flagK in test_flagK_list:
        for key, value_list in output_proper_result.items():
            for value in value_list:
                key_to_match, sub_key_to_match = value[0]
                if key_to_match in test_flagK:
                    if (  (sub_key_to_match == [] ) and (test_flagK[key_to_match] == () )  ) or sub_key_to_match in test_flagK[key_to_match]:  # 修改：处理空元组的情况
                        matches.append((key, value))
    return matches


def all_equal(iterable):
    """
    判断 iterable 中的所有元素是否都相等。

    参数:
    iterable: 可迭代对象。

    返回:
    如果所有元素都相等，返回 True；否则返回 False。
    """
    iterator = iter(iterable)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)



def do_check_inputComplexOutput_proper_functions(proper_functions, task: Dict, flags: Dict[str, List[bool]], flagsNtrain):

    print('do_check_input___ComplexOutput___proper_functions')

    prepare_diff(task, flags)

    train_data = task['train']
    test_data = task['test']

    # flags = flags

    # if len(flagsN) != len(train_data):
    #     raise ValueError("flagsN 的长度必须与 train_data 的长度一致。")

    # I = train_data[0]['input']
    # O = train_data[0]['output']

    # height_i, width_i = height(I), width(I)    # 输入对象的高度和宽度
    # height_o, width_o = height(O), width(O)    # 输出对象的高度和宽度

    # if height_o > height_i and width_o > width_i:
    #     height_ratio = int(height_o / height_i)
    # elif height_o < height_i and width_o < width_i:
    #     height_ratio = int(height_i / height_o)
    # else:
    #     height_ratio = 0

    # get proper and  args

    for fun in proper_functions:
        # for fun in [vmirror]:

        if "half" in fun.__name__ or "mirror" in fun.__name__:
            flags["out_in"] = True
        # else:
        #     ##!!!!!! set false after use  half not concat
        #     flags["out_in"] = False

        args = []

        # success = True
        for i, data_pair in enumerate(train_data):
            # data_pair = train_data[2]
            input_grid = data_pair['input']
            output_grid = data_pair['output']

            # ##flagK = flagsNtrain[i]
            # numbcolor = palette(output_grid)
            # if len(numbcolor) == 1:
            #     # compare_flagK_dicts will error when numbcolor is 1 use []
            #     flagsNtrain[i]["output_allone_color"] = [next(iter(numbcolor))]

            # fun(output_grid)
            if flags["out_in"] == True:
                transformed = safe_execute(fun, output_grid, *args)
                if transformed == input_grid:
                    # flagsNtrain[i]["out_in_fun"].append(fun)
                    if fun not in flags["out_in_fun"]:
                        flags["out_in_fun"].append(fun)

                    # continue

                if transformed == output_grid:
                    # flagsNtrain[i]["out_out_fun"].append(fun)
                    if fun not in flags["out_out_fun"]:
                        flags["out_out_fun"].append(fun)

                    # continue
                # if transformed == otherfun(input_grid):

            # fun(input_grid)
            transformed = safe_execute(fun, input_grid, *args)
            if transformed == output_grid:
                # flagsNtrain[i]["in_out_fun"].append(fun)
                if fun not in flags["in_out_fun"]:
                    flags["in_out_fun"].append(fun)

                # continue

            if transformed == input_grid:
                # flagsNtrain[i]["in_in_fun"].append(fun)
                if fun not in flags["in_in_fun"]:
                    flags["in_in_fun"].append(fun)

                # continue

            # else:
            # print(f"failed : {fun.__name__}")
        #     success = False
        #     break
        # if success:
        #     print(f"ok____ : {fun.__name__}")
        # else:
        #     # print(f"failed : {fun.__name__}")
        #     pass
        flags["out_in"] = False
    print('do_check_input___ComplexOutput___proper_functions')
    return flags if flags else [False]


def is_proper_finding(task, flagsNtrain):
    train_data = task['train']
    test_data = task['test']
    flags = initialize_flags()

    result = is_underfill_corners(task, flags)
    if result:
        return result
    result = is_objectComplete_change_color(task, flags, True)
    if result:
        return result
    result = check_largest_objects_dimensions(train_data[1]['input'])
    if result:
        flags["can_partition"] = True

    # result = is_mirror_hole_get_args(task, flags)
    # if result:
    #     flags["is_get_mirror_hole"] = result
    #     get_mirror_hole(I, color=0)

    for i, data_pair in enumerate(train_data):
        # data_pair = train_data[1]
        #! 上面已经初始化了
        flagK = flagsNtrain[i]

        I = input_grid = data_pair['input']
        O = output_grid = data_pair['output']

        # height_i, width_i = height(I), width(I)    # 输入对象的高度和宽度
        # height_o, width_o = height(O), width(O)    # 输出对象的高度和宽度

        # height_ratio = height_o / height_i
        # width_ratio = width_o / width_i

        # flagK["height_ratio"].append(height_ratio)
        # flagK["width_ratio"].append(width_ratio)

        # 提取输入对象特征
        # update_objects_proper_flags(input_grid, output_grid, flagK)

        # 处理信息更新 findflags[i]
        update_proper_in_out_flags(input_grid, output_grid, flagK)

        #################################################################

        # flagsNtrain[i] = flagK

    ################################################
    height_ratios = [flagK["height_ratio"][0]
                     for flagK in flagsNtrain if "height_ratio" in flagK and flagK["height_ratio"]]
    width_ratios = [flagK["width_ratio"][0]
                    for flagK in flagsNtrain if "width_ratio" in flagK and flagK["width_ratio"]]
    # all_in_out_fun = [flagK["in_out_fun"]
    #                   for flagK in flagsNtrain if "in_out_fun" in flagK and flagK["in_out_fun"]]
    all_in_out_fun = {
        fun for flagK in flagsNtrain if "in_out_fun" in flagK for fun in flagK["in_out_fun"]}

    # 38 7b7f7511
    if len(set(height_ratios)) == 1:
        pass
    elif len(set(height_ratios)) == 2:
        if len(set(width_ratios)) == 2:
            if len(all_in_out_fun) == 4:
                # if (0.5 in height_ratios and ((lefthalf and righthalf) in flag-i-k['in_out_fun']) and 0.5 in width_ratio and (tophalf and bottomhalf) in flagK['in_out_fun']:
                # if (lefthalf and righthalf) in flagK['in_out_fun']
                for flagK in flagsNtrain:
                    if 0.5 in height_ratios and 0.5 in width_ratios:
                        in_out_fun = flagK.get("in_out_fun", [])
                        if lefthalf in in_out_fun and righthalf in in_out_fun:
                            for other_flagK in flagsNtrain:
                                if other_flagK != flagK:
                                    other_in_out_fun = other_flagK.get(
                                        "in_out_fun", [])
                                    if tophalf in other_in_out_fun and bottomhalf in other_in_out_fun:
                                        result = do_portrait_half(test_data[0]['input'])
                                        if result:
                                            return result
    #                                 return
    #                             return
    #                     return
    #                 return
    #         return
    #     return
    # return
    # grouped_results, proper2 = compare_flagK_dicts(flagsNtrain)
    # if grouped_results:
    #     return



def prepare_funlist_and_call_do_test(fun_process_list: List[List[Any]],
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
    print("Fun List : calling is prepare_funlist_and_call_do_test")
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

    if flags["same_diff_is_frontier"]:
        color = flags["fill_frontier_color"]
        return [[do_frontier, [color]]]

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
                if top_half_left_quarter in flags["out_in_fun"]:
                    # # 处理 vmirror + top_half_left_quarter 的情况
                    flags["use_fun2"] = [True]
                    flags["use_fun3"] = [True]
                    flags["use_fun4"] = [True]
                    return [
                        [vmirror, []],          # hmirror 函数，无参数
                        # vconcat 函数，有参数 'pin' 和 'in'
                        [hconcat, ['in', 'pin']],
                        [hmirror, []],          # hmirror 函数，无参数
                        [vconcat, ['in', 'pin']]  # vconcat 函数，有参数 'pin' 和 'in'
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
