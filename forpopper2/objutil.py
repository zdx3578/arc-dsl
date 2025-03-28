from itertools import product
from typing import FrozenSet, Tuple, Union
from dsl import *
from dataclasses import dataclass

from dsl2 import *
from arc_types import *
import pandas as pd
pd.set_option('display.max_rows', None)      # 显示所有行
pd.set_option('display.max_columns', None)   # 显示所有列
pd.set_option('display.width', 1000)         # 调整输出宽度
pd.set_option('display.max_colwidth', None)  # 显示完整列内容
from copy import deepcopy

from objutil2plus import *
import sys
sys.path.append("bateson/")
from bateson import bateson_algorithm






managerid = IdManager()


def process_single_data(task: List[Any]) -> bool:
    # analysys_in_out_pattern_000(task)
    # result = analysys_in_out_pattern(task)
    # if result:
    #     return True

    analysys_out_out_pattern(task)



def analysys_out_out_pattern(task) -> bool:
    train_data = task['train']
    test_data = task['test']
    successful_obj_pairs = []
    df = pd.DataFrame(columns=columns_outout)
    temp_pd_data = []

    for paramid, out_param in enumerate(param_combinations2):  # 遍历 param_combinations
        obj_id_sets=[]
        for pair_id, data_pair in enumerate(train_data):
            I = input_grid = data_pair['input']
            O = output_grid = data_pair.get('output')  # 使用 get 方法获取 output，默认为 None
            height_i, width_i = height(I), width(I)    # 输入对象的高度和宽度
            height_o, width_o = height(O), width(O)

            out_obj_set = objects_info_from_one_params(the_pair_id=pair_id,in_or_out="out",grid=O, bools=out_param, hw=(height_o, width_o) )
            for out_obj in out_obj_set:  # 遍历 out_obj_set
                display_diff_color_ofa_matrices(out_obj.obj)
                display_matrices(out_obj.obj)
                bateson_algorithm(out_obj.obj)


            in_obj_set = objects_info_from_one_params(the_pair_id=pair_id,in_or_out="in",grid=I, bools=out_param, hw=(height_i, width_i) )
            len_out_obj_set = len(out_obj_set),
            len_in_obj_set = len(in_obj_set)
            print(paramid, "\nOutput parameters:", out_param,"pair id : ",  pair_id  , "  | Number of output objects:", len_out_obj_set)
            # for out_obj in out_obj_set:  # 遍历 out_obj_set
            #     display_matrices(out_obj.obj)
            obj_ids = count_obj_ids(out_obj_set,"obj_ID")
            print(f"\nout_obj_set 的 obj_ID 种类:\n {obj_ids}")
            obj_id_sets.append(obj_ids)
        ###!!! todo same id or same position?? same 00 or same 000
        all_same = all(obj_id_sets[0] == obj_id_set for obj_id_set in obj_id_sets) and (len(obj_id_sets[0]) > 1)
        if all_same:
            print("\n\n相同 > 1  所有 out_obj_set 的 obj_ID 种类相同！paramid : ", paramid, "Output parameters:", out_param)
            print("\n\n\n")
        else:
            print("\n\n不同 or = 1 存在不同的 obj_ID 种类！")
            print("\n\n\n")


def analysys_in_out_pattern(task) -> bool:
    # Initialize accumulators
    train_data = task['train']

    successful_params = []
    all_transformations = {}



    # Iterate through parameter combinations and accumulate successes
    for paramid, out_param in enumerate(param_combinations):

        # Check all training pairs with current parameter
        param_works_for_all_pairs = True
        success_pairs = []
        for pair_id, data_pair in enumerate(train_data):
            I = input_grid = data_pair['input']
            O = output_grid = data_pair.get('output')
            height_i, width_i = height(I), width(I)    # 输入对象的高度和宽度
            height_o, width_o = height(O), width(O)

            out_obj_set = objects_info_from_one_params(the_pair_id=pair_id,in_or_out="out",grid=O, bools=out_param, hw=(height_o, width_o) )
            # input_obj_set = objects_info_from_one_params(the_pair_id=pair_id,in_or_out="in",grid=I, bools=out_param, hw=(height_i, width_i) )

            input_obj_set = input_obj_setsparam = all_objects_from_grid_all_parma(the_pair_id=pair_id,in_or_out="in",grid=I, hw=(height_i, width_i))

            successful_obj = []
            # Check that all output objects are solvable with current parameter
            all_out_obj_satisfied = True
            for out_obj in out_obj_set:  # 遍历 out_obj_set
                # display_diff_color_ofa_matrices(out_obj.obj)
                # display_matrices(out_obj.obj)
                found_valid_in_outobj = False
                if not found_valid_in_outobj:
                    for in_obj in input_obj_set:  # 遍历 input_obj_set
                        if in_obj.obj == out_obj.obj:  # 如果找到满足条件的 in_obj
                            found_valid_in_outobj = True
                            tempdata = (lessforprintobj(out_obj), lessforprintobj(in_obj), "same_same")
                            successful_obj.append(tempdata)
                            break  # 存在一个满足条件即可退出内层循环
                #! 先平移  00  再其他旋转等  same_same",match_op
                if not found_valid_in_outobj:
                    for in_obj in input_obj_set:
                        if in_obj.obj_00 == out_obj.obj_00:  # 如果找到满足条件的 in_obj
                            found_valid_in_outobj = True
                            tempdata = (lessforprintobj(out_obj), lessforprintobj(in_obj), "same00_same00",'move')
                            successful_obj.append(tempdata)
                            break  # 存在一个满足条件即可退出内层循环
                if not found_valid_in_outobj:
                    for in_obj in input_obj_set:
                        match_op = next((name for name, res in out_obj.obj_ops if res == in_obj.obj), None)
                        if match_op is not None:
                            found_valid_in_outobj = True
                            tempdata = (lessforprintobj(out_obj), lessforprintobj(in_obj), "same_same",match_op)
                            successful_obj.append(tempdata)
                            break
                if not found_valid_in_outobj:
                    for in_obj in input_obj_set:
                        match_op = next((name for name, res in out_obj.obj_ops if res == in_obj.obj_00), None)
                        if match_op is not None:
                            found_valid_in_outobj = True
                            tempdata = (lessforprintobj(out_obj), lessforprintobj(in_obj), "same00_same",match_op)
                            successful_obj.append(tempdata)
                            break
                if not found_valid_in_outobj:
                    for in_obj in input_obj_set:
                        if in_obj.obj_000 == out_obj.obj_000:  # 如果找到满足条件的 in_obj
                            found_valid_in_outobj = True
                            tempdata = (lessforprintobj(out_obj), lessforprintobj(in_obj), "same00_0",'move+color')
                            successful_obj.append(tempdata)
                            break  # 存在一个满足条件即可退出内循环
                        # elif in_obj.obj_000 in out_obj.obj000_ops: ##any(x in out_obj.obj000_ops for x in in_obj.obj000_ops):    # 至少存在一个共同元素
                if not found_valid_in_outobj:
                    for in_obj in input_obj_set:
                        match_op = next((name for name, res in out_obj.obj000_ops if res == in_obj.obj_000), None)
                        if match_op is not None:
                            found_valid_in_outobj = True
                            tempdata = (lessforprintobj(out_obj), lessforprintobj(in_obj), "same00_0",match_op)
                            successful_obj.append(tempdata)
                            break
                if not found_valid_in_outobj:
                    for in_obj in input_obj_set:
                        if any(x in out_obj.obj000_ops for x in in_obj.obj000_ops):    # 至少存在一个共同元素
                            # in_obj.obj_00 == out_obj.obj_00:
                            found_valid_in_outobj = True
                            tempdata = (lessforprintobj(out_obj), lessforprintobj(in_obj), "extend222")
                            successful_obj.append(tempdata)
                            break
                if not found_valid_in_outobj:  # 如果没有找到满足条件的 in_obj
                    all_out_obj_satisfied = False
                    break  # 跳出中间层循环
            sorted_successful_obj = sorted(    successful_obj,    key=lambda x: int(x[0][3].split(' : ')[1])  )
            success_pairs.append((paramid, out_param, pair_id ,sorted_successful_obj))
            # success_pairs.append((successful_obj,))

            if not all_out_obj_satisfied:
                # Some output object was unsolvable - current parameter doesn't work for all pairs
                param_works_for_all_pairs = False
                break  # Move to the next parameter combination

        if param_works_for_all_pairs:
            # Current parameter worked for all training pairs - accumulate it
            # successful_params.append((paramid, out_param,success_pairs))
            successful_params.append((success_pairs))
            print(" - -  pretty_print")
            pretty_print(success_pairs)
            result = find_rule(success_pairs,task)
            if result:
                return True


    # print("\n\n all  all  pretty_print")
    # pretty_print(successful_params)


    # Construct the final result
    # success = len(successful_params) > 0
    return






def find_rule(df,task):
    test_data = task['test']
    op1, op2 = None, None
    found_mismatch = False
    for d in df:
        obj_pairs = d[3]
        if all(obj_pairs[0][2] == obj_pair[2] for obj_pair in obj_pairs) :
            # print("all obj one pair have the same -- obj:",obj_pairs[0][2])
            op1 = obj_pairs[0][2] if op1 is None else op1
            if op1 != obj_pairs[0][2]:
                found_mismatch = True
                op1, op2 = None, None
                break # 说明不是同一个 obj op


            if  ( len(obj_pairs[0]) > 3 and obj_pairs[0][3] ) :
                if all(obj_pairs[0][3] == obj_pair[3] for obj_pair in obj_pairs) :
                    # print("all obj one pair have the same -- operator:",obj_pairs[0][3])
                    op2 = obj_pairs[0][3] if op2 is None else op2
                    if op2 != obj_pairs[0][3]:
                        op1, op2 = None, None
                        found_mismatch = True
                        break # 说明不是同一个 obj op
        if found_mismatch:
            break  # 跳出最外层的 for 循环
    if op1 and op2 :
        print("op1: ",op1)
        print("op2: ",op2)
        # print("found rule")
        return apply_rule(op1,op2,task, df)
        # return op1, op2

#! 罗列所有的规则
def apply_rule(op1,op2,task,df):
    test_data = task['test']
    data_pair = test_data[0]
    I = input_grid = data_pair['input']
    O = output_grid = data_pair.get('output')  # 使用 get 方法获取 output，默认为 None
    background  = None
    height_widht = None
    if all(df[0][3][0][0][6] == obj_pairs[3][0][0][6] for obj_pairs in df) :
        print(f"all obj one pair have the background: {df[0][3][0][0][6]}, test var {df[0][3][0][0][2]}, {df[0][3][0][0][3]}")
        background = df[0][3][0][0][6]
    else:
        background = 0
    if all(df[0][3][0][0][7] == obj_pairs[3][0][0][7] for obj_pairs in df) :
        print(f"all obj one pair have the height_widht: {df[0][3][0][0][7]}, test var {df[0][3][-1][0][4]}, {df[0][3][-1][0][3]}")
        height_widht = df[0][3][0][0][7]

    if op1 == "same00_same00" and op2 == "move":
        # print("apply_rule: same00_same00 and move")
        # 确认移动位置在多pair之间是相同的，相同id 相同的位置 ，假设 df[0][3] 是一个列表或可迭代对象，且每个元素需要匹配其内部的某个位置，例如 [0][0][3]
        if all(
            all(d[3][i][0][3] == df[0][3][i][0][3] for d in df) and all(d[3][i][0][3] == df[0][3][i][0][3] for d in df)
            for i in range(len(df[0][3]))
            ):
            # print(f"\n\n！相同对象在相同的位置 ,test var: {df[0][3][0][0][2]}")
            input_obj_set = objects_info_from_one_params(the_pair_id="test",in_or_out="in",grid=I, bools=df[0][3][0][0][2], hw=("test") )
            moved_objs = move_in_obj_based_on_out(df,input_obj_set)
            output = paint_objects(moved_objs, background,height_widht)
            # display_matrices(output)
            assert output == O
            print("                   !  ok  !                     apply_rule: same00_same00 and move ok ")
            if output == O:
                return True
        # return  ( assert output == O)
    if op1 == "same00_same" and op2 == "vmirror":         #7468f01a
        #! if one op just run one op
        if all( len(d[3]) == 1 for d in df) :
            print
            input_obj_set = objects_info_from_one_params(the_pair_id="",in_or_out="in",grid=I, bools=df[0][3][0][1][2], hw=("") )
            obj = input_obj_set[0].obj
            obj00 = shift_pure_obj_to_00(obj)
            op2fun = globals()[op2]
            out = op2fun(obj00)
            output = object_to_grid(out)
            assert output == O
            if output == O:
                    return True

        print


    elif op1 == "same_same" :
        # print(" - -  pretty_print")
        pretty_print(df)
        print(op2)
        # function = findfunction(op2    getattr(solvers_module, f'solve_{key}')
        if all( len(d[3]) == 1 for d in df) :
            op2fun = globals()[op2]
            out = op2fun(I)
            assert out == O
            if out == O:
                    return True




    elif op1 == "same00_0" and op2 == "move+color":
        print("apply_rule: same00_0 and move+color")






@dataclass
class ObjInf:
    pair_id: Integer
    in_or_out: str
    objparam: Tuple[bool, bool, bool]  # 3个bool
    obj: Objects       # 假设这是一个通用对象
    obj_00: Objects    # 假设这是一个通用对象
    obj_ID: int
    obj_000: Objects   # 假设这是一个通用对象
    grid_H_W: Tuple[Integer, Integer]    # 假设是一个 (height, width) 的元组
    bounding_box: Tuple[Integer, Integer, Integer, Integer]    # 列表 [minr, minc, maxr, maxc]
    color_ranking: tuple(IntegerTuple)  # 从大到小的 多对( color count , color );
    background: int
    obj000_ops:list
    obj_ops:list





def lessforprintobj(obj):
    return (obj.pair_id,obj.in_or_out,obj.objparam,obj.obj_ID,obj.bounding_box, obj.color_ranking, obj.background, obj.grid_H_W)

def objects_fromone_params(the_pair_id: int, in_or_out: str, grid: Grid, bools: Tuple[bool, bool, bool],hw:list) -> Objects:
    b1, b2, b3 = bools  # 解包布尔值
    return objects( grid, b1, b2, b3)



def objects_info_from_one_params(the_pair_id: int, in_or_out: str, grid: Grid, bools: Tuple[bool, bool, bool],hw:list) -> Objects:
    b1, b2, b3 = bools  # 解包布尔值
    # return objects( grid, b1, b2, b3)
    result = []
    bg = mostcolor(grid)
    for obj in objects(grid, b1, b2, b3):
        # 对每个 obj，计算对应平移后的版本
        # 假设 obj 本身是一个表示对象的集合；如果不是，则请调整调用方式
        obj00 = shift_pure_obj_to_00(obj)
        obj000 = shift_pure_obj_to_0_0_0(obj)
        new_obj = ObjInf(
            pair_id='pair_id: '+str(the_pair_id),
            in_or_out=in_or_out,
            objparam=bools,  # 使用传入的布尔值
            obj=obj,         # 原始对象
            obj_00=obj00,
            obj_000=obj000,
            # obj_ID=managerid.get_id("OBJshape", obj000),
            obj_ID="objID : "+str(managerid.get_id("OBJshape", obj000)),
            grid_H_W=hw,            # 默认值，根据需要调整
            bounding_box=(uppermost(obj), leftmost(obj), lowermost(obj), rightmost(obj)),   # 默认值，根据需要调整
            color_ranking=palette(obj)    ,   # 默认空 tuple
            background = bg,
            obj000_ops=extend_obj(obj000),
            obj_ops = extend_obj(obj)
        )
        result.append(new_obj)
    return result

def all_objects_from_grid_all_parma(the_pair_id: int, in_or_out: str, grid: Grid, hw: list):
    result = []  # 初始化空列表存放所有结果
    for params in param_combinations:
        objs = objects_info_from_one_params(the_pair_id, in_or_out, grid, params, hw)
        result.extend(objs)  # 将当前参数组合的结果加入 result
    return result

def objop(obj,op):
    return grid_to_object(op(object_to_grid(obj)))

def s_filtered(s) :
    return frozenset(e for e in s if e[0] is not None)


def extend_obj(obj):
    """
    对传入的 obj 分别进行不同的变换，并返回一个包含
    (操作名称, 变换结果) 的元组，每个操作结果经过 s_filtered 处理（如果需要）。
    """
    transformations = [
        ("vmirror", vmirror),
        ("cmirror", cmirror),
        ("hmirror", hmirror),
        ("dmirror", dmirror),
        ("rot90", lambda o: s_filtered(objop(o, rot90))),
        ("rot180", lambda o: s_filtered(objop(o, rot180))),
        ("rot270", lambda o: s_filtered(objop(o, rot270))),
    ]
    results = tuple((name, func(obj)) for name, func in transformations)
    return results
