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


columns = [
    "pair_id", "out", "outparam", "output_id",
    "input_id",
    "outbounding_box", "outcolor",
    # "pair_id2",
    "in", "inparam",
    # "input_id",
    "inbounding_box", "incolor",
    "label"
]
columnsoutout = [
    "pair_id", "out", "outparam", "output_id",
    # "input_id",
    "outbounding_box", "outcolor",
    # "pair_id2",
    # "in", "inparam",
    # "input_id",
    # "inbounding_box", "incolor",
    # "label"
]


def parsed_pd_data(raw_data):
    data = {
        "pair_id": raw_data[0][0],
        "out": raw_data[0][1],
        "outparam": raw_data[0][2],
        "output_id": int(raw_data[0][3].split()[-1]),
        "outbounding_box": raw_data[0][4],
        "outcolor": next(iter(raw_data[0][5])),
        # "pair_id_2": raw_data[1][0],
        # "pair_id_2": raw_data[0][0],
        "in": raw_data[1][1],
        "inparam": str(raw_data[1][2]),
        "input_id": int(raw_data[1][3].split()[-1]),
        "inbounding_box": raw_data[1][4],
        "incolor": next(iter(raw_data[1][5])),
        "label": raw_data[2]
    }
    if len(raw_data) > 3 and raw_data[3]:
        data["operation"] = raw_data[3]
    return data

def parsed_pd_outout_data(raw_data):
    data = {
        "pair_id": raw_data[0][0],
        "out": raw_data[0][1],
        "outparam": raw_data[0][2],
        "output_id": int(raw_data[0][3].split()[-1]),
        "outbounding_box": raw_data[0][4],
        "outcolor": next(iter(raw_data[0][5])),
        # "pair_id_2": raw_data[1][0],
        # "pair_id_2": raw_data[0][0],
        # "in": raw_data[1][1],
        # "inparam": str(raw_data[1][2]),
        # "input_id": int(raw_data[1][3].split()[-1]),
        # "inbounding_box": raw_data[1][4],
        # "incolor": next(iter(raw_data[1][5])),
        # "label": raw_data[2]
    }
    if len(raw_data) > 3 and raw_data[3]:
        data["operation"] = raw_data[3]
    return data

class IdManager:
    def __init__(self):
        # 初始化字段：tables 用于存储各 category 下的值与 ID 映射；next_id 用于记录下一个可用的 ID
        self.tables = {}    # 例如: {'shape': {'shape_1': 1, 'shape_2': 2, ...}}
        self.next_id = {}   # 例如: {'shape': 1}

    def get_id(self, category, value):
        """
        获取 category 分类下 value 对应的 ID，
        如果 value 不存在，则分配新的 ID。
        """
        if isinstance(value, set):
            value = frozenset(value)

        if category not in self.tables:
            self.tables[category] = {}
            self.next_id[category] = 1

        category_table = self.tables[category]

        if value not in category_table:
            category_table[value] = self.next_id[category]
            self.next_id[category] += 1

        return category_table[value]

    def print_all_ids(self):
        """
        打印所有类别下的所有对象及其对应的 ID。
        """
        for category, category_table in self.tables.items():
            print(f"\n\nCategory: {category}, length: {len(category_table)} \n")
            for value, id_val in category_table.items():
                print(f"ID : {id_val} -> Object content -> : \n                  {value}")

    def reset(self):
        """
        清空所有数据
        """
        self.tables = {}
        self.next_id = {}
        print("All data has been reset.")

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
    extend:list
    extend2:list


managerid = IdManager()


def process_single_data(task: List[Any]) -> bool:
    # task1 = deepcopy(task)
    # task2 = deepcopy(task)
    # print(task1)
    analysys_in_out_pattern(task)
    # print("\n -----------------------------------------------------9999999999 ")
    # print(task2)
    analysys_out_out_pattern(task)



def analysys_out_out_pattern(task) -> bool:
    train_data = task['train']
    test_data = task['test']
    successful_obj_pairs = []
    df = pd.DataFrame(columns=columnsoutout)
    temp_pd_data = []

    for i, data_pair in enumerate(train_data):
        I = input_grid = data_pair['input']
        O = output_grid = data_pair.get('output')  # 使用 get 方法获取 output，默认为 None
        height_i, width_i = height(I), width(I)    # 输入对象的高度和宽度
        height_o, width_o = height(O), width(O)

        for paramid, out_param in enumerate(param_combinations):  # 遍历 param_combinations
            out_obj_set = output_objects_with_params(the_pair_id=i,in_or_out="out",grid=O, bools=out_param, hw=(height_o, width_o) )
            in_obj_set = output_objects_with_params(the_pair_id=i,in_or_out="in",grid=I, bools=out_param, hw=(height_i, width_i) )
            len_out_obj_set = len(out_obj_set),
            len_in_obj_set = len(in_obj_set)
            print("\n\nOutput parameters:", out_param, "| Number of output objects:", len_out_obj_set)
            for out_obj in out_obj_set:  # 遍历 out_obj_set
                parsed_data = parsed_pd_outout_data(out_obj)
                temp_pd_data.append(parsed_data)
            for in_obj in in_obj_set:  # 遍历 out_obj_set
                parsed_data = parsed_pd_outout_data(in_obj)
                temp_pd_data.append(parsed_data)
            df = pd.concat([df, pd.DataFrame(temp_pd_data)], ignore_index=True)
    print(df)
    # foranalysisshow(df)



def analysys_in_out_pattern(task: List[Any]) -> bool:
    ##首先是input  output模式分析
    train_data = task['train']
    test_data = task['test']
    successful_obj_pairs = []
    df = pd.DataFrame(columns=columns)
    temp_pd_data = []

    for i, data_pair in enumerate(train_data):
        I = input_grid = data_pair['input']
        O = output_grid = data_pair.get('output')  # 使用 get 方法获取 output，默认为 None

        height_i, width_i = height(I), width(I)    # 输入对象的高度和宽度
        height_o, width_o = height(O), width(O)

        input_obj_set = all_objects_from_grid(                the_pair_id=i,
                in_or_out="in",                grid=I, hw=(height_i, width_i) #,height_o, width_o)
            )

        successful_params = []
        for paramid, out_param in enumerate(param_combinations):  # 遍历 param_combinations
            all_out_obj_satisfied = True

            out_obj_set = output_objects_with_params(the_pair_id=i,
                                                    in_or_out="out",
                                                    grid=O, bools=out_param, hw=(height_o, width_o) )
            len_out_obj_set = len(out_obj_set)
            # if_duplicate_out_obj_set =
            print("\n\nOutput parameters:", out_param, "| Number of output objects:", len_out_obj_set)
            #调试  print 查看
            # for out_obj in out_obj_set:  # 遍历 out_obj_set
            #     # display_diff_matrices(out_obj.obj)
            #     display_matrices(out_obj.obj)


            successful_obj = []
            for out_obj in out_obj_set:  # 遍历 out_obj_set
                # display_diff_matrices(out_obj.obj)
                # display_matrices(out_obj.obj)
                found_valid_in_outobj = False
                if not found_valid_in_outobj:
                    for in_obj in input_obj_set:  # 遍历 input_obj_set
                        if in_obj.obj == out_obj.obj:  # 如果找到满足条件的 in_obj
                            found_valid_in_outobj = True
                            tempdata = (lessforprintobj(out_obj), lessforprintobj(in_obj), "same")
                            successful_obj.append(tempdata)
                            #successful_obj_pairs.append(tempdata)
                            parsed_data = parsed_pd_data(tempdata)
                            temp_pd_data.append(parsed_data)
                            break  # 存在一个满足条件即可退出内层循环
                if not found_valid_in_outobj:
                    for in_obj in input_obj_set:
                        if in_obj.obj_00 == out_obj.obj_00:  # 如果找到满足条件的 in_obj
                            found_valid_in_outobj = True
                            tempdata = (lessforprintobj(out_obj), lessforprintobj(in_obj), "same00")
                            successful_obj.append(tempdata)
                            #successful_obj_pairs.append(tempdata)
                            parsed_data = parsed_pd_data(tempdata)
                            temp_pd_data.append(parsed_data)
                            break  # 存在一个满足条件即可退出内层循环
                if not found_valid_in_outobj:
                    for in_obj in input_obj_set:

                        match_op = next((name for name, res in out_obj.extend2 if res == in_obj.obj), None)
                        if match_op is not None:
                            found_valid_in_outobj = True
                            tempdata = (lessforprintobj(out_obj), lessforprintobj(in_obj), "same",match_op)
                            successful_obj.append(tempdata)
                            #successful_obj_pairs.append(tempdata)
                            parsed_data = parsed_pd_data(tempdata)
                            temp_pd_data.append(parsed_data)
                            break
                if not found_valid_in_outobj:
                    for in_obj in input_obj_set:

                        match_op = next((name for name, res in out_obj.extend2 if res == in_obj.obj_00), None)
                        if match_op is not None:
                            found_valid_in_outobj = True
                            tempdata = (lessforprintobj(out_obj), lessforprintobj(in_obj), "same00",match_op)
                            successful_obj.append(tempdata)
                            #successful_obj_pairs.append(tempdata)
                            parsed_data = parsed_pd_data(tempdata)
                            temp_pd_data.append(parsed_data)
                            break
                if not found_valid_in_outobj:
                    for in_obj in input_obj_set:

                        if in_obj.obj_000 == out_obj.obj_000:  # 如果找到满足条件的 in_obj
                            found_valid_in_outobj = True
                            tempdata = (lessforprintobj(out_obj), lessforprintobj(in_obj), "same00_0")
                            successful_obj.append(tempdata)
                            #successful_obj_pairs.append(tempdata)
                            parsed_data = parsed_pd_data(tempdata)
                            temp_pd_data.append(parsed_data)
                            break  # 存在一个满足条件即可退出内循环
                        # elif in_obj.obj_000 in out_obj.extend: ##any(x in out_obj.extend for x in in_obj.extend):    # 至少存在一个共同元素
                if not found_valid_in_outobj:
                    for in_obj in input_obj_set:
                        match_op = next((name for name, res in out_obj.extend if res == in_obj.obj_000), None)
                        if match_op is not None:
                            found_valid_in_outobj = True
                            tempdata = (lessforprintobj(out_obj), lessforprintobj(in_obj), "same00_0",match_op)
                            successful_obj.append(tempdata)
                            #successful_obj_pairs.append(tempdata)
                            parsed_data = parsed_pd_data(tempdata)
                            temp_pd_data.append(parsed_data)
                            break
                if not found_valid_in_outobj:
                    for in_obj in input_obj_set:
                        if any(x in out_obj.extend for x in in_obj.extend):    # 至少存在一个共同元素
                            # in_obj.obj_00 == out_obj.obj_00:
                            found_valid_in_outobj = True
                            tempdata = (lessforprintobj(out_obj), lessforprintobj(in_obj), "extend222")
                            successful_obj.append(tempdata)
                            #successful_obj_pairs.append(tempdata)
                            parsed_data = parsed_pd_data(tempdata)
                            temp_pd_data.append(parsed_data)
                            break
                if not found_valid_in_outobj:  # 如果没有找到满足条件的 in_obj
                    all_out_obj_satisfied = False
                    break  # 跳出中间层循环
            # printlist(successful_obj)
            if all_out_obj_satisfied:  # 如果所有 out_obj 都满足
                # successful_params.append((successful_obj,))
                successful_params.append((paramid, out_param, len_out_obj_set ,successful_obj))  # 累计成功的参数组合

        # 检查是否至少有一个成功
        if not successful_params:  # 如果没有找到任何成功的参数组合
            return False  # 直接返回 False，表示失败
        print("\n\npretty_print")
        # pretty_print(successful_params)

        # print("\n\nforprintlist")
        # forprintlist(successful_params)
    # print("\n\nsuccessful_obj_pairs")
    # printlist(successful_obj_pairs)
    # print("lenght of successful_obj_pairs: ", len(successful_obj_pairs))
    print("\n\n")
    df = pd.concat([df, pd.DataFrame(temp_pd_data)], ignore_index=True)
    foranalysisshow(df)

    # print("\n按 outparam 和 pair_id 排序后的 DataFrame:")
    # print(sorted_df)
    # print(df)
    return True  # 所有 pair 都成功

def foranalysisshow(df):
    value_counts = df['outparam'].value_counts()

    sorted_value_counts = value_counts.sort_values(ascending=True)
    print("\n按 outparam count  排序后的 DataFrame:")
    print(sorted_value_counts)
    print("\n")

    df['out_param_count'] = df['outparam'].map(value_counts)
    sorted_df = df.sort_values(by=["out_param_count","outparam", "pair_id", "output_id"], ascending=[True, True, True, True])

    column_widths = {col: max(sorted_df[col].astype(str).apply(len).max(), len(col)) + 3 for col in sorted_df.columns}

    # 输出时检测 outparam 的变化并插入空行
    previous_outparam = None
    output_lines = []
    header = "".join(f"{col:^{column_widths[col]}}" for col in sorted_df.columns)

    output_lines.append(header)
    for _, row in sorted_df.iterrows():
        current_outparam = row['outparam']
        # 如果 outparam 发生变化，插入一个空行
        if previous_outparam is not None and current_outparam != previous_outparam:
            output_lines.append("")  # 插入空行
            print("\n".join(output_lines))
        # 添加当前行到输出
        # output_line = "\t".join(str(row[col]) for col in sorted_df.columns)
        formatted_row = []
        for col in sorted_df.columns:
            value = str(row[col])
            width222 = column_widths[col]
            formatted_row.append(f"{value:^{width222}}")
        output_lines.append("".join(formatted_row))
        # 更新 previous_outparam
        previous_outparam = current_outparam
    # 打印最终输出
    print("\n".join(output_lines))


def lessforprintobj(obj):
    return (obj.pair_id,obj.in_or_out,obj.objparam,'ID is '+str(obj.obj_ID),obj.bounding_box, obj.color_ranking)

# printlist = lambda x: print("\n".join(map(str, x)))
def printlist(x):
    # for l in list:
    #     print(l)
    print("\n\n".join(map(str, x)))
    # lambda x: print("\n".join(map(str, x)))

def forprintlist(xx):
    for x in xx:
        print("\n\n")
        print("\n\n".join(map(str, x)))

param_combinations: List[Tuple[bool, bool, bool]] = [
    (False, False, False),
    (False, False, True),
    (False, True, False),
    (False, True, True),
    (True, True, False),
    (True, True, True),
    (True, False, False),
    (True, False, True)
]

def objects_with_params(the_pair_id: int, in_or_out: str, grid: Grid, bools: Tuple[bool, bool, bool],hw:list) -> Objects:
    b1, b2, b3 = bools  # 解包布尔值
    return objects( grid, b1, b2, b3)

def output_objects_with_params(the_pair_id: int, in_or_out: str, grid: Grid, bools: Tuple[bool, bool, bool],hw:list) -> Objects:
    b1, b2, b3 = bools  # 解包布尔值
    # return objects( grid, b1, b2, b3)
    result = []
    for obj in objects(grid, b1, b2, b3):
        # 对每个 obj，计算对应平移后的版本
        # 假设 obj 本身是一个表示对象的集合；如果不是，则请调整调用方式
        obj00 = shift_pure_obj_to_00(obj)
        obj000 = shift_pure_obj_to_0_0_0(obj)
        new_obj = ObjInf(
            pair_id=the_pair_id,
            in_or_out=in_or_out,
            objparam=bools,  # 使用传入的布尔值
            obj=obj,         # 原始对象
            obj_00=obj00,
            obj_000=obj000,
            obj_ID=managerid.get_id("OBJshape", obj000),

            grid_H_W=hw,            # 默认值，根据需要调整
            bounding_box=(uppermost(obj), leftmost(obj), lowermost(obj), rightmost(obj)),   # 默认值，根据需要调整
            color_ranking=palette(obj)    ,   # 默认空 tuple
            extend=extend_obj(obj000),
            extend2 = extend_obj(obj)
        )
        result.append(new_obj)
    return result

# all_objects_from_grid 函数
def all_objects_from_grid(the_pair_id: int, in_or_out: str, grid: Grid, hw:list) -> FrozenSet[Object]:
    acc: FrozenSet[Object] = frozenset()  # 初始化空集合
    for params in param_combinations:
        acc = acc.union(objects_with_params(the_pair_id, in_or_out, grid, params,hw))
        # print()
    result = []
    for obj in acc:
        # 对每个 obj，计算对应平移后的版本
        # 假设 obj 本身是一个表示对象的集合；如果不是，则请调整调用方式
        obj00 = shift_pure_obj_to_00(obj)
        obj000 = shift_pure_obj_to_0_0_0(obj)
        new_obj = ObjInf(
            pair_id=the_pair_id,
            in_or_out=in_or_out,
            objparam="all",  # 使用传入的布尔值
            obj=obj,         # 原始对象
            obj_00=obj00,
            obj_000=obj000,
            obj_ID=managerid.get_id("OBJshape", obj000),
            grid_H_W=hw,            # 默认值，根据需要调整
            bounding_box=(uppermost(obj), leftmost(obj), lowermost(obj), rightmost(obj)),    # 默认值，根据需要调整
            color_ranking=palette(obj)    ,     # 默认空 tuple
            extend=extend_obj(obj000),
            extend2 = extend_obj(obj)
        )
        result.append(new_obj)
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


def shift_pure_obj_to_0_0_0(obj):
    """
    对于纯对象集合（以 set 表示），将所有对象坐标平移到 (0,0) 并将颜色重设为 0.
    假设每个对象 e 格式为 (color, (r, c))，其中 color, r, c 为整数.
    """
    if not obj:
        return set()
    obj_list = list(obj)
    # 提取所有 (r, c)
    rc_list = [e[1] for e in obj_list]
    min_row = min(r for r, c in rc_list)
    min_col = min(c for r, c in rc_list)
    new_set = set()
    for e in obj_list:
        # 忽略原始 color，统一设置为 0
        _, (r, c) = e
        new_obj = (0, (r - min_row, c - min_col))
        new_set.add(new_obj)
    return new_set


def shift_pure_obj_to_00(obj):
    """
    对于纯对象集合，将所有对象坐标平移到 (0,0)，但保持原始颜色不变.
    假设每个对象 e 格式为 (color, (r, c)).
    """
    if not obj:
        return set()
    obj_list = list(obj)
    rc_list = [e[1] for e in obj_list]
    min_row = min(r for r, c in rc_list)
    min_col = min(c for r, c in rc_list)
    new_set = set()
    for e in obj_list:
        color, (r, c) = e
        new_obj = (color, (r - min_row, c - min_col))
        new_set.add(new_obj)
    return new_set




def pretty_print(dataall, indent=2):
    """
    格式化打印嵌套数据结构，每个字段一行，嵌套列表的每个子列表也分行打印。
    :param dataall: 要打印的数据（可以是元组、列表、集合、字典等）
    :param indent: 当前缩进级别（用于递归调用）
    """
    # 定义缩进字符串
    indent_str = " " * (indent * 5)

    if isinstance(dataall, (tuple, list)):
        # 如果是元组或列表
        print(indent_str + "[")
        for data in dataall:
            if isinstance(data, (tuple, list)) and len(data) == 4:
                # 如果是包含四个字段的元组 (paramid, out_param, len_out_obj_set, successful_obj)
                paramid, out_param, len_out_obj_set, successful_obj = data
                print(indent_str + " " * 4 + f"paramid: {paramid}")
                print(indent_str + " " * 4 + f"out_param: {out_param}")
                print(indent_str + " " * 4 + f"len_out_obj_set: {len_out_obj_set}")
                print(indent_str + " " * 4 + "successful_obj:")
                # 打印 successful_obj 中的每个元素
                if isinstance(successful_obj, (tuple, list)):
                    print(indent_str + " " * 8 + "[")
                    for item in successful_obj:
                        # print(indent_str + str(item))
                        print(indent_str + " " * 12 + str(item))
                        # pretty_print(item, indent + 3)  # 递归打印 successful_obj 中的每个元素
                    print(indent_str + " " * 8 + "]")
                else:
                    print(indent_str + " " * 8 + str(successful_obj))
            else:
                # 对其他类型的元组或列表递归打印
                pretty_print(data, indent + 1)
        print(indent_str + "]")
    elif isinstance(dataall, frozenset):
        # 如果是 frozenset，转换为列表后递归打印
        print(indent_str + "frozenset({")
        pretty_print(sorted(dataall), indent + 1)  # 排序以便输出更整齐
        print(indent_str + "})")
    elif isinstance(dataall, dict):
        # 如果是字典，逐键值对打印
        for key, value in dataall.items():
            print(indent_str + f"{key}:")
            pretty_print(value, indent + 1)
    else:
        # 普通数据类型直接打印
        print(indent_str + str(dataall))

