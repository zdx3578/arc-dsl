from itertools import product
from typing import FrozenSet, Tuple, Union
from dsl import *
from dataclasses import dataclass
from arc_types import *



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





managerid = IdManager()

def process_single_data(task: List[Any]) -> bool:
    train_data = task['train']
    test_data = task['test']

    successful_obj_pairs = []
    for i, data_pair in enumerate(train_data):
        I = input_grid = data_pair['input']
        O = output_grid = data_pair.get('output')  # 使用 get 方法获取 output，默认为 None

        height_i, width_i = height(I), width(I)    # 输入对象的高度和宽度
        height_o, width_o = height(O), width(O)

        input_obj_set = all_objects_from_grid(                the_pair_id=i,
                in_or_out="in",                grid=I, hw=(height_i, width_i) #,height_o, width_o)
            )

        successful_params = []
        for out_param in param_combinations:  # 遍历 param_combinations
            all_out_obj_satisfied = True

            out_obj_set = output_objects_with_params(the_pair_id=i,
                                                    in_or_out="out",
                                                    grid=O, bools=out_param, hw=(height_o, width_o) )

            successful_obj = []
            for out_obj in out_obj_set:  # 遍历 out_obj_set
                found_valid_in_obj = False
                for in_obj in input_obj_set:  # 遍历 input_obj_set
                    if in_obj.obj_000 == out_obj.obj_000:  # 如果找到满足条件的 in_obj
                        found_valid_in_obj = True
                        successful_obj.append((lessforprintobj(in_obj), lessforprintobj(out_obj),"same"))
                        successful_obj_pairs.append((lessforprintobj(in_obj), lessforprintobj(out_obj), "same"))
                        break  # 存在一个满足条件即可退出内层循环
                    elif any(x in out_obj.extend for x in in_obj.extend):    # 至少存在一个共同元素
                        # in_obj.obj_00 == out_obj.obj_00:
                        found_valid_in_obj = True
                        successful_obj.append((lessforprintobj(in_obj), lessforprintobj(out_obj),"extend"))
                        successful_obj_pairs.append((lessforprintobj(in_obj), lessforprintobj(out_obj), "extend"))
                        break
                if not found_valid_in_obj:  # 如果没有找到满足条件的 in_obj
                    all_out_obj_satisfied = False
                    break  # 跳出中间层循环
            printlist(successful_obj)
            if all_out_obj_satisfied:  # 如果所有 out_obj 都满足
                successful_params.append(successful_obj)  # 累计成功的参数组合

        # 检查是否至少有一个成功
        if not successful_params:  # 如果没有找到任何成功的参数组合
            return False  # 直接返回 False，表示失败
    printlist(successful_params)
    printlist(successful_obj_pairs)
    return True  # 所有 pair 都成功

def lessforprintobj(obj):
    return (obj.pair_id,obj.in_or_out,obj.objparam,obj.obj_ID,obj.bounding_box)

# printlist = lambda x: print("\n".join(map(str, x)))
def printlist(x):
    # for l in list:
    #     print(l)
    lambda x: print("\n".join(map(str, x)))

param_combinations: List[Tuple[bool, bool, bool]] = [
    (False, False, False),
    (False, False, True),
    (False, True, False),
    (False, True, True),
    (True, False, False),
    (True, False, True),
    (True, True, False),
    (True, True, True)
]

# objects_with_params 函数
# def objects_with_params(the_pair_id: int, in_or_out: str, grid: Grid, bools: Tuple[bool, bool, bool]) -> Objects:
    # b1, b2, b3 = bools  # 解包布尔值
    # return objects( grid, b1, b2, b3)  #the_pair_id, in_or_out,

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
            color_ranking=False    ,   # 默认空 tuple
            extend=extend_obj(obj000)
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
            objparam="inallparam",  # 使用传入的布尔值
            obj=obj,         # 原始对象
            obj_00=obj00,
            obj_000=obj000,
            obj_ID=managerid.get_id("OBJshape", obj000),
            grid_H_W=hw,            # 默认值，根据需要调整
            bounding_box=(uppermost(obj), leftmost(obj), lowermost(obj), rightmost(obj)),    # 默认值，根据需要调整
            color_ranking=False    ,     # 默认空 tuple
            extend=extend_obj(obj000)
        )
        result.append(new_obj)
    return result


def objop(obj,op):
    return grid_to_object(op(object_to_grid(obj)))

def s_filtered(s) :
    return frozenset(e for e in s if e[0] is not None)

def extend_obj(obj):
    # return (objop(obj,vmirror),objop(obj,cmirror),objop(obj,hmirror),objop(obj,dmirror),objop(obj,rot90),objop(obj,rot180),objop(obj,rot270))
    return (vmirror(obj),cmirror(obj),hmirror(obj),dmirror(obj),s_filtered(objop(obj,rot90)),s_filtered(objop(obj,rot180)),s_filtered(objop(obj,rot270) ) )

def all_objects_00_c0_from_objs(the_pair_id: int, in_or_out: str, all_objs):
    return {shift_obj_to_0_0_0(the_pair_id,in_or_out,obj) for obj in all_objs}


def shift_obj_to_0_0_0(id: int, in_or_out: str, objbig):
    """
    将 objbig 平移到 (0,0) ，如果是 ObjInf 类型则同时生成 obj00 和 obj000;
    如果是纯 set，则直接调用 shift_pure_obj_to_0_0_0.
    """
    # 假设 ObjInf 类型已定义
    if isinstance(objbig, ObjInf):
        orig_obj = objbig.obj  # 从 ObjInf 中提取原始对象集合
        obj000 = shift_pure_obj_to_0_0_0(orig_obj)
        obj00  = shift_pure_obj_to_00(orig_obj)
        return makeshift_ObjInf(objbig, obj00, obj000)
    elif True :
        orig_obj = objbig
        obj000 = shift_pure_obj_to_0_0_0(orig_obj)
        obj00  = shift_pure_obj_to_00(orig_obj)
        return makeshift_ObjInf(id,in_or_out, objbig, obj00, obj000)

        # return shift_pure_obj_to_0_0_0(objbig)
    else:
        raise ValueError(f"shift_obj_to_0_0_0: unsupported argument type {objbig}")


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



# 创建全局的 managerid 实例，方便其他地方直接使用


def makeshift_ObjInf(the_pair_id: int, in_or_out: str, objinf, obj00, obj000):
    """
    如果 objinf 已经是 ObjInf 实例，则更新其 obj00、obj000 和 obj_ID；
    否则，根据传入的参数创建一个新的 ObjInf 对象。
    """
    if isinstance(objinf, ObjInf):
        objinf.obj00 = obj00
        objinf.obj000 = obj000
        objinf.obj_ID = managerid.get_id("OBJshape", obj000)
        return objinf
    else:
        # objinf 是 frozenset 或其他类型，这里使用默认值填充未提供的字段，
        # 请根据实际情况调整默认值。
        return ObjInf(
            pair_id=the_pair_id,
            in_or_out=in_or_out,
            objparam=(False, False, False),    # 默认值，根据需要调整
            obj=objinf,                        # 原始对象集合（frozenset）
            obj_00=obj00,
            obj_ID=managerid.get_id("OBJshape", obj000),
            obj_000=obj000,
            grid_H_W=(0, 0),                   # 默认值
            bounding_box=(0, 0, 0, 0),           # 默认值
            color_ranking=tuple()              # 默认空tuple
        )

