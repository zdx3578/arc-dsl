
from dsl import *
from constants import *
import dsl2
import inspect
from arc_types import *
import logging



def is_arg1_is_arg2_subgrid(grid1: Grid, grid2: Grid) -> Union[Tuple[bool, str, Tuple[int, int], Tuple[int, int]], bool]:
    """
    检查 grid1 是否是 grid2 的子网格。

    参数:
    - grid1: Grid - 第一个矩形网格。
    - grid2: Grid - 第二个矩形网格。

    返回:
    - bool: 如果 grid1 是 grid2 的子网格，返回 True；否则返回 False。
    """
    h1, w1 = len(grid1), len(grid1[0])
    h2, w2 = len(grid2), len(grid2[0])

    # 检查 grid1 的尺寸是否小于或等于 grid2
    if h1 > h2 or w1 > w2:
        return False

    # 遍历 grid2，检查是否存在与 grid1 匹配的子网格
    for i in range(h2 - h1 + 1):
        for j in range(w2 - w1 + 1):
            match = True
            for x in range(h1):
                for y in range(w1):
                    if grid1[x][y] != grid2[i + x][j + y]:
                        match = False
                        break
                if not match:
                    break
            if match:
                # return (True, 'crop', (i, j),(h1, w1))
                return True

    return False



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
columns_outout = [
    "pair_id", "out", "outparam", "output_id",
    # "input_id",
    "outbounding_box", "outcolor",
]
def parsed_pd_data(raw_data):
    data = {
        "pair_id": raw_data[0][0],
        "out": raw_data[0][1],
        "outparam": raw_data[0][2],
        "output_id": (raw_data[0][3].split()[-1]),
        "outbounding_box": raw_data[0][4],
        "outcolor": next(iter(raw_data[0][5])),
        # "pair_id_2": raw_data[1][0],
        # "pair_id_2": raw_data[0][0],
        "in": raw_data[1][1],
        "inparam": str(raw_data[1][2]),
        "input_id": (raw_data[1][3].split()[-1]),
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
        # "output_id": int(raw_data[0][3].split()[-1]),
        "output_id": (raw_data[0][3].split()[-1]),
        "outbounding_box": raw_data[0][4],
        "outcolor": next(iter(raw_data[0][5])),
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

# def count_obj_ids(obj_set,colum):
#     return set(obj.colum for obj in obj_set)
def count_obj_ids(obj_set, attr_name):
    return {getattr(obj, attr_name) for obj in obj_set}

def show_count_col(df,col_id):
            value_counts = df[col_id].value_counts()
            sorted_value_counts = value_counts.sort_values(ascending=True)
            print("\n","count_number  排序:  ",col_id)
            print(sorted_value_counts)
            print("\n")




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




def pretty_print(dataall, indent=1):
    """
    格式化打印嵌套数据结构，每个字段一行，嵌套列表的每个子列表也分行打印。
    :param dataall: 要打印的数据（可以是元组、列表、集合、字典等）
    :param indent: 当前缩进级别（用于递归调用）
    """
    # 定义缩进字符串
    indent_str = " " * (indent * 3)
    print("\n\npretty_print function")

    if isinstance(dataall, (tuple, list)):
        # 如果是元组或列表
        print(indent_str + "[")
        for data in dataall:
            if isinstance(data, (tuple, list)) and len(data) == 4:
                # 如果是包含四个字段的元组 (paramid, out_param, len_out_obj_set, successful_obj)
                paramid, out_param, pair_id, successful_obj = data
                print(indent_str + " " * 4 + f"paramid: {paramid}")
                print(indent_str + " " * 4 + f"out_param: {out_param}")
                print(indent_str + " " * 4 + f"pair_id:: {pair_id}")
                print(indent_str + " " * 4 + "successful_obj:")
                # 打印 successful_obj 中的每个元素
                if isinstance(successful_obj, (tuple, list)):
                    print(indent_str + " " * 8 + "[")
                    for item in successful_obj:
                        # print(indent_str + str(item))
                        print(indent_str + " " * 8 + str(item))
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



def foranalysisshow(df):
    value_counts = df['outparam'].value_counts()

    sorted_value_counts = value_counts.sort_values(ascending=True)
    print("\n按 outparam count  排序:")
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
            # print("\n".join(output_lines))
        # 添加当前行到输出

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
    (True, False, True)  ]

param_combinations2: List[Tuple[bool, bool, bool]] = [
    (True, True, False),
    (True, True, True),
    (True, False, False),
    (True, False, True),
    (False, False, False),
    (False, False, True),
    (False, True, False),
    (False, True, True) ]


# all_objects_from_grid 函数
def all_objects_from_grid(the_pair_id: int, in_or_out: str, grid: Grid, hw:list) -> FrozenSet[Object]:
    acc: FrozenSet[Object] = frozenset()  # 初始化空集合
    for params in param_combinations:
        acc = acc.union(objects_fromone_params(the_pair_id, in_or_out, grid, params,hw))
        # print()
    result = []
    bg = mostcolor(grid)
    for obj in acc:
        # 对每个 obj，计算对应平移后的版本
        # 假设 obj 本身是一个表示对象的集合；如果不是，则请调整调用方式
        obj00 = shift_pure_obj_to_00(obj)
        obj000 = shift_pure_obj_to_0_0_0(obj)
        new_obj = ObjInf(
            pair_id='pair_id: '+str(the_pair_id),
            in_or_out=in_or_out,
            objparam="all",  # 使用传入的布尔值
            obj=obj,         # 原始对象
            obj_00=obj00,
            obj_000=obj000,
            # obj_ID=managerid.get_id("OBJshape", obj000),
            obj_ID="obj-ID:"+str(managerid.get_id("OBJshape", obj000)),
            grid_H_W=hw,            # 默认值，根据需要调整
            bounding_box=(uppermost(obj), leftmost(obj), lowermost(obj), rightmost(obj)),    # 默认值，根据需要调整
            color_ranking=palette(obj)    ,     # 默认空 tuple
            background = bg,
            obj000_ops=extend_obj(obj000),
            obj_ops = extend_obj(obj)
        )
        result.append(new_obj)
    return result

def paint_objects(obj_set, background,hw):
    if background is None:
        background = -1
    grid = canvas(background, hw)
    grid = [list(row) for row in grid]

    for obj in obj_set:
        for node in obj:
            color, (r, c) = node
            grid[r][c] = color
    tpl = tuple(tuple(inner) for inner in grid)
    return tpl


def move_in_obj_based_on_out(successful_obj,new_in_obj_list):
    pairs = successful_obj[0][3]
    moved_objs = []
    for pair in pairs:
        out_obj_info, in_obj_info, op_type, op = pair
        # if in_obj_info[3] == out_obj_info[3]:
        bbox = out_obj_info[4]  # bounding_box
        # print(f"匹配到 obj_ID {in_obj_info[3]}，使用 bounding_box {bbox} 来移动 in_obj.obj")

        in_obj = get_in_obj_by_id(in_obj_info[3],new_in_obj_list)
        if in_obj is not None:
            in_obj.obj = shift_object_by_bbox(in_obj.obj,in_obj.bounding_box, bbox)
            # print(f"in_obj.obj 已经移动到新位置: {in_obj.obj}")
            moved_objs.append(in_obj.obj)
    # pretty_print(moved_objs)
    return moved_objs

def shift_object_by_bbox(obj_set,bbox0, bbox):
    dr0, dc0 = bbox0[0], bbox0[1]
    dr, dc = bbox[0], bbox[1]
    new_obj_set = set()
    for color, (r, c) in obj_set:
        new_obj_set.add((color, (r -dr0 + dr, c -dc0 + dc)))
    return new_obj_set


def get_in_obj_by_id(obj_id, global_in_obj_list):
    for obj in global_in_obj_list:
        if obj.obj_ID == obj_id:
            return obj
    return None


