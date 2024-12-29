import dsl
import constants
import arc_types
import os
import json
import inspect
import tqdm
import sys
import logging
import traceback
import heapq
from collections import deque
from searchStrategy import *  # 从 searchARC-search.py 中导入所有内容
import re
import random
from collections import defaultdict
import solvers_is_judge as solvers

class TypeExtractor:
    def __init__(self, file_path):
        """
        初始化时解析文件内容，构建类型定义映射，并生成包含关系图。
        :param file_path: 类型定义文件路径
        """
        self.type_definitions = self._load_types(file_path)
        self.reverse_dependencies = self._build_reverse_dependencies()

    def _load_types(self, file_path):
        """
        从文件中加载所有的类型定义。
        :param file_path: 类型定义的字典 {类型名称: 类型定义字符串}
        """
        type_definitions = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                stripped_line = line.strip()
                # 匹配类型定义：等号前是类型名称，等号后是定义
                match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$', stripped_line)
                if match:
                    type_name, type_definition = match.groups()
                    type_definitions[type_name] = type_definition
        return type_definitions

    def _extract_union_members(self, definition):
        """
        提取 Union 中的成员类型。
        :param definition: 类型定义字符串
        :return: Union 中的成员类型列表
        """
        union_match = re.match(r'^Union\[(.+)\]$', definition)
        if union_match:
            members = union_match.group(1)
            return [member.strip() for member in members.split(',')]
        return []

    def _build_reverse_dependencies(self):
        """
        构建反向依赖关系图，记录每个类型被哪些类型引用。
        :return: 一个字典，键为类型名称，值为引用它的类型列表。
        """
        reverse_dependencies = defaultdict(list)
        for type_name, definition in self.type_definitions.items():
            for member in self._extract_union_members(definition):
                reverse_dependencies[member].append(type_name)
        return reverse_dependencies

    def find_recursive_types(self, type_name):
        """
        递归查找指定类型包含的所有类型，最终展开为所有直接和间接成员类型。
        :param type_name: 要查找的类型名称
        :return: 包含的所有类型名称列表（已递归展开，去重）。
        """
        if (type_name not in self.type_definitions):
            return []  # 如果类型未定义，返回空列表

        # 获取该类型的定义
        definition = self.type_definitions[type_name]
        included_types = self._extract_union_members(definition)

        # 递归解析子类型中的 Union
        expanded_types = []
        for member in included_types:
            expanded_types.extend(self.find_recursive_types(member))
            expanded_types.append(member)

        # 去重并保持顺序
        return list(dict.fromkeys(expanded_types))

    def query_type(self, type_name):
        """
        查询一个类型，返回包含的所有类型（递归展开）以及引用该类型的所有类型，直接返回处理后的字符串列表。
        :param type_name: 要查询的类型名称
        :return: 包含的类型和引用它的类型组成的去重列表，例如：['grid', 'element', 'piece']
        """
        type_name_lower = type_name.lower()  # 转为小写进行匹配
        matched_types = [
            name for name, definition in self.type_definitions.items()
            if type_name_lower in name.lower() or type_name_lower in definition.lower()
        ]

        if not matched_types:
            raise ValueError(f"Type '{type_name}' not found in definitions.")

        # 包含的类型（递归展开）
        includes = self.find_recursive_types(matched_types[0])

        # 引用该类型的类型
        included_by = self.reverse_dependencies.get(matched_types[0], [])

        # 合并自身、包含的类型和引用它的类型，去重并保持顺序
        result = [matched_types[0]] + includes + included_by
        return [item.lower() for item in dict.fromkeys(result)]



    extract_types = query_type  # 为了兼容之前的调用方式


# 示例使用
arc_types_path = 'arc_types.py'  # 替换为实际路径

# 初始化时一次性加载文件内容
type_extractor = TypeExtractor(arc_types_path)



class Operator:
    def __init__(self, name, function_name, inverse_function_name=None, applicable_types=None, dsl_registry=None):
        self.name = name
        self.function_name = function_name
        self.inverse_function_name = inverse_function_name
        self.applicable_types = applicable_types if applicable_types else []
        self.dsl_registry = dsl_registry

    def apply(self, state):
        input_types = state.get_type()  # 现在是类型列表
        applicable_types = set(input_types) & set(self.applicable_types)
        if not applicable_types:
            return []
        new_states = []
        for input_type in applicable_types:
            results = self.dsl_registry.call_function([input_type], state.data)  # 修改：接收所有结果
            for new_data, output_type, func_name in results:
                new_state = State(new_data, output_type, parent=state, action=func_name)  # 使用 func_name 记录动作
                new_states.append(new_state)
        return new_states

    def invert(self, state):
        if self.inverse_function_name:
            for input_type in state.get_type():
                output_type = self.get_output_type(input_type)
                if output_type:
                    func_name = self.dsl_registry.get_functions([input_type], output_type)[0]
                    new_data = self.dsl_registry.call_function(func_name, state.data)
                    if new_data:
                        return State(new_data, output_type, parent=state, action=self.inverse_function_name)
        raise NotImplementedError(f"Operator {self.name} does not have an inverse function")


class OperatorLayer:
    def __init__(self, operators):
        self.operators = operators  # List of Operator objects

    @classmethod
    def from_config(cls, config_functions):
        operators = []
        for func in config_functions:
            op = Operator(func['name'], func['function'], func.get('inverse_function'), func.get('applicable_types'))
            operators.append(op)
        return cls(operators)

    def get_applicable_operators(self, state):
        applicable_ops = []
        for op in self.operators:
            if state.get_type() in op.applicable_types:
                applicable_ops.append(op)
        return applicable_ops

class SearchAlgorithm:
    def __init__(self, operator_layer):
        self.operator_layer = operator_layer

    def search(self, task, direction='forward'):
        raise NotImplementedError


def get_is_functions(code_file):
    """
    从指定的代码文件中获取所有以 'is_' 开头的函数和以 'solve_' 开头的函数的名称列表。
    """
    with open(code_file, 'r', encoding='utf-8') as file:
        code = file.read()
    # 提取所有以 'is_' 开头的函数名称
    is_functions = re.findall(r'def (is_\w+)\s*\(', code)
    # 提取所有以 'solve_' 开头的函数名称
    # solve_functions = re.findall(r'def (solve_\w+)\s*\(', code)
    return is_functions ##+ solve_functions

def validate_args(args_list):
            """验证参数列表中的所有参数是否相同"""
            if not args_list:
                return True
            first_args = args_list[0]
            return all(args == first_args for args in args_list)

def is_checking(task):
    """
    遍历所有的 'is_' 函数，验证任务的训练数据。
    对于验证成功的函数，提取其子函数。
    """
    ## classified_functions_file = '/home/zdx/github/VSAHDC/arc-dsl/forprolog/classDSLresult2.json'
    code_file = '/Users/zhangdexiang/github/VSAHDC/arc-dsl/solvers_is_judge.py'  # 指定包含 is 函数的文件
    is_functions = get_is_functions(code_file)
    valid_functions = []
    functionargs = []

    ### todo todo now just one correct is_function_name
    for is_function_name in is_functions:
        # 从模块中获取函数对象
        is_function = getattr(solvers, is_function_name)
        if not is_function:
            continue  # 如果函数不存在，跳过

        success = True
        for pair in task['train']:
            input_grid = pair['input']
            output_grid = pair['output']
            # 调用 is 函数进行验证
            try:
                result = is_function(input_grid, output_grid)

                if result == False:  # 直接返回 False 的情况
                    success = False
                    break
                elif result == True:  # 直接返回 True 的情况
                    pass
                elif isinstance(result, tuple) and result[0] == True:  # 返回元组的情况
                    functionargs.append(result[1:])
                else:
                    raise ValueError("Unexpected return value from is_function")


                # if result ==  False:
                #     success = False
                #     break
                # else:
                #     functionargs.append(result[1:])
                #     # args = result[2:]
            except Exception as e:
                # logging.error("捕获到异常：%s", e)
                # logging.error("详细错误信息：\n%s", traceback.format_exc())
                success = False
                break


        if success:
            if functionargs:
                if validate_args(functionargs):
                    return (1,functionargs)
                else:
                    print("Warning: Function arguments are not consistent across results")
                    return None
            else:
                # 如果在所有训练数据对上验证成功，记录函数名称
                # valid_functions.append(is_function_name)
                # 提取该函数的子函数
                subclasses = get_function_subclass(is_function_name, code_file)
                print(f"Function '{is_function_name}' passed validation. Subclasses: {subclasses}")
                return (2,subclasses)

def get_function_subclass(function_name, code_file):
    """
    获取指定函数中调用的子函数名称列表,包括列表定义中的函数。
    """
    with open(code_file, 'r', encoding='utf-8') as file:
        code = file.read()

    # 提取指定函数的完整定义
    # pattern = rf'def {'is_upscale'}\s*\(.*?\):([\s\S]*?)(?=\n\ndef |\n#|\Z)'
    # pattern = rf"def {function_name}\(.*?\)\s*(->\s*\w+)?\s*:(.*?)(?=\n\s*def\s|\Z)"
    # pattern = rf"def {function_name}\(I\):.*?(?=\n\s*def |$)"
    pattern = rf'''
        def\s+{function_name}    # 函数名
        \s*\(                    # 左括号
        [^)]*                    # 参数部分（非贪婪）
        \)\s*                    # 右括号
        (?:->[ \t]*\w+)?        # 可选的返回类型
        \s*:\s*                 # 冒号
        (?:(?!def\s).)*         # 函数体（非贪婪）
    '''.strip()

    match = re.search(pattern, code, re.VERBOSE | re.DOTALL)

    # match = re.search(pattern, code)
    if not match:
        print(f"Function {function_name} not found.")
        return []

    function_body = match.group(0)

    # 收集所有找到的函数名
    functions = set()

    # 提取函数体中直接调用的函数
    direct_calls = re.findall(r'\b(\w+)\s*\(', function_body)
    functions.update(direct_calls)

    # 提取列表定义中的函数名
    # 匹配形如 [hmirror, vmirror] 或 [hconcat, vconcat] 的列表定义
    list_definitions = re.findall(r'\[([\w,\s]+)\]', function_body)
    for definition in list_definitions:
        # 分割并清理每个函数名
        list_functions = [f.strip() for f in definition.split(',') if f.strip()]
        functions.update(list_functions)

    # 去除自身和内置函数
    functions.discard(function_name)

    return list(functions)

def compute_difference(data1, data2):
    """
    计算两个状态的数据差异。
    """
    ## is check, object 对象，color相关 函数，shape,positon, same shape point, same color point,
    ## 两个grid的差异，两个object的差异  旋转对象是一类，镜像对象是一类，填充对象是一类，颜色变换是一类，形状变换是一类，位置变换是一类，大小变换是一类，合并对象是一类，分割对象是一类，删除对象是一类，增加对象是一类，交换对象是一类，重复对象是一类，

    if isinstance(data1, list) and isinstance(data2, list):
        # 假设是grid，计算不同元素的数量
        diff = sum(1 for row1, row2 in zip(data1, data2) for v1, v2 in zip(row1, row2) if v1 != v2)
        return diff
    elif isinstance(data1, set) and isinstance(data2, set):
        # 假设是object，计算对称差集的大小
        diff = len(data1.symmetric_difference(data2))
        return diff
    else:
        return float('inf')

# class DifferenceAnalyzer:
#     def analyze_difference(self, task):
#         difference = compute_difference(start_state.data, goal_state.data)
#         # 根据差异大小或类型选择搜索方向
#         if difference == 0:
#             return 'none'
#         elif difference < 10:
#             return 'forward'
#         else:
#             return 'bidirectional'


import ast

class DSLFunctionRegistry:
    def __init__(self, classified_functions_file):
        self.classified_functions = self.load_classified_functions(classified_functions_file)
        # 动态加载 DSL 模块中的所有函数
        import dsl
        self.dsl_functions = {func: getattr(dsl, func) for func in dir(dsl) if callable(getattr(dsl, func))}

    def load_classified_functions(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 将键从字符串转换为元组
        classified_functions = {}
        for key_str, functions in data.items():
            key_tuple = ast.literal_eval(key_str)
            classified_functions[key_tuple] = functions
        from collections import OrderedDict

        # 将普通字典转换为有序字典并排序
        classified_functions = OrderedDict(sorted(classified_functions.items(), key=lambda x: x[0]))
        return classified_functions

    def call_function(self, input_types, state_data):
        """
        根据输入类型，在 classified_functions 中查找对应的函数，
        然后从 DSL 中加载实际的函数并调用，返回所有成功的结果，包括函数名称。
        """
        matching_functions = self.get_functions(input_types)
        results = []  # 修改：收集所有成功结果
        for func_name in matching_functions:
            if (func_name in self.dsl_functions):
                func = self.dsl_functions[func_name]
                try:
                    # 调用函数并获取返回值 ！！！！！！！！！！！print

                    # print(func_name)
                    new_data = func(state_data)
                    # 获取函数的输出类型
                    output_type = self.get_output_type(func_name)
                    if new_data is not None and output_type is not None:
                        results.append((new_data, output_type, func_name))  # 收集结果并包含函数名称
                except Exception as e:
                    logging.error("捕获到异常：%s", e)
                    logging.error("详细错误信息：\n%s", traceback.format_exc())
                    # pass  # 记录日志后继续
        return results  # 返回所有成功的结果，包括函数名称

    def get_functions(self, input_types):
        matching_functions = []
        for key, functions in self.classified_functions.items():
            key_input_types, _ = key
            if tuple(input_types) == key_input_types:
                matching_functions.extend(functions)
        return matching_functions

    def get_output_type(self, function_name):
        for key, functions in self.classified_functions.items():
            if function_name in functions:
                _, output_type = key
                return output_type
        return None

    def create_bidirectional_search(self):
        """创建双向搜索实例"""
        from searchStra2 import BidirectionalSearch
        return BidirectionalSearch(self)

def run_search(task, classified_functions_file):
    """主搜索入口函数"""
    # 初始化DSL注册器
    dsl_registry = DSLFunctionRegistry(classified_functions_file)

    # 检查是否有可用的特化函数
    whitelist = is_checking(task)
    if whitelist:
        # 使用特化函数进行搜索
        search_strategy = SearchStrategy(dsl_registry, enable_whitelist=False, whitelist=whitelist)
        result = search_strategy.search(task)
        if result:
            return result

    # 如果特化搜索失败,尝试双向树搜索
    print("Trying bidirectional tree search...")
    for pair in task['train']:
        # 创建初始状态和目标状态
        input_state = State(pair['input'], 'grid')
        output_state = State(pair['output'], 'grid')

        # 创建双向搜索实例
        search = dsl_registry.create_bidirectional_search()

        # 执行搜索
        solution = search.find_path(input_state, output_state)
        if solution:
            # 验证解决方案
            if search.validate_solution(task, solution):
                print("Found valid solution via tree search!")
                return solution

    return None

class Controller:
    def __init__(self, operator_layer, search_algorithm, difference_analyzer):
        self.operator_layer = operator_layer
        self.search_algorithm = search_algorithm
        self.difference_analyzer = difference_analyzer

    def run(self, task):

        difference = self.difference_analyzer.analyze(task)
        # Decide search direction based on difference
        path = self.search_algorithm.search(task, direction='forward')
        if path:
            print("Found path:", path)
        else:
            print("No solution found")

# Placeholder for configuration management
class ConfigManager:
    def __init__(self, config_file):
        self.config = self._load_config(config_file)

    def _load_config(self, config_file):
        # Implement configuration loading
        pass

    def get_proper_functions(self):
        return self.config.get('proper_functions', [])

    def get_search_algorithm(self):
        return self.config.get('search_algorithm', 'BFS')

import solvers_is_judge as solvers

def get_data(train=True):
    # /home/zdx/github/VSAHDC/arc-agi/data
    path = f'/Users/zhangdexiang/github/arc-agi/data/{
        "training" if train else "evaluation"}'
    data = {}
    for fn in os.listdir(path):
        if not fn.endswith('.json'):
            continue  # 只处理 JSON 文件
        with open(f'{path}/{fn}') as f:
            data[fn.rstrip('.json')] = json.load(f)

    def ast(g): return tuple(tuple(r) for r in g)
    return {
        'train': {k: [{
            'input': ast(e['input']),
            'output': ast(e['output']),
        } for e in v['train']] for k, v in data.items()},
        'test': {k: [{
            'input': ast(e['input']),
            'output': ast(e['output']),
        } for e in v['test']] for k, v in data.items()}
    }

def get_solver_functions():
    with open('solvers.py', 'r', encoding='utf-8') as file:
        code = file.read()
    pattern = r"def solve_([a-fA-F0-9]+)\(I\):"
    import re
    # 获取所有匹配的函数名
    solver_functions_name = re.findall(pattern, code)  # 修改变量名为 solver_functions
    return solver_functions_name

if __name__ == '__main__':
    data = get_data(train=True)

    # 获取所有需要处理的任务ID
    solver_functions_name = get_solver_functions()

    for i, key in enumerate(solver_functions_name, start=1):
        key = '97a05b5b'
        if i != 1:
            break
        print(f"\nProcessing task {i}: {key}")

        # 构建任务
        task = {
            'train': data['train'][key],
            'test': data['test'][key]
        }

        # 运行搜索
        classified_functions_file = '/Users/zhangdexiang/github/VSAHDC/arc-dsl/forprolog/classDSLresult2.json'
        solution = run_search(task, classified_functions_file)

        if solution:
            print(f"Success! Found solution for task {key}")
            print("Solution:", solution)
        else:
            print(f"Failed to find solution for task {key}")