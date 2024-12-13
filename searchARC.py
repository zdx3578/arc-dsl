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
from searchARC_search import *  # 从 searchARC-search.py 中导入所有内容


class State:
    def __init__(self, data, type, parent=None, action=None):
        self.data = data
        self.type = type
        self.parent = parent      # 新增：记录父状态
        self.action = action      # 新增：记录产生该状态的操作符
        self.hash = self.compute_hash()

    def compute_hash(self):
        """
        计算状态的哈希值，用于重复检测。
        """
        return hash((self.type, self._data_hash()))

    def _data_hash(self):
        if isinstance(self.data, list):
            return tuple(map(tuple, self.data))
        elif isinstance(self.data, set) or isinstance(self.data, frozenset):
            return frozenset(self.data)
        else:
            return hash(self.data)

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return self.type == other.type and self.data == other.data

    def get_type(self):
        return self.type


class GridState(State):
# class State:
    def __init__(self, data, state_type='grid'):
        self.data = data
        self.type = state_type  # 状态类型：'grid'、'object' 等
        self.hash = self.compute_hash()

    def compute_hash(self):
        """
        计算状态的哈希值，用于重复检测。
        """
        if self.type == 'grid':
            return hash(tuple(map(tuple, self.data)))
        elif self.type == 'object':
            return hash(frozenset(self.data))
        else:
            return hash(self.data)

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return self.data == other.data and self.type == other.type

    def get_size(self):
        return (len(self.grid), len(self.grid[0]))

class Operator:
    def __init__(self, name, function_name, inverse_function_name=None, applicable_types=None, dsl_registry=None):
        self.name = name
        self.function_name = function_name
        self.inverse_function_name = inverse_function_name
        self.applicable_types = applicable_types if applicable_types else []
        self.dsl_registry = dsl_registry

    def apply(self, state):
        input_type = state.get_type()
        input_types = [input_type]
        # 调用 dsl_registry 的 call_function 方法
        new_data, output_type = self.dsl_registry.call_function(input_types, state.data)
        if new_data is not None and output_type is not None:
            new_state = State(new_data, output_type)
            return [new_state]
        return []


    def invert(self, state):
        if self.inverse_function_name:
            input_type = state.get_type()
            output_type = self.get_output_type(input_type)
            functions = self.dsl_registry.get_functions([input_type], output_type)
            if not functions:
                raise TypeError(f"No applicable function found for input type {input_type} and output type {output_type}")
            func_name = functions[0]
            new_data = self.dsl_registry.call_function(func_name, state.data)
            return GridState(new_data, output_type)
        else:
            raise NotImplementedError(f"Operator {self.name} does not have an inverse function")

    # def get_output_type(self, input_type):
    #     # 根据输入类型确定输出类型，这里可以根据实际需求进行修改
    #     if input_type == 'grid':
    #         return 'indices'
    #     elif input_type == 'integer':
    #         return 'object'
    #     else:
    #         return 'any' ????????????????????????


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




def compute_difference(data1, data2):
    """
    计算两个状态的数据差异。
    """
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

class DifferenceAnalyzer:
    def analyze_difference(self, task):
        difference = compute_difference(start_state.data, goal_state.data)
        # 根据差异大小或类型选择搜索方向
        if difference == 0:
            return 'none'
        elif difference < 10:
            return 'forward'
        else:
            return 'bidirectional'


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
        return classified_functions

    def call_function(self, input_types, state_data):
        """
        根据输入类型，在 classified_functions 中查找对应的函数，
        然后从 DSL 中加载实际的函数并调用。
        """
        matching_functions = self.get_functions(input_types)
        for func_name in matching_functions:
            if func_name in self.dsl_functions:
                func = self.dsl_functions[func_name]
                try:
                    # 调用函数并获取返回值
                    new_data = func(state_data)
                    # 获取函数的输出类型
                    output_type = self.get_output_type(func_name)
                    return new_data, output_type
                except Exception as e:
                    pass  # 可以记录日志或忽略异常，尝试下一个函数
        return None, None

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

if __name__ == '__main__':
    data = get_data(train=True)

    with open('solvers.py', 'r', encoding='utf-8') as file:
        code = file.read()
    pattern = r"def solve_([a-fA-F0-9]+)\(I\):"
    import re
    # 获取所有匹配的函数名
    solvers = re.findall(pattern, code)

    for i, key in enumerate(solvers, start=1):

        # key = 'c3f564a4'

        print(i, key)
        task = {}
        task['train'] = data['train'][key]
        task['test'] = data['test'][key]



        # Configuration management
        # config_manager = ConfigManager('config.py')
        # proper_functions = config_manager.get_proper_functions()
        # operator_layer = OperatorLayer.from_config(proper_functions)

        # Search algorithm selection
        # search_algorithm_name = config_manager.get_search_algorithm()
        # if search_algorithm_name == 'BFS':
        #     search_algorithm = BFS(operator_layer)


        # Difference analyzer
        difference_analyzer = DifferenceAnalyzer()

        classified_functions_file = '/Users/zhangdexiang/github/VSAHDC/arc-dsl/forprolog/classDSLresult2.json'
        dsl_registry = DSLFunctionRegistry(classified_functions_file)

        search_algorithm = SearchStrategy(dsl_registry)
        search_algorithm.search(task)


        # Controller
        # controller = Controller(operator_layer, search_algorithm, difference_analyzer, dsl_registry)

        # # Run the search
        # success = controller.run(task)
        # if success:
        #     print("All training pairs successfully transformed.")
        # else:
        #     print("Failed to transform some training pairs.")


        # assert solution == task['test'][0]['output']