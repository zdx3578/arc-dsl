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
# from searchStrategy import *  # 从 searchARC-search.py 中导入所有内容
import re

import re
from collections import defaultdict

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
        if type_name not in self.type_definitions:
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


class TypeExtractor0:
    def __init__(self, file_path):
        """
        初始化时一次性读取文件内容，并解析出所有的类型定义。
        :param file_path: 类型定义文件路径
        """
        self.type_definitions = self._load_types(file_path)

    def _load_types(self, file_path):
        """
        从文件中加载所有的类型定义。
        :param file_path: 类型定义文件路径
        :return: 类型定义的字典 {类型名称: 类型定义字符串}
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

    def extract_types(self, keyword):
        """
        根据关键字查找包含该关键字的类型。
        :param keyword: 查找的关键字
        :return: 包含关键字的类型名称列表
        """
        keyword_lower = keyword.lower()
        matched_types = [
            name.lower() for name, definition in self.type_definitions.items()  # 转换为小写
            if keyword_lower in name.lower() or keyword_lower in definition.lower()
        ]
        return matched_types


# 示例使用
arc_types_path = 'arc_types.py'  # 替换为实际路径

# 初始化时一次性加载文件内容
type_extractor = TypeExtractor(arc_types_path)

# 查找包含 'grid' 的类型
# types = type_extractor.extract_types(type)

class State:
    def __init__(self, data, type, parent=None, action=None):
        self.data = data
        self.types  = type_extractor.extract_types(type)  # 修改：支持多个类型
        self.parent = parent      # 新增：记录父状态
        self.action = action      # 新增：记录产生该状态的操作符
        self.hash = self.compute_hash()

    def compute_hash(self):
        """
        计算状态的哈希值，用于重复检测。
        """
        return hash((tuple(sorted(self.types)), self._data_hash()))  # 修改：使用多个类型

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
        return set(self.types) == set(other.types) and self.data == other.data  # 修改：比较类型集
    def __lt__(self, other):
        return (self.types, self.data) < (other.types, other.data)


    def get_type(self):
        return self.types  # 修改：返回类型列表


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

class SearchStrategy:
    def __init__(self, dsl_registry):
        self.dsl_registry = dsl_registry
    #     self.operators = self.load_operators()

    # def load_operators(self):
    #     # 从 DSL 注册表中加载操作符
    #     operators = []
    #     for key, functions in self.dsl_registry.classified_functions.items():
    #         # key_str = str(key)  # 确保 key 是字符串类型
    #         input_types, output_type = key
    #         for func_name in functions:
    #             op = Operator(func_name, func_name, applicable_types=input_types, dsl_registry=self.dsl_registry)
    #             operators.append(op)
    #     return operators

    def search(self, task, strategy='a_star', direction='bidirectional'):
        if strategy == 'a_star':
            if direction == 'forward':
                solution = self.a_star_search(task)
            elif direction == 'backward':
                solution = self.a_star_search(task, reverse=True)
            elif direction == 'bidirectional':
                solution = self.bidirectional_a_star_search(task, self.heuristic)
            else:
                raise ValueError("未实现的搜索策略")

            # 如果找到了解决方案，打印函数序列
            if solution:
                actions = solution  # 修改：solution 现在只包含 actions
                print("成功的状态转换过程的函数序列:")
                print(actions)

                # 使用记录的函数序列对测试数据进行验证
                self.validate_test_data(task, actions)
            else:
                print("未找到解决方案")

    def bidirectional_a_star_search(self, task, heuristic):
        actions_list = []

        for pair in task['train']:
            start_state = State(pair['input'], 'grid')  # 包含类型信息
            goal_state = State(pair['output'], 'grid')  # 包含类型信息

            solution = self._search_single_pair(start_state, goal_state, heuristic)
            if solution is None:
                print("未找到训练数据对的解决方案")
                return None
            else:
                path, actions = solution
                # 过滤掉 None 值
                filtered_actions = [action for action in actions if action]
                actions_list.append(filtered_actions)

        # 检查是否存在适用于所有训练数据对的共用操作符序列
        common_actions = actions_list[0]
        for actions in actions_list[1:]:
            if actions != common_actions:
                print("无法找到适用于所有训练数据对的共用函数序列")
                return None

        # 如果找到共用的操作符序列，进行测试验证
        print("找到适用于所有训练数据对的共用函数序列:", common_actions)
        self.validate_test_data(task, common_actions)
        return common_actions  # 修改：只返回 common_actions

    def _search_single_pair(self, start_state, goal_state, heuristic):
        open_set_start = []
        open_set_goal = []
        heapq.heappush(open_set_start, (0, start_state))
        heapq.heappush(open_set_goal, (0, goal_state))

        came_from_start = {}
        came_from_goal = {}

        g_score_start = {start_state: 0}
        g_score_goal = {goal_state: 0}

        f_score_start = {start_state: heuristic(start_state, goal_state)}
        f_score_goal = {goal_state: heuristic(goal_state, start_state)}

        closed_set_start = set()
        closed_set_goal = set()

        while open_set_start and open_set_goal:
            _, current_start = heapq.heappop(open_set_start)
            closed_set_start.add(current_start)

            _, current_goal = heapq.heappop(open_set_goal)
            closed_set_goal.add(current_goal)

            # 仅在会合点类型为 'grid' 时认为路径成功
            if (current_start in closed_set_goal and 'grid' in current_start.get_type()) or \
               (current_goal in closed_set_start and 'grid' in current_goal.get_type()):
                meeting_point = current_start if current_start in closed_set_goal else current_goal
                return self.reconstruct_bidirectional_path(came_from_start, came_from_goal, meeting_point)

            neighbors_start = self.get_neighbors(current_start)
            for neighbor in neighbors_start:
                if neighbor in closed_set_start:
                    continue
                tentative_g_score = g_score_start[current_start] + 1
                if neighbor not in g_score_start or tentative_g_score < g_score_start[neighbor]:
                    came_from_start[neighbor] = current_start  # 修改：记录父状态
                    g_score_start[neighbor] = tentative_g_score
                    f_score_start[neighbor] = tentative_g_score + heuristic(neighbor, goal_state)
                    heapq.heappush(open_set_start, (f_score_start[neighbor], neighbor))

            neighbors_goal = self.get_neighbors(current_goal, reverse=False)
            for neighbor in neighbors_goal:
                if neighbor in closed_set_goal:
                    continue
                tentative_g_score = g_score_goal[current_goal] + 1
                if neighbor not in g_score_goal or tentative_g_score < g_score_goal[neighbor]:
                    came_from_goal[neighbor] = current_goal   # 修改：记录父状态
                    g_score_goal[neighbor] = tentative_g_score
                    f_score_goal[neighbor] = tentative_g_score + heuristic(neighbor, start_state)
                    heapq.heappush(open_set_goal, (f_score_goal[neighbor], neighbor))

        return None

    def reconstruct_bidirectional_path(self, came_from_start, came_from_goal, meeting_point):
        path_start = []
        actions_start = []
        state = meeting_point
        while state in came_from_start:
            path_start.append(state)
            actions_start.append(state.action)   # 新增：记录操作符
            state = state.parent
        path_start.reverse()
        actions_start.reverse()

        path_goal = []
        actions_goal = []
        state = meeting_point
        while state in came_from_goal:
            state = came_from_goal[state]
            path_goal.append(state)
            actions_goal.append(state.action)   # 新增：记录操作符

        # 合并路径和操作符
        full_path = path_start + path_goal
        full_actions = actions_start + actions_goal

        return full_path, full_actions   # 修改：返回操作符序列


    # def apply(self, state):
    #     input_types = state.get_type()  # 现在是类型列表
    #     applicable_types = set(input_types) & set(self.applicable_types)
    #     if not applicable_types:
    #         return []
    #     new_states = []
    #     for input_type in applicable_types:
    #         results = self.dsl_registry.call_function([input_type], state.data)  # 修改：接收所有结果
    #         for new_data, output_type, func_name in results:
    #             new_state = State(new_data, output_type, parent=state, action=func_name)  # 使用 func_name 记录动作
    #             new_states.append(new_state)
    #     return new_states

    # def get_neighbors(self, state, reverse=False):
    #     neighbors = []

    #     applicable_types = self.get_applicable_types(state, op.applicable_types)  # 使用辅助方法
    #     if not applicable_types:
    #         continue
    #     # if reverse and op.inverse_function_name:
    #     #     new_states = op.invert(state)
    #     else:
    #         new_states = self.apply(state)
    #     for new_state in new_states:
    #         neighbors.append(new_state)  # 修改：操作符内部已记录父状态和操作符
    #     return neighbors

    def get_neighbors(self, state, reverse=False):
        """直接通过 dsl_registry 获取所有可能的状态转换"""
        input_types = state.get_type()
        neighbors = []

        if reverse:
            # 处理反向搜索的逻辑，如果需要
            # TODO: 实现反向搜索逻辑
            pass
        else:
            for input_type in input_types:
                results = self.dsl_registry.call_function([input_type], state.data)  # 修改：接收所有结果
                for new_data, output_type, func_name in results:
                    new_state = State(new_data, output_type, parent=state, action=func_name)  # 使用 func_name 记录动作
                    neighbors.append(new_state)
        return neighbors

    def heuristic(self, state, goal_state):
        return compute_difference(state.data, goal_state.data)

    def validate_test_data(self, task, actions):
        for pair in task['test']:
            state = State(pair['input'], 'grid')
            for action in actions:
                op = self.get_operator_by_name(action)
                new_states = op.apply(state)
                if new_states:
                    # 更新状态，处理类型转换和中间结果
                    state = new_states[0]
                else:
                    print(f"函数 {action} 无法应用于当前状态")
                    break
            # 应用 'asindices' 转换
            state = self.apply_asindices_if_needed(state)
            # 比较最终输出结果
            if state.data == pair['output']:
                print("测试数据验证成功，输出与预期一致")
            else:
                print("测试数据验证失败，输出与预期不一致")

    def apply_asindices_if_needed(self, state):
        """
        如果需要，应用 'asindices' 函数将状态转换为 'grid' 类型。
            op = self.get_operator_by_name('asindices')"""

        if 'grid' not in state.get_type():
            op = self.get_operator_by_name('asindices')
            if op:
                new_states = op.apply(state)
                if new_states:
                    return new_states[0]
                else:
                    print("函数 asindices 无法应用于当前状态")
        return state

    def convert_to_grid(self, state):
        """
        将非 'grid' 类型的状态转换为 'grid' 类型。
        需要根据具体的上下文和可用的函数来实现。
        """
        # 示例：如果状态类型是 'indices'，尝试转换为 'grid'
        if 'indices' in state.get_type():
            # 假设有一个函数可以将 indices 转换为 grid，例如 indices_to_grid
            new_data = indices_to_grid(state.data)
            return State(new_data, 'grid')
        else:
            # 无法转换，返回原状态
            return state

    def get_operator_by_name(self, name):
        for op in self.operators:
            if op.name == name:
                return op
        return None

    def get_applicable_types(self, state_or_input_types, applicable_types):
        if isinstance(state_or_input_types, State):
            input_types = state_or_input_types.get_type()
        else:
            input_types = state_or_input_types
        return set(input_types) & set(applicable_types)


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
        然后从 DSL 中加载实际的函数并调用，返回所有成功的结果，包括函数名称。
        """
        matching_functions = self.get_functions(input_types)
        results = []  # 修改：收集所有成功结果
        for func_name in matching_functions:
            if func_name in self.dsl_functions:
                func = self.dsl_functions[func_name]
                try:
                    # 调用函数并获取返回值
                    new_data = func(state_data)
                    # 获取函数的输出类型
                    output_type = self.get_output_type(func_name)
                    if new_data is not None and output_type is not None:
                        results.append((new_data, output_type, func_name))  # 收集结果并包含函数名称
                except Exception as e:
                    logging.error("捕获到异常：%s", e)
                    logging.error("详细错误信息：\n%s", traceback.format_exc())
                    pass  # 记录日志后继续
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