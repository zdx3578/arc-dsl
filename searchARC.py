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


class State:
    def __init__(self):
        pass

    def __hash__(self):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def get_type(self):
        raise NotImplementedError

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
    def __init__(self, name, function, inverse_function=None, applicable_types=None):
        self.name = name
        self.function = function
        self.inverse_function = inverse_function
        self.applicable_types = applicable_types if applicable_types else []

    def apply(self, state):
        if state.get_type() in self.applicable_types:
            new_grid = self.function(state.grid)
            return GridState(new_grid)
        else:
            raise TypeError(f"Operator {self.name} not applicable to state type {state.get_type()}")

    def invert(self, state):
        if self.inverse_function:
            if state.get_type() in self.applicable_types:
                new_grid = self.inverse_function(state.grid)
                return GridState(new_grid)
            else:
                raise TypeError(f"Operator {self.name} not applicable to state type {state.get_type()}")
        else:
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

from collections import deque

class BFS:
    def __init__(self, operator_layer):
        self.operator_layer = operator_layer

    def search(self, task, direction='forward'):
        visited = set()
        queue = deque()
        queue.append((start_state, []))  # State and path
        while queue:
            current_state, path = queue.popleft()
            if current_state == goal_state:
                return path
            if current_state in visited:
                continue
            visited.add(current_state)
            applicable_ops = self.operator_layer.get_applicable_operators(current_state)
            for op in applicable_ops:
                try:
                    next_state = op.apply(current_state)
                except:
                    continue
                if next_state not in visited:
                    queue.append((next_state, path + [op.name]))
        return None  # No solution found

import heapq

class SearchStrategy:
    def __init__(self, operators):
        self.operators = operators

    def search(self, task, strategy='a_star', direction='bidirectional'):
        if strategy == 'a_star':
            if direction == 'forward':
                return self.a_star_search(task)
            elif direction == 'backward':
                return self.a_star_search(task, reverse=True)
            elif direction == 'bidirectional':
                return self.bidirectional_a_star_search(task, self.heuristic, self.get_neighbors_forward, self.get_neighbors_backward)
        # 可以添加其他策略的实现
        else:
            raise ValueError("未实现的搜索策略")

    def a_star_search(self, task, reverse=False):
        """
        实现 A* 搜索算法。
        """
        start_state = pair['input']
        goal_state = pair['output']
        open_set = []
        heapq.heappush(open_set, (0, start_state))
        came_from = {}
        cost_so_far = {start_state: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_state:
                return self.reconstruct_path(came_from, current)

            if reverse:
                neighbors = self.get_neighbors_backward(current)
            else:
                neighbors = self.get_neighbors_forward(current)

            for neighbor in neighbors:
                new_cost = cost_so_far[current] + 1  # 假设每个操作的代价为1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal_state)
                    heapq.heappush(open_set, (priority, neighbor))
                    came_from[neighbor] = current
        return None


    def bidirectional_a_star_search(self, task, heuristic, get_neighbors_forward, get_neighbors_backward):
        """
        使用双向 A* 搜索算法在状态空间中寻找从 start_state 到 goal_state 的路径。

        参数:
        - task: 包含多对训练数据的任务列表，格式为 [{'input': input1, 'output': output1}, {'input': input2, 'output': output2}, ...]
        - heuristic: 启发式函数，估计从当前状态到目标状态的代价
        - get_neighbors_forward: 函数，返回当前状态的正向邻居状态
        - get_neighbors_backward: 函数，返回当前状态的反向邻居状态

        返回:
        - path: 从 start_state 到 goal_state 的路径，如果所有训练对都成功转换，则返回路径，否则返回 None
        """
        train_data = task['train']
        test_data = task['test']
        for i, data_pair in enumerate(train_data):
            start_state = data_pair['input']
            goal_state = data_pair['output']

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
                # 从正向搜索的 open set 中取出代价最小的节点
                _, current_start = heapq.heappop(open_set_start)
                closed_set_start.add(current_start)

                # 从反向搜索的 open set 中取出代价最小的节点
                _, current_goal = heapq.heappop(open_set_goal)
                closed_set_goal.add(current_goal)

                # 检查是否有交集
                if current_start in closed_set_goal:
                    return self.reconstruct_bidirectional_path(came_from_start, came_from_goal, current_start)
                if current_goal in closed_set_start:
                    return self.reconstruct_bidirectional_path(came_from_start, came_from_goal, current_goal)

                # 正向搜索扩展邻居节点
                neighbors_start = get_neighbors_forward(current_start)
                for neighbor in neighbors_start:
                    if neighbor in closed_set_start:
                        continue
                    tentative_g_score = g_score_start[current_start] + 1  # 假设每个操作的代价为1
                    if neighbor not in g_score_start or tentative_g_score < g_score_start[neighbor]:
                        came_from_start[neighbor] = current_start
                        g_score_start[neighbor] = tentative_g_score
                        f_score_start[neighbor] = tentative_g_score + heuristic(neighbor, goal_state)
                        heapq.heappush(open_set_start, (f_score_start[neighbor], neighbor))

                # 反向搜索扩展前驱节点
                neighbors_goal = get_neighbors_backward(current_goal)
                for neighbor in neighbors_goal:
                    if neighbor in closed_set_goal:
                        continue
                    tentative_g_score = g_score_goal[current_goal] + 1  # 假设每个操作的代价为1
                    if neighbor not in g_score_goal or tentative_g_score < g_score_goal[neighbor]:
                        came_from_goal[neighbor] = current_goal
                        g_score_goal[neighbor] = tentative_g_score
                        f_score_goal[neighbor] = tentative_g_score + heuristic(neighbor, start_state)
                        heapq.heappush(open_set_goal, (f_score_goal[neighbor], neighbor))

            # 如果某个训练对无法找到路径，则返回 None
            return None

        # 如果所有训练对都成功转换，则返回路径
        return self.reconstruct_bidirectional_path(came_from_start, came_from_goal, current_start)

    def reconstruct_bidirectional_path(self, came_from_start, came_from_goal, meeting_point):
        """
        重建从初始状态到目标状态的路径。

        参数:
        - came_from_start: 字典，记录正向搜索每个状态的前驱状态
        - came_from_goal: 字典，记录反向搜索每个状态的前驱状态
        - meeting_point: 正向搜索和反向搜索的相遇点

        返回:
        - path: 从初始状态到目标状态的路径
        """
        path_start = [meeting_point]
        while meeting_point in came_from_start:
            meeting_point = came_from_start[meeting_point]
            path_start.append(meeting_point)
        path_start.reverse()

        path_goal = []
        meeting_point = path_start[-1]
        while meeting_point in came_from_goal:
            meeting_point = came_from_goal[meeting_point]
            path_goal.append(meeting_point)

        return path_start + path_goal


    def get_neighbors_forward(self, state):
        neighbors = []
        for op in self.operators:
            next_state = op.apply(state)
            if next_state:
                neighbors.append(next_state)
        return neighbors

    def get_neighbors_backward(self, state):
        neighbors = []
        for op in self.operators:
            next_state = op.apply_inverse(state)
            if next_state:
                neighbors.append(next_state)
        return neighbors

    def heuristic(self, state, goal_state):
        # 示例启发式函数，可以根据具体问题定义
        return compute_difference(state, goal_state)

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


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


class DSLFunctionRegistry:
    def __init__(self, classified_functions_file):
        self.classified_functions = self.load_classified_functions(classified_functions_file)

    def load_classified_functions(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            classified_functions = json.load(f)

        # 将字符串键转换回元组键
        classified_functions = {eval(key): value for key, value in classified_functions.items()}
        # print(f"加载的分类函数: {classified_functions}")  # 调试信息
        return classified_functions

    def get_functions(self, input_types, output_type):
        key = str((tuple(input_types), output_type))
        return self.classified_functions.get(key, [])

    def call_function(self, func_name, *args):
        # 动态导入 DSL 文件中的函数并调用
        module_name = 'dsl_module'  # 替换为实际的 DSL 模块名
        module = __import__(module_name)
        func = getattr(module, func_name)
        return func(*args)

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