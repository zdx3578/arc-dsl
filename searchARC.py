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

    def search(self, start_state, goal_state, direction='forward'):
        raise NotImplementedError

from collections import deque

class BFS:
    def __init__(self, operator_layer):
        self.operator_layer = operator_layer

    def search(self, start_state, goal_state, direction='forward'):
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

    def search(self, start_state, goal_state, strategy='a_star', direction='forward'):
        if strategy == 'a_star':
            if direction == 'forward':
                return self.a_star_search(start_state, goal_state)
            elif direction == 'backward':
                return self.a_star_search(goal_state, start_state, reverse=True)
            elif direction == 'bidirectional':
                return self.bidirectional_a_star_search(start_state, goal_state)
        # 可以添加其他策略的实现
        else:
            raise ValueError("未实现的搜索策略")

    def a_star_search(self, start_state, goal_state, reverse=False):
        """
        实现 A* 搜索算法。
        """
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

    def bidirectional_a_star_search(self, start_state, goal_state):
        """
        实现双向 A* 搜索算法。
        """
        # 正向搜索集
        open_set_start = []
        heapq.heappush(open_set_start, (0, start_state))
        came_from_start = {}
        cost_so_far_start = {start_state: 0}

        # 反向搜索集
        open_set_goal = []
        heapq.heappush(open_set_goal, (0, goal_state))
        came_from_goal = {}
        cost_so_far_goal = {goal_state: 0}

        # 已访问节点集
        visited_start = set()
        visited_goal = set()

        while open_set_start and open_set_goal:
            # 正向搜索一步
            _, current_start = heapq.heappop(open_set_start)
            visited_start.add(current_start)

            neighbors_start = self.get_neighbors_forward(current_start)
            for neighbor in neighbors_start:
                if neighbor in visited_start:
                    continue
                new_cost = cost_so_far_start[current_start] + 1
                if neighbor not in cost_so_far_start or new_cost < cost_so_far_start[neighbor]:
                    cost_so_far_start[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal_state)
                    heapq.heappush(open_set_start, (priority, neighbor))
                    came_from_start[neighbor] = current_start

                if neighbor in visited_goal:
                    return self.reconstruct_bidirectional_path(came_from_start, came_from_goal, neighbor)

            # 反向搜索一步
            _, current_goal = heapq.heappop(open_set_goal)
            visited_goal.add(current_goal)

            neighbors_goal = self.get_neighbors_backward(current_goal)
            for neighbor in neighbors_goal:
                if neighbor in visited_goal:
                    continue
                new_cost = cost_so_far_goal[current_goal] + 1
                if neighbor not in cost_so_far_goal or new_cost < cost_so_far_goal[neighbor]:
                    cost_so_far_goal[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, start_state)
                    heapq.heappush(open_set_goal, (priority, neighbor))
                    came_from_goal[neighbor] = current_goal

                if neighbor in visited_start:
                    return self.reconstruct_bidirectional_path(came_from_start, came_from_goal, neighbor)

        return None

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
        return compute_difference(state.data, goal_state.data)

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def reconstruct_bidirectional_path(self, came_from_start, came_from_goal, meeting_point):
        path_start = []
        current = meeting_point
        while current in came_from_start:
            current = came_from_start[current]
            path_start.append(current)
        path_start.reverse()

        path_goal = []
        current = meeting_point
        while current in came_from_goal:
            current = came_from_goal[current]
            path_goal.append(current)

        full_path = path_start + [meeting_point] + path_goal
        return full_path

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
    def analyze_difference(self, start_state, goal_state):
        difference = compute_difference(start_state.data, goal_state.data)
        # 根据差异大小或类型选择搜索方向
        if difference == 0:
            return 'none'
        elif difference < 10:
            return 'forward'
        else:
            return 'bidirectional'


# class DifferenceAnalyzer:
#     def analyze(self, start_state, goal_state):
#         if start_state.get_type() != goal_state.get_type():
#             raise TypeError("States are of different types")
#         if start_state.get_type() == 'grid':
#             difference = self._analyze_grid(start_state.grid, goal_state.grid)
#         else:
#             difference = {}
#         return difference

#     def _analyze_grid(self, start_grid, goal_grid):
#         # Implement grid difference analysis
#         pass

class Controller:
    def __init__(self, operator_layer, search_algorithm, difference_analyzer):
        self.operator_layer = operator_layer
        self.search_algorithm = search_algorithm
        self.difference_analyzer = difference_analyzer

    def run(self, start_state, goal_state):
        difference = self.difference_analyzer.analyze(start_state, goal_state)
        # Decide search direction based on difference
        path = self.search_algorithm.search(start_state, goal_state, direction='forward')
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

if __name__ == '__main__':
    # Sample data loading
    start_grid = [[0, 0], [0, 1]]  # Initial grid state
    goal_grid = [[0, 1], [0, 1]]   # Goal grid state
    start_state = GridState(start_grid)
    goal_state = GridState(goal_grid)

    # Configuration management
    config_manager = ConfigManager('config.py')
    proper_functions = config_manager.get_proper_functions()
    operator_layer = OperatorLayer.from_config(proper_functions)

    # Search algorithm selection
    search_algorithm_name = config_manager.get_search_algorithm()
    if search_algorithm_name == 'BFS':
        search_algorithm = BFS(operator_layer)

    # Difference analyzer
    difference_analyzer = DifferenceAnalyzer()

    # Controller
    controller = Controller(operator_layer, search_algorithm, difference_analyzer)

    # Run the search
    controller.run(start_state, goal_state)