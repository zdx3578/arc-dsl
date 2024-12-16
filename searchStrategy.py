from searchARC import *
import searchARC
import logging


class State:
    def __init__(self, data, type, parent=None, action=None, parameters=None):
        self.data = data
        self.types = type_extractor.extract_types(type)  # 修改：支持多个类型
        self.parent = parent      # 新增：记录父状态
        self.action = action      # 新增：记录产生该状态的操作符
        self.parameters = parameters if parameters else []
        self.hash = self.compute_hash()

    def compute_hash(self):
        """
        计算状态的哈希值，用于重复检测。
        """
        return hash((tuple(sorted(self.types)), tuple(self.parameters), self._data_hash()))  # 修改：包含参数

    def __eq__(self, other):
        return (
            set(self.types) == set(other.types) and
            self.data == other.data and
            self.parameters == other.parameters  # 修改：比较参数
        )

    def _data_hash(self):
        if isinstance(self.data, list):
            return tuple(map(tuple, self.data))
        elif isinstance(self.data, set) or isinstance(self.data, frozenset):
            return frozenset(self.data)
        else:
            return hash(self.data)

    def __hash__(self):
        return self.hash

    def __lt__(self, other):
        return random.choice([True, False])

    def get_type(self):
        return self.types  # 修改：返回类型列表



class SearchStrategy:
    def __init__(self, dsl_registry, enable_whitelist=True):
        self.dsl_registry = dsl_registry
        # 定义函数白名单，默认包含所有 DSL 函数
        self.function_whitelist = set(self.dsl_registry.dsl_functions.keys())
        # 批量移除不需要的函数
        functions_to_remove = [
            'add',
            'subtract',
            'multiply',
            'divide',
            'tojvec',
            'toivec'
        ]
        for func in functions_to_remove:
            self.function_whitelist.discard(func)  # 使用 discard 防止函数不存在时报错

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
                actions = solution  # 修改：解包路径和操作序列
                print(" ！ ！ ！ ！ ！ 成功的状态转换过程的函数序列:",actions)
                # print(actions)

                # 使用记录的函数序列对测试数据进行验证
                # self.validate_test_data(task, actions)
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
        print(" - - 找到适用于所有训练数据对的共用函数序列:", common_actions)
        if self.validate_test_data(task, common_actions) :
            return common_actions  # 修改：只返回 common_actions

    def data_in_closed_set(self, state_data, closed_set):
        """
        检查 state_data 是否存在于 closed_set 中的某个元素的 data 中。
        """
        for state in closed_set:
            if state.data == state_data:
                return True
        return False

    def _search_single_pair(self, start_state, goal_state, heuristic):
        max_depth = 10  # 最大搜索深度，可以根据需要调整
        came_from = {}
        original_data = start_state.data  # 设置原始数据
        current_states = [start_state] + [State(i, 'integer') for i in range(10)] + [
            State((0, 0), 'integertuple'),
            State((0, 1), 'integertuple'),
            State((1, 0), 'integertuple'),
            State((-1, 0), 'integertuple'),
            State((0, -1), 'integertuple'),
            State((1, 1), 'integertuple'),
            State((-1, -1), 'integertuple'),
            State((-1, 1), 'integertuple'),
            State((1, -1), 'integertuple'),
            State((0, 2), 'integertuple'),
            State((2, 0), 'integertuple'),
            State((2, 2), 'integertuple'),
            State((3, 3), 'integertuple')
        ]
        visited = set(current_states)  # 新增：记录已访问的状态

        for depth in range(max_depth):
            print(f"当前深度：{depth}")
            neighbors = self.get_neighbors(current_states, start_state)  # 修改：传入 start_state
            if not neighbors:
                break  # 没有新的邻居，停止搜索
            next_states = []
            for neighbor in neighbors:
                if neighbor.data == goal_state.data:
                    # 在调用 reconstruct_path 前，先更新 came_from
                    if neighbor not in came_from:
                        came_from[neighbor] = neighbor.parent
                    return self.reconstruct_path(came_from, neighbor, original_data)
                if neighbor not in visited:
                    came_from[neighbor] = neighbor.parent
                    visited.add(neighbor)  # 新增：标记状态为已访问
                    next_states.append(neighbor)
            current_states = next_states  # 准备生成下一层的邻居
        return None  # 未找到解

    def get_neighbors(self, current_states, start_state):
        """生成下一层的邻居状态，支持多参数函数和状态组合。"""
        neighbors = []
        state_type_map = defaultdict(list)
        original_data = start_state.data
        attempted_combinations = set()  # 用于记录已尝试的函数和参数组合

        for state in current_states:
            for t in state.get_type():
                state_type_map[t].append(state)

        # 遍历 DSL 中的函数，根据输入类型匹配
        for key, func_names in self.dsl_registry.classified_functions.items():
            input_types, output_type = key
            # 仅使用白名单中的函数
            func_list = [fn for fn in func_names if fn in self.function_whitelist]
            if not func_list:
                continue  # 当前类型组合下无可用函数，跳过

            possible_states_lists = []
            for input_type in input_types:
                if input_type in state_type_map:
                    possible_states_lists.append(state_type_map[input_type])
                else:
                    break
            else:
                # 限制每个输入类型的状态数量，避免组合过多
                # limited_states_lists = [states[:5] for states in possible_states_lists]  # 取前5个状态
                limited_states_lists = possible_states_lists
                from itertools import product
                for states_combination in product(*limited_states_lists):
                    args = [state.data for state in states_combination]
                    for func_name in func_list:
                        combination_key = (func_name, tuple(args))
                        if combination_key in attempted_combinations:
                            continue  # 已经尝试过该函数和参数组合，跳过
                        attempted_combinations.add(combination_key)
                        func = self.dsl_registry.dsl_functions.get(func_name)
                        if func:
                            try:
                                print(f"--尝试应用函数 {func_name}  arg {args} len: {len(neighbors)}")
                                new_data = func(*args)
                                if new_data is not None:
                                    # 保存所有参数，后续在 reconstruct_path 中处理
                                    parameters = []
                                    for arg in args:
                                        if arg == original_data:
                                            parameters.append((True, 'is_origin_data'))
                                        else:
                                            parameters.append((False, arg))
                                    new_state = State(new_data, output_type, parent=states_combination,
                                                      action=func_name, parameters=parameters)
                                    # 计算启发式值，选择性加入 neighbors
                                    # heuristic_value = self.heuristic(new_state, start_state)
                                    # 假设设定一个合理的阈值，如 10
                                    heuristic_value = 0
                                    if heuristic_value < 10:
                                        neighbors.append(new_state)
                            except Exception as e:
                                # print(f"函数 {func_name} 应用时出错: {e}")
                                # logging.error(f"函数 {func_name} 应用时出错: {e}")
                                pass
        return neighbors

    def reconstruct_path(self, came_from, current_state, original_data):
        """回溯路径，生成操作序列和路径。同时处理参数列表。"""
        path = []
        actions = []

        while current_state in came_from:
            # 检查参数列表,提取额外参数
            if current_state.parameters:
                actions.append((current_state.action, current_state.parameters))
            else:
                actions.append((current_state.action, []))

            path.append(current_state)
            current_state = came_from[current_state]

        path.reverse()
        actions.reverse()
        return path, actions

    def heuristic(self, state, goal_state):
        return compute_difference(state.data, goal_state.data)

    def validate_test_data(self, task, actions):
        for pair in task['test']:
            state = State(pair['input'], 'grid')
            for action, parameters in actions:
                func = self.dsl_registry.dsl_functions.get(action)
                if func:
                    args = []
                    for q, value in parameters:
                        if q:
                            args.append(state.data)
                        else:
                            if action == 'upscale':
                                if isinstance(value, tuple):
                                    value = value[1]  # 提取元组中的值
                            args.append(value)
                    try:
                        new_data = func(*args)
                        if new_data is not None:
                            state = State(new_data, 'grid', parent=state, action=action, parameters=parameters)
                        else:
                            print(f"函数 {action} 无法应用于当前状态")
                            break
                    except Exception as e:
                        print(f"函数 {action} 执行时出错: {e}")
                        logging.error("捕获到异常：%s", e)
                        logging.error("详细错误信息：\n%s", traceback.format_exc())
                        break
                else:
                    print(f"未找到操作符 {action}")
                    break
            # 比较最终输出结果
            if state.data == pair['output']:
                print(" - - 测试数据验证成功，输出与预期一致")

            else:
                print("测试数据验证失败，输出与预期---不一致")
        return True

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
        for op in self.dsl_registry.dsl_functions.values():
            if op.__name__ == name:
                return Operator(name, name, dsl_registry=self.dsl_registry)
        return None

    def get_applicable_types(self, state_or_input_types, applicable_types):
        if isinstance(state_or_input_types, State):
            input_types = state_or_input_types.get_type()
        else:
            input_types = state_or_input_types
        return set(input_types) & set(applicable_types)

