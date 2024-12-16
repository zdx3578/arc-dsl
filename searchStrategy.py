from searchARC import *
import searchARC
from functools import partial
from typing import List, Tuple, Set, Any, Callable
import logging


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

    def search(self, task, strategy="a_star", direction="bidirectional"):
        if strategy == "a_star":
            if direction == "forward":
                solution = self.a_star_search(task)
            elif direction == "backward":
                solution = self.a_star_search(task, reverse=True)
            elif direction == "bidirectional":
                solution = self.bidirectional_a_star_search(task, self.heuristic)
            else:
                raise ValueError("未实现的搜索策略")

            # 如果找到了解决方案，打印函数序列
            if solution:
                actions = solution  # 修改：解包路径和操作序列
                print("成功的状态转换过程的函数序列:")
                print(actions)

                # 使用记录的函数序列对测试数据进行验证
                # self.validate_test_data(task, actions)
            else:
                print("未找到解决方案")

    def bidirectional_a_star_search(self, task, heuristic):
        actions_list = []

        for pair in task["train"]:
            start_state = State(pair["input"], "grid")  # 包含类型信息
            goal_state = State(pair["output"], "grid")  # 包含类型信息

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

    def data_in_closed_set(self, state_data, closed_set):
        """
        检查 state_data 是否存在于 closed_set 中的某个元素的 data 中。
        """
        for state in closed_set:
            if state.data == state_data:
                return True
        return False

    def _search_single_pair(self, start_state, goal_state, heuristic):
        max_depth = 10
        came_from = {}
        visited_states = set()  # 新增：追踪已访问状态
        current_states = [start_state]
        visited_states.add(str(start_state.data))  # 将起始状态标记为已访问

        for depth in range(max_depth):
            print(f"\n=== 当前搜索深度：{depth} ===")
            print(f"当前状态数量：{len(current_states)}")

            if not current_states:
                print("当前层没有状态可以扩展")
                break

            next_states = []
            neighbors = self.get_neighbors(current_states, start_state)
            print(f"本层生成的邻居数量: {len(neighbors)}")

            for neighbor in neighbors:
                # 转换状态数据为字符串用于比较
                neighbor_str = str(neighbor.data)

                if neighbor_str not in visited_states:
                    visited_states.add(neighbor_str)

                    if neighbor.data == goal_state.data:
                        print("找到目标状态！")
                        came_from[neighbor] = neighbor.parent
                        return self.reconstruct_path(came_from, neighbor)

                    came_from[neighbor] = neighbor.parent
                    next_states.append(neighbor)
                    # print(f"添加新的未访问状态，当前next_states大小: {len(next_states)}")

            if not next_states:
                print("没有新的状态可以扩展，搜索终止")
                break

            print(f"进入下一层，状态数量: {len(next_states)}")
            current_states = next_states  # 更新当前状态集

        print("达到最大深度或无法找到解决方案")
        return None

    def get_neighbors(self, current_states, start_state):
        """生成下一层的邻居状态，支持多参数函数和状态组合。"""
        neighbors = []
        state_type_map = defaultdict(list)
        original_state = start_state

        # 添加调试信息
        print("开始生成邻居状态:")
        print(f"输入状态数量: {len(current_states)}")

        for state in current_states:
            for t in state.get_type():
                state_type_map[t].append(state)

        # 添加调试信息
        # print(f"状态类型映射: {dict(state_type_map)}")

        # 遍历 DSL 中的函数，根据输入类型匹配
        for key, func_names in self.dsl_registry.classified_functions.items():
            input_types, output_type = key
            # 添加调试信息
            # print(f"尝试函数组: {func_names}")
            # print(f"需要的输入类型: {input_types}")

            func_list = func_names
            possible_states_lists = []
            for input_type in input_types:
                if input_type in state_type_map:
                    possible_states_lists.append(state_type_map[input_type])
                else:
                    break
            else:
                # 生成所有可能的状态组合
                from itertools import product

                for states_combination in product(*possible_states_lists):
                    args = [state.data for state in states_combination]
                    for func_name in func_list:
                        if (
                            func_name in self.dsl_registry.dsl_functions
                            and func_name != "extract_all_boxes"
                        ):
                            func = self.dsl_registry.dsl_functions[func_name]
                            try:
                                new_data = func(*args)
                                if new_data is not None:
                                    # 保存所有参数，后续在reconstruct_path中处理
                                    new_state = State(
                                        new_data,
                                        output_type,
                                        parent=states_combination,
                                        action=func_name,
                                        parameters=args,
                                    )
                                    neighbors.append(new_state)
                            except Exception as e:
                                pass
        if len(neighbors) > 0:
            print(f"成功生成 {len(neighbors)} 个新邻居")
        return neighbors

    def reconstruct_path(self, came_from, current_state):
        """使用函数式方式重构路径重建逻辑"""

        def collect_original_data(state, acc_set: Set) -> Set:
            """纯函数：收集原始输入数据"""
            if state not in came_from:
                return acc_set
            parent = state.parent
            if not parent:
                return acc_set

            new_data = set()
            if isinstance(parent, (list, tuple)):
                new_data.update(p.data for p in parent if p.parent is None)
            elif parent.parent is None:
                new_data.add(parent.data)

            return collect_original_data(came_from[state], acc_set | new_data)

        def create_param_info(params, original_datas: Set) -> Tuple[List, List]:
            """纯函数：创建参数信息"""
            positions = []
            values = []

            for i, param in enumerate(params or []):
                is_original = param in original_datas
                positions.append(i if is_original else None)
                values.append(None if is_original else param)

            return (
                [p for p in positions if p is not None],  # 原始输入位置
                values  # 参数值，原始输入位置为None
            )

        def build_action(state, original_datas: Set) -> Tuple:
            """纯函数：构建动作信息"""
            if not state.parameters:
                return (state.action, [], [])
            param_positions, param_values = create_param_info(state.parameters, original_datas)
            return (state.action, param_values, param_positions)

        # 获取原始数据集
        original_datas = collect_original_data(current_state, set())

        # 构建路径和动作
        path = []
        actions = []
        while current_state in came_from:
            path.append(current_state)
            actions.append(build_action(current_state, original_datas))
            current_state = came_from[current_state]

        return list(reversed(path)), list(reversed(actions))

    def heuristic(self, state, goal_state):
        return compute_difference(state.data, goal_state.data)

    def validate_test_data(self, task, actions):
        """使用函数式方式重构验证逻辑"""

        def apply_action(state: State, action_info: Tuple) -> State:
            """纯函数：应用单个动作到状态"""
            action_name, param_values, input_positions = action_info
            func = self.dsl_registry.dsl_functions.get(action_name)

            if not func:
                raise ValueError(f"未找到操作符 {action_name}")

            # 构建参数列表
            args = param_values.copy()
            for pos in input_positions:
                args[pos] = state.data

            try:
                result = func(*args)
                if result is None:
                    raise ValueError(f"函数 {action_name} 无法应用于当前状态")

                return State(result, "grid", parent=state, action=action_name, parameters=args)

            except Exception as e:
                logging.error(f"函数 {action_name} 执行时出错: {e}")
                logging.error(f"详细错误信息：\n{traceback.format_exc()}")
                raise

        def validate_single_pair(pair, actions):
            """纯函数：验证单个输入输出对"""
            initial_state = State(pair["input"], "grid")
            try:
                final_state = reduce(apply_action, actions, initial_state)
                return final_state.data == pair["output"]
            except Exception as e:
                print(f"验证过程出错: {e}")
                return False

        # 验证所有测试数据
        results = map(lambda pair: validate_single_pair(pair, actions), task["test"])

        # 处理验证结果
        for success in results:
            print("测试数据验证成功，输出与预期一致" if success else "测试数据验证失败，输出与预期不一致")

    # def apply_asindices_if_needed(self, state):
    #     """
    #     如果需要，应用 'asindices' 函数将状态转换为 'grid' 类型。
    #     """
    #     if 'grid' not in state.get_type():
    #         func = self.dsl_registry.dsl_functions.get('asindices')
    #         if func:
    #             try:
    #                 new_data = func(state.data)
    #                 if new_data是 not None:
    #                     return State(new_data, 'grid', parent=state, action='asindices')
    #                 else:
    #                     print("函数 asindices 无法应用于当前状态")
    #             except Exception as e:
    #                 print(f"函数 asindices 执行时出错: e")
    #     return state

    def convert_to_grid(self, state):
        """
        将非 'grid' 类型的状态转换为 'grid' 类型。
        需要根据具体的上下文和可用的函数来实现。
        """
        # 示例：如果状态类型是 'indices'，尝试转换为 'grid'
        if "indices" in state.get_type():
            # 假设有一个函数可以将 indices 转换为 grid，例如 indices_to_grid
            new_data = indices_to_grid(state.data)
            return State(new_data, "grid")
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
