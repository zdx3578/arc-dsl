from searchARC import *
# import searchARC


# class State:
#     def __init__(self, data, type, parent=None, action=None):
#         self.data = data
#         self.type = type
#         self.parent = parent      # 新增：记录父状态
#         self.action = action      # 新增：记录产生该状态的操作符


class SearchStrategy:
    def __init__(self, dsl_registry):
        self.dsl_registry = dsl_registry
        self.operators = self.load_operators()

    def load_operators(self):
        # 从 DSL 注册表中加载操作符
        operators = []
        for key, functions in self.dsl_registry.classified_functions.items():
            # key_str = str(key)  # 确保 key 是字符串类型
            input_types, output_type = key
            for func_name in functions:
                op = Operator(func_name, func_name, applicable_types=input_types, dsl_registry=self.dsl_registry)
                operators.append(op)
        return operators

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

            neighbors_goal = self.get_neighbors(current_goal, reverse=True)
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

    def get_neighbors(self, state, reverse=False):
        neighbors = []
        for op in self.operators:
            applicable_types = set(state.get_type()) & set(op.applicable_types)  # 修改：处理多类型
            if not applicable_types:
                continue
            if reverse and op.inverse_function_name:
                new_states = op.invert(state)
            else:
                new_states = op.apply(state)
            for new_state in new_states:
                neighbors.append(new_state)  # 修改：操作符内部已记录父状态和操作符
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

# 添加用于转换 indices 到 grid 的函数
def indices_to_grid(indices):
    grid = [[0 for _ in range(max_y)] for _ in range(max_x)]
    """
    将 indices 转换为 grid。
    """
    # 根据 indices 生成 grid，这里需要根据具体需求实现
    # 示例实现，假设 indices 是一组坐标，生成包含这些坐标的 grid
    max_x = max(idx[0] for idx in indices) + 1
    max_y = max(idx[1] for idx in indices) + 1
    grid = [[0 for _ in range(max_y)] for _ in range(max_x)]
    for x, y in indices:
        grid[x][y] = 1  # 或者其他值，视情况而定
    return grid
