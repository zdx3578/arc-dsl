from searchARC import *
# import searchARC


from searchARC import *
# import searchARC


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
                return self.a_star_search(task)
            elif direction == 'backward':
                return self.a_star_search(task, reverse=True)
            elif direction == 'bidirectional':
                return self.bidirectional_a_star_search(task, self.heuristic)
        else:
            raise ValueError("未实现的搜索策略")

    def bidirectional_a_star_search(self, task, heuristic):
        for pair in task['train']:
            start_state = State(pair['input'], 'grid')  # 包含类型信息
            goal_state = State(pair['output'], 'grid')  # 包含类型信息

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

                if current_start in closed_set_goal:
                    return self.reconstruct_bidirectional_path(came_from_start, came_from_goal, current_start)
                if current_goal in closed_set_start:
                    return self.reconstruct_bidirectional_path(came_from_start, came_from_goal, current_goal)

                neighbors_start = self.get_neighbors(current_start)
                for neighbor in neighbors_start:
                    if neighbor in closed_set_start:
                        continue
                    tentative_g_score = g_score_start[current_start] + 1
                    if neighbor not in g_score_start or tentative_g_score < g_score_start[neighbor]:
                        came_from_start[neighbor] = current_start
                        g_score_start[neighbor] = tentative_g_score
                        f_score_start[neighbor] = tentative_g_score + heuristic(neighbor, goal_state)
                        heapq.heappush(open_set_start, (f_score_start[neighbor], neighbor))

                neighbors_goal = self.get_neighbors(current_goal, reverse=True)
                for neighbor in neighbors_goal:
                    if neighbor in closed_set_goal:
                        continue
                    tentative_g_score = g_score_goal[current_goal] + 1
                    if neighbor not in g_score_goal or tentative_g_score < g_score_goal[neighbor]:
                        came_from_goal[neighbor] = current_goal
                        g_score_goal[neighbor] = tentative_g_score
                        f_score_goal[neighbor] = tentative_g_score + heuristic(neighbor, start_state)
                        heapq.heappush(open_set_goal, (f_score_goal[neighbor], neighbor))

            return None

        return self.reconstruct_bidirectional_path(came_from_start, came_from_goal, current_start)

    def reconstruct_bidirectional_path(self, came_from_start, came_from_goal, meeting_point):
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

    def get_neighbors(self, state, reverse=False):
        neighbors = []
        for op in self.operators:
            if state.get_type() in op.applicable_types:
                if reverse and op.inverse_function_name:
                    new_states = op.invert(state)  # 使用反向函数
                else:
                    new_states = op.apply(state)  # 使用正向函数
                neighbors.extend(new_states)
        return neighbors

    def heuristic(self, state, goal_state):
        return compute_difference(state.data, goal_state.data)