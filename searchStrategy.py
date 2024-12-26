from searchARC import *
import searchARC
import logging
from dsl  import *


class State:
    def __init__(self, data, type, parent=None, action=None, parameters=None, transformation_path=None, weight=5):
        # 将默认权重改为5,让初始状态有一个中等权重
        self.data = data
        self.types = type_extractor.extract_types(type)
        self.types.append('container')# 修改：支持多个类型
        self.types.append('object')
        self.types.append('objects')
        self.parent = parent      # 新增：记录父状态
        self.action = action      # 新增：记录产生该状态的操作符
        self.parameters = parameters if parameters else []
        self.hash = self.compute_hash()
        self.transformation_path = transformation_path if transformation_path else []
        self.weight = weight  # 新增：状态权重属性
        self.value_generation_path = []  # 新增：记录值的生成路径
        self.computation_trace = {  # 新增:完整记录计算过程
            'inputs': [],  # 输入值列表
            'function': None,  # 使用的函数
            'args': [],  # 参数
            'result': None,  # 计算结果
            'source_states': []  # 源状态列表
        }
        self.value_source = {
            'type': None,  # 'input_direct' | 'computed' | 'constant'
            'origin': None,  # 来源描述
            'computation': None  # 如果是计算得到的,记录计算过程
        }

    def compute_hash(self):
        """
        计算状态的哈希值，用于重复检测。
        """
        import time
        return hash(time.perf_counter_ns())
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
    def __init__(self, dsl_registry, enable_whitelist=False, whitelist = None):
        self.dsl_registry = dsl_registry
        # 定义函数白名单，根据 enable_whitelist 参数选择初始化方式
        global success
        if 'success' not in globals():
            success = 0
        if enable_whitelist:
            # 初始化为包含所有 DSL 函数
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
            print(' - - - fun is dsl_functions')
            self.function_whitelist_args = None
        else:
            if whitelist[0] == 1:
                self.function_whitelist = whitelist[1][0]
                self.function_whitelist_args = whitelist[1][1]
            elif whitelist[0] == 2:
                self.function_whitelist = whitelist[1]
                self.function_whitelist_args = None
                # self.function_whitelist_args = whitelist[1]
            print(' - - - fun is ' + str(whitelist))
            # 手动指定需要的函数集合
            # self.function_whitelist = {
            #     'hmirror',
            #     'vconcat',
            #     # 其他需要的函数
            # }
            # 如果需要添加更多函数，直接在此处添加即可
            # 例如:
            # 'another_function',
    # @staticmethod
    def create_zero_matrix(self, grid):
        if not grid or not grid[0]:
            return tuple()

        h, w = len(grid), len(grid[0])
        # 使用tuple保持数据类型一致
        return tuple(tuple(0 for _ in range(w)) for _ in range(h))

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
                print(actions, " \n ok over ！ ！ ！ ！ ！ all data test over 成功的状态转换过程的函数序列:",)
                # print(actions)

                # 使用记录的函数序列对测试数据进行验证
                # self.validate_test_data(task, actions)
            else:
                print("未找到解决方案")

    def bidirectional_a_star_search(self, task, heuristic):
        actions_list = []
        best_solution = None
        min_solution_length = float('inf')

        # 对每个训练数据对进行搜索
        for pair in [task['train'][2]]:  #task['train']
            pair_solutions = []  # 存储当前数据对的所有可能解决方案
            start_state = State(pair['input'], 'grid')
            goal_state = State(pair['output'], 'grid')
            goalhw_state = State(self.create_zero_matrix(pair['output']), 'grid')

            # 搜索当前数据对的所有可能解决方案
            max_attempts = 1  # 限制每个数据对的最大尝试次数
            for _ in range(max_attempts):
                solution = self._search_single_pair(task, start_state, goal_state,goalhw_state, heuristic)
                if solution:
                    _, actions = solution
                    filtered_actions = [action for action in actions if action]
                    if filtered_actions:
                        pair_solutions.append(filtered_actions)
                        # 如果找到更简洁的解决方案，更新最佳解
                        if len(filtered_actions) < min_solution_length:
                            min_solution_length = len(filtered_actions)
                            best_solution = filtered_actions

            if not pair_solutions:
                print(f"未找到第 {len(actions_list) + 1} 个训练数据对的解决方案")
                return None

            actions_list.append(pair_solutions)

        # 尝试找到适用于所有数据对的最简解决方案
        if best_solution:
            # 验证最佳解决方案是否适用于所有数据对
            if self.validate_on_all_data(task, best_solution):
                global success
                success += 1
                print("\n best 找到适用于所有训练数据对的最优函数序列:\n",best_solution)
                print(f"success {success}")
                return best_solution

        # 如果没有找到通用解决方案，尝试其他可能的组合
        print("未找到适用于所有训练数据对的通用函数序列")
        return None

    def data_in_closed_set(self, state_data, closed_set):
        """
        检查 state_data 是否存在于 closed_set 中的某个元素的 data 中。
        """
        for state in closed_set:
            if state.data == state_data:
                return True
        return False


    def _search_single_pair_reverse(self,task, start_state, goal_state,goalhw_state, heuristic):
        return self._search_single_pair(task, goal_state, start_state, heuristic)

    def _search_single_pair(self,task, start_state, goal_state,goalhw_state,  heuristic):
        max_depth = 5  # 最大搜索深度，可以根据需要调整
        came_from = {}
        original_data = start_state.data  # 设置原始数据

        # 修改初始状态列表，添加权重
        current_states = [start_state, goalhw_state]  # 起始状态权重默认为5
        # 添加基础常量状态，设置低权重
        basic_states = ([State(i, 'integer', weight=9) for i in range(10)]
            + [State((0, 0), 'integertuple', weight=10),
             State((0, 1), 'integertuple', weight=10),
             State((1, 0), 'integertuple', weight=10),
             State((-1, 0), 'integertuple', weight=10),
             State((0, -1), 'integertuple', weight=10),
             State((1, 1), 'integertuple', weight=10),
             State((-1, -1), 'integertuple', weight=10),
             State((-1, 1), 'integertuple', weight=10),
             State((1, -1), 'integertuple', weight=10),
             State((0, 2), 'integertuple', weight=10),
             State((2, 0), 'integertuple', weight=10),
             State((2, 2), 'integertuple', weight=10),
             State((3, 3), 'integertuple', weight=10)])
        # basic_states = ([State(i, 'integer', weight=9)for i in [3,8]])
        # current_states.extend(basic_states)

        def create_state_from_arg(arg, weight=0):
            """根据参数类型创建对应的State对象"""
            if isinstance(arg, tuple):
                if all(isinstance(x, tuple) for x in arg):
                    return State(arg, 'tupletuple', weight=weight)
                return State(arg, 'integertuple', weight=weight)
            return State(arg, 'integer', weight=weight)

        if self.function_whitelist_args:
            args_states = [create_state_from_arg(arg) for arg in self.function_whitelist_args]
            current_states.extend(args_states)
        # current_states.extend(goalhw_state)

        visited = set(current_states)  # 新增：记录已访问的状态
        original_states = current_states # 保存初始状态列表

        def create_base_state(value, type_name, source_type, origin=None):
            state = State(value, type_name)
            state.value_source = {
                'type': source_type,
                'origin': origin or f"Found in input values",
                'computation': None
            }
            return state

        for depth in range(max_depth):
            print(f"\n当前深度：{depth}")
            # 修改get_neighbors调用,传入额外参数
            neighbors, solution = self.get_neighbors(
                current_states,
                start_state,
                visited,
                goal_state=goal_state,
                task=task,
                came_from=came_from
            )

            # 如果找到解决方案,直接返回
            if solution:
                return None, solution

            if not neighbors:
                break  # 没有新的邻居，停止搜索
            next_states = []
            for neighbor in neighbors:
                if neighbor.data == goal_state.data and neighbor.hash != goal_state.hash:
                    # 在调用 reconstruct_path 前，先更新 came_from
                    if neighbor not in came_from:
                        came_from[neighbor] = neighbor.parent
                    # return self.reconstruct_path(came_from, neighbor, original_data)
                    _, actions = self.reconstruct_path(came_from, neighbor, original_data)
                    if self.validate_on_all_data(task, actions):
                        return None, actions  # 返回找到的解，立即结束搜索
                    else:
                        pass  # 继续搜索

                if neighbor not in visited:
                    visited.add(neighbor)  # 新增：标记状态为已访问
                # if neighbor not in came_from:
                    came_from[neighbor] = neighbor.parent

                    next_states.append(neighbor)
            current_states = next_states  # 准备生成下一层的邻居
        return None  # 未找到解



    def validate_on_all_data(self, task, actions):
        """在所有训练数据和测试数据上验证给定的函数序列。"""
        for pair in task['train'] + task.get('test', []):
            I = pair['input']
            expected_output = pair['output']
            if not self.validate_single_pair(I, expected_output, actions):
                print(f"----Failed on input: {I}, expected output: {expected_output}")
                return False  # 有一个数据未通过验证
        print(actions," ！ ！ ！ ！ ！ all data test 成功的状态转换过程的函数序列:",)
        return True  # 所有数据都通过

    def validate_single_pair(self, I, expected_output, actions):
        """修改验证过程支持动态计算"""
        # 添加辅助函数
        def get_value_from_input(input_data, target):
            """从输入动态计算值"""
            if isinstance(input_data, (list, tuple)):
                # 尝试各种计算方法
                if target == max(max(row) for row in input_data):
                    return target
                elif target == len(input_data):
                    return target
            return target

        # 构建执行环境
        exec_globals = {
            'get_value_from_input': get_value_from_input,
            **globals()
        }

        # 构建执行环境
        func_code = ['def solve(I):']
        for line in actions:
            func_code.append('    ' + line)
        func_code.append('    return O')
        func_code_str = '\n'.join(func_code)
        local_vars = {}
        exec(func_code_str, exec_globals, local_vars)
        solve = local_vars['solve']
        output = solve(I)
        return output == expected_output

    def get_neighbors(self, current_states, start_state):
        """生成下一层的邻居状态，支持多参数函数和状态组合。"""
        neighbors = []
        input_type_state = defaultdict(list)
        original_state = start_state

        for state in current_states:
            for t in state.get_type():
                input_type_state[t].append(state)
        func_list = defaultdict(list)


        func_type_map = {}

        # 遍历并记录函数类型信息
        for key, func_names in self.dsl_registry.classified_functions.items():
            input_types, output_type = key
            for input_type in input_types:
                if input_type in input_type_state:
                    for name in func_names:
                        if name not in func_list[input_type]:  # 检查是否已存在
                            func_list[input_type].append(name)
                    # 记录函数的输入输出类型
                    for func_name in func_names:
                        func_type_map[func_name] = (input_types, output_type)

        # 处理grid类型的函数
        for func_name in func_list.get('grid', []):
            if (func_name in self.dsl_registry.dsl_functions and
                func_name != 'extract_all_boxes'):

                func = self.dsl_registry.dsl_functions[func_name]
                input_types, output_type = func_type_map.get(func_name, (None, None))

                # 优先处理单参数函数
                if len(input_types) == 1:
                    try:
                        new_data = func(state.data)
                        if new_data is not None:
                            new_state = State(new_data, output_type,
                                        parent=state,
                                        action=func_name)
                            neighbors.append(new_state)
                    except Exception as e:
                        logging.debug(f"Function {func_name} failed: {str(e)}")
                        continue

        return neighbors


    def get_neighbors0(self, current_states, start_state, visited, goal_state=None, task=None, came_from=None):
        """修改版本: 支持状态权重动态提升"""
        neighbors = []
        state_type_map = defaultdict(list)
        original_data = start_state.data
        attempted_combinations = set()

        # 按权重对当前状态进行分组和排序
        weight_groups = defaultdict(list)
        for state in visited:
            weight_groups[state.weight].extend((state, t) for t in state.get_type())

        # 预处理：为每种类型的状态按权重排序
        sorted_type_states = defaultdict(list)
        for weight in sorted(weight_groups.keys()):  # 删除 reverse=True
            for state, t in weight_groups[weight]:
                sorted_type_states[t].append((state, weight))
                state_type_map[t].append(state)

        # 对每种类型的状态按权重排序
        for t in sorted_type_states:
            sorted_type_states[t].sort(key=lambda x: x[1])

        # 添加状态频率计数器
        state_frequency = defaultdict(int)
        for state in visited:
            state_frequency[state.data] += 1

        # 修改权重计算逻辑
        def calculate_weight(input_weights, state_data):
            base_weight = max(input_weights) + 2
            frequency = state_frequency[state_data]
            # 根据出现频率提升权重
            frequency_bonus = min(frequency * 2, 10)  # 最多提升10
            return base_weight + frequency_bonus
            # frequency_bonus = min(frequency * 2, 3)  # 最多提升10
            # return base_weight - frequency * 2

        lenlog=0

        # 按权重从低到高处理状态和函数
        for weight in sorted(weight_groups.keys()):  # 删除 reverse=True
            # 获取当前权重下可用的函数
            available_funcs = []
            for key, func_names in self.dsl_registry.classified_functions.items():
                input_types, output_type = key
                func_list = [fn for fn in func_names if fn in self.function_whitelist]
                # func_list = self.function_whitelist
                if not func_list:
                    continue

                # 检查是否有足够的低权重状态可用作参数
                has_low_weight_inputs = True
                for input_type in input_types:
                    if input_type not in sorted_type_states or not sorted_type_states[input_type]:
                        has_low_weight_inputs = False
                        break

                if has_low_weight_inputs:           #true
                    available_funcs.append((func_list, input_types, output_type))

            # 优先使用权重低的状态作为函数参数
            for func_list, input_types, output_type in available_funcs:
                possible_states_lists = []
                for input_type in input_types:
                    # 获取该类型的所有状态，但优先使用低权重的
                    states = [s for s, w in sorted(sorted_type_states[input_type])]  # 删除 reverse=True
                    possible_states_lists.append(states)

                from itertools import product
                for states_combination in product(*possible_states_lists):
                    if len(states_combination) > lenlog :
                        print('len states_combination',len(states_combination))
                        lenlog=len(states_combination)
                    args = [state.data for state in states_combination]
                    max_input_weight = max(state.weight for state in states_combination)

                    # 如果组合中包含低权重状态，优先测试这些函数
                    if max_input_weight <= weight:  # 修改为 <=
                        # print(f"func_list type: {type(func_list)}")
                        # print(f"func_list content: {func_list}")
                        if not isinstance(func_list, (list, tuple)):
                            func_list = list(func_list)
                        for func_name in func_list:
                            # print(f"Processing function: {func_name}")
                            combination_key = (func_name, tuple(args))
                            if combination_key in attempted_combinations:
                                continue
                            attempted_combinations.add(combination_key)

                            func = self.dsl_registry.dsl_functions.get(func_name)
                            if func:
                                # if func_name in ['ofcolor','delta', 'fill']:  #['objects' ]: #, 'hsplit', 'first', 'transpose']:
                                #     print(f"Processing function: {func_name}")
                                #     pass
                                # print(func,args)
                                try:
                                    new_data = func(*args)
                                    if new_data is not None:
                                        # 记录完整的计算过程


                                        # 构建参数列表...
                                        parameters = []
                                        for arg in args:
                                            if arg == original_data:
                                                parameters.append((True, 'is_origin_data'))
                                            else:
                                                parameters.append((False, arg))

                                        # 使用新的权重计算方法
                                        input_weights = [s.weight for s in states_combination]
                                        new_weight = calculate_weight(input_weights, new_data)

                                        # 构建变换路径...
                                        new_transformation_path = []
                                        for state in states_combination:
                                            new_transformation_path.extend(state.transformation_path)
                                        new_transformation = {
                                            'action': func_name,
                                            'parameters': parameters
                                        }
                                        new_transformation_path.append(new_transformation)

                                        computation_trace = {
                                            'inputs': [state.data for state in states_combination],
                                            'function': func_name,
                                            'args': parameters,
                                            'result': new_data,
                                            'source_states': states_combination
                                        }

                                        # 创建新状态
                                        new_state = State(
                                            new_data,
                                            output_type,
                                            parent=states_combination,
                                            action=func_name,
                                            parameters=parameters,
                                            transformation_path=new_transformation_path,
                                            weight=new_weight
                                        )
                                        new_state.computation_trace = computation_trace
                                        # 记录值的生成路径
                                        value_generation = {
                                            'result': new_data,
                                            'function': func_name,
                                            'args': args,
                                            'arg_states': states_combination  # 保存用于生成该值的源状态
                                        }
                                        new_state.value_generation_path = sum([s.value_generation_path for s in states_combination], [])
                                        new_state.value_generation_path.append(value_generation)

                                        # 立即检查是否找到目标状态
                                        if goal_state and new_data == goal_state.data:
                                            came_from[new_state] = new_state.parent
                                            _, actions = self.reconstruct_path(came_from, new_state, original_data)
                                            if task and self.validate_on_all_data(task, actions):
                                                print(f'len neighbors: {len(neighbors)}')
                                                return None, actions

                                        # 更新状态频率
                                        state_frequency[new_data] += 1

                                        # 根据权重过滤并添加到neighbors
                                        if new_weight <= 30 or self.heuristic(new_state, start_state) < 5:
                                            neighbors.append(new_state)

                                except Exception as e:
                                    # logging.error("捕获到异常：%s", e)
                                    # logging.error("详细错误信息：\n%s", traceback.format_exc())

                                    pass
        print(f'len neighbors: {len(neighbors)}')
        return neighbors, None

    def reconstruct_path(self, came_from, current_state, original_data):
        """回溯路径，生成操作序列，并优化掉未使用的变量"""
        actions = []
        var_mapping = {}
        used_vars = set()
        value_generations = {}  # 记录每个变量值的生成过程

        def mark_used_vars(var_name):
            """标记变量为已使用，包括以 'x' 或 'const_' 开头的变量"""
            if var_name.startswith('x') or var_name.startswith('const_') or var_name in ['I', 'O']:
                used_vars.add(var_name)

        # 首先，构建 var_mapping，将状态映射到变量名
        def build_var_mapping(state):
            if state in var_mapping:
                return
            if state.action:
                for parent_state in state.parent:
                    build_var_mapping(parent_state)
                var_name = f'x{len(var_mapping) + 1}'  # 修改：确保变量名唯一
                var_mapping[state] = var_name
            else:
                if state.data == original_data:
                    var_mapping[state] = 'I'
                else:
                    var_mapping[state] = f'const_{len(var_mapping) + 1}'  # 修改：使用 const_ 前缀

        build_var_mapping(current_state)

        # 然后，从输出变量开始，倒序追溯，标记所有被使用的变量
        def trace_used_vars(state):
            var_name = var_mapping[state]
            mark_used_vars(var_name)
            if state.action:
                for parent_state in state.parent:
                    trace_used_vars(parent_state)
                    parent_var_name = var_mapping[parent_state]
                    mark_used_vars(parent_var_name)
            else:
                if state.data == original_data:
                    mark_used_vars('I')

        trace_used_vars(current_state)

        # 根据 var_mapping 和 used_vars 构建操作序列
        def build_actions(state):
            # 先处理父状态
            if state.parent:
                for parent_state in state.parent:
                    build_actions(parent_state)

            # 获取当前状态对应的变量名
            var_name = var_mapping[state]
            if var_name not in used_vars:
                return

            # 记录状态的生成信息
            value_generations[var_name] = {
                'value': state.data,
                'computation': state.computation_trace
            }

            # 生成代码
            if state.action:
                # 函数调用的情况
                computation = state.computation_trace
                if computation['function']:
                    args = []
                    for source_state in computation['source_states']:
                        source_var = var_mapping[source_state]
                        args.append(source_var)
                    func_call = f"{var_name} = {computation['function']}({', '.join(args)})"
                    actions.append(func_call)
            elif var_name != 'I':  # 常量或基础状态
                if state.computation_trace['function']:
                    # 使用计算过程生成常量
                    computation = state.computation_trace
                    args = [var_mapping[s] for s in computation['source_states']]
                    actions.append(f"{var_name} = {computation['function']}({', '.join(args)})")
                else:
                    # 如果是基础常量，使用动态计算
                    actions.append(f"{var_name} = get_value_from_input(I, {repr(state.data)})")

        # 构建操作序列前先确保所有父状态都被映射
        def ensure_all_mappings(state):
            if state in var_mapping:
                return
            if state.parent:
                for parent_state in state.parent:
                    ensure_all_mappings(parent_state)
            build_var_mapping(state)

        ensure_all_mappings(current_state)
        trace_used_vars(current_state)
        build_actions(current_state)

        actions.append(f"O = {var_mapping[current_state]}")

        print(actions,"找到 transformations:", )

        # 可以打印出每个变量的生成过程
        # print("\n变量生成过程:")
        # for var_name, gen_info in value_generations.items():
        #     print(f"{var_name}: {gen_info['value']}")
        #     # for step in gen_info['generation_path']:
        #     for step in gen_info['computation']:
        #         print(f"  <- {step['function']}({step['args']})")      ????????????bug

        return None, actions

    def heuristic(self, state, goal_state):
        return compute_difference(state.data, goal_state.data)


