from searchARC import *
import searchARC
from dsl import *
from searchStrategy import *

class StateNode:
    """状态树的节点"""
    def __init__(self, state, depth=0, parent=None):
        self.state = state          # 状态信息
        self.depth = depth          # 节点深度
        self.parent = parent        # 父节点
        self.children = []          # 子节点
        self.generation_func = None # 生成该状态的函数
        self.gen_params = []        # 生成参数
        self.matched_node = None    # 映射到目标树的节点
        self.computation_trace = {  # 从searchStrategy借鉴,记录计算过程
            'inputs': [],
            'function': None,
            'args': [],
            'result': None,
            'source_states': []
        }
        self.value_source = {  # 记录值的来源
            'type': None,  # 'input_direct' | 'computed' | 'constant'
            'origin': None,
            'computation': None
        }

    def add_child(self, child_state, func=None, params=None):
        """添加子节点"""
        child = StateNode(child_state, self.depth + 1, self)
        child.generation_func = func
        child.gen_params = params
        self.children.append(child)
        return child

    def get_path_to_root(self):
        """获取到根节点的路径"""
        path = []
        current = self
        while current:
            path.append(current)
            current = current.parent
        return path[::-1]



class StateTree:
    """状态树结构,支持按类型组织和搜索状态"""
    def __init__(self, root_state):
        self.root = StateNode(root_state)
        self.all_nodes = {root_state: self.root}
        self.leaf_nodes = set([self.root])
        self.max_depth = 5

        # 按类型索引状态
        self.type_index = defaultdict(list)  # 类型 -> [状态]
        # 记录每个类型的实例数量
        self.type_counts = defaultdict(int)
        self.add_to_type_index(root_state)

    def add_to_type_index(self, state):
        """将状态添加到类型索引中"""
        for type_name in state.types:
            self.type_index[type_name].append(state)
            self.type_counts[type_name] += 1

    def get_states_by_type(self, type_name, max_count=None):
        """获取指定类型的所有状态"""
        states = self.type_index.get(type_name, [])
        if max_count:
            return states[:max_count]
        return states

    def get_additional_args(self, required_types):
        """智能获取符合类型要求的参数组合"""
        args_candidates = []
        for type_name in required_types:
            # 获取该类型的所有可用状态
            type_states = self.get_states_by_type(type_name)
            if not type_states:
                continue

            # 按照状态特征进行排序
            sorted_states = self.rank_states(type_states, type_name)
            args_candidates.append(sorted_states)

        # 生成参数组合
        from itertools import product
        max_combinations = 10  # 限制组合数量
        return list(product(*args_candidates))[:max_combinations]

    def rank_states(self, states, type_name):
        """对状态进行排序"""
        if type_name == 'grid':
            # 网格类型按大小排序
            return sorted(states, key=lambda s: len(s.data) * len(s.data[0]) if s.data else 0)
        elif type_name == 'integer':
            # 整数类型按值排序
            return sorted(states, key=lambda s: abs(s.data))
        elif type_name == 'object':
            # 对象类型按复杂度排序
            return sorted(states, key=lambda s: len(s.data))
        else:
            return states

    def add_node(self, node):
        """添加新节点时同时更新类型索引"""
        self.all_nodes[node.state] = node
        self.leaf_nodes.add(node)
        if node.parent:
            self.leaf_nodes.discard(node.parent)
        self.add_to_type_index(node.state)

    def get_type_stats(self):
        """获取类型统计信息"""
        return {
            'type_counts': dict(self.type_counts),
            'total_nodes': len(self.all_nodes),
            'leaf_nodes': len(self.leaf_nodes)
        }

    def find_similar_states(self, state, threshold=0.8):
        """查找相似状态"""
        similar_states = []
        state_types = set(state.types)

        # 首先在相同类型中查找
        for type_name in state_types:
            candidates = self.get_states_by_type(type_name)
            for candidate in candidates:
                if self.compute_similarity(state, candidate) >= threshold:
                    similar_states.append(candidate)

        return similar_states

    def compute_similarity(self, state1, state2):
        """计算两个状态的相似度"""
        # 类型相似度
        type_similarity = len(set(state1.types) & set(state2.types)) / len(set(state1.types) | set(state2.types))

        # 数据相似度
        data_similarity = 0.0
        if isinstance(state1.data, type(state2.data)):
            if isinstance(state1.data, (list, tuple)):
                # 对于网格类型,计算重叠率
                data_similarity = self.compute_grid_similarity(state1.data, state2.data)
            else:
                # 对于其他类型,使用简单比较
                data_similarity = 1.0 if state1.data == state2.data else 0.0

        # 综合相似度
        return (type_similarity + data_similarity) / 2

    def compute_grid_similarity(self, grid1, grid2):
        """计算两个网格的相似度"""
        if not grid1 or not grid2:
            return 0.0

        # 计算共同元素的比例
        common = sum(r1.count(v) for r1 in grid1 for v in r1 if any(r2.count(v) for r2 in grid2))
        total = sum(len(r) for r in grid1) + sum(len(r) for r in grid2)
        return 2 * common / total if total else 0.0


    def expand_node(self, node, dsl_reg, visited_states=None):
        """扩展节点,借鉴 searchStrategy 中的函数分组处理逻辑"""
        if visited_states is None:
            visited_states = set()

        if node.depth >= self.max_depth:
            return []

        new_nodes = []
        state_type_map = defaultdict(list)
        attempted_combinations = set()

        # # 按类型对函数分组
        function_groups = defaultdict(list)
        # for key, func_names in dsl_reg.classified_functions.items():
        #     input_types, output_type = key
        #     # 检查函数类型是否匹配当前节点
        #     if any(t in node.state.get_type() for t in input_types):
        #         for func_name in func_names:
        #             function_groups[input_types].append((func_name, output_type))
        meaning_fun = set()

        # 读取文件并提取函数名
        with open('/Users/zhangdexiang/github/VSAHDC/arc-dsl/forprolog/dsl_meaning_class.py', 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('from') and not line.startswith('import'):
                    func_name = line.split('#')[0].strip()  # 去掉注释部分
                    meaning_fun.add(func_name)

        # 大模型改进这段，meaning file 提取函数
        for key, func_names in dsl_reg.classified_functions.items():
            input_types, output_type = key
            # 检查函数类型是否匹配当前节点
            if any(t in node.state.get_type() for t in input_types):
                for func_name in func_names:
                    if func_name in meaning_fun:  #
                        function_groups[input_types].append((func_name, output_type))

        # 按优先级处理不同类型的函数
        # priority_order = [
        #     ('grid',),           # 基础网格操作
        #     ('patch',),
        #     ('element',),
        #     ('object',),         # 对象操作
        #     ('grid', 'grid'),    # 双网格操作
        #     ('object', 'grid'),  # 对象-网格组合操作
        #     ('grid', 'integer'), # 网格-数值组合操作

        # ]

        # 按优先级处理函数组
        # for type_key in priority_order:
        #     if type_key not in function_groups:
        #         continue

        for type_key, funcs in function_groups.items():

            for func_name, output_type in funcs:
                try:
                    # 获取函数实例
                    func = dsl_reg.dsl_functions.get(func_name)
                    if not func:
                        continue

                    # 根据函数类型准备参数
                    if len(type_key) == 1:  # 单参数函数
                        try:
                            result = func(node.state.data)
                            if result is not None and result not in visited_states:
                                # 记录计算过程
                                computation_trace = {
                                    'inputs': [node.state.data],
                                    'function': func_name,
                                    'args': [node.state.data],
                                    'result': result,
                                    'source_states': [node.state]
                                }

                                child = self.create_child_node(
                                    node, result, output_type,
                                    func_name, [node.state],
                                    computation_trace
                                )

                                if child and self.is_valid_state(child):
                                    self.add_node(child)
                                    new_nodes.append(child)
                                    visited_states.add(result)
                        except Exception as e:
                                logging.error("捕获到异常：%s", e)
                                logging.error("详细错误信息：\n%s", traceback.format_exc())
                                continue

                    else:  # 多参数函数
                        # 获取其他参数的可能值
                        additional_args = self.get_additional_args(type_key[1:])
                        for args in additional_args:
                            all_args = [node.state.data] + args
                            if (func_name, tuple(all_args)) in attempted_combinations:
                                continue

                            attempted_combinations.add((func_name, tuple(all_args)))
                            try:
                                result = func(*all_args)
                                if result is not None and result not in visited_states:
                                    # 记录完整的计算过程
                                    computation_trace = {
                                        'inputs': all_args,
                                        'function': func_name,
                                        'args': all_args,
                                        'result': result,
                                        'source_states': [node.state] + [arg for arg in args]
                                    }

                                    child = self.create_child_node(
                                        node, result, output_type,
                                        func_name, computation_trace['source_states'],
                                        computation_trace
                                    )

                                    if child and self.is_valid_state(child):
                                        self.add_node(child)
                                        new_nodes.append(child)
                                        visited_states.add(result)

                            except Exception as e:
                                logging.error("捕获到异常：%s", e)
                                logging.error("详细错误信息：\n%s", traceback.format_exc())
                                continue

                except Exception as e:
                    logging.error("捕获到异常：%s", e)
                    logging.error("详细错误信息：\n%s", traceback.format_exc())
                    continue

        return new_nodes

    def create_child_node(self, parent, data, type_name, func_name, source_states, computation_trace):
        """创建子节点并设置相关属性"""
        child = parent.add_child(State(data, type_name), func_name, source_states)
        child.computation_trace = computation_trace
        child.value_source = {
            'type': 'computed',
            'origin': f"Computed by {func_name}",
            'computation': computation_trace
        }
        return child

    def is_valid_state(self, state):
        """验证状态是否有效"""
        # 可以添加更多验证规则
        return True



class BidirectionalSearch:
    """双向搜索"""
    def __init__(self, dsl_registry):
        self.dsl_registry = dsl_registry
        self.forward_tree = None   # 从输入开始的树
        self.backward_tree = None  # 从输出开始的树
        self.required_connections = []  # 新增:记录需要连接的所有子模块

    def find_path(self, input_state, output_state):
        """寻找输入到输出的路径"""
        # 分析输出状态,获取所有需要连接的子模块
        self.analyze_output_components(output_state)

        self.forward_tree = StateTree(input_state)
        self.backward_tree = StateTree(output_state)

        visited_forward = set()
        visited_backward = set()
        found_connections = {}  # 记录已找到的连接

        while self.forward_tree.leaf_nodes and self.backward_tree.leaf_nodes:
            if len(self.forward_tree.leaf_nodes) <= len(self.backward_tree.leaf_nodes):
                new_connections = self.expand_tree(
                    self.forward_tree, self.backward_tree,
                    visited_forward, is_forward=True)
            else:
                new_connections = self.expand_tree(
                    self.backward_tree, self.forward_tree,
                    visited_backward, is_forward=False)

            if new_connections:
                found_connections.update(new_connections)
                # 检查是否所有必需的连接都找到了
                if self.check_all_connections_found(found_connections):
                    return self.construct_complete_solution(found_connections)

        return None

    # same as expand tree !!!!!!!!!
    def analyze_input_components(self, input_state):
        result = self.analyze_output_components(input_state)
        return result

    def analyze_output_components(self, output_state):
        """分析输出状态中的子模块"""
        self.required_connections = []
        grid = output_state.data

        # 获取grid的基本维度信息
        if isinstance(grid, (list, tuple)):
            height = len(grid)
            width = len(grid[0]) if height > 0 else 0
            self.required_connections.append({
                'type': 'dimension',
                'height': height,
                'width': width
            })

        # 使用 dsl.objects 提取对象
        extracted_objects = dsl.objects(grid,
                                     univalued=True,  # 单一颜色的对象
                                     diagonal=False,   # 不考虑对角连接
                                     without_bg=True)  # 忽略背景

        for obj in extracted_objects:
            # self.extract_obj_feature(obj)


            self.required_connections.append({
                'type': 'object',
                'data': obj,
                'connected': False
            })

        # 识别其他特征
        features = self.extract_features(grid)
        for feature in features:
            self.required_connections.append({
                'type': 'feature',
                'data': feature,
                'connected': False
            })

    # def extract_obj_feature(self, obj):
        # obj_type_input_fun = findfun(meaningfun,typeobj,)


    def extract_features(self, grid):
        """提取 grid 的完整特征"""
        features = []

        # 1. 颜色相关特征
        features.extend([
            ('color_distribution', dsl.colorcount(grid, color))
            for color in dsl.palette(grid)
        ])
        features.append(('most_color', dsl.mostcolor(grid)))
        features.append(('least_color', dsl.leastcolor(grid)))
        features.append(('num_colors', dsl.numcolors(grid)))

        # 2. 形状相关特征
        features.append(('shape', dsl.shape(grid)))
        features.append(('height', dsl.height(grid)))
        features.append(('width', dsl.width(grid)))
        features.append(('hw_ratio', dsl.hwratio(grid)))
        features.append(('is_square', dsl.is_square(grid)))
        features.append(('is_portrait', dsl.portrait(grid)))

        # 3. 对称性特征
        features.append(('vmirror_symmetric', grid == dsl.vmirror(grid)))
        features.append(('hmirror_symmetric', grid == dsl.hmirror(grid)))
        features.append(('dmirror_symmetric', grid == dsl.dmirror(grid)))

        # 4. 对象分析
        objects = dsl.objects(grid, univalued=True, diagonal=False, without_bg=True)
        for obj in objects:
            obj_features = {
                'color': dsl.color(obj),
                'size': len(obj),
                'box': dsl.is_box(obj),
                'position': dsl.centerofmass(obj),
                'borders': dsl.bordering(obj, grid),
                'is_line': dsl.hline(obj) or dsl.vline(obj)
            }
            features.append(('object', obj_features))

        # 5. 周期性特征
        if len(objects) > 0:
            for obj in objects:
                features.append(('hperiod', dsl.hperiod(obj)))
                features.append(('vperiod', dsl.vperiod(obj)))

        # 6. 结构特征
        features.append(('has_frontiers', bool(dsl.frontiers(grid))))

        # 7. 分区特征
        features.extend([
            ('upper_third', dsl.upper_third(grid)),
            ('middle_third', dsl.middle_third(grid)),
            ('lower_third', dsl.lower_third(grid)),
            ('left_third', dsl.left_third(grid)),
            ('center_third', dsl.center_third(grid)),
            ('right_third', dsl.right_third(grid))
        ])

        return features

    def match_feature(self, feature1, feature2):
        """增强特征匹配逻辑"""
        feature_type1, data1 = feature1
        feature_type2, data2 = feature2

        if feature_type1 != feature_type2:
            return False

        if feature_type1 == 'color_distribution':
            return data1 == data2
        elif feature_type1 == 'object':
            # 对象特征的匹配需要考虑多个属性
            return (data1['color'] == data2['color'] and
                   data1['size'] == data2['size'] and
                   data1['box'] == data2['box'] and
                   data1['is_line'] == data2['is_line'])
        else:
            return data1 == data2

    # 大模型改进这段，树结构的parent reach no leaf node的检查
    def check_all_connections_found(self, connections):
        """检查是否所有必需的连接都已找到"""
        for req in self.required_connections:
            if req['type'] == 'object':
                if not any(self.match_object(req['data'], conn)
                          for conn in connections.values()):
                    return False
            elif req['type'] == 'feature':
                if not any(self.match_feature(req['data'], conn)
                          for conn in connections.values()):
                    return False
        return True

    def construct_complete_solution(self, connections):
        """构造完整的解决方案"""
        # 按依赖关系排序连接
        ordered_connections = self.order_connections(connections)

        actions = []
        for conn in ordered_connections:
            sub_actions = self.construct_sub_solution(conn)
            actions.extend(sub_actions)

        return actions

    def expand_tree(self, tree_to_expand, other_tree, visited, is_forward):
        """扩展一棵树并检查是否可以连接到另一棵树"""
        current_leaves = list(tree_to_expand.leaf_nodes)
        for leaf in current_leaves:
            new_nodes = tree_to_expand.expand_node(leaf, self.dsl_registry, visited)

            # 检查新节点是否可以连接到另一棵树
            for new_node in new_nodes:
                for other_node in other_tree.all_nodes.values():
                    if self.can_connect(new_node, other_node, is_forward):
                        # 返回一个字典而不是元组
                        connection_id = f"connection_{len(visited)}"
                        if is_forward:
                            return {connection_id: (new_node, other_node)}
                        else:
                            return {connection_id: (other_node, new_node)}

        return {}  # 如果没有找到连接，返回空字典

    def can_connect(self, node1, node2, is_forward):
        """检查两个节点是否可以连接"""
        # 实现连接条件检查逻辑
        return node1.state.data == node2.state.data

    def construct_solution(self, connection):
        """构造解决方案"""
        forward_node, backward_node = connection

        # 获取正向路径
        forward_path = forward_node.get_path_to_root()

        # 获取反向路径
        backward_path = backward_node.get_path_to_root()[::-1]

        # 构造转换序列,采用searchStrategy的格式
        actions = []
        var_mapping = {}
        var_counter = 1

        def add_node_action(node, is_forward=True):
            if node.parent:
                # 为节点创建变量名
                var_name = f'x{var_counter}'
                var_mapping[node] = var_name

                # 获取参数变量名
                param_vars = []
                for param in node.gen_params:
                    if param not in var_mapping:
                        const_name = f'const_{len(var_mapping) + 1}'
                        var_mapping[param] = const_name
                        # 添加常量定义
                        if param.value_source['type'] == 'computed':
                            # 使用计算过程生成常量
                            comp = param.computation_trace
                            args = [var_mapping[s] for s in comp['source_states']]
                            actions.append(f"{const_name} = {comp['function']}({', '.join(args)})")
                        else:
                            # 使用值的实际来源
                            actions.append(f"{const_name} = {param.value_source['origin']}")
                    param_vars.append(var_mapping[param])

                # 添加函数调用
                actions.append(f"{var_name} = {node.generation_func}({', '.join(param_vars)})")
                var_counter += 1

        # 添加正向转换
        for node in forward_path[1:]:
            add_node_action(node, True)

        # 添加反向转换
        for node in backward_path[1:]:
            add_node_action(node, False)

        # 添加最终输出赋值
        actions.append(f"O = {var_mapping[forward_path[-1]]}")

        return actions

    def validate_solution(self, task, actions):
        """验证解决方案,采用searchStrategy的验证逻辑"""
        def validate_single_pair(I, expected_output, actions):
            func_code = ['def solve(I):']
            for line in actions:
                func_code.append('    ' + line)
            func_code.append('    return O')

            local_vars = {}
            exec('\n'.join(func_code), globals(), local_vars)
            solve = local_vars['solve']
            output = solve(I)
            return output == expected_output

        # 验证所有测试数据
        for pair in task['train'] + task.get('test', []):
            if not validate_single_pair(pair['input'], pair['output'], actions):
                return False
        return True
