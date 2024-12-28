from searchARC import *
import searchARC
from dsl import *

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
    """状态树结构"""
    def __init__(self, root_state):
        self.root = StateNode(root_state)
        self.all_nodes = {root_state: self.root}
        self.leaf_nodes = set([self.root])
        self.max_depth = 5

    def expand_node(self, node, dsl_funcs, visited_states=None):
        """扩展节点"""
        if visited_states is None:
            visited_states = set()

        if node.depth >= self.max_depth:
            return []

        new_nodes = []
        for func_name, func in dsl_funcs.items():
            try:
                result = func(node.state.data)
                if result is not None and result not in visited_states:
                    new_state = State(result, 'grid')
                    # 记录计算过程
                    computation_trace = {
                        'inputs': [node.state.data],
                        'function': func_name,
                        'args': [node.state.data],
                        'result': result,
                        'source_states': [node.state]
                    }

                    child = node.add_child(new_state, func_name, [node.state])
                    child.computation_trace = computation_trace
                    child.value_source = {
                        'type': 'computed',
                        'origin': f"Computed by {func_name}",
                        'computation': computation_trace
                    }
                    self.all_nodes[new_state] = child
                    self.leaf_nodes.add(child)
                    self.leaf_nodes.discard(node)
                    visited_states.add(result)
                    new_nodes.append(child)
            except:
                continue
        return new_nodes

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

    def analyze_output_components(self, output_state):
        """分析输出状态中的子模块"""
        self.required_connections = []

        # 获取grid的基本属性
        if isinstance(output_state.data, (list, tuple)):
            height = len(output_state.data)
            width = len(output_state.data[0]) if height > 0 else 0

            # 添加基本维度要求
            self.required_connections.append({
                'type': 'dimension',
                'height': height,
                'width': width
            })

        # 识别对象/颜色块
        objects = self.extract_objects(output_state.data)
        for obj in objects:
            self.required_connections.append({
                'type': 'object',
                'data': obj,
                'connected': False
            })

        # 识别其他特征(如颜色分布、对称性等)
        features = self.extract_features(output_state.data)
        for feature in features:
            self.required_connections.append({
                'type': 'feature',
                'data': feature,
                'connected': False
            })

    def extract_objects(self, grid):
        """从grid中提取独立对象"""
        objects = []
        visited = set()

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if (i,j) not in visited and grid[i][j] != 0:
                    obj = self.flood_fill(grid, i, j, visited)
                    if obj:
                        objects.append(obj)
        return objects

    def flood_fill(self, grid, i, j, visited):
        """使用flood fill算法提取连通对象"""
        if not (0 <= i < len(grid) and 0 <= j < len(grid[0])):
            return None

        if (i,j) in visited or grid[i][j] == 0:
            return None

        object_points = [(i,j)]
        visited.add((i,j))
        color = grid[i][j]

        # 遍历相邻点
        for ni, nj in [(i+1,j), (i-1,j), (i,j+1), (i,j-1)]:
            if (ni,nj) not in visited and \
               0 <= ni < len(grid) and \
               0 <= nj < len(grid[0]) and \
               grid[ni][nj] == color:
                sub_obj = self.flood_fill(grid, ni, nj, visited)
                if sub_obj:
                    object_points.extend(sub_obj)

        return object_points

    def extract_features(self, grid):
        """提取grid的特征"""
        features = []

        # 颜色分布
        color_counts = defaultdict(int)
        for row in grid:
            for cell in row:
                color_counts[cell] += 1
        features.append(('color_distribution', color_counts))

        # 对称性
        if self.check_symmetry(grid, 'horizontal'):
            features.append(('symmetry', 'horizontal'))
        if self.check_symmetry(grid, 'vertical'):
            features.append(('symmetry', 'vertical'))

        return features

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
            new_nodes = tree_to_expand.expand_node(leaf, self.dsl_registry.dsl_functions, visited)

            # 检查新节点是否可以连接到另一棵树
            for new_node in new_nodes:
                for other_node in other_tree.all_nodes.values():
                    if self.can_connect(new_node, other_node, is_forward):
                        return (new_node, other_node) if is_forward else (other_node, new_node)

        return None

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
