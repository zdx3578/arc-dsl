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
                    child = node.add_child(new_state, func_name, [node.state])
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

    def find_path(self, input_state, output_state):
        """寻找输入到输出的路径"""
        self.forward_tree = StateTree(input_state)
        self.backward_tree = StateTree(output_state)

        visited_forward = set()
        visited_backward = set()

        while self.forward_tree.leaf_nodes and self.backward_tree.leaf_nodes:
            # 交替扩展两棵树
            if len(self.forward_tree.leaf_nodes) <= len(self.backward_tree.leaf_nodes):
                connection = self.expand_tree(self.forward_tree, self.backward_tree,
                                           visited_forward, is_forward=True)
            else:
                connection = self.expand_tree(self.backward_tree, self.forward_tree,
                                           visited_backward, is_forward=False)

            if connection:
                return self.construct_solution(connection)

        return None

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

        # 构造完整的转换序列
        transformations = []

        # 添加正向转换
        for node in forward_path[1:]:  # 跳过根节点
            transformations.append({
                'function': node.generation_func,
                'params': [p.data for p in node.gen_params]
            })

        # 添加反向转换
        for node in backward_path[1:]:  # 跳过根节点
            transformations.append({
                'function': node.generation_func,
                'params': [p.data for p in node.gen_params]
            })

        return transformations
