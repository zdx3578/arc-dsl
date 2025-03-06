import torch
import torchhd
import numpy as np
from typing import List, Tuple, Dict

class VSAGridEncoder:
    def __init__(self, dimension=1024):
        """初始化VSA编码器，设置向量维度"""
        self.dimension = dimension
        self.color_vectors = {}  # 缓存颜色的向量表示
        self.position_vectors = {}  # 缓存位置的向量表示
        self.shape_vectors = {}  # 缓存形状的向量表示

    def _get_color_vector(self, color):
        """获取颜色的向量表示，相同颜色返回相同向量"""
        if color not in self.color_vectors:
            self.color_vectors[color] = torchhd.random(self.dimension)
        return self.color_vectors[color]

    def _get_position_vector(self, row, col):
        """获取位置的向量表示"""
        position = (row, col)
        if position not in self.position_vectors:
            # 使用位置编码创建唯一向量
            row_vec = torchhd.random(self.dimension)
            col_vec = torchhd.random(self.dimension)
            # 使用绑定操作结合行列信息
            self.position_vectors[position] = torchhd.bind(row_vec, col_vec)
        return self.position_vectors[position]

    def encode_object(self, obj):
        """编码单个对象（一组像素点）"""
        # 初始化为零向量
        obj_vector = torch.zeros(self.dimension)

        # 遍历对象中的每个点
        for color, (row, col) in obj:
            # 获取颜色和位置的向量表示
            color_vec = self._get_color_vector(color)
            pos_vec = self._get_position_vector(row, col)

            # 绑定颜色和位置信息
            pixel_vec = torchhd.bind(color_vec, pos_vec)

            # 捆绑（累加）到对象向量
            obj_vector = torchhd.bundle(obj_vector, pixel_vec)

        # 归一化
        if torch.norm(obj_vector) > 0:
            obj_vector = obj_vector / torch.norm(obj_vector)

        return obj_vector

    def encode_grid(self, grid):
        """编码整个网格"""
        # 将网格转换为对象
        height, width = len(grid), len(grid[0])
        grid_vector = torch.zeros(self.dimension)

        # 直接编码格子
        for i in range(height):
            for j in range(width):
                color = grid[i][j]
                if color != 0:  # 忽略背景色0
                    color_vec = self._get_color_vector(color)
                    pos_vec = self._get_position_vector(i, j)

                    # 绑定颜色和位置
                    pixel_vec = torchhd.bind(color_vec, pos_vec)

                    # 捆绑到网格向量
                    grid_vector = torchhd.bundle(grid_vector, pixel_vec)

        # 归一化
        if torch.norm(grid_vector) > 0:
            grid_vector = grid_vector / torch.norm(grid_vector)

        return grid_vector

    def encode_object_relation(self, obj1, obj2):
        """编码两个对象之间的关系"""
        obj1_vec = self.encode_object(obj1)
        obj2_vec = self.encode_object(obj2)

        # 使用绑定操作编码关系
        relation_vec = torchhd.bind(obj1_vec, obj2_vec)

        return relation_vec

    def similarity(self, vec1, vec2):
        """计算两个向量的相似度"""
        return torchhd.cosine_similarity(vec1, vec2)

    def transform_encoding(self, obj_vec, transform_name):
        """对对象编码应用变换"""
        # 获取或创建变换向量
        if transform_name not in self.shape_vectors:
            self.shape_vectors[transform_name] = torchhd.random(self.dimension)

        # 绑定变换到对象
        transformed_vec = torchhd.bind(obj_vec, self.shape_vectors[transform_name])

        return transformed_vec

def integrate_with_objutil(encoder, task):
    """将VSA编码器集成到现有的objutil框架"""
    train_data = task['train']
    test_data = task['test']

    # 对训练数据的每个对进行编码
    encoded_pairs = []
    for pair_id, data_pair in enumerate(train_data):
        input_grid = data_pair['input']
        output_grid = data_pair['output']

        # 编码输入输出网格
        input_vec = encoder.encode_grid(input_grid)
        output_vec = encoder.encode_grid(output_grid)

        # 编码变换关系
        transform_vec = torchhd.bind(input_vec, output_vec)

        encoded_pairs.append({
            'pair_id': pair_id,
            'input_vec': input_vec,
            'output_vec': output_vec,
            'transform_vec': transform_vec
        })

    # 检查所有训练对的变换向量是否相似
    if len(encoded_pairs) > 1:
        similarities = []
        for i in range(len(encoded_pairs)-1):
            sim = encoder.similarity(
                encoded_pairs[i]['transform_vec'],
                encoded_pairs[i+1]['transform_vec']
            )
            similarities.append(sim.item())

        # 如果变换向量相似，尝试应用到测试数据
        if all(s > 0.7 for s in similarities):  # 设置相似度阈值
            test_input = test_data[0]['input']
            test_input_vec = encoder.encode_grid(test_input)

            # 应用从训练数据中学到的变换
            pred_output_vec = torchhd.bind(
                test_input_vec,
                torchhd.unbind(encoded_pairs[0]['input_vec'], encoded_pairs[0]['transform_vec'])
            )

            # 这里需要从向量解码回网格，这是VSA的一个挑战
            # 可能需要额外的解码机制如kNN或查找表

    return encoded_pairs