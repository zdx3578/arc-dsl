from collections import defaultdict

def bateson_algorithm(data):
    """
    实现 Bateson 算法，处理二维线段数据。
    :param data: frozenset 格式的输入数据，每个元素为 (color, (x, y))
    :return: 排序后的数据、符号化特征列表和概念字典
    """
    # 1. 颜色比较函数
    def compare_colors(c1, c2):
        """
        比较两个颜色值，返回符号化结果。
        :param c1: 第一个颜色值
        :param c2: 第二个颜色值
        :return: 符号化结果 ("=" 或 "!=")
        """
        return "=" if c1 == c2 else "!="

    # 2. 点的比较函数
    def compare_points(p1, p2):
        """
        比较两个点的属性，生成符号化特征。
        :param p1: 第一个点 (color, (x, y))
        :param p2: 第二个点 (color, (x, y))
        :return: 符号化特征集合
        """
        c1, (x1, y1) = p1
        c2, (x2, y2) = p2
        feature = {}

        # 颜色比较
        feature["C"] = compare_colors(c1, c2)

        # X 坐标比较
        if x1 < x2:
            feature["X"] = "<"
        elif x1 > x2:
            feature["X"] = ">"
        else:
            feature["X"] = "="

        # Y 坐标比较
        if y1 < y2:
            feature["Y"] = "<"
        elif y1 > y2:
            feature["Y"] = ">"
        else:
            feature["Y"] = "="

        return frozenset(feature.items())

    # 3. 数据排序
    sorted_data = sorted(data, key=lambda point: (point[1][1], point[1][0]))

    # 4. 符号化特征提取
    features = []
    for i in range(len(sorted_data) - 1):
        features.append(compare_points(sorted_data[i], sorted_data[i + 1]))

    # 5. 生成概念
    concepts = defaultdict(list)
    for i, feature in enumerate(features):
        concepts[feature].append((i, i + 1))

    print("Sorted Data:", sorted_data)
    print("Features:", features)
    print("Concepts:")
    for intent, extent in concepts.items():
        print(f"Intent: {intent}, Extent: {extent}")

    # 返回结果
    return {
        "sorted_data": sorted_data,
        "features": features,
        "concepts": dict(concepts)
    }

# 示例输入数据
# data = frozenset({
#     (0, (1, 2)),
#     (1, (0, 0)),
#     (2, (1, 0)),
#     (3, (2, 0)),
#     (4, (0, 1)),
#     (5, (1, 1)),
#     (6, (2, 1))
# })

# # 调用 Bateson 算法函数
# result = bateson_algorithm(data)

# # 打印结果
# print("Sorted Data:", result["sorted_data"])
# print("Features:", result["features"])
# print("Concepts:")
# for intent, extent in result["concepts"].items():
#     print(f"Intent: {intent}, Extent: {extent}")