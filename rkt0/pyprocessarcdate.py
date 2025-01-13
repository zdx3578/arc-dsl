#!/usr/bin/env python3
# process_json.py

import sys
import json
from dsl import objects  # 假设 dsl.py 在同目录，里面有 objects 函数

def main():
    # 1. 从命令行参数获取 JSON 文件路径
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No JSON file path provided"}))
        sys.exit(1)

    json_path = sys.argv[1]

    try:
        # 2. 读取并解析 JSON 文件内容
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 假设 JSON 中有这几个键值 (根据你的需求调整)
        # data = {
        #   "grid": [...],
        #   "univalued": true/false,
        #   "diagonal": true/false,
        #   "without_bg": true/false
        # }

        grid = data["grid"]
        univalued = data["univalued"]
        diagonal = data["diagonal"]
        without_bg = data["without_bg"]

        # 3. 调用你的 objects(...) 函数进行处理
        result = objects(grid, univalued, diagonal, without_bg)

        # 4. 把 result 转为 JSON 输出
        # 如果 result 中包含不可 JSON 序列化的内容，需要自定义转换逻辑
        print(json.dumps(result))

    except Exception as e:
        # 若有报错，把错误信息转成 JSON 输出
        error_obj = {"error": str(e)}
        print(json.dumps(error_obj))
        sys.exit(1)


if __name__ == "__main__":
    main()
