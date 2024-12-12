import re
import sys
import ast
import json
from collections import defaultdict

def extract_functions_from_dsl(dsl_file):
    with open(dsl_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 正则表达式匹配函数定义，包括多行和类型注解
    function_pattern = re.compile(
        r'def\s+(\w+)\s*\(\s*((?:[^()]|\([^()]*\))*)\s*\)\s*(?:->\s*([\w\[\], ]+))?\s*:',
        re.DOTALL
    )
    functions = function_pattern.findall(content)
    print(f"提取到的函数: {functions}")  # 调试信息
    return functions

def extract_types_from_dsl(dsl_file):
    with open(dsl_file, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=dsl_file)

    types = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # 提取返回类型
            if node.returns:
                if isinstance(node.returns, ast.Name):
                    types.add(node.returns.id)
                elif isinstance(node.returns, ast.Subscript):
                    types.add(ast.unparse(node.returns).replace(' ', ''))
            # 提取参数类型
            for arg in node.args.args:
                if arg.annotation:
                    if isinstance(arg.annotation, ast.Name):
                        types.add(arg.annotation.id)
                    elif isinstance(arg.annotation, ast.Subscript):
                        types.add(ast.unparse(arg.annotation).replace(' ', ''))

    print(f"提取到的类型: {types}")  # 调试信息
    return types

def map_type(py_type, type_mapping):
    # 从动态生成的类型映射字典中获取映射
    mapped_type = type_mapping.get(py_type, 'any')
    print(f"映射类型: {py_type} -> {mapped_type}")  # 调试信息
    return mapped_type

def classify_functions(functions, type_mapping):
    classified_functions = defaultdict(list)
    for func_name, params, return_type in functions:
        # 清理参数列表
        param_list = [param.strip() for param in params.replace('\n', '').split(',') if param.strip()]
        param_types = []
        for param in param_list:
            parts = param.split(':')
            if len(parts) == 2:
                typ = parts[1].strip()
                param_types.append(map_type(typ, type_mapping))
            else:
                param_types.append('any')  # 默认类型

        # 获取返回类型
        if return_type:
            return_type = map_type(return_type.strip(), type_mapping)
        else:
            return_type = 'any'

        # 分类函数
        key = (tuple(param_types), return_type)
        key_str = str(key)  # 将元组键转换为字符串键
        classified_functions[key_str].append(func_name)

    print(f"分类后的函数: {classified_functions}")  # 调试信息
    return classified_functions

def print_classified_functions(classified_functions):
    total_functions = 0
    for key, funcs in classified_functions.items():
        param_types, return_type = eval(key)  # 将字符串键转换回元组键
        print(f"\n\n#输入类型: {param_types}, 返回类型: {return_type}")
        print(f"#分类下的函数个数: {len(funcs)}")
        print("#函数列表:")
        print(f"[")
        for func in funcs:
            print(f"  {func},")
        print(f"]")

        total_functions += len(funcs)
    return total_functions

def generate_type_mapping(types):
    # 自动生成类型映射，假设 Prolog 类型名为小写的 Python 类型名
    # 可以根据需要调整映射规则
    type_mapping = {py_type: py_type.lower() for py_type in types}
    print(f"生成的类型映射: {type_mapping}")  # 调试信息
    return type_mapping

def save_classified_functions(classified_functions, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(classified_functions, f, ensure_ascii=False, indent=4)
    print(f"分类后的函数已保存到 {output_file}")  # 调试信息

def load_classified_functions(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        classified_functions = json.load(f)

    # 将字符串键转换回元组键
    classified_functions = {eval(key): value for key, value in classified_functions.items()}
    print(f"加载的分类函数: {classified_functions}")  # 调试信息
    return classified_functions

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python filename.py <dsl.py 的路径> <输出文件路径>")
        sys.exit(1)

    dsl_file = sys.argv[1]
    output_file = sys.argv[2]

    # 提取函数
    functions = extract_functions_from_dsl(dsl_file)
    original_function_count = len(functions)
    print(f"from dsl import * \nfrom arc_types import * \n\n#原本函数的总个数: {original_function_count}\n")

    if not functions:
        print("在指定的 dsl.py 文件中未找到函数。请检查文件内容。")
    else:
        # 提取所有类型
        types = extract_types_from_dsl(dsl_file)
        # 生成类型映射
        type_mapping = generate_type_mapping(types)
        # 可选：打印类型映射，便于验证
        print("#自动生成的类型映射:")
        for py_type, prolog_type in type_mapping.items():
            print(f"'{py_type}': '{prolog_type}'")
        print()

        # 分类函数
        classified_functions = classify_functions(functions, type_mapping)
        # 保存分类后的函数
        save_classified_functions(classified_functions, output_file)
        print(f"分类后的函数已保存到 {output_file}")



# import json

# class DSLFunctionRegistry:
#     def __init__(self, classified_functions_file):
#         self.classified_functions = self.load_classified_functions(classified_functions_file)

#     def load_classified_functions(self, input_file):
#         with open(input_file, 'r', encoding='utf-8') as f:
#             classified_functions = json.load(f)

#         # 将字符串键转换回元组键
#         classified_functions = {eval(key): value for key, value in classified_functions.items()}
#         print(f"加载的分类函数: {classified_functions}")  # 调试信息
#         return classified_functions

#     def get_functions(self, input_types, output_type):
#         key = str((tuple(input_types), output_type))
#         return self.classified_functions.get(key, [])

#     def call_function(self, func_name, *args):
#         # 动态导入 DSL 文件中的函数并调用
#         module_name = 'dsl_module'  # 替换为实际的 DSL 模块名
#         module = __import__(module_name)
#         func = getattr(module, func_name)
#         return func(*args)

# # 示例使用
# if __name__ == '__main__':
#     classified_functions_file = 'path/to/classified_functions.json'
#     registry = DSLFunctionRegistry(classified_functions_file)

#     # 获取输入类型为 ('grid',) 输出类型为 'grid' 的函数
#     input_types = ['grid']
#     output_type = 'grid'
#     functions = registry.get_functions(input_types, output_type)
#     print(f"Functions with input types {input_types} and output type {output_type}: {functions}")

#     # 调用其中一个函数
#     if functions:
#         func_name = functions[0]
#         result = registry.call_function(func_name, [[0, 0], [0, 1]])
#         print(f"Result of calling {func_name}: {result}")