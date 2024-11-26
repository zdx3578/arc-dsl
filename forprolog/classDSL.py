import re
import sys
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

    return functions

def map_type(py_type):
    # 简单映射 Python 类型到 Prolog 类型，可以根据需要扩展
    type_mapping = {
        'Any': 'any',
        'Grid': 'grid',
        'Indices': 'indices',
        'IntegerTuple': 'integertuple',
        'Patch': 'patch',
        'Piece': 'piece',
        'Object': 'object',
        'Integer': 'integer',
        'Box': 'box',
        'Result': 'result'
        # 添加更多类型映射
    }
    return type_mapping.get(py_type, 'any')

def classify_functions(functions):
    classified_functions = defaultdict(list)
    for func_name, params, return_type in functions:
        # 清理参数列表
        param_list = [param.strip() for param in params.replace('\n', '').split(',') if param.strip()]
        param_types = []
        for param in param_list:
            parts = param.split(':')
            if len(parts) == 2:
                typ = parts[1].strip()
                param_types.append(map_type(typ))
            else:
                param_types.append('any')  # 默认类型

        # 获取返回类型
        if return_type:
            return_type = map_type(return_type.strip())
        else:
            return_type = 'any'

        # 分类函数
        key = (tuple(param_types), return_type)
        classified_functions[key].append(func_name)

    return classified_functions

def print_classified_functions(classified_functions):
    total_functions = 0
    for key, funcs in classified_functions.items():
        param_types, return_type = key
        print(f"输入类型: {param_types}, 返回类型: {return_type}")
        print("函数列表:")
        for func in funcs:
            print(f"  - {func}")
        print(f"分类下的函数个数: {len(funcs)}\n")
        total_functions += len(funcs)
    return total_functions

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python dslToBk.py <dsl.py 的路径>")
        sys.exit(1)

    dsl_file = sys.argv[1]

    functions = extract_functions_from_dsl(dsl_file)
    original_function_count = len(functions)
    print(f"原本函数的总个数: {original_function_count}\n")

    if not functions:
        print("在指定的 dsl.py 文件中未找到函数。请检查文件内容。")
    else:
        classified_functions = classify_functions(functions)
        classified_function_count = print_classified_functions(classified_functions)
        print(f"分类后输出的函数总个数: {classified_function_count}")