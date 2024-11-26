# import re
# import sys

# def extract_functions_from_dsl(dsl_file):
#     with open(dsl_file, 'r', encoding='utf-8') as f:
#         content = f.read()

#     # 正则表达式匹配函数定义，包括多行和类型注解
#     function_pattern = re.compile(
#         r'def\s+(\w+)\s*\(\s*((?:[^()]|\([^()]*\))*)\s*\)\s*(?:->\s*([\w\[\], ]+))?\s*:',
#         re.DOTALL
#     )
#     functions = function_pattern.findall(content)

#     return functions

# def generate_prolog_predicates(functions):
#     prolog_definitions = []
#     type_declarations = []
#     for func_name, params, return_type in functions:
#         # 清理参数列表
#         param_list = [param.strip() for param in params.replace('\n', '').split(',') if param.strip()]
#         param_names = []
#         param_types = []
#         for param in param_list:
#             parts = param.split(':')
#             if len(parts) == 2:
#                 name = parts[0].strip()
#                 typ = parts[1].strip()
#                 param_names.append(name)
#                 param_types.append(typ.lower())
#             else:
#                 param_names.append(parts[0].strip())
#                 param_types.append('any')  # 默认类型

#         # 准备 Prolog 参数
#         prolog_params = ', '.join(param_names)
#         arity = len(param_names)
#         directions = ['in'] * arity

#         # 如果有返回类型，添加输出参数
#         if return_type:
#             return_type = return_type.strip().lower()
#             prolog_params += ', Result'
#             param_types.append(return_type)
#             directions.append('out')
#             arity += 1
#         else:
#             return_type = 'any'

#         # 定义辅助谓词
#         inputs = ', '.join(param_names)
#         prolog_definitions.append(
#             f"{func_name}({prolog_params}) :-\n"
#             f"    call_python_function('dsl.{func_name}', [{inputs}], Result)."
#         )
#         # 添加 body_pred 和 direction 声明
#         prolog_definitions.append(f"body_pred({func_name}, {arity}).")
#         prolog_definitions.append(f"direction({func_name}, ({', '.join(directions)})).")
#         prolog_definitions.append("")  # 添加空行以增加可读性

#         # 添加类型声明
#         prolog_types = ', '.join(param_types)
#         type_declarations.append(f"type({func_name}, ({prolog_types})).")

#     return prolog_definitions, type_declarations

# def write_to_files(prolog_definitions, type_declarations, bk_file, bias_file):
#     with open(bk_file, 'a', encoding='utf-8') as f_bk:
#         f_bk.write('\n')
#         for line in prolog_definitions:
#             f_bk.write(line + '\n')

#     with open(bias_file, 'a', encoding='utf-8') as f_bias:
#         f_bias.write('\n')
#         for line in type_declarations:
#             f_bias.write(line + '\n')

#     print(f"辅助谓词已写入 {bk_file}")
#     print(f"类型声明已写入 {bias_file}")

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("用法: python dslToBk.py <dsl.py 的路径> [bk.pl 的路径] [bias.pl 的路径]")
#         sys.exit(1)

#     dsl_file = sys.argv[1]
#     bk_file = sys.argv[2] if len(sys.argv) > 2 else 'bk.pl'
#     bias_file = sys.argv[3] if len(sys.argv) > 3 else 'bias.pl'

#     functions = extract_functions_from_dsl(dsl_file)
#     if not functions:
#         print("在指定的 dsl.py 文件中未找到函数。请检查文件内容。")
#     else:
#         prolog_definitions, type_declarations = generate_prolog_predicates(functions)
#         write_to_files(prolog_definitions, type_declarations, bk_file, bias_file)






import re
import sys

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

def generate_prolog_predicates(functions):
    prolog_definitions = []
    type_declarations = []
    for func_name, params, return_type in functions:
        # 打印当前处理的函数，便于检查
        print(f"正在处理函数: {func_name}")
        print(f"参数列表:\n{params.strip()}")
        if return_type:
            print(f"返回类型: {return_type.strip()}")
        else:
            print("返回类型: 无")
        print()

        # 清理参数列表
        param_list = [param.strip() for param in params.replace('\n', '').split(',') if param.strip()]
        param_names = []
        param_types = []
        for param in param_list:
            parts = param.split(':')
            if len(parts) == 2:
                name = parts[0].strip()
                typ = parts[1].strip()
                param_names.append(name)
                prolog_type = map_type(typ)
                param_types.append(prolog_type)
            else:
                name = parts[0].strip()
                param_names.append(name)
                param_types.append('any')  # 默认类型

        # 准备 Prolog 参数和方向
        if return_type:
            prolog_params = ', '.join(param_names) + ', Result'
            arity = len(param_names) + 1
            directions = ['in'] * len(param_names) + ['out']
            return_type_prolog = map_type(return_type.strip())
            param_types.append(return_type_prolog)
        else:
            prolog_params = ', '.join(param_names)
            arity = len(param_names)
            directions = ['in'] * arity

        # 定义辅助谓词
        if return_type:
            prolog_definitions.append(
                f"{func_name}({prolog_params}) :-\n"
                f"    call_python_function('{func_name}', [{', '.join(param_names)}], Result)."
            )
        else:
            prolog_definitions.append(
                f"{func_name}({prolog_params}) :-\n"
                f"    call_python_function('{func_name}', [{', '.join(param_names)}], _)."
            )

        # 添加 body_pred 和 direction 声明
        prolog_definitions.append(f"body_pred({func_name}, {arity}).")
        prolog_definitions.append(f"direction({func_name}, ({', '.join(directions)})).")
        prolog_definitions.append("")  # 添加空行以增加可读性

        # 添加类型声明
        prolog_types = ', '.join(param_types)
        type_declarations.append(f"type({func_name}, ({prolog_types})).")

    return prolog_definitions, type_declarations

def write_to_files(prolog_definitions, type_declarations, bk_file, bias_file):
    with open(bk_file, 'a', encoding='utf-8') as f_bk:
        f_bk.write('\n')
        for line in prolog_definitions:
            f_bk.write(line + '\n')

    with open(bias_file, 'a', encoding='utf-8') as f_bias:
        f_bias.write('\n')
        for line in type_declarations:
            f_bias.write(line + '\n')

    print(f"辅助谓词已写入 {bk_file}")
    print(f"类型声明已写入 {bias_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python dslToBk.py <dsl.py 的路径> [bk.pl 的路径] [bias.pl 的路径]")
        sys.exit(1)

    dsl_file = sys.argv[1]
    bk_file = sys.argv[2] if len(sys.argv) > 2 else 'bk.pl'
    bias_file = sys.argv[3] if len(sys.argv) > 3 else 'bias.pl'

    functions = extract_functions_from_dsl(dsl_file)
    if not functions:
        print("在指定的 dsl.py 文件中未找到函数。请检查文件内容。")
    else:
        prolog_definitions, type_declarations = generate_prolog_predicates(functions)
        write_to_files(prolog_definitions, type_declarations, bk_file, bias_file)