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

def generate_prolog_predicates(functions):
    prolog_definitions = []
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
        param_names = [param.split(':')[0].strip() for param in param_list]

        # 准备 Prolog 参数
        if return_type:
            inputs = ', '.join(param_names)
            prolog_params = ', '.join(param_names + ['Result'])
            arity = len(param_names) + 1  # 加上输出参数
            directions = ['in'] * len(param_names) + ['out']

            # 定义辅助谓词
            prolog_definitions.append(
                f"{func_name}({prolog_params}) :-\n"
                f"    call_python_function('dsl.{func_name}', [{inputs}], Result)."
            )
        else:
            inputs = ', '.join(param_names)
            prolog_params = ', '.join(param_names)
            arity = len(param_names)
            directions = ['in'] * arity

            # 定义辅助谓词（无返回值）
            prolog_definitions.append(
                f"{func_name}({prolog_params}) :-\n"
                f"    call_python_function('dsl.{func_name}', [{inputs}], _)."
            )

        # 添加 body_pred 和 direction 声明
        prolog_definitions.append(f"body_pred({func_name}, {arity}).")
        prolog_definitions.append(f"direction({func_name}, ({', '.join(directions)})).")
        prolog_definitions.append("")  # 添加空行以增加可读性

    return prolog_definitions

def write_to_bk(prolog_definitions, bk_file):
    with open(bk_file, 'a', encoding='utf-8') as f:
        f.write('\n')
        for line in prolog_definitions:
            f.write(line + '\n')
    print(f"Prolog 代码已写入 {bk_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python generate_bk.py <dsl.py 的路径> [bk.pl 的路径]")
        sys.exit(1)

    dsl_file = sys.argv[1]

    if len(sys.argv) > 2:
        bk_file = sys.argv[2]
    else:
        bk_file = 'bk.pl'

    functions = extract_functions_from_dsl(dsl_file)
    if not functions:
        print("在指定的 dsl.py 文件中未找到函数。请检查文件内容。")
    else:
        prolog_definitions = generate_prolog_predicates(functions)
        write_to_bk(prolog_definitions, bk_file)