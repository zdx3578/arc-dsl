import re
import sys

def extract_functions_from_dsl(dsl_file):
    """
    从 Python dsl 文件中提取函数签名 (名称, 参数, 返回类型)。
    匹配形如: def func_name(params) -> return_type:
    """
    with open(dsl_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 正则表达式：匹配 def func_name(params) -> return_type:
    function_pattern = re.compile(
        r'def\s+(\w+)\s*\(\s*((?:[^()]|\([^()]*\))*)\s*\)\s*(?:->\s*([\w\[\], ]+))?\s*:',
        re.DOTALL
    )
    functions = function_pattern.findall(content)
    return functions

def map_type(py_type):
    """
    直接映射 Python DSL 类型到 Prolog 端的同名或近似名。
    保持 DSL 的语义标签，不强行合并类型。
    """
    # 常见类型的映射表；可根据实际 DSL 中的定义增补
    type_mapping = {
        'Any': 'any',
        'Boolean': 'boolean',
        'Integer': 'integer',
        'IntegerTuple': 'integer_tuple',
        'Numerical': 'numerical',
        'Grid': 'grid',
        'Object': 'object',
        'Indices': 'indices',
        'IndicesSet': 'indices_set',
        'Patch': 'patch',
        'Element': 'element',
        'Piece': 'piece',
        'TupleTuple': 'tupletuple',
        'Container': 'container',
        'ContainerContainer': 'containercontainer',
    }
    # 去除空格/方括号等（非常简单的处理，如果是 Union[...] 等需更复杂解析）
    py_type = py_type.strip().replace('[', '').replace(']', '').split(',')[0]
    return type_mapping.get(py_type, 'any')

def generate_prolog_predicates(functions):
    """
    根据提取到的函数信息，生成:
      - prolog_definitions: 用于 bk.pl 的辅助谓词定义 + body_pred/direction + modeb(...)
      - type_declarations : 用于 bias.pl 的类型声明
    """
    prolog_definitions = []
    type_declarations = []

    for func_name, params, return_type in functions:
        print(f"正在处理函数: {func_name}")
        print(f"参数列表:\n{params.strip()}")
        if return_type:
            print(f"返回类型: {return_type.strip()}")
        else:
            print("返回类型: 无")
        print()

        # 拆分参数
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
                # 无显式类型注解
                name = parts[0].strip()
                param_names.append(name)
                param_types.append('any')

        # 判断返回类型是否存在
        if return_type:
            return_type_prolog = map_type(return_type.strip())
            # Prolog 参数列表 (包含一个Output: Result)
            prolog_params = ', '.join(param_names) + ', Result'
            # 计算谓词元数
            arity = len(param_names) + 1
            # 方向声明
            directions = ['in'] * len(param_names) + ['out']
            # 将返回类型也加入 param_types
            param_types.append(return_type_prolog)
        else:
            prolog_params = ', '.join(param_names)
            arity = len(param_names)
            directions = ['in'] * arity

        # ------ 生成bk.pl中的谓词定义 ------
        # 如果函数有返回类型，就将 call_python_function 调用的结果绑定到Result；否则用 '_'
        if return_type:
            bk_clause = (
                f"{func_name}({prolog_params}) :-\n"
                f"    call_python_function('{func_name}', [{', '.join(param_names)}], Result)."
            )
        else:
            bk_clause = (
                f"{func_name}({prolog_params}) :-\n"
                f"    call_python_function('{func_name}', [{', '.join(param_names)}], _)."
            )
        prolog_definitions.append(bk_clause)

        # body_pred & direction
        prolog_definitions.append(f"body_pred({func_name}, {arity}).")
        prolog_definitions.append(f"direction({func_name}, ({', '.join(directions)})).")

        # ------ 生成 `modeb(*, func_name(...))` 声明 ------
        # 例如: modeb(*, func_name(+grid, -grid)).
        mode_list = []
        for dir_, ptype in zip(directions, param_types):
            sign = '+' if dir_ == 'in' else '-'
            mode_list.append(f"{sign}{ptype}")
        mode_string = ', '.join(mode_list)
        prolog_definitions.append(f"modeb(*, {func_name}({mode_string})).\n")

        # ------ 生成bias.pl中的类型声明 ------
        # 例如: type(func_name, (grid, grid, integer)) 等
        prolog_types = ', '.join(param_types)
        type_decl = f"type({func_name}, ({prolog_types}))."
        type_declarations.append(type_decl)

    return prolog_definitions, type_declarations

def write_to_files(prolog_definitions, type_declarations, bk_file, bias_file):
    """将辅助谓词写入bk_file, 将类型声明写入bias_file。"""
    with open(bk_file, 'a', encoding='utf-8') as f_bk:
        f_bk.write('\n\n% ------------  自动生成的辅助谓词与 modeb 声明  ------------\n\n')
        for line in prolog_definitions:
            f_bk.write(line + '\n')

    with open(bias_file, 'a', encoding='utf-8') as f_bias:
        f_bias.write('\n\n% ------------  自动生成的类型声明  ------------\n\n')
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

    # 提取函数
    functions = extract_functions_from_dsl(dsl_file)
    if not functions:
        print("在指定的 dsl.py 文件中未找到函数。请检查文件内容。")
        sys.exit(0)

    # 生成bk和bias所需的定义
    prolog_definitions, type_declarations = generate_prolog_predicates(functions)

    # 写入文件
    write_to_files(prolog_definitions, type_declarations, bk_file, bias_file)