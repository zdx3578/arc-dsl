import re
from collections import defaultdict

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

def write_file(file_path, lines):
    with open(file_path, 'w') as file:
        file.writelines(lines)

def normalize_function_body(lines, start_index):
    body_lines = []
    for line in lines[start_index:]:
        if line.startswith("def ") or line.startswith("#"):
            break
        # 忽略参数，只保留函数调用和其他代码
        normalized_line = re.sub(r'\(.*?\)', '()', line.strip())
        if normalized_line:
            body_lines.append(normalized_line)
    return body_lines

def find_duplicate_functions(lines):
    function_pattern = re.compile(r'def (\w+)\((.*?)\):')
    functions = defaultdict(list)

    for i, line in enumerate(lines):
        match = function_pattern.match(line)
        if match:
            func_name, func_params = match.groups()
            func_body = normalize_function_body(lines, i + 1)
            func_body_str = "\n".join(func_body)
            functions[func_body_str].append((func_name, func_params, i))

    return functions

def merge_functions(lines, functions):
    merged_lines = []
    merged_lines2 = []
    func_counter = 1
    for func_body, func_list in functions.items():
        if len(func_list) > 1:
            # 保留第一个函数定义，删除其余的
            main_func_name, main_func_params, main_func_index = func_list[0]
            main_func_id = main_func_name.replace('solve_', '')
            merged_lines.append(f"# 第 {func_counter} 个函数  {main_func_id}\n")
            merged_lines2.append(f" {main_func_id}\n")
            merged_lines.append(lines[main_func_index])
            merged_lines.extend(lines[main_func_index + 1:main_func_index + 1 + len(func_body.split('\n'))])
            for func_name, func_params, func_index in func_list[1:]:
                func_id = func_name.replace('solve_', '')
                merged_lines.append(f"# {func_id}({func_params}) 合并到 {main_func_id}\n")
                merged_lines.append(f"{func_name} = {main_func_name}\n")
        else:
            func_name, func_params, func_index = func_list[0]
            func_id = func_name.replace('solve_', '')
            merged_lines.append(f"# 第 {func_counter} 个函数  {func_id}\n")
            merged_lines2.append(f" {func_id}\n")
            merged_lines.append(lines[func_index])
            func_body = normalize_function_body(lines, func_index + 1)
            merged_lines.extend(lines[func_index + 1:func_index + 1 + len(func_body)])
        func_counter += 1
        merged_lines.append("\n\n")  # 添加两行间隔

    return merged_lines, merged_lines2

def sort_functions_by_body(functions):
    sorted_functions = sorted(functions.items(), key=lambda item: item[0])
    return sorted_functions

def main():
    file_path = '../solvers.py'
    lines = read_file(file_path)

    # 保留 import 语句
    import_lines = [line for line in lines if line.startswith("import ") or line.startswith("from ")]

    functions = find_duplicate_functions(lines)
    sorted_functions = sort_functions_by_body(functions)
    merged_lines, merged_lines2  = merge_functions(lines, dict(sorted_functions))

    # 将 import 语句添加到合并后的文件开头
    merged_lines = import_lines + ["\n"] + merged_lines

    write_file('solvers_merged2.py', merged_lines)
    write_file('solvers_merged3.txt', merged_lines2)

if __name__ == "__main__":
    main()