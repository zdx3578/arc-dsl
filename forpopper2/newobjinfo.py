import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append("/Users/zhangdexiang/github/VSAHDC/arc-dsl/")

import json
import inspect
import tqdm
from datetime import datetime

import logging
import traceback

from solvers2 import *
# import solvers_is_judge
import solvers_is_judge as solvers
import tests
import dsl
import constants
import arc_types
from objutil import *




# 将 constants.py 所在路径添加到 Python 路径
sys.path.append(current_dir)
sys.path.append("/Users/zhangdexiang/github/VSAHDC/arc-dsl/")

def get_data(train=True):
    # /home/zdx/github/VSAHDC/arc-agi/data
    path = f'/Users/zhangdexiang/github/arc-agi/data/{
        "training" if train else "evaluation"}'
    data = {}
    for fn in os.listdir(path):
        if not fn.endswith('.json'):
            continue  # 只处理 JSON 文件
        with open(f'{path}/{fn}') as f:
            data[fn.rstrip('.json')] = json.load(f)

    def ast(g): return tuple(tuple(r) for r in g)
    return {
        'train': {k: [{
            'input': ast(e['input']),
            'output': ast(e['output']),
        } for e in v['train']] for k, v in data.items()},
        'test': {k: [{
            'input': ast(e['input']),
            'output': ast(e['output']),
        } for e in v['test']] for k, v in data.items()}
    }



def test_solvers_correctness(data, solvers_module):
    """ tests the implemented solvers for correctness """

    with open('solvers.py', 'r', encoding='utf-8') as file:
        code = file.read()
    pattern = r"def solve_([a-fA-F0-9]+)\(I\):"
    import re
    # 获取所有匹配的函数名
    solvers = re.findall(pattern, code)
    n_correct = 0
    n = len(data["train"])
    # for key in range(1): # tqdm.tqdm(data['train'].keys(), total=n):
    print()
    print()
    print()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(now)
    print()
    print()
    for i, key in enumerate(solvers, start=1):
    # for i, key in enumerate(['a8c38be5',"8403a5d5"]):

        # key = 'a8c38be5'      # in out
        # key =    "b775ac94"
        # key = 'c3f564a4'
        # key = '8403a5d5'  # out out
        # key = "68b16354"  hmirror
        # key = '7468f01a'
        # key = "25d8a9c8"
        key = "d22278a0"

        print(i, key)
        task = {}
        task['train'] = data['train'][key]
        task['test'] = data['test'][key]
        try:
            # solver = getattr(solvers_module, f'solve_{key}')
            # preparetask(task)
            if n_correct % 2 == 0:
                print()
            if process_single_data(task) :

            # solve_arc_task(task)

            # for ex in task['train']:
            #     # prepare_diff(ex['input'],ex['output'])
            #     # assert solver(ex['input']) == ex['output']
            #     assert solver(ex['output']) == ex['output']
                n_correct += 1
                print()
                # print()
                print(n_correct, "  .  success", "  .  .  .  .  .  .  .  .   ", n_correct)
                print()
            print()

            print()

        except Exception as e:
            logging.error("捕获到异常：%s", e)
            logging.error("详细错误信息：\n%s", traceback.format_exc())
    print(f'{n_correct} out of {n} tasks solved correctly.')


def main():
    data = get_data(train=True)
    # run_dsl_tests(dsl, tests)
    # test_solvers_formatting(solvers, dsl)
    test_solvers_correctness(data, solvers)


if __name__ == '__main__':
    main()
