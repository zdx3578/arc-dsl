


% 引用 Janus 库以嵌入 Python
:- use_module(library(janus)).

% 声明 discontiguous 谓词以消除警告
:- discontiguous body_pred/2.
:- discontiguous direction/2.

% 初始化 Janus 并导入 dsl 模块和 forpopperbk 模块
:- initialization(py_import('dsl', [as(dsl)])).
:- initialization(py_import('forpopperbk', [as(fpbk)])).

% 定义辅助谓词，例如 fill_box_color

fill_box_color(Input, Output) :-
    py_call(fpbk:fill_box_color(Input), Output).

asindices(Grid, Indices) :-
    py_call(dsl:asindices(Grid), Indices).

box(Indices, Box) :-
    py_call(dsl:box(Indices), Box).

fill(Grid, Value, Indices, FilledGrid) :-
    py_call(dsl:fill((Grid, Value, Indices)), FilledGrid).

% 从 dsl.py 和 forpopperbk.py 文件中提取的函数定义和方向信息

body_pred(fill_box_color, 2).
direction(fill_box_color, (in, out)).

body_pred(asindices, 2).
direction(asindices, (in, out)).

body_pred(box, 2).
direction(box, (in, out)).

body_pred(fill, 4).
direction(fill, (in, in, in, out)).

% 继续添加其他辅助谓词...