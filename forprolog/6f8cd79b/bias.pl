max_vars(7).
max_body(10).

% 定义目标谓词
head_pred(program, 2).
type(program, (grid, grid)).
direction(program, (in, out)).

% 定义辅助谓词
body_pred(asindices, 2).
type(asindices, (grid, indices)).
direction(asindices, (in, out)).

body_pred(box, 2).
type(box, (indices, indices)).
direction(box, (in, out)).

body_pred(fill, 4).
type(fill, (grid, integer, indices, grid)).
direction(fill, (in, in, in, out)).

body_pred(fill_box_color, 2).
type(fill_box_color, (grid, grid)). % 请根据实际类型替换 input_type 和 output_type
direction(fill_box_color, (in, out)).

% 继续为其他辅助谓词添加类型和方向声明