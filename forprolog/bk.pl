% 引用 py_swip 库以调用 Python 函数
:- use_module(library(pyswip)).

% 导入 dsl 模块
:- py_import_module('dsl').

% 通用调用 Python 函数的谓词

call_python_function(Function, Input, Output) :-
	py_call(Function, [Input], Output).


% ------------  自动生成的辅助谓词与 modeb 声明  ------------

identity(x, Result) :-
    call_python_function('identity', [x], Result).
body_pred(identity, 2).
direction(identity, (in, out)).
modeb(*, identity(+any, -any)).

add(a, b, Result) :-
    call_python_function('add', [a, b], Result).
body_pred(add, 3).
direction(add, (in, in, out)).
modeb(*, add(+numerical, +numerical, -numerical)).

subtract(a, b, Result) :-
    call_python_function('subtract', [a, b], Result).
body_pred(subtract, 3).
direction(subtract, (in, in, out)).
modeb(*, subtract(+numerical, +numerical, -numerical)).

multiply(a, b, Result) :-
    call_python_function('multiply', [a, b], Result).
body_pred(multiply, 3).
direction(multiply, (in, in, out)).
modeb(*, multiply(+numerical, +numerical, -numerical)).

divide(a, b, Result) :-
    call_python_function('divide', [a, b], Result).
body_pred(divide, 3).
direction(divide, (in, in, out)).
modeb(*, divide(+numerical, +numerical, -numerical)).

invert(n, Result) :-
    call_python_function('invert', [n], Result).
body_pred(invert, 2).
direction(invert, (in, out)).
modeb(*, invert(+numerical, -numerical)).

even(n, Result) :-
    call_python_function('even', [n], Result).
body_pred(even, 2).
direction(even, (in, out)).
modeb(*, even(+integer, -boolean)).

double(n, Result) :-
    call_python_function('double', [n], Result).
body_pred(double, 2).
direction(double, (in, out)).
modeb(*, double(+numerical, -numerical)).

halve(n, Result) :-
    call_python_function('halve', [n], Result).
body_pred(halve, 2).
direction(halve, (in, out)).
modeb(*, halve(+numerical, -numerical)).

flip(b, Result) :-
    call_python_function('flip', [b], Result).
body_pred(flip, 2).
direction(flip, (in, out)).
modeb(*, flip(+boolean, -boolean)).

equality(a, b, Result) :-
    call_python_function('equality', [a, b], Result).
body_pred(equality, 3).
direction(equality, (in, in, out)).
modeb(*, equality(+any, +any, -boolean)).

contained(value, container, Result) :-
    call_python_function('contained', [value, container], Result).
body_pred(contained, 3).
direction(contained, (in, in, out)).
modeb(*, contained(+any, +container, -boolean)).

combine(a, b, Result) :-
    call_python_function('combine', [a, b], Result).
body_pred(combine, 3).
direction(combine, (in, in, out)).
modeb(*, combine(+container, +container, -container)).

intersection(a, b, Result) :-
    call_python_function('intersection', [a, b], Result).
body_pred(intersection, 3).
direction(intersection, (in, in, out)).
modeb(*, intersection(+any, +any, -any)).

difference(a, b, Result) :-
    call_python_function('difference', [a, b], Result).
body_pred(difference, 3).
direction(difference, (in, in, out)).
modeb(*, difference(+any, +any, -any)).

advanced_difference(a, b, Result) :-
    call_python_function('advanced_difference', [a, b], Result).
body_pred(advanced_difference, 3).
direction(advanced_difference, (in, in, out)).
modeb(*, advanced_difference(+any, +any, -any)).

dedupe(tup, Result) :-
    call_python_function('dedupe', [tup], Result).
body_pred(dedupe, 2).
direction(dedupe, (in, out)).
modeb(*, dedupe(+any, -any)).

order(container, compfunc, Result) :-
    call_python_function('order', [container, compfunc], Result).
body_pred(order, 3).
direction(order, (in, in, out)).
modeb(*, order(+container, +any, -any)).

repeat(item, num, Result) :-
    call_python_function('repeat', [item, num], Result).
body_pred(repeat, 3).
direction(repeat, (in, in, out)).
modeb(*, repeat(+any, +integer, -any)).

greater(a, b, Result) :-
    call_python_function('greater', [a, b], Result).
body_pred(greater, 3).
direction(greater, (in, in, out)).
modeb(*, greater(+integer, +integer, -boolean)).

size(container, Result) :-
    call_python_function('size', [container], Result).
body_pred(size, 2).
direction(size, (in, out)).
modeb(*, size(+container, -integer)).

merge(containers, Result) :-
    call_python_function('merge', [containers], Result).
body_pred(merge, 2).
direction(merge, (in, out)).
modeb(*, merge(+containercontainer, -container)).

maximum(container, Result) :-
    call_python_function('maximum', [container], Result).
body_pred(maximum, 2).
direction(maximum, (in, out)).
modeb(*, maximum(+any, -integer)).

minimum(container, Result) :-
    call_python_function('minimum', [container], Result).
body_pred(minimum, 2).
direction(minimum, (in, out)).
modeb(*, minimum(+any, -integer)).

valmax(container, compfunc, Result) :-
    call_python_function('valmax', [container, compfunc], Result).
body_pred(valmax, 3).
direction(valmax, (in, in, out)).
modeb(*, valmax(+container, +any, -integer)).

valmin(container, compfunc, Result) :-
    call_python_function('valmin', [container, compfunc], Result).
body_pred(valmin, 3).
direction(valmin, (in, in, out)).
modeb(*, valmin(+container, +any, -integer)).

argmax(container, compfunc, Result) :-
    call_python_function('argmax', [container, compfunc], Result).
body_pred(argmax, 3).
direction(argmax, (in, in, out)).
modeb(*, argmax(+container, +any, -any)).

argmin(container, compfunc, Result) :-
    call_python_function('argmin', [container, compfunc], Result).
body_pred(argmin, 3).
direction(argmin, (in, in, out)).
modeb(*, argmin(+container, +any, -any)).

mostcommon(container, Result) :-
    call_python_function('mostcommon', [container], Result).
body_pred(mostcommon, 2).
direction(mostcommon, (in, out)).
modeb(*, mostcommon(+container, -any)).

leastcommon(container, Result) :-
    call_python_function('leastcommon', [container], Result).
body_pred(leastcommon, 2).
direction(leastcommon, (in, out)).
modeb(*, leastcommon(+container, -any)).

initset(value, Result) :-
    call_python_function('initset', [value], Result).
body_pred(initset, 2).
direction(initset, (in, out)).
modeb(*, initset(+any, -any)).

both(a, b, Result) :-
    call_python_function('both', [a, b], Result).
body_pred(both, 3).
direction(both, (in, in, out)).
modeb(*, both(+boolean, +boolean, -boolean)).

either(a, b, Result) :-
    call_python_function('either', [a, b], Result).
body_pred(either, 3).
direction(either, (in, in, out)).
modeb(*, either(+boolean, +boolean, -boolean)).

increment(x, Result) :-
    call_python_function('increment', [x], Result).
body_pred(increment, 2).
direction(increment, (in, out)).
modeb(*, increment(+numerical, -numerical)).

decrement(x, Result) :-
    call_python_function('decrement', [x], Result).
body_pred(decrement, 2).
direction(decrement, (in, out)).
modeb(*, decrement(+numerical, -numerical)).

crement(x, Result) :-
    call_python_function('crement', [x], Result).
body_pred(crement, 2).
direction(crement, (in, out)).
modeb(*, crement(+numerical, -numerical)).

sign(x, Result) :-
    call_python_function('sign', [x], Result).
body_pred(sign, 2).
direction(sign, (in, out)).
modeb(*, sign(+numerical, -numerical)).

positive(x, Result) :-
    call_python_function('positive', [x], Result).
body_pred(positive, 2).
direction(positive, (in, out)).
modeb(*, positive(+integer, -boolean)).

toivec(#     i, Result) :-
    call_python_function('toivec', [#     i], Result).
body_pred(toivec, 2).
direction(toivec, (in, out)).
modeb(*, toivec(+any, -integer_tuple)).

tojvec(#     j, Result) :-
    call_python_function('tojvec', [#     j], Result).
body_pred(tojvec, 2).
direction(tojvec, (in, out)).
modeb(*, tojvec(+any, -integer_tuple)).

sfilter(container, condition, Result) :-
    call_python_function('sfilter', [container, condition], Result).
body_pred(sfilter, 3).
direction(sfilter, (in, in, out)).
modeb(*, sfilter(+container, +any, -container)).

mfilter(container, function, Result) :-
    call_python_function('mfilter', [container, function], Result).
body_pred(mfilter, 3).
direction(mfilter, (in, in, out)).
modeb(*, mfilter(+container, +any, -any)).

extract(container, condition, Result) :-
    call_python_function('extract', [container, condition], Result).
body_pred(extract, 3).
direction(extract, (in, in, out)).
modeb(*, extract(+container, +any, -any)).

totuple(container, Result) :-
    call_python_function('totuple', [container], Result).
body_pred(totuple, 2).
direction(totuple, (in, out)).
modeb(*, totuple(+any, -any)).

first(container, Result) :-
    call_python_function('first', [container], Result).
body_pred(first, 2).
direction(first, (in, out)).
modeb(*, first(+container, -any)).

last(container, Result) :-
    call_python_function('last', [container], Result).
body_pred(last, 2).
direction(last, (in, out)).
modeb(*, last(+container, -any)).

insert(value, container, Result) :-
    call_python_function('insert', [value, container], Result).
body_pred(insert, 3).
direction(insert, (in, in, out)).
modeb(*, insert(+any, +any, -any)).

remove(value, container, Result) :-
    call_python_function('remove', [value, container], Result).
body_pred(remove, 3).
direction(remove, (in, in, out)).
modeb(*, remove(+any, +container, -container)).

other(container, value, Result) :-
    call_python_function('other', [container, value], Result).
body_pred(other, 3).
direction(other, (in, in, out)).
modeb(*, other(+container, +any, -any)).

interval(start, stop, step, Result) :-
    call_python_function('interval', [start, stop, step], Result).
body_pred(interval, 4).
direction(interval, (in, in, in, out)).
modeb(*, interval(+integer, +integer, +integer, -any)).

astuple(a, b, Result) :-
    call_python_function('astuple', [a, b], Result).
body_pred(astuple, 3).
direction(astuple, (in, in, out)).
modeb(*, astuple(+integer, +integer, -integer_tuple)).

product(a, b, Result) :-
    call_python_function('product', [a, b], Result).
body_pred(product, 3).
direction(product, (in, in, out)).
modeb(*, product(+container, +container, -any)).

pair(a, b, Result) :-
    call_python_function('pair', [a, b], Result).
body_pred(pair, 3).
direction(pair, (in, in, out)).
modeb(*, pair(+any, +any, -tupletuple)).

branch(condition, a, b, Result) :-
    call_python_function('branch', [condition, a, b], Result).
body_pred(branch, 4).
direction(branch, (in, in, in, out)).
modeb(*, branch(+boolean, +any, +any, -any)).

compose(outer, inner, Result) :-
    call_python_function('compose', [outer, inner], Result).
body_pred(compose, 3).
direction(compose, (in, in, out)).
modeb(*, compose(+any, +any, -any)).

chain(h, g, f, Result) :-
    call_python_function('chain', [h, g, f], Result).
body_pred(chain, 4).
direction(chain, (in, in, in, out)).
modeb(*, chain(+any, +any, +any, -any)).

matcher(function, target, Result) :-
    call_python_function('matcher', [function, target], Result).
body_pred(matcher, 3).
direction(matcher, (in, in, out)).
modeb(*, matcher(+any, +any, -any)).

rbind(function, fixed, Result) :-
    call_python_function('rbind', [function, fixed], Result).
body_pred(rbind, 3).
direction(rbind, (in, in, out)).
modeb(*, rbind(+any, +any, -any)).

lbind(function, fixed, Result) :-
    call_python_function('lbind', [function, fixed], Result).
body_pred(lbind, 3).
direction(lbind, (in, in, out)).
modeb(*, lbind(+any, +any, -any)).

power(function, n, Result) :-
    call_python_function('power', [function, n], Result).
body_pred(power, 3).
direction(power, (in, in, out)).
modeb(*, power(+any, +integer, -any)).

fork(outer, a, b, Result) :-
    call_python_function('fork', [outer, a, b], Result).
body_pred(fork, 4).
direction(fork, (in, in, in, out)).
modeb(*, fork(+any, +any, +any, -any)).

apply(function, container, Result) :-
    call_python_function('apply', [function, container], Result).
body_pred(apply, 3).
direction(apply, (in, in, out)).
modeb(*, apply(+any, +container, -container)).

rapply(functions, value, Result) :-
    call_python_function('rapply', [functions, value], Result).
body_pred(rapply, 3).
direction(rapply, (in, in, out)).
modeb(*, rapply(+container, +any, -container)).

mapply(function, container, Result) :-
    call_python_function('mapply', [function, container], Result).
body_pred(mapply, 3).
direction(mapply, (in, in, out)).
modeb(*, mapply(+any, +containercontainer, -any)).

papply(function, a, b, Result) :-
    call_python_function('papply', [function, a, b], Result).
body_pred(papply, 4).
direction(papply, (in, in, in, out)).
modeb(*, papply(+any, +any, +any, -any)).

mpapply(function, a, b, Result) :-
    call_python_function('mpapply', [function, a, b], Result).
body_pred(mpapply, 4).
direction(mpapply, (in, in, in, out)).
modeb(*, mpapply(+any, +any, +any, -any)).

prapply(function, a, b, Result) :-
    call_python_function('prapply', [function, a, b], Result).
body_pred(prapply, 4).
direction(prapply, (in, in, in, out)).
modeb(*, prapply(+any, +container, +container, -any)).

mostcolor(element, Result) :-
    call_python_function('mostcolor', [element], Result).
body_pred(mostcolor, 2).
direction(mostcolor, (in, out)).
modeb(*, mostcolor(+element, -integer)).

leastcolor(element, Result) :-
    call_python_function('leastcolor', [element], Result).
body_pred(leastcolor, 2).
direction(leastcolor, (in, out)).
modeb(*, leastcolor(+element, -integer)).

height(piece, Result) :-
    call_python_function('height', [piece], Result).
body_pred(height, 2).
direction(height, (in, out)).
modeb(*, height(+piece, -integer)).

width(piece, Result) :-
    call_python_function('width', [piece], Result).
body_pred(width, 2).
direction(width, (in, out)).
modeb(*, width(+piece, -integer)).

hwratio(piece, Result) :-
    call_python_function('hwratio', [piece], Result).
body_pred(hwratio, 2).
direction(hwratio, (in, out)).
modeb(*, hwratio(+piece, -integer)).

hratio(piece, piece2, Result) :-
    call_python_function('hratio', [piece, piece2], Result).
body_pred(hratio, 3).
direction(hratio, (in, in, out)).
modeb(*, hratio(+piece, +piece, -integer)).

wratio(piece, piece2, Result) :-
    call_python_function('wratio', [piece, piece2], Result).
body_pred(wratio, 3).
direction(wratio, (in, in, out)).
modeb(*, wratio(+piece, +piece, -integer)).

hratioI(piece, piece2, Result) :-
    call_python_function('hratioI', [piece, piece2], Result).
body_pred(hratioI, 3).
direction(hratioI, (in, in, out)).
modeb(*, hratioI(+piece, +piece, -integer)).

wratioI(piece, piece2, Result) :-
    call_python_function('wratioI', [piece, piece2], Result).
body_pred(wratioI, 3).
direction(wratioI, (in, in, out)).
modeb(*, wratioI(+piece, +piece, -integer)).

shape(piece, Result) :-
    call_python_function('shape', [piece], Result).
body_pred(shape, 2).
direction(shape, (in, out)).
modeb(*, shape(+piece, -integer_tuple)).

portrait(piece, Result) :-
    call_python_function('portrait', [piece], Result).
body_pred(portrait, 2).
direction(portrait, (in, out)).
modeb(*, portrait(+piece, -boolean)).

colorcount(element, value, Result) :-
    call_python_function('colorcount', [element, value], Result).
body_pred(colorcount, 3).
direction(colorcount, (in, in, out)).
modeb(*, colorcount(+element, +integer, -integer)).

all_colorcount(element, Result) :-
    call_python_function('all_colorcount', [element], Result).
body_pred(all_colorcount, 2).
direction(all_colorcount, (in, out)).
modeb(*, all_colorcount(+element, -any)).

colorfilter(objs, value, Result) :-
    call_python_function('colorfilter', [objs, value], Result).
body_pred(colorfilter, 3).
direction(colorfilter, (in, in, out)).
modeb(*, colorfilter(+any, +integer, -any)).

sizefilter(container, n, Result) :-
    call_python_function('sizefilter', [container, n], Result).
body_pred(sizefilter, 3).
direction(sizefilter, (in, in, out)).
modeb(*, sizefilter(+container, +integer, -any)).

asindices_patch(patch, Result) :-
    call_python_function('asindices_patch', [patch], Result).
body_pred(asindices_patch, 2).
direction(asindices_patch, (in, out)).
modeb(*, asindices_patch(+patch, -indices)).

asindices(grid, Result) :-
    call_python_function('asindices', [grid], Result).
body_pred(asindices, 2).
direction(asindices, (in, out)).
modeb(*, asindices(+grid, -indices)).

ofcolor(grid, value, Result) :-
    call_python_function('ofcolor', [grid, value], Result).
body_pred(ofcolor, 3).
direction(ofcolor, (in, in, out)).
modeb(*, ofcolor(+grid, +integer, -indices)).

ulcorner(patch, Result) :-
    call_python_function('ulcorner', [patch], Result).
body_pred(ulcorner, 2).
direction(ulcorner, (in, out)).
modeb(*, ulcorner(+patch, -integer_tuple)).

urcorner(patch, Result) :-
    call_python_function('urcorner', [patch], Result).
body_pred(urcorner, 2).
direction(urcorner, (in, out)).
modeb(*, urcorner(+patch, -integer_tuple)).

llcorner(patch, Result) :-
    call_python_function('llcorner', [patch], Result).
body_pred(llcorner, 2).
direction(llcorner, (in, out)).
modeb(*, llcorner(+patch, -integer_tuple)).

lrcorner(patch, Result) :-
    call_python_function('lrcorner', [patch], Result).
body_pred(lrcorner, 2).
direction(lrcorner, (in, out)).
modeb(*, lrcorner(+patch, -integer_tuple)).

crop(grid, start, dims, Result) :-
    call_python_function('crop', [grid, start, dims], Result).
body_pred(crop, 4).
direction(crop, (in, in, in, out)).
modeb(*, crop(+grid, +integer_tuple, +integer_tuple, -grid)).

toindices(patch, Result) :-
    call_python_function('toindices', [patch], Result).
body_pred(toindices, 2).
direction(toindices, (in, out)).
modeb(*, toindices(+patch, -indices)).

recolor(value, patch, Result) :-
    call_python_function('recolor', [value, patch], Result).
body_pred(recolor, 3).
direction(recolor, (in, in, out)).
modeb(*, recolor(+integer, +patch, -object)).

shift(patch, directions, Result) :-
    call_python_function('shift', [patch, directions], Result).
body_pred(shift, 3).
direction(shift, (in, in, out)).
modeb(*, shift(+patch, +integer_tuple, -patch)).

normalize(patch, Result) :-
    call_python_function('normalize', [patch], Result).
body_pred(normalize, 2).
direction(normalize, (in, out)).
modeb(*, normalize(+patch, -patch)).

dneighbors(loc, Result) :-
    call_python_function('dneighbors', [loc], Result).
body_pred(dneighbors, 2).
direction(dneighbors, (in, out)).
modeb(*, dneighbors(+integer_tuple, -indices)).

ineighbors(loc, Result) :-
    call_python_function('ineighbors', [loc], Result).
body_pred(ineighbors, 2).
direction(ineighbors, (in, out)).
modeb(*, ineighbors(+integer_tuple, -indices)).

neighbors(loc, Result) :-
    call_python_function('neighbors', [loc], Result).
body_pred(neighbors, 2).
direction(neighbors, (in, out)).
modeb(*, neighbors(+integer_tuple, -indices)).

objects(grid, univalued, diagonal, without_bg, Result) :-
    call_python_function('objects', [grid, univalued, diagonal, without_bg], Result).
body_pred(objects, 5).
direction(objects, (in, in, in, in, out)).
modeb(*, objects(+grid, +boolean, +boolean, +boolean, -any)).

partition(grid, Result) :-
    call_python_function('partition', [grid], Result).
body_pred(partition, 2).
direction(partition, (in, out)).
modeb(*, partition(+grid, -any)).

fgpartition(grid, Result) :-
    call_python_function('fgpartition', [grid], Result).
body_pred(fgpartition, 2).
direction(fgpartition, (in, out)).
modeb(*, fgpartition(+grid, -any)).

uppermost(patch, Result) :-
    call_python_function('uppermost', [patch], Result).
body_pred(uppermost, 2).
direction(uppermost, (in, out)).
modeb(*, uppermost(+patch, -integer)).

lowermost(patch, Result) :-
    call_python_function('lowermost', [patch], Result).
body_pred(lowermost, 2).
direction(lowermost, (in, out)).
modeb(*, lowermost(+patch, -integer)).

leftmost(patch, Result) :-
    call_python_function('leftmost', [patch], Result).
body_pred(leftmost, 2).
direction(leftmost, (in, out)).
modeb(*, leftmost(+patch, -integer)).

rightmost(patch, Result) :-
    call_python_function('rightmost', [patch], Result).
body_pred(rightmost, 2).
direction(rightmost, (in, out)).
modeb(*, rightmost(+patch, -integer)).

square(piece, Result) :-
    call_python_function('square', [piece], Result).
body_pred(square, 2).
direction(square, (in, out)).
modeb(*, square(+piece, -boolean)).

is_square(piece, Result) :-
    call_python_function('is_square', [piece], Result).
body_pred(is_square, 2).
direction(is_square, (in, out)).
modeb(*, is_square(+piece, -boolean)).

vline(patch, Result) :-
    call_python_function('vline', [patch], Result).
body_pred(vline, 2).
direction(vline, (in, out)).
modeb(*, vline(+patch, -boolean)).

hline(patch, Result) :-
    call_python_function('hline', [patch], Result).
body_pred(hline, 2).
direction(hline, (in, out)).
modeb(*, hline(+patch, -boolean)).

sorted_frozenset(fset, Result) :-
    call_python_function('sorted_frozenset', [fset], Result).
body_pred(sorted_frozenset, 2).
direction(sorted_frozenset, (in, out)).
modeb(*, sorted_frozenset(+any, -any)).

hmatching(a, b, Result) :-
    call_python_function('hmatching', [a, b], Result).
body_pred(hmatching, 3).
direction(hmatching, (in, in, out)).
modeb(*, hmatching(+patch, +patch, -boolean)).

vmatching(a, b, Result) :-
    call_python_function('vmatching', [a, b], Result).
body_pred(vmatching, 3).
direction(vmatching, (in, in, out)).
modeb(*, vmatching(+patch, +patch, -boolean)).

manhattan(a, b, diagonal, Result) :-
    call_python_function('manhattan', [a, b, diagonal], Result).
body_pred(manhattan, 4).
direction(manhattan, (in, in, in, out)).
modeb(*, manhattan(+patch, +patch, +any, -integer)).

adjacent(a, b, diagonal, Result) :-
    call_python_function('adjacent', [a, b, diagonal], Result).
body_pred(adjacent, 4).
direction(adjacent, (in, in, in, out)).
modeb(*, adjacent(+patch, +patch, +any, -boolean)).

bordering(patch, grid, Result) :-
    call_python_function('bordering', [patch, grid], Result).
body_pred(bordering, 3).
direction(bordering, (in, in, out)).
modeb(*, bordering(+patch, +grid, -boolean)).

centerofmass(patch, Result) :-
    call_python_function('centerofmass', [patch], Result).
body_pred(centerofmass, 2).
direction(centerofmass, (in, out)).
modeb(*, centerofmass(+patch, -integer_tuple)).

palette(element, Result) :-
    call_python_function('palette', [element], Result).
body_pred(palette, 2).
direction(palette, (in, out)).
modeb(*, palette(+element, -any)).

numcolors(element, Result) :-
    call_python_function('numcolors', [element], Result).
body_pred(numcolors, 2).
direction(numcolors, (in, out)).
modeb(*, numcolors(+element, -any)).

color(obj, Result) :-
    call_python_function('color', [obj], Result).
body_pred(color, 2).
direction(color, (in, out)).
modeb(*, color(+object, -integer)).

toobject(patch, grid, Result) :-
    call_python_function('toobject', [patch, grid], Result).
body_pred(toobject, 3).
direction(toobject, (in, in, out)).
modeb(*, toobject(+patch, +grid, -object)).

asobject(grid, Result) :-
    call_python_function('asobject', [grid], Result).
body_pred(asobject, 2).
direction(asobject, (in, out)).
modeb(*, asobject(+grid, -object)).

object_to_grid(obj, Result) :-
    call_python_function('object_to_grid', [obj], Result).
body_pred(object_to_grid, 2).
direction(object_to_grid, (in, out)).
modeb(*, object_to_grid(+object, -grid)).

rot90(grid, Result) :-
    call_python_function('rot90', [grid], Result).
body_pred(rot90, 2).
direction(rot90, (in, out)).
modeb(*, rot90(+grid, -grid)).

rot180(grid, Result) :-
    call_python_function('rot180', [grid], Result).
body_pred(rot180, 2).
direction(rot180, (in, out)).
modeb(*, rot180(+grid, -grid)).

upper_third(grid, Result) :-
    call_python_function('upper_third', [grid], Result).
body_pred(upper_third, 2).
direction(upper_third, (in, out)).
modeb(*, upper_third(+grid, -grid)).

middle_third(grid, Result) :-
    call_python_function('middle_third', [grid], Result).
body_pred(middle_third, 2).
direction(middle_third, (in, out)).
modeb(*, middle_third(+grid, -grid)).

lower_third(grid, Result) :-
    call_python_function('lower_third', [grid], Result).
body_pred(lower_third, 2).
direction(lower_third, (in, out)).
modeb(*, lower_third(+grid, -grid)).

left_third(grid, Result) :-
    call_python_function('left_third', [grid], Result).
body_pred(left_third, 2).
direction(left_third, (in, out)).
modeb(*, left_third(+grid, -grid)).

center_third(grid, Result) :-
    call_python_function('center_third', [grid], Result).
body_pred(center_third, 2).
direction(center_third, (in, out)).
modeb(*, center_third(+grid, -grid)).

right_third(grid, Result) :-
    call_python_function('right_third', [grid], Result).
body_pred(right_third, 2).
direction(right_third, (in, out)).
modeb(*, right_third(+grid, -grid)).

rot270(grid, Result) :-
    call_python_function('rot270', [grid], Result).
body_pred(rot270, 2).
direction(rot270, (in, out)).
modeb(*, rot270(+grid, -grid)).

hmirror(piece, Result) :-
    call_python_function('hmirror', [piece], Result).
body_pred(hmirror, 2).
direction(hmirror, (in, out)).
modeb(*, hmirror(+piece, -piece)).

vmirror(piece, Result) :-
    call_python_function('vmirror', [piece], Result).
body_pred(vmirror, 2).
direction(vmirror, (in, out)).
modeb(*, vmirror(+piece, -piece)).

dmirror(piece, Result) :-
    call_python_function('dmirror', [piece], Result).
body_pred(dmirror, 2).
direction(dmirror, (in, out)).
modeb(*, dmirror(+piece, -piece)).

cmirror(piece, Result) :-
    call_python_function('cmirror', [piece], Result).
body_pred(cmirror, 2).
direction(cmirror, (in, out)).
modeb(*, cmirror(+piece, -piece)).

fill(grid, value, patch, Result) :-
    call_python_function('fill', [grid, value, patch], Result).
body_pred(fill, 4).
direction(fill, (in, in, in, out)).
modeb(*, fill(+grid, +integer, +patch, -grid)).

paint(grid, obj, Result) :-
    call_python_function('paint', [grid, obj], Result).
body_pred(paint, 3).
direction(paint, (in, in, out)).
modeb(*, paint(+grid, +object, -grid)).

underfill(grid, value, patch, Result) :-
    call_python_function('underfill', [grid, value, patch], Result).
body_pred(underfill, 4).
direction(underfill, (in, in, in, out)).
modeb(*, underfill(+grid, +integer, +patch, -grid)).

underpaint(grid, obj, Result) :-
    call_python_function('underpaint', [grid, obj], Result).
body_pred(underpaint, 3).
direction(underpaint, (in, in, out)).
modeb(*, underpaint(+grid, +object, -grid)).

hupscale(grid, factor, Result) :-
    call_python_function('hupscale', [grid, factor], Result).
body_pred(hupscale, 3).
direction(hupscale, (in, in, out)).
modeb(*, hupscale(+grid, +integer, -grid)).

vupscale(grid, factor, Result) :-
    call_python_function('vupscale', [grid, factor], Result).
body_pred(vupscale, 3).
direction(vupscale, (in, in, out)).
modeb(*, vupscale(+grid, +integer, -grid)).

upscale(element, factor, Result) :-
    call_python_function('upscale', [element, factor], Result).
body_pred(upscale, 3).
direction(upscale, (in, in, out)).
modeb(*, upscale(+element, +integer, -element)).

get_mode(values, float]], Result) :-
    call_python_function('get_mode', [values, float]]], Result).
body_pred(get_mode, 3).
direction(get_mode, (in, in, out)).
modeb(*, get_mode(+any, +any, -any)).

downscale(grid, factor, Result) :-
    call_python_function('downscale', [grid, factor], Result).
body_pred(downscale, 3).
direction(downscale, (in, in, out)).
modeb(*, downscale(+grid, +integer, -grid)).

hconcat(a, b, Result) :-
    call_python_function('hconcat', [a, b], Result).
body_pred(hconcat, 3).
direction(hconcat, (in, in, out)).
modeb(*, hconcat(+grid, +grid, -grid)).

vconcat(a, b, Result) :-
    call_python_function('vconcat', [a, b], Result).
body_pred(vconcat, 3).
direction(vconcat, (in, in, out)).
modeb(*, vconcat(+grid, +grid, -grid)).

subgrid(patch, grid, Result) :-
    call_python_function('subgrid', [patch, grid], Result).
body_pred(subgrid, 3).
direction(subgrid, (in, in, out)).
modeb(*, subgrid(+patch, +grid, -grid)).

hsplit(grid, n, Result) :-
    call_python_function('hsplit', [grid, n], Result).
body_pred(hsplit, 3).
direction(hsplit, (in, in, out)).
modeb(*, hsplit(+grid, +integer, -any)).

vsplit(grid, n, Result) :-
    call_python_function('vsplit', [grid, n], Result).
body_pred(vsplit, 3).
direction(vsplit, (in, in, out)).
modeb(*, vsplit(+grid, +integer, -any)).

cellwise(a, b, fallback, Result) :-
    call_python_function('cellwise', [a, b, fallback], Result).
body_pred(cellwise, 4).
direction(cellwise, (in, in, in, out)).
modeb(*, cellwise(+grid, +grid, +integer, -grid)).

replace(grid, replacee, replacer, Result) :-
    call_python_function('replace', [grid, replacee, replacer], Result).
body_pred(replace, 4).
direction(replace, (in, in, in, out)).
modeb(*, replace(+grid, +integer, +integer, -grid)).

switch(grid, a, b, Result) :-
    call_python_function('switch', [grid, a, b], Result).
body_pred(switch, 4).
direction(switch, (in, in, in, out)).
modeb(*, switch(+grid, +integer, +integer, -grid)).

center(patch, Result) :-
    call_python_function('center', [patch], Result).
body_pred(center, 2).
direction(center, (in, out)).
modeb(*, center(+patch, -integer_tuple)).

position(a, b, Result) :-
    call_python_function('position', [a, b], Result).
body_pred(position, 3).
direction(position, (in, in, out)).
modeb(*, position(+patch, +patch, -integer_tuple)).

color_at_location(grid, loc, Result) :-
    call_python_function('color_at_location', [grid, loc], Result).
body_pred(color_at_location, 3).
direction(color_at_location, (in, in, out)).
modeb(*, color_at_location(+grid, +integer_tuple, -integer)).

canvas(value, dimensions, Result) :-
    call_python_function('canvas', [value, dimensions], Result).
body_pred(canvas, 3).
direction(canvas, (in, in, out)).
modeb(*, canvas(+integer, +integer_tuple, -grid)).

corners(patch, Result) :-
    call_python_function('corners', [patch], Result).
body_pred(corners, 2).
direction(corners, (in, out)).
modeb(*, corners(+patch, -indices)).

connect(a, b, Result) :-
    call_python_function('connect', [a, b], Result).
body_pred(connect, 3).
direction(connect, (in, in, out)).
modeb(*, connect(+integer_tuple, +integer_tuple, -indices)).

cover(grid, patch, Result) :-
    call_python_function('cover', [grid, patch], Result).
body_pred(cover, 3).
direction(cover, (in, in, out)).
modeb(*, cover(+grid, +patch, -grid)).

trim(grid, Result) :-
    call_python_function('trim', [grid], Result).
body_pred(trim, 2).
direction(trim, (in, out)).
modeb(*, trim(+grid, -grid)).

move(grid, obj, offset, Result) :-
    call_python_function('move', [grid, obj, offset], Result).
body_pred(move, 4).
direction(move, (in, in, in, out)).
modeb(*, move(+grid, +object, +integer_tuple, -grid)).

tophalf(grid, Result) :-
    call_python_function('tophalf', [grid], Result).
body_pred(tophalf, 2).
direction(tophalf, (in, out)).
modeb(*, tophalf(+grid, -grid)).

bottomhalf(grid, Result) :-
    call_python_function('bottomhalf', [grid], Result).
body_pred(bottomhalf, 2).
direction(bottomhalf, (in, out)).
modeb(*, bottomhalf(+grid, -grid)).

lefthalf(grid, Result) :-
    call_python_function('lefthalf', [grid], Result).
body_pred(lefthalf, 2).
direction(lefthalf, (in, out)).
modeb(*, lefthalf(+grid, -grid)).

righthalf(grid, Result) :-
    call_python_function('righthalf', [grid], Result).
body_pred(righthalf, 2).
direction(righthalf, (in, out)).
modeb(*, righthalf(+grid, -grid)).

vfrontier(#     location, Result) :-
    call_python_function('vfrontier', [#     location], Result).
body_pred(vfrontier, 2).
direction(vfrontier, (in, out)).
modeb(*, vfrontier(+any, -indices)).

hfrontier(#     location, Result) :-
    call_python_function('hfrontier', [#     location], Result).
body_pred(hfrontier, 2).
direction(hfrontier, (in, out)).
modeb(*, hfrontier(+any, -indices)).

backdrop(patch, Result) :-
    call_python_function('backdrop', [patch], Result).
body_pred(backdrop, 2).
direction(backdrop, (in, out)).
modeb(*, backdrop(+patch, -indices)).

delta(patch, Result) :-
    call_python_function('delta', [patch], Result).
body_pred(delta, 2).
direction(delta, (in, out)).
modeb(*, delta(+patch, -indices)).

gravitate(source, destination, Result) :-
    call_python_function('gravitate', [source, destination], Result).
body_pred(gravitate, 3).
direction(gravitate, (in, in, out)).
modeb(*, gravitate(+patch, +patch, -integer_tuple)).

inbox(patch, Result) :-
    call_python_function('inbox', [patch], Result).
body_pred(inbox, 2).
direction(inbox, (in, out)).
modeb(*, inbox(+patch, -indices)).

inbox0(patch, Result) :-
    call_python_function('inbox0', [patch], Result).
body_pred(inbox0, 2).
direction(inbox0, (in, out)).
modeb(*, inbox0(+patch, -indices)).

extract_all_boxes(patch, Result) :-
    call_python_function('extract_all_boxes', [patch], Result).
body_pred(extract_all_boxes, 2).
direction(extract_all_boxes, (in, out)).
modeb(*, extract_all_boxes(+patch, -any)).

outbox(patch, Result) :-
    call_python_function('outbox', [patch], Result).
body_pred(outbox, 2).
direction(outbox, (in, out)).
modeb(*, outbox(+patch, -indices)).

box(patch, Result) :-
    call_python_function('box', [patch], Result).
body_pred(box, 2).
direction(box, (in, out)).
modeb(*, box(+patch, -indices)).

shoot(start, direction, Result) :-
    call_python_function('shoot', [start, direction], Result).
body_pred(shoot, 3).
direction(shoot, (in, in, out)).
modeb(*, shoot(+integer_tuple, +integer_tuple, -indices)).

occurrences(grid, obj, Result) :-
    call_python_function('occurrences', [grid, obj], Result).
body_pred(occurrences, 3).
direction(occurrences, (in, in, out)).
modeb(*, occurrences(+grid, +object, -indices)).

frontiers(#     grid, Result) :-
    call_python_function('frontiers', [#     grid], Result).
body_pred(frontiers, 2).
direction(frontiers, (in, out)).
modeb(*, frontiers(+any, -any)).

frontiers(grid, Result) :-
    call_python_function('frontiers', [grid], Result).
body_pred(frontiers, 2).
direction(frontiers, (in, out)).
modeb(*, frontiers(+grid, -any)).

split_by_frontiers(grid, Result) :-
    call_python_function('split_by_frontiers', [grid], Result).
body_pred(split_by_frontiers, 2).
direction(split_by_frontiers, (in, out)).
modeb(*, split_by_frontiers(+grid, -any)).

compress(grid, Result) :-
    call_python_function('compress', [grid], Result).
body_pred(compress, 2).
direction(compress, (in, out)).
modeb(*, compress(+grid, -grid)).

hperiod(obj, Result) :-
    call_python_function('hperiod', [obj], Result).
body_pred(hperiod, 2).
direction(hperiod, (in, out)).
modeb(*, hperiod(+object, -integer)).

vperiod(obj, Result) :-
    call_python_function('vperiod', [obj], Result).
body_pred(vperiod, 2).
direction(vperiod, (in, out)).
modeb(*, vperiod(+object, -integer)).

period(obj, Result) :-
    call_python_function('period', [obj], Result).
body_pred(period, 2).
direction(period, (in, out)).
modeb(*, period(+object, -any)).

