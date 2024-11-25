% 引用 py_swip 库以调用 Python 函数
:- use_module(library(py_swip)).

% 导入 dsl 模块
:- py_import_module('dsl').

% 通用调用 Python 函数的谓词

call_python_function(Function, Input, Output) :-
	py_call(Function, [Input], Output).

identity(x, Result) :-
    call_python_function('dsl.identity', [x], Result).
body_pred(identity, 2).
direction(identity, (in, out)).

add(a, b, Result) :-
    call_python_function('dsl.add', [a, b], Result).
body_pred(add, 3).
direction(add, (in, in, out)).

subtract(a, b, Result) :-
    call_python_function('dsl.subtract', [a, b], Result).
body_pred(subtract, 3).
direction(subtract, (in, in, out)).

multiply(a, b, Result) :-
    call_python_function('dsl.multiply', [a, b], Result).
body_pred(multiply, 3).
direction(multiply, (in, in, out)).

divide(a, b, Result) :-
    call_python_function('dsl.divide', [a, b], Result).
body_pred(divide, 3).
direction(divide, (in, in, out)).

invert(n, Result) :-
    call_python_function('dsl.invert', [n], Result).
body_pred(invert, 2).
direction(invert, (in, out)).

even(n, Result) :-
    call_python_function('dsl.even', [n], Result).
body_pred(even, 2).
direction(even, (in, out)).

double(n, Result) :-
    call_python_function('dsl.double', [n], Result).
body_pred(double, 2).
direction(double, (in, out)).

halve(n, Result) :-
    call_python_function('dsl.halve', [n], Result).
body_pred(halve, 2).
direction(halve, (in, out)).

flip(b, Result) :-
    call_python_function('dsl.flip', [b], Result).
body_pred(flip, 2).
direction(flip, (in, out)).

equality(a, b, Result) :-
    call_python_function('dsl.equality', [a, b], Result).
body_pred(equality, 3).
direction(equality, (in, in, out)).

contained(value, container, Result) :-
    call_python_function('dsl.contained', [value, container], Result).
body_pred(contained, 3).
direction(contained, (in, in, out)).

combine(a, b, Result) :-
    call_python_function('dsl.combine', [a, b], Result).
body_pred(combine, 3).
direction(combine, (in, in, out)).

intersection(a, b, Result) :-
    call_python_function('dsl.intersection', [a, b], Result).
body_pred(intersection, 3).
direction(intersection, (in, in, out)).

difference(a, b, Result) :-
    call_python_function('dsl.difference', [a, b], Result).
body_pred(difference, 3).
direction(difference, (in, in, out)).

dedupe(tup, Result) :-
    call_python_function('dsl.dedupe', [tup], Result).
body_pred(dedupe, 2).
direction(dedupe, (in, out)).

order(container, compfunc, Result) :-
    call_python_function('dsl.order', [container, compfunc], Result).
body_pred(order, 3).
direction(order, (in, in, out)).

repeat(item, num, Result) :-
    call_python_function('dsl.repeat', [item, num], Result).
body_pred(repeat, 3).
direction(repeat, (in, in, out)).

greater(a, b, Result) :-
    call_python_function('dsl.greater', [a, b], Result).
body_pred(greater, 3).
direction(greater, (in, in, out)).

size(container, Result) :-
    call_python_function('dsl.size', [container], Result).
body_pred(size, 2).
direction(size, (in, out)).

merge(containers, Result) :-
    call_python_function('dsl.merge', [containers], Result).
body_pred(merge, 2).
direction(merge, (in, out)).

maximum(container, Result) :-
    call_python_function('dsl.maximum', [container], Result).
body_pred(maximum, 2).
direction(maximum, (in, out)).

minimum(container, Result) :-
    call_python_function('dsl.minimum', [container], Result).
body_pred(minimum, 2).
direction(minimum, (in, out)).

valmax(container, compfunc, Result) :-
    call_python_function('dsl.valmax', [container, compfunc], Result).
body_pred(valmax, 3).
direction(valmax, (in, in, out)).

valmin(container, compfunc, Result) :-
    call_python_function('dsl.valmin', [container, compfunc], Result).
body_pred(valmin, 3).
direction(valmin, (in, in, out)).

argmax(container, compfunc, Result) :-
    call_python_function('dsl.argmax', [container, compfunc], Result).
body_pred(argmax, 3).
direction(argmax, (in, in, out)).

argmin(container, compfunc, Result) :-
    call_python_function('dsl.argmin', [container, compfunc], Result).
body_pred(argmin, 3).
direction(argmin, (in, in, out)).

mostcommon(container, Result) :-
    call_python_function('dsl.mostcommon', [container], Result).
body_pred(mostcommon, 2).
direction(mostcommon, (in, out)).

leastcommon(container, Result) :-
    call_python_function('dsl.leastcommon', [container], Result).
body_pred(leastcommon, 2).
direction(leastcommon, (in, out)).

initset(value, Result) :-
    call_python_function('dsl.initset', [value], Result).
body_pred(initset, 2).
direction(initset, (in, out)).

both(a, b, Result) :-
    call_python_function('dsl.both', [a, b], Result).
body_pred(both, 3).
direction(both, (in, in, out)).

either(a, b, Result) :-
    call_python_function('dsl.either', [a, b], Result).
body_pred(either, 3).
direction(either, (in, in, out)).

increment(x, Result) :-
    call_python_function('dsl.increment', [x], Result).
body_pred(increment, 2).
direction(increment, (in, out)).

decrement(x, Result) :-
    call_python_function('dsl.decrement', [x], Result).
body_pred(decrement, 2).
direction(decrement, (in, out)).

crement(x, Result) :-
    call_python_function('dsl.crement', [x], Result).
body_pred(crement, 2).
direction(crement, (in, out)).

sign(x, Result) :-
    call_python_function('dsl.sign', [x], Result).
body_pred(sign, 2).
direction(sign, (in, out)).

positive(x, Result) :-
    call_python_function('dsl.positive', [x], Result).
body_pred(positive, 2).
direction(positive, (in, out)).

toivec(i, Result) :-
    call_python_function('dsl.toivec', [i], Result).
body_pred(toivec, 2).
direction(toivec, (in, out)).

tojvec(j, Result) :-
    call_python_function('dsl.tojvec', [j], Result).
body_pred(tojvec, 2).
direction(tojvec, (in, out)).

sfilter(container, condition, Result) :-
    call_python_function('dsl.sfilter', [container, condition], Result).
body_pred(sfilter, 3).
direction(sfilter, (in, in, out)).

mfilter(container, function, Result) :-
    call_python_function('dsl.mfilter', [container, function], Result).
body_pred(mfilter, 3).
direction(mfilter, (in, in, out)).

extract(container, condition, Result) :-
    call_python_function('dsl.extract', [container, condition], Result).
body_pred(extract, 3).
direction(extract, (in, in, out)).

totuple(container, Result) :-
    call_python_function('dsl.totuple', [container], Result).
body_pred(totuple, 2).
direction(totuple, (in, out)).

first(container, Result) :-
    call_python_function('dsl.first', [container], Result).
body_pred(first, 2).
direction(first, (in, out)).

last(container, Result) :-
    call_python_function('dsl.last', [container], Result).
body_pred(last, 2).
direction(last, (in, out)).

insert(value, container, Result) :-
    call_python_function('dsl.insert', [value, container], Result).
body_pred(insert, 3).
direction(insert, (in, in, out)).

remove(value, container, Result) :-
    call_python_function('dsl.remove', [value, container], Result).
body_pred(remove, 3).
direction(remove, (in, in, out)).

other(container, value, Result) :-
    call_python_function('dsl.other', [container, value], Result).
body_pred(other, 3).
direction(other, (in, in, out)).

interval(start, stop, step, Result) :-
    call_python_function('dsl.interval', [start, stop, step], Result).
body_pred(interval, 4).
direction(interval, (in, in, in, out)).

astuple(a, b, Result) :-
    call_python_function('dsl.astuple', [a, b], Result).
body_pred(astuple, 3).
direction(astuple, (in, in, out)).

product(a, b, Result) :-
    call_python_function('dsl.product', [a, b], Result).
body_pred(product, 3).
direction(product, (in, in, out)).

pair(a, b, Result) :-
    call_python_function('dsl.pair', [a, b], Result).
body_pred(pair, 3).
direction(pair, (in, in, out)).

branch(condition, a, b, Result) :-
    call_python_function('dsl.branch', [condition, a, b], Result).
body_pred(branch, 4).
direction(branch, (in, in, in, out)).

compose(outer, inner, Result) :-
    call_python_function('dsl.compose', [outer, inner], Result).
body_pred(compose, 3).
direction(compose, (in, in, out)).

chain(h, g, f, Result) :-
    call_python_function('dsl.chain', [h, g, f], Result).
body_pred(chain, 4).
direction(chain, (in, in, in, out)).

matcher(function, target, Result) :-
    call_python_function('dsl.matcher', [function, target], Result).
body_pred(matcher, 3).
direction(matcher, (in, in, out)).

rbind(function, fixed, Result) :-
    call_python_function('dsl.rbind', [function, fixed], Result).
body_pred(rbind, 3).
direction(rbind, (in, in, out)).

lbind(function, fixed, Result) :-
    call_python_function('dsl.lbind', [function, fixed], Result).
body_pred(lbind, 3).
direction(lbind, (in, in, out)).

power(function, n, Result) :-
    call_python_function('dsl.power', [function, n], Result).
body_pred(power, 3).
direction(power, (in, in, out)).

fork(outer, a, b, Result) :-
    call_python_function('dsl.fork', [outer, a, b], Result).
body_pred(fork, 4).
direction(fork, (in, in, in, out)).

apply(function, container, Result) :-
    call_python_function('dsl.apply', [function, container], Result).
body_pred(apply, 3).
direction(apply, (in, in, out)).

rapply(functions, value, Result) :-
    call_python_function('dsl.rapply', [functions, value], Result).
body_pred(rapply, 3).
direction(rapply, (in, in, out)).

mapply(function, container, Result) :-
    call_python_function('dsl.mapply', [function, container], Result).
body_pred(mapply, 3).
direction(mapply, (in, in, out)).

papply(function, a, b, Result) :-
    call_python_function('dsl.papply', [function, a, b], Result).
body_pred(papply, 4).
direction(papply, (in, in, in, out)).

mpapply(function, a, b, Result) :-
    call_python_function('dsl.mpapply', [function, a, b], Result).
body_pred(mpapply, 4).
direction(mpapply, (in, in, in, out)).

prapply(function, a, b, Result) :-
    call_python_function('dsl.prapply', [function, a, b], Result).
body_pred(prapply, 4).
direction(prapply, (in, in, in, out)).

mostcolor(element, Result) :-
    call_python_function('dsl.mostcolor', [element], Result).
body_pred(mostcolor, 2).
direction(mostcolor, (in, out)).

leastcolor(element, Result) :-
    call_python_function('dsl.leastcolor', [element], Result).
body_pred(leastcolor, 2).
direction(leastcolor, (in, out)).

height(piece, Result) :-
    call_python_function('dsl.height', [piece], Result).
body_pred(height, 2).
direction(height, (in, out)).

width(piece, Result) :-
    call_python_function('dsl.width', [piece], Result).
body_pred(width, 2).
direction(width, (in, out)).

shape(piece, Result) :-
    call_python_function('dsl.shape', [piece], Result).
body_pred(shape, 2).
direction(shape, (in, out)).

portrait(piece, Result) :-
    call_python_function('dsl.portrait', [piece], Result).
body_pred(portrait, 2).
direction(portrait, (in, out)).

colorcount(element, value, Result) :-
    call_python_function('dsl.colorcount', [element, value], Result).
body_pred(colorcount, 3).
direction(colorcount, (in, in, out)).

colorfilter(objs, value, Result) :-
    call_python_function('dsl.colorfilter', [objs, value], Result).
body_pred(colorfilter, 3).
direction(colorfilter, (in, in, out)).

sizefilter(container, n, Result) :-
    call_python_function('dsl.sizefilter', [container, n], Result).
body_pred(sizefilter, 3).
direction(sizefilter, (in, in, out)).

asindices(grid, Result) :-
    call_python_function('dsl.asindices', [grid], Result).
body_pred(asindices, 2).
direction(asindices, (in, out)).

ofcolor(grid, value, Result) :-
    call_python_function('dsl.ofcolor', [grid, value], Result).
body_pred(ofcolor, 3).
direction(ofcolor, (in, in, out)).

ulcorner(patch, Result) :-
    call_python_function('dsl.ulcorner', [patch], Result).
body_pred(ulcorner, 2).
direction(ulcorner, (in, out)).

urcorner(patch, Result) :-
    call_python_function('dsl.urcorner', [patch], Result).
body_pred(urcorner, 2).
direction(urcorner, (in, out)).

llcorner(patch, Result) :-
    call_python_function('dsl.llcorner', [patch], Result).
body_pred(llcorner, 2).
direction(llcorner, (in, out)).

lrcorner(patch, Result) :-
    call_python_function('dsl.lrcorner', [patch], Result).
body_pred(lrcorner, 2).
direction(lrcorner, (in, out)).

crop(grid, start, dims, Result) :-
    call_python_function('dsl.crop', [grid, start, dims], Result).
body_pred(crop, 4).
direction(crop, (in, in, in, out)).

toindices(patch, Result) :-
    call_python_function('dsl.toindices', [patch], Result).
body_pred(toindices, 2).
direction(toindices, (in, out)).

recolor(value, patch, Result) :-
    call_python_function('dsl.recolor', [value, patch], Result).
body_pred(recolor, 3).
direction(recolor, (in, in, out)).

shift(patch, directions, Result) :-
    call_python_function('dsl.shift', [patch, directions], Result).
body_pred(shift, 3).
direction(shift, (in, in, out)).

normalize(patch, Result) :-
    call_python_function('dsl.normalize', [patch], Result).
body_pred(normalize, 2).
direction(normalize, (in, out)).

dneighbors(loc, Result) :-
    call_python_function('dsl.dneighbors', [loc], Result).
body_pred(dneighbors, 2).
direction(dneighbors, (in, out)).

ineighbors(loc, Result) :-
    call_python_function('dsl.ineighbors', [loc], Result).
body_pred(ineighbors, 2).
direction(ineighbors, (in, out)).

neighbors(loc, Result) :-
    call_python_function('dsl.neighbors', [loc], Result).
body_pred(neighbors, 2).
direction(neighbors, (in, out)).

objects(grid, univalued, diagonal, without_bg, Result) :-
    call_python_function('dsl.objects', [grid, univalued, diagonal, without_bg], Result).
body_pred(objects, 5).
direction(objects, (in, in, in, in, out)).

partition(grid, Result) :-
    call_python_function('dsl.partition', [grid], Result).
body_pred(partition, 2).
direction(partition, (in, out)).

fgpartition(grid, Result) :-
    call_python_function('dsl.fgpartition', [grid], Result).
body_pred(fgpartition, 2).
direction(fgpartition, (in, out)).

uppermost(patch, Result) :-
    call_python_function('dsl.uppermost', [patch], Result).
body_pred(uppermost, 2).
direction(uppermost, (in, out)).

lowermost(patch, Result) :-
    call_python_function('dsl.lowermost', [patch], Result).
body_pred(lowermost, 2).
direction(lowermost, (in, out)).

leftmost(patch, Result) :-
    call_python_function('dsl.leftmost', [patch], Result).
body_pred(leftmost, 2).
direction(leftmost, (in, out)).

rightmost(patch, Result) :-
    call_python_function('dsl.rightmost', [patch], Result).
body_pred(rightmost, 2).
direction(rightmost, (in, out)).

square(piece, Result) :-
    call_python_function('dsl.square', [piece], Result).
body_pred(square, 2).
direction(square, (in, out)).

vline(patch, Result) :-
    call_python_function('dsl.vline', [patch], Result).
body_pred(vline, 2).
direction(vline, (in, out)).

hline(patch, Result) :-
    call_python_function('dsl.hline', [patch], Result).
body_pred(hline, 2).
direction(hline, (in, out)).

hmatching(a, b, Result) :-
    call_python_function('dsl.hmatching', [a, b], Result).
body_pred(hmatching, 3).
direction(hmatching, (in, in, out)).

vmatching(a, b, Result) :-
    call_python_function('dsl.vmatching', [a, b], Result).
body_pred(vmatching, 3).
direction(vmatching, (in, in, out)).

manhattan(a, b, Result) :-
    call_python_function('dsl.manhattan', [a, b], Result).
body_pred(manhattan, 3).
direction(manhattan, (in, in, out)).

adjacent(a, b, Result) :-
    call_python_function('dsl.adjacent', [a, b], Result).
body_pred(adjacent, 3).
direction(adjacent, (in, in, out)).

bordering(patch, grid, Result) :-
    call_python_function('dsl.bordering', [patch, grid], Result).
body_pred(bordering, 3).
direction(bordering, (in, in, out)).

centerofmass(patch, Result) :-
    call_python_function('dsl.centerofmass', [patch], Result).
body_pred(centerofmass, 2).
direction(centerofmass, (in, out)).

palette(element, Result) :-
    call_python_function('dsl.palette', [element], Result).
body_pred(palette, 2).
direction(palette, (in, out)).

numcolors(element, Result) :-
    call_python_function('dsl.numcolors', [element], Result).
body_pred(numcolors, 2).
direction(numcolors, (in, out)).

color(obj, Result) :-
    call_python_function('dsl.color', [obj], Result).
body_pred(color, 2).
direction(color, (in, out)).

toobject(patch, grid, Result) :-
    call_python_function('dsl.toobject', [patch, grid], Result).
body_pred(toobject, 3).
direction(toobject, (in, in, out)).

asobject(grid, Result) :-
    call_python_function('dsl.asobject', [grid], Result).
body_pred(asobject, 2).
direction(asobject, (in, out)).

rot90(grid, Result) :-
    call_python_function('dsl.rot90', [grid], Result).
body_pred(rot90, 2).
direction(rot90, (in, out)).

rot180(grid, Result) :-
    call_python_function('dsl.rot180', [grid], Result).
body_pred(rot180, 2).
direction(rot180, (in, out)).

rot270(grid, Result) :-
    call_python_function('dsl.rot270', [grid], Result).
body_pred(rot270, 2).
direction(rot270, (in, out)).

hmirror(piece, Result) :-
    call_python_function('dsl.hmirror', [piece], Result).
body_pred(hmirror, 2).
direction(hmirror, (in, out)).

vmirror(piece, Result) :-
    call_python_function('dsl.vmirror', [piece], Result).
body_pred(vmirror, 2).
direction(vmirror, (in, out)).

dmirror(piece, Result) :-
    call_python_function('dsl.dmirror', [piece], Result).
body_pred(dmirror, 2).
direction(dmirror, (in, out)).

cmirror(piece, Result) :-
    call_python_function('dsl.cmirror', [piece], Result).
body_pred(cmirror, 2).
direction(cmirror, (in, out)).

fill(grid, value, patch, Result) :-
    call_python_function('dsl.fill', [grid, value, patch], Result).
body_pred(fill, 4).
direction(fill, (in, in, in, out)).

paint(grid, obj, Result) :-
    call_python_function('dsl.paint', [grid, obj], Result).
body_pred(paint, 3).
direction(paint, (in, in, out)).

underfill(grid, value, patch, Result) :-
    call_python_function('dsl.underfill', [grid, value, patch], Result).
body_pred(underfill, 4).
direction(underfill, (in, in, in, out)).

underpaint(grid, obj, Result) :-
    call_python_function('dsl.underpaint', [grid, obj], Result).
body_pred(underpaint, 3).
direction(underpaint, (in, in, out)).

hupscale(grid, factor, Result) :-
    call_python_function('dsl.hupscale', [grid, factor], Result).
body_pred(hupscale, 3).
direction(hupscale, (in, in, out)).

vupscale(grid, factor, Result) :-
    call_python_function('dsl.vupscale', [grid, factor], Result).
body_pred(vupscale, 3).
direction(vupscale, (in, in, out)).

upscale(element, factor, Result) :-
    call_python_function('dsl.upscale', [element, factor], Result).
body_pred(upscale, 3).
direction(upscale, (in, in, out)).

downscale(grid, factor, Result) :-
    call_python_function('dsl.downscale', [grid, factor], Result).
body_pred(downscale, 3).
direction(downscale, (in, in, out)).

hconcat(a, b, Result) :-
    call_python_function('dsl.hconcat', [a, b], Result).
body_pred(hconcat, 3).
direction(hconcat, (in, in, out)).

vconcat(a, b, Result) :-
    call_python_function('dsl.vconcat', [a, b], Result).
body_pred(vconcat, 3).
direction(vconcat, (in, in, out)).

subgrid(patch, grid, Result) :-
    call_python_function('dsl.subgrid', [patch, grid], Result).
body_pred(subgrid, 3).
direction(subgrid, (in, in, out)).

hsplit(grid, n, Result) :-
    call_python_function('dsl.hsplit', [grid, n], Result).
body_pred(hsplit, 3).
direction(hsplit, (in, in, out)).

vsplit(grid, n, Result) :-
    call_python_function('dsl.vsplit', [grid, n], Result).
body_pred(vsplit, 3).
direction(vsplit, (in, in, out)).

cellwise(a, b, fallback, Result) :-
    call_python_function('dsl.cellwise', [a, b, fallback], Result).
body_pred(cellwise, 4).
direction(cellwise, (in, in, in, out)).

replace(grid, replacee, replacer, Result) :-
    call_python_function('dsl.replace', [grid, replacee, replacer], Result).
body_pred(replace, 4).
direction(replace, (in, in, in, out)).

switch(grid, a, b, Result) :-
    call_python_function('dsl.switch', [grid, a, b], Result).
body_pred(switch, 4).
direction(switch, (in, in, in, out)).

center(patch, Result) :-
    call_python_function('dsl.center', [patch], Result).
body_pred(center, 2).
direction(center, (in, out)).

position(a, b, Result) :-
    call_python_function('dsl.position', [a, b], Result).
body_pred(position, 3).
direction(position, (in, in, out)).

index(grid, loc, Result) :-
    call_python_function('dsl.index', [grid, loc], Result).
body_pred(index, 3).
direction(index, (in, in, out)).

canvas(value, dimensions, Result) :-
    call_python_function('dsl.canvas', [value, dimensions], Result).
body_pred(canvas, 3).
direction(canvas, (in, in, out)).

corners(patch, Result) :-
    call_python_function('dsl.corners', [patch], Result).
body_pred(corners, 2).
direction(corners, (in, out)).

connect(a, b, Result) :-
    call_python_function('dsl.connect', [a, b], Result).
body_pred(connect, 3).
direction(connect, (in, in, out)).

cover(grid, patch, Result) :-
    call_python_function('dsl.cover', [grid, patch], Result).
body_pred(cover, 3).
direction(cover, (in, in, out)).

trim(grid, Result) :-
    call_python_function('dsl.trim', [grid], Result).
body_pred(trim, 2).
direction(trim, (in, out)).

move(grid, obj, offset, Result) :-
    call_python_function('dsl.move', [grid, obj, offset], Result).
body_pred(move, 4).
direction(move, (in, in, in, out)).

tophalf(grid, Result) :-
    call_python_function('dsl.tophalf', [grid], Result).
body_pred(tophalf, 2).
direction(tophalf, (in, out)).

bottomhalf(grid, Result) :-
    call_python_function('dsl.bottomhalf', [grid], Result).
body_pred(bottomhalf, 2).
direction(bottomhalf, (in, out)).

lefthalf(grid, Result) :-
    call_python_function('dsl.lefthalf', [grid], Result).
body_pred(lefthalf, 2).
direction(lefthalf, (in, out)).

righthalf(grid, Result) :-
    call_python_function('dsl.righthalf', [grid], Result).
body_pred(righthalf, 2).
direction(righthalf, (in, out)).

vfrontier(location, Result) :-
    call_python_function('dsl.vfrontier', [location], Result).
body_pred(vfrontier, 2).
direction(vfrontier, (in, out)).

hfrontier(location, Result) :-
    call_python_function('dsl.hfrontier', [location], Result).
body_pred(hfrontier, 2).
direction(hfrontier, (in, out)).

backdrop(patch, Result) :-
    call_python_function('dsl.backdrop', [patch], Result).
body_pred(backdrop, 2).
direction(backdrop, (in, out)).

delta(patch, Result) :-
    call_python_function('dsl.delta', [patch], Result).
body_pred(delta, 2).
direction(delta, (in, out)).

gravitate(source, destination, Result) :-
    call_python_function('dsl.gravitate', [source, destination], Result).
body_pred(gravitate, 3).
direction(gravitate, (in, in, out)).

inbox(patch, Result) :-
    call_python_function('dsl.inbox', [patch], Result).
body_pred(inbox, 2).
direction(inbox, (in, out)).

outbox(patch, Result) :-
    call_python_function('dsl.outbox', [patch], Result).
body_pred(outbox, 2).
direction(outbox, (in, out)).

box(patch, Result) :-
    call_python_function('dsl.box', [patch], Result).
body_pred(box, 2).
direction(box, (in, out)).

shoot(start, direction, Result) :-
    call_python_function('dsl.shoot', [start, direction], Result).
body_pred(shoot, 3).
direction(shoot, (in, in, out)).

occurrences(grid, obj, Result) :-
    call_python_function('dsl.occurrences', [grid, obj], Result).
body_pred(occurrences, 3).
direction(occurrences, (in, in, out)).

frontiers(grid, Result) :-
    call_python_function('dsl.frontiers', [grid], Result).
body_pred(frontiers, 2).
direction(frontiers, (in, out)).

compress(grid, Result) :-
    call_python_function('dsl.compress', [grid], Result).
body_pred(compress, 2).
direction(compress, (in, out)).

hperiod(obj, Result) :-
    call_python_function('dsl.hperiod', [obj], Result).
body_pred(hperiod, 2).
direction(hperiod, (in, out)).

vperiod(obj, Result) :-
    call_python_function('dsl.vperiod', [obj], Result).
body_pred(vperiod, 2).
direction(vperiod, (in, out)).

