max_vars(7).
max_body(10).

% 定义目标谓词
head_pred(program, 2).
type(program, (grid, grid)).
direction(program, (in, out)).


% ------------  自动生成的类型声明  ------------

type(identity, (any, any)).
type(add, (numerical, numerical, numerical)).
type(subtract, (numerical, numerical, numerical)).
type(multiply, (numerical, numerical, numerical)).
type(divide, (numerical, numerical, numerical)).
type(invert, (numerical, numerical)).
type(even, (integer, boolean)).
type(double, (numerical, numerical)).
type(halve, (numerical, numerical)).
type(flip, (boolean, boolean)).
type(equality, (any, any, boolean)).
type(contained, (any, container, boolean)).
type(combine, (container, container, container)).
type(intersection, (any, any, any)).
type(difference, (any, any, any)).
type(advanced_difference, (any, any, any)).
type(dedupe, (any, any)).
type(order, (container, any, any)).
type(repeat, (any, integer, any)).
type(greater, (integer, integer, boolean)).
type(size, (container, integer)).
type(merge, (containercontainer, container)).
type(maximum, (any, integer)).
type(minimum, (any, integer)).
type(valmax, (container, any, integer)).
type(valmin, (container, any, integer)).
type(argmax, (container, any, any)).
type(argmin, (container, any, any)).
type(mostcommon, (container, any)).
type(leastcommon, (container, any)).
type(initset, (any, any)).
type(both, (boolean, boolean, boolean)).
type(either, (boolean, boolean, boolean)).
type(increment, (numerical, numerical)).
type(decrement, (numerical, numerical)).
type(crement, (numerical, numerical)).
type(sign, (numerical, numerical)).
type(positive, (integer, boolean)).
type(toivec, (any, integer_tuple)).
type(tojvec, (any, integer_tuple)).
type(sfilter, (container, any, container)).
type(mfilter, (container, any, any)).
type(extract, (container, any, any)).
type(totuple, (any, any)).
type(first, (container, any)).
type(last, (container, any)).
type(insert, (any, any, any)).
type(remove, (any, container, container)).
type(other, (container, any, any)).
type(interval, (integer, integer, integer, any)).
type(astuple, (integer, integer, integer_tuple)).
type(product, (container, container, any)).
type(pair, (any, any, tupletuple)).
type(branch, (boolean, any, any, any)).
type(compose, (any, any, any)).
type(chain, (any, any, any, any)).
type(matcher, (any, any, any)).
type(rbind, (any, any, any)).
type(lbind, (any, any, any)).
type(power, (any, integer, any)).
type(fork, (any, any, any, any)).
type(apply, (any, container, container)).
type(rapply, (container, any, container)).
type(mapply, (any, containercontainer, any)).
type(papply, (any, any, any, any)).
type(mpapply, (any, any, any, any)).
type(prapply, (any, container, container, any)).
type(mostcolor, (element, integer)).
type(leastcolor, (element, integer)).
type(height, (piece, integer)).
type(width, (piece, integer)).
type(hwratio, (piece, integer)).
type(hratio, (piece, piece, integer)).
type(wratio, (piece, piece, integer)).
type(hratioI, (piece, piece, integer)).
type(wratioI, (piece, piece, integer)).
type(shape, (piece, integer_tuple)).
type(portrait, (piece, boolean)).
type(colorcount, (element, integer, integer)).
type(all_colorcount, (element, any)).
type(colorfilter, (any, integer, any)).
type(sizefilter, (container, integer, any)).
type(asindices_patch, (patch, indices)).
type(asindices, (grid, indices)).
type(ofcolor, (grid, integer, indices)).
type(ulcorner, (patch, integer_tuple)).
type(urcorner, (patch, integer_tuple)).
type(llcorner, (patch, integer_tuple)).
type(lrcorner, (patch, integer_tuple)).
type(crop, (grid, integer_tuple, integer_tuple, grid)).
type(toindices, (patch, indices)).
type(recolor, (integer, patch, object)).
type(shift, (patch, integer_tuple, patch)).
type(normalize, (patch, patch)).
type(dneighbors, (integer_tuple, indices)).
type(ineighbors, (integer_tuple, indices)).
type(neighbors, (integer_tuple, indices)).
type(objects, (grid, boolean, boolean, boolean, any)).
type(partition, (grid, any)).
type(fgpartition, (grid, any)).
type(uppermost, (patch, integer)).
type(lowermost, (patch, integer)).
type(leftmost, (patch, integer)).
type(rightmost, (patch, integer)).
type(square, (piece, boolean)).
type(is_square, (piece, boolean)).
type(vline, (patch, boolean)).
type(hline, (patch, boolean)).
type(sorted_frozenset, (any, any)).
type(hmatching, (patch, patch, boolean)).
type(vmatching, (patch, patch, boolean)).
type(manhattan, (patch, patch, any, integer)).
type(adjacent, (patch, patch, any, boolean)).
type(bordering, (patch, grid, boolean)).
type(centerofmass, (patch, integer_tuple)).
type(palette, (element, any)).
type(numcolors, (element, any)).
type(color, (object, integer)).
type(toobject, (patch, grid, object)).
type(asobject, (grid, object)).
type(object_to_grid, (object, grid)).
type(rot90, (grid, grid)).
type(rot180, (grid, grid)).
type(upper_third, (grid, grid)).
type(middle_third, (grid, grid)).
type(lower_third, (grid, grid)).
type(left_third, (grid, grid)).
type(center_third, (grid, grid)).
type(right_third, (grid, grid)).
type(rot270, (grid, grid)).
type(hmirror, (piece, piece)).
type(vmirror, (piece, piece)).
type(dmirror, (piece, piece)).
type(cmirror, (piece, piece)).
type(fill, (grid, integer, patch, grid)).
type(paint, (grid, object, grid)).
type(underfill, (grid, integer, patch, grid)).
type(underpaint, (grid, object, grid)).
type(hupscale, (grid, integer, grid)).
type(vupscale, (grid, integer, grid)).
type(upscale, (element, integer, element)).
type(get_mode, (any, any, any)).
type(downscale, (grid, integer, grid)).
type(hconcat, (grid, grid, grid)).
type(vconcat, (grid, grid, grid)).
type(subgrid, (patch, grid, grid)).
type(hsplit, (grid, integer, any)).
type(vsplit, (grid, integer, any)).
type(cellwise, (grid, grid, integer, grid)).
type(replace, (grid, integer, integer, grid)).
type(switch, (grid, integer, integer, grid)).
type(center, (patch, integer_tuple)).
type(position, (patch, patch, integer_tuple)).
type(color_at_location, (grid, integer_tuple, integer)).
type(canvas, (integer, integer_tuple, grid)).
type(corners, (patch, indices)).
type(connect, (integer_tuple, integer_tuple, indices)).
type(cover, (grid, patch, grid)).
type(trim, (grid, grid)).
type(move, (grid, object, integer_tuple, grid)).
type(tophalf, (grid, grid)).
type(bottomhalf, (grid, grid)).
type(lefthalf, (grid, grid)).
type(righthalf, (grid, grid)).
type(vfrontier, (any, indices)).
type(hfrontier, (any, indices)).
type(backdrop, (patch, indices)).
type(delta, (patch, indices)).
type(gravitate, (patch, patch, integer_tuple)).
type(inbox, (patch, indices)).
type(inbox0, (patch, indices)).
type(extract_all_boxes, (patch, any)).
type(outbox, (patch, indices)).
type(box, (patch, indices)).
type(shoot, (integer_tuple, integer_tuple, indices)).
type(occurrences, (grid, object, indices)).
type(frontiers, (any, any)).
type(frontiers, (grid, any)).
type(split_by_frontiers, (grid, any)).
type(compress, (grid, grid)).
type(hperiod, (object, integer)).
type(vperiod, (object, integer)).
type(period, (object, any)).
