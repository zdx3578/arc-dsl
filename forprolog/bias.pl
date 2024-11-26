
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
type(intersection, (frozenset, frozenset, frozenset)).
type(difference, (frozenset, frozenset, frozenset)).
type(dedupe, (tuple, tuple)).
type(order, (container, callable, tuple)).
type(repeat, (any, integer, tuple)).
type(greater, (integer, integer, boolean)).
type(size, (container, integer)).
type(merge, (containercontainer, container)).
type(maximum, (integerset, integer)).
type(minimum, (integerset, integer)).
type(valmax, (container, callable, integer)).
type(valmin, (container, callable, integer)).
type(argmax, (container, callable, any)).
type(argmin, (container, callable, any)).
type(mostcommon, (container, any)).
type(leastcommon, (container, any)).
type(initset, (any, frozenset)).
type(both, (boolean, boolean, boolean)).
type(either, (boolean, boolean, boolean)).
type(increment, (numerical, numerical)).
type(decrement, (numerical, numerical)).
type(crement, (numerical, numerical)).
type(sign, (numerical, numerical)).
type(positive, (integer, boolean)).
type(toivec, (integer, integertuple)).
type(tojvec, (integer, integertuple)).
type(sfilter, (container, callable, container)).
type(mfilter, (container, callable, frozenset)).
type(extract, (container, callable, any)).
type(totuple, (frozenset, tuple)).
type(first, (container, any)).
type(last, (container, any)).
type(insert, (any, frozenset, frozenset)).
type(remove, (any, container, container)).
type(other, (container, any, any)).
type(interval, (integer, integer, integer, tuple)).
type(astuple, (integer, integer, integertuple)).
type(product, (container, container, frozenset)).
type(pair, (tuple, tuple, tupletuple)).
type(branch, (boolean, any, any, any)).
type(compose, (callable, callable, callable)).
type(chain, (callable, callable, callable, callable)).
type(matcher, (callable, any, callable)).
type(rbind, (callable, any, callable)).
type(lbind, (callable, any, callable)).
type(power, (callable, integer, callable)).
type(fork, (callable, callable, callable, callable)).
type(apply, (callable, container, container)).
type(rapply, (container, any, container)).
type(mapply, (callable, containercontainer, frozenset)).
type(papply, (callable, tuple, tuple, tuple)).
type(mpapply, (callable, tuple, tuple, tuple)).
type(prapply, (any, container, container, frozenset)).
type(mostcolor, (element, integer)).
type(leastcolor, (element, integer)).
type(height, (piece, integer)).
type(width, (piece, integer)).
type(shape, (piece, integertuple)).
type(portrait, (piece, boolean)).
type(colorcount, (element, integer, integer)).
type(colorfilter, (objects, integer, objects)).
type(sizefilter, (container, integer, frozenset)).
type(asindices, (grid, indices)).
type(ofcolor, (grid, integer, indices)).
type(ulcorner, (patch, integertuple)).
type(urcorner, (patch, integertuple)).
type(llcorner, (patch, integertuple)).
type(lrcorner, (patch, integertuple)).
type(crop, (grid, integertuple, integertuple, grid)).
type(toindices, (patch, indices)).
type(recolor, (integer, patch, object)).
type(shift, (patch, integertuple, patch)).
type(normalize, (patch, patch)).
type(dneighbors, (integertuple, indices)).
type(ineighbors, (integertuple, indices)).
type(neighbors, (integertuple, indices)).
type(objects, (grid, boolean, boolean, boolean, objects)).
type(partition, (grid, objects)).
type(fgpartition, (grid, objects)).
type(uppermost, (patch, integer)).
type(lowermost, (patch, integer)).
type(leftmost, (patch, integer)).
type(rightmost, (patch, integer)).
type(square, (piece, boolean)).
type(vline, (patch, boolean)).
type(hline, (patch, boolean)).
type(hmatching, (patch, patch, boolean)).
type(vmatching, (patch, patch, boolean)).
type(manhattan, (patch, patch, integer)).
type(adjacent, (patch, patch, boolean)).
type(bordering, (patch, grid, boolean)).
type(centerofmass, (patch, integertuple)).
type(palette, (element, integerset)).
type(numcolors, (element, integerset)).
type(color, (object, integer)).
type(toobject, (patch, grid, object)).
type(asobject, (grid, object)).
type(rot90, (grid, grid)).
type(rot180, (grid, grid)).
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
type(downscale, (grid, integer, grid)).
type(hconcat, (grid, grid, grid)).
type(vconcat, (grid, grid, grid)).
type(subgrid, (patch, grid, grid)).
type(hsplit, (grid, integer, tuple)).
type(vsplit, (grid, integer, tuple)).
type(cellwise, (grid, grid, integer, grid)).
type(replace, (grid, integer, integer, grid)).
type(switch, (grid, integer, integer, grid)).
type(center, (patch, integertuple)).
type(position, (patch, patch, integertuple)).
type(index, (grid, integertuple, integer)).
type(canvas, (integer, integertuple, grid)).
type(corners, (patch, indices)).
type(connect, (integertuple, integertuple, indices)).
type(cover, (grid, patch, grid)).
type(trim, (grid, grid)).
type(move, (grid, object, integertuple, grid)).
type(tophalf, (grid, grid)).
type(bottomhalf, (grid, grid)).
type(lefthalf, (grid, grid)).
type(righthalf, (grid, grid)).
type(vfrontier, (integertuple, indices)).
type(hfrontier, (integertuple, indices)).
type(backdrop, (patch, indices)).
type(delta, (patch, indices)).
type(gravitate, (patch, patch, integertuple)).
type(inbox, (patch, indices)).
type(outbox, (patch, indices)).
type(box, (patch, indices)).
type(shoot, (integertuple, integertuple, indices)).
type(occurrences, (grid, object, indices)).
type(frontiers, (grid, objects)).
type(compress, (grid, grid)).
type(hperiod, (object, integer)).
type(vperiod, (object, integer)).
