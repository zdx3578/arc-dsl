from dsl import *

#原本函数的总个数: 160

#自动生成的类型映射:
  'IntegerSet': 'integerset'
  'Object': 'object'
  'IntegerTuple': 'integertuple'
  'ContainerContainer': 'containercontainer'
  'Grid': 'grid'
  'Container': 'container'
  'Element': 'element'
  'Piece': 'piece'
  'Indices': 'indices'
  'Numerical': 'numerical'
  'Patch': 'patch'
  'Objects': 'objects'
  'Integer': 'integer'
  'FrozenSet': 'frozenset'
  'Tuple': 'tuple'
  'Boolean': 'boolean'
  'Any': 'any'
  'Callable': 'callable'
  'TupleTuple': 'tupletuple'



#输入类型: ('any',), 返回类型: any
#分类下的函数个数: 1
#函数列表:
  - identity


#输入类型: ('numerical', 'numerical'), 返回类型: numerical
#分类下的函数个数: 4
#函数列表:
  - add
  - subtract
  - multiply
  - divide


#输入类型: ('numerical',), 返回类型: numerical
#分类下的函数个数: 7
#函数列表:
  - invert
  - double
  - halve
  - increment
  - decrement
  - crement
  - sign


#输入类型: ('integer',), 返回类型: boolean
#分类下的函数个数: 2
#函数列表:
  - even
  - positive


#输入类型: ('boolean',), 返回类型: boolean
#分类下的函数个数: 1
#函数列表:
  - flip


#输入类型: ('any', 'any'), 返回类型: boolean
#分类下的函数个数: 1
#函数列表:
  - equality


#输入类型: ('any', 'container'), 返回类型: boolean
#分类下的函数个数: 1
#函数列表:
  - contained


#输入类型: ('container', 'container'), 返回类型: container
#分类下的函数个数: 1
#函数列表:
  - combine


#输入类型: ('frozenset', 'frozenset'), 返回类型: frozenset
#分类下的函数个数: 2
#函数列表:
  - intersection
  - difference


#输入类型: ('tuple',), 返回类型: tuple
#分类下的函数个数: 1
#函数列表:
  - dedupe


#输入类型: ('container', 'callable'), 返回类型: tuple
#分类下的函数个数: 1
#函数列表:
  - order


#输入类型: ('any', 'integer'), 返回类型: tuple
#分类下的函数个数: 1
#函数列表:
  - repeat


#输入类型: ('integer', 'integer'), 返回类型: boolean
#分类下的函数个数: 1
#函数列表:
  - greater


#输入类型: ('container',), 返回类型: integer
#分类下的函数个数: 1
#函数列表:
  - size


#输入类型: ('containercontainer',), 返回类型: container
#分类下的函数个数: 1
#函数列表:
  - merge


#输入类型: ('integerset',), 返回类型: integer
#分类下的函数个数: 2
#函数列表:
  - maximum
  - minimum


#输入类型: ('container', 'callable'), 返回类型: integer
#分类下的函数个数: 2
#函数列表:
  - valmax
  - valmin


#输入类型: ('container', 'callable'), 返回类型: any
#分类下的函数个数: 3
#函数列表:
  - argmax
  - argmin
  - extract


#输入类型: ('container',), 返回类型: any
#分类下的函数个数: 4
#函数列表:
  - mostcommon
  - leastcommon
  - first
  - last


#输入类型: ('any',), 返回类型: frozenset
#分类下的函数个数: 1
#函数列表:
  - initset


#输入类型: ('boolean', 'boolean'), 返回类型: boolean
#分类下的函数个数: 2
#函数列表:
  - both
  - either


#输入类型: ('integer',), 返回类型: integertuple
#分类下的函数个数: 2
#函数列表:
  - toivec
  - tojvec


#输入类型: ('container', 'callable'), 返回类型: container
#分类下的函数个数: 1
#函数列表:
  - sfilter


#输入类型: ('container', 'callable'), 返回类型: frozenset
#分类下的函数个数: 1
#函数列表:
  - mfilter


#输入类型: ('frozenset',), 返回类型: tuple
#分类下的函数个数: 1
#函数列表:
  - totuple


#输入类型: ('any', 'frozenset'), 返回类型: frozenset
#分类下的函数个数: 1
#函数列表:
  - insert


#输入类型: ('any', 'container'), 返回类型: container
#分类下的函数个数: 1
#函数列表:
  - remove


#输入类型: ('container', 'any'), 返回类型: any
#分类下的函数个数: 1
#函数列表:
  - other


#输入类型: ('integer', 'integer', 'integer'), 返回类型: tuple
#分类下的函数个数: 1
#函数列表:
  - interval


#输入类型: ('integer', 'integer'), 返回类型: integertuple
#分类下的函数个数: 1
#函数列表:
  - astuple


#输入类型: ('container', 'container'), 返回类型: frozenset
#分类下的函数个数: 1
#函数列表:
  - product


#输入类型: ('tuple', 'tuple'), 返回类型: tupletuple
#分类下的函数个数: 1
#函数列表:
  - pair


#输入类型: ('boolean', 'any', 'any'), 返回类型: any
#分类下的函数个数: 1
#函数列表:
  - branch


#输入类型: ('callable', 'callable'), 返回类型: callable
#分类下的函数个数: 1
#函数列表:
  - compose


#输入类型: ('callable', 'callable', 'callable'), 返回类型: callable
#分类下的函数个数: 2
#函数列表:
  - chain
  - fork


#输入类型: ('callable', 'any'), 返回类型: callable
#分类下的函数个数: 3
#函数列表:
  - matcher
  - rbind
  - lbind


#输入类型: ('callable', 'integer'), 返回类型: callable
#分类下的函数个数: 1
#函数列表:
  - power


#输入类型: ('callable', 'container'), 返回类型: container
#分类下的函数个数: 1
#函数列表:
  - apply


#输入类型: ('container', 'any'), 返回类型: container
#分类下的函数个数: 1
#函数列表:
  - rapply


#输入类型: ('callable', 'containercontainer'), 返回类型: frozenset
#分类下的函数个数: 1
#函数列表:
  - mapply


#输入类型: ('callable', 'tuple', 'tuple'), 返回类型: tuple
#分类下的函数个数: 2
#函数列表:
  - papply
  - mpapply


#输入类型: ('any', 'container', 'container'), 返回类型: frozenset
#分类下的函数个数: 1
#函数列表:
  - prapply


#输入类型: ('element',), 返回类型: integer
#分类下的函数个数: 2
#函数列表:
  - mostcolor
  - leastcolor


#输入类型: ('piece',), 返回类型: integer
#分类下的函数个数: 2
#函数列表:
  - height
  - width


#输入类型: ('piece',), 返回类型: integertuple
#分类下的函数个数: 1
#函数列表:
  - shape


#输入类型: ('piece',), 返回类型: boolean
#分类下的函数个数: 2
#函数列表:
  - portrait
  - square


#输入类型: ('element', 'integer'), 返回类型: integer
#分类下的函数个数: 1
#函数列表:
  - colorcount


#输入类型: ('objects', 'integer'), 返回类型: objects
#分类下的函数个数: 1
#函数列表:
  - colorfilter


#输入类型: ('container', 'integer'), 返回类型: frozenset
#分类下的函数个数: 1
#函数列表:
  - sizefilter


#输入类型: ('grid',), 返回类型: indices
#分类下的函数个数: 1
#函数列表:
  - asindices


#输入类型: ('grid', 'integer'), 返回类型: indices
#分类下的函数个数: 1
#函数列表:
  - ofcolor


#输入类型: ('patch',), 返回类型: integertuple
#分类下的函数个数: 6
#函数列表:
  - ulcorner
  - urcorner
  - llcorner
  - lrcorner
  - centerofmass
  - center


#输入类型: ('grid', 'integertuple', 'integertuple'), 返回类型: grid
#分类下的函数个数: 1
#函数列表:
  - crop


#输入类型: ('patch',), 返回类型: indices
#分类下的函数个数: 7
#函数列表:
  - toindices
  - corners
  - backdrop
  - delta
  - inbox
  - outbox
  - box


#输入类型: ('integer', 'patch'), 返回类型: object
#分类下的函数个数: 1
#函数列表:
  - recolor


#输入类型: ('patch', 'integertuple'), 返回类型: patch
#分类下的函数个数: 1
#函数列表:
  - shift


#输入类型: ('patch',), 返回类型: patch
#分类下的函数个数: 1
#函数列表:
  - normalize


#输入类型: ('integertuple',), 返回类型: indices
#分类下的函数个数: 5
#函数列表:
  - dneighbors
  - ineighbors
  - neighbors
  - vfrontier
  - hfrontier


#输入类型: ('grid', 'boolean', 'boolean', 'boolean'), 返回类型: objects
#分类下的函数个数: 1
#函数列表:
  - objects


#输入类型: ('grid',), 返回类型: objects
#分类下的函数个数: 3
#函数列表:
  - partition
  - fgpartition
  - frontiers


#输入类型: ('patch',), 返回类型: integer
#分类下的函数个数: 4
#函数列表:
  - uppermost
  - lowermost
  - leftmost
  - rightmost


#输入类型: ('patch',), 返回类型: boolean
#分类下的函数个数: 2
#函数列表:
  - vline
  - hline


#输入类型: ('patch', 'patch'), 返回类型: boolean
#分类下的函数个数: 3
#函数列表:
  - hmatching
  - vmatching
  - adjacent


#输入类型: ('patch', 'patch'), 返回类型: integer
#分类下的函数个数: 1
#函数列表:
  - manhattan


#输入类型: ('patch', 'grid'), 返回类型: boolean
#分类下的函数个数: 1
#函数列表:
  - bordering


#输入类型: ('element',), 返回类型: integerset
#分类下的函数个数: 2
#函数列表:
  - palette
  - numcolors


#输入类型: ('object',), 返回类型: integer
#分类下的函数个数: 3
#函数列表:
  - color
  - hperiod
  - vperiod


#输入类型: ('patch', 'grid'), 返回类型: object
#分类下的函数个数: 1
#函数列表:
  - toobject


#输入类型: ('grid',), 返回类型: object
#分类下的函数个数: 1
#函数列表:
  - asobject


#输入类型: ('grid',), 返回类型: grid
#分类下的函数个数: 9
#函数列表:
  - rot90
  - rot180
  - rot270
  - trim
  - tophalf
  - bottomhalf
  - lefthalf
  - righthalf
  - compress


#输入类型: ('piece',), 返回类型: piece
#分类下的函数个数: 4
#函数列表:
  - hmirror
  - vmirror
  - dmirror
  - cmirror


#输入类型: ('grid', 'integer', 'patch'), 返回类型: grid
#分类下的函数个数: 2
#函数列表:
  - fill
  - underfill


#输入类型: ('grid', 'object'), 返回类型: grid
#分类下的函数个数: 2
#函数列表:
  - paint
  - underpaint


#输入类型: ('grid', 'integer'), 返回类型: grid
#分类下的函数个数: 3
#函数列表:
  - hupscale
  - vupscale
  - downscale


#输入类型: ('element', 'integer'), 返回类型: element
#分类下的函数个数: 1
#函数列表:
  - upscale


#输入类型: ('grid', 'grid'), 返回类型: grid
#分类下的函数个数: 2
#函数列表:
  - hconcat
  - vconcat


#输入类型: ('patch', 'grid'), 返回类型: grid
#分类下的函数个数: 1
#函数列表:
  - subgrid


#输入类型: ('grid', 'integer'), 返回类型: tuple
#分类下的函数个数: 2
#函数列表:
  - hsplit
  - vsplit


#输入类型: ('grid', 'grid', 'integer'), 返回类型: grid
#分类下的函数个数: 1
#函数列表:
  - cellwise


#输入类型: ('grid', 'integer', 'integer'), 返回类型: grid
#分类下的函数个数: 2
#函数列表:
  - replace
  - switch


#输入类型: ('patch', 'patch'), 返回类型: integertuple
#分类下的函数个数: 2
#函数列表:
  - position
  - gravitate


#输入类型: ('grid', 'integertuple'), 返回类型: integer
#分类下的函数个数: 1
#函数列表:
  - index


#输入类型: ('integer', 'integertuple'), 返回类型: grid
#分类下的函数个数: 1
#函数列表:
  - canvas


#输入类型: ('integertuple', 'integertuple'), 返回类型: indices
#分类下的函数个数: 2
#函数列表:
  - connect
  - shoot


#输入类型: ('grid', 'patch'), 返回类型: grid
#分类下的函数个数: 1
#函数列表:
  - cover


#输入类型: ('grid', 'object', 'integertuple'), 返回类型: grid
#分类下的函数个数: 1
#函数列表:
  - move


#输入类型: ('grid', 'object'), 返回类型: indices
#分类下的函数个数: 1
#函数列表:
  - occurrences
分类后输出的函数总个数: 160
