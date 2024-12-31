from dsl import *

#1.两个grid的差异
cellwise#网格单元格差异
difference

#2.两个object的差异
difference
adjacent#判断两个对象是否相邻
hmatching#行匹配判断两个patch是否在行上有重叠
vmatching#列匹配
manhattan#曼哈顿距离
objects
partition
fgpartition
frontiers
delta

tophalf
bottomhalf
lefthalf
righthalf
upper_third
middle_third
lower_third
left_third
center_third
right_third

#对象相关：
asobject
toobject
toindices

contained

#3.旋转对象
rot90
rot180
rot270

#4.镜像对象
hmirror
vmirror
dmirror
cmirror

#5.填充对象 and 裁剪对象
compress
fill
underfill
canvas


#6.颜色相关(包括color关键字)

mostcolor
leastcolor
colorcount
colorfilter
ofcolor
color
palette
recolor
numcolors



replace#颜色替换
switch#颜色交换
palette#获取调色板
mostcolor#获取最多的颜色
leastcolor#获取最少的颜色
ofcolor#获取指定颜色的位置
colorcount#统计颜色数量
recolor#重新着色
color
ofcolor

colorfilter
color_indices



#7.形状变换

box
backdrop
inbox
outbox
shape
width
height
hwratio
hratio
wratio
hratioI
wratioI
portrait
square
is_square
extract_all_boxes
is_box
is_negative_diagonal
is_positive_diagonal
is_square
is_valid_empty_box

hline
vline
vfrontier
hfrontier
frontiers
compress

llcorner
lrcorner
ulcorner
urcorner
corners

#8.位置变换

shift
move
normalize
position
toindices
asindices
gravitate
bordering
uppermost
lowermost
leftmost
rightmost

center
centerofmass

neighbors
ineighbors
dneighbors
vperiod
hperiod



#9.大小变换

upscale
downscale
hupscale
vupscale

size
sizefilter


#10.合并对象

hconcat
vconcat
combine
merge


#11.分割对象

hsplit
vsplit
crop
split

subgrid
intersection

first
last


#12.删除对象

remove
cover


#13.增加对象

insert
paint


#14.交换对象 color

switch


#15.重复对象

repeat
hperiod
vperiod
