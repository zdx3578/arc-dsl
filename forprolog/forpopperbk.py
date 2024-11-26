from dsl import *
from constants import *

def fill_box_color(I):
    # print("------------- input is", I)
    color = 8
    x1 = asindices(I)
    # print("-------------Output from asindices:", x1)
    x2 = box(x1)
    # print("-------------Output from box:", x2)
    O = fill(I, color, x2)
    # print("-------------Output from fill:", O)
    return O
def get_box(I):
    # print("------------- input is", I)
    # color = 8
    x1 = asindices(I)
    # print("-------------Output from asindices:", x1)
    return box(x1)

# # def is_fill_box_color(I,color=8):
# def fill_box_color(I):
#     print(" ------------- input is ",I)
#     color = 8
#     x1 = asindices(I)
#     x2 = box(x1)
#     O = fill(I, color,x2)
#     # O = fill(I, color, box(asindices(I)))
#     return O