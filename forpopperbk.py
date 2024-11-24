from dsl import *
from constants import *



def is_fill_box_color(I,color=8):
    # x1 = asindices(I)
    # # x0 = outbox(x1)
    # x2 = box(x1)
    O = fill(I, color, box(asindices(I)))
    return O