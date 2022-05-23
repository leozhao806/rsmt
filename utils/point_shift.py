import numpy as np

def decompos_inputs(inputs, t=7):
    xs = inputs[:,:,0]
    ys = inputs[:,:,1]
    if t >= 4:
        temp = xs
        xs = ys
        ys = temp
    if t % 2 == 1:
        xs = 1 - xs
    if t % 4 >= 2:
        ys = 1 - ys
    return np.stack([xs, ys], -1)