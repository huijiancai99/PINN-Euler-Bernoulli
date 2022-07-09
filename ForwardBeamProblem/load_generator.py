import numpy as np

def uniform_load(x, magnitude):
    return np.ones((x.shape[0], x.shape[1])) * magnitude

def triangular_load(x, magnitude, zero_end):
    left_zero_end_load = magnitude / x[-1, -1] * x
    if zero_end:
        return left_zero_end_load
    else:
        return magnitude - left_zero_end_load

def custom_load(x, custom_fun):

    return custom_fun(x)
