import numpy as np

def uniform_load(x, magnitude):
    return np.ones((x.shape[0], x.shape[1])) * magnitude

def uniform_varying_load(x, bound, magnitude, zero_end):
    left_zero_end_load = magnitude / bound * x
    if zero_end:
        return left_zero_end_load
    else:
        return magnitude - left_zero_end_load

def custom_load(x, custom_fun):

    return custom_fun(x)

def concentrated_load(magnitude, loc, x, a=0.1):
    return magnitude * dirac_delta(loc, x, a)

def dirac_delta(loc, x, a=0.1):
    
    return np.exp(-np.square((x - loc) / a)) / (a * np.sqrt(np.pi))

def concentrated_moment(magnitude, loc, x, a=0.1, b=0.1):
    
    cf = magnitude / b
    left = -concentrated_load(cf, loc - b / 2, x, a)
    right = concentrated_load(cf, loc + b / 2, x, a)
    
    return left + right