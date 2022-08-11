import numpy as np

#simply-supported cases#

def ss_uniform_loads(x, magnitude, bound, EI):
    L = bound[1] - bound[0]
    return magnitude * x * (x - L) / (2 * EI)

def ss_uniform_varying_loads(x, magnitude, bound, EI, zero_end=1):
    L = bound[1] - bound[0]
    return magnitude * x * (np.power(x, 2) - L ** 2) / (6 * EI * L)

def ss_concentrated_force_in(x, magnitude, bound, EI, x1):
    curv = np.zeros((0, 1))
    L = bound[1] - bound[0]
    for coord in x:
        if coord < x1:
            phi = -magnitude * coord / (2 * EI)
        else:
            phi = -magnitude * (L - coord) / (2 * EI)
        curv = np.vstack((curv, phi))

    return curv

def ss_concentrated_moment(x, magnitude, bound, EI, right=1):
    L = bound[1] - bound[0]
    return magnitude * x / (EI * L)

def ss_concentrated_moment_in(x, magnitude, bound, EI, x1):
    curv = np.zeros((0, 1))
    L = bound[1] - bound[0]
    for coord in x:
        if coord < x1:
            phi = magnitude * coord / (EI * L)
        else:
            phi = magnitude * (coord - L) / (EI * L)
        curv = np.vstack((curv, phi))
    return curv

# cantilever cases#


def ca_uniform_loads(x, magnitude, bound, EI):
    L = bound[1] - bound[0]
    return magnitude * np.power((x - L), 2) / (2 * EI)

def ca_uniform_varying_loads(x, magnitude, bound, EI, zero_end=1):
    L = bound[1] - bound[0]
    return -magnitude * np.power((x - L), 3) / (6 * EI * L)

def ca_concentrated_force(x, magnitude, bound, EI, right=1):
    L = bound[1] - bound[0]
    return magnitude * (L - x) / EI

def ca_concentrated_force_in(x, magnitude, bound, EI, x1):
    curv = np.zeros((0, 1))
    L = bound[1] - bound[0]
    for coord in x:
        if coord < x1:
            phi = -magnitude * (x1 - coord) / EI
        else:
            phi = 0
        curv = np.vstack((curv, phi))
    return curv

def ca_concentrated_moment(x, magnitude, bound, EI, right=1):
    return magnitude / EI

def ca_concentrated_moment_in(x, magnitude, bound, EI, x1):
    curv = np.zeros((0, 1))
    L = bound[1] - bound[0]
    for coord in x:
        if coord < x1:
            phi = -magnitude / EI
        else:
            phi = 0
        curv = np.vstack((curv, phi))

    return curv


