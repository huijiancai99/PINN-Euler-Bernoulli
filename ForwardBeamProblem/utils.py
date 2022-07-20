import numpy as np
from pyDOE import lhs
from math import floor
# Utility functions for unpacking boundary conditions

def unpack_bc(bc, domain, EI, N_f):
    bc_dict, index_dict, left_coord, right_coord = {}, {}, [], []
    accurate_domain = []
    ct = 0
    length = domain[1] - domain[0]
    offset = length / N_f
    for i in range(bc.shape[0]):
        if bc[i, 0] in domain:
            bc_dict[bc[i, 0]] = np.hstack([bc[i, 1:], np.array([0])])
            index_dict[bc[i, 0]] = ct
            ct += 1
        else:
            bc_dict[bc[i, 0] - offset] = np.hstack([bc[i, 1:], np.array([-1])])
            bc_dict[bc[i, 0] + offset] = np.hstack([bc[i, 1:], np.array([1])])
            index_dict[bc[i, 0] - offset] = ct
            index_dict[bc[i, 0] + offset] = ct + 1
            ct += 2
            left_coord.append(bc[i, 0] - offset)
            right_coord.append(bc[i, 0] + offset)
    
    if left_coord:
        accurate_domain.append([domain[0], min(left_coord)])
        left_coord.remove(min(left_coord))
    
        while left_coord:
            left, right = min(left_coord), min(right_coord)
            accurate_domain.append([left, right])
            left_coord.remove(left)
            right_coord.remove(right)
        
        accurate_domain.append([right_coord[0], domain[1]])
        
    else:
        accurate_domain.append([domain[0], domain[1]])
    
    return bc_dict, index_dict, accurate_domain

def assemble_boundary_set(bc_dict, raw_domain, EI):
    # find boundary points with non-nan values
    X_u_train = [[], [], [], []]
    label_train = [[], [], [], []]
    for i in range(4):
        coords = [coord for coord in raw_domain if not np.isnan(bc_dict[coord][i])]
        labels = [bc_dict[coord][i] for coord in raw_domain if not np.isnan(bc_dict[coord][i])]
        length = len(coords)
        coords_arr = np.reshape(np.array(coords, dtype='float32'), (length, 1))
        labels_arr = np.reshape(np.array(labels, dtype='float32'), (length, 1))
        X_u_train[i] = coords_arr
        label_train[i] = labels_arr
    label_train[2] /= EI
    label_train[3] /= EI
    return X_u_train, label_train

def assemble_training_set(X_u_train, accurate_domain, N_f):
    # TODO: divide by EI
    domain_length = [sub_domain[1] - sub_domain[0] for sub_domain in accurate_domain]
    points_per_domain = [floor(length / sum(domain_length) * N_f) for length in domain_length]
    points_per_domain[-1] = N_f - sum(points_per_domain[0:-1])
    X_f_train = []
    for i in range(len(accurate_domain)):
        sub_domain = accurate_domain[i]
        X_u_sub = np.array([[sub_domain[0]], [sub_domain[1]]])
        colloc_points = sub_domain[0] + (sub_domain[1] - sub_domain[0]) * lhs(1, points_per_domain[i])
        X_f_train.append(np.vstack((X_u_sub, colloc_points)))
    return X_f_train

def assemble_continuity_set(bc_dict, raw_domain, EI):
    label_con = [[], [], [], []]
    for i in range(4):
        labels = [bc_dict[coord][i] for coord in bc_dict.keys() if bc_dict[coord][4] == 1 and not np.isnan(bc_dict[coord][i])]
        length = len(labels)
        labels_arr = np.reshape(np.array(labels, dtype='float32'), (length, 1))
        label_con[i] = labels_arr
    label_con[2] /= EI
    label_con[3] /= EI
    return label_con

def uniform_load(x, magnitude):
    loads = []
    for segment in x:
        loads.append(np.ones((segment.shape[0], segment.shape[1])) * magnitude)
    return loads

def uniform_varying_load(x, bound, magnitude, zero_end):
    left_zero_end_load = magnitude / bound * x
    if zero_end:
        return left_zero_end_load
    else:
        return magnitude - left_zero_end_load

def coord_partition(x, bc_dict):
    l = np.array([key for key in bc_dict.keys() if bc_dict[key][-1] == -1])
    r = np.array([key for key in bc_dict.keys() if bc_dict[key][-1] == 1])
    partition = (l + r) / 2
    if partition:
        start_index = 0
        x_part = []
        for part_pt in partition:
            end_index = np.argmin(np.abs(x - part_pt))
            if x[end_index] >= part_pt:
                end_index -= 1
            x_part.append(x[start_index:end_index, :])
            start_index = end_index
        x_part.append(x[end_index:, :])
    else:
        x_part = [x]
    
    return x_part

def custom_load(x, custom_fun):

    return custom_fun(x)