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
    
    accurate_domain.append([domain[0], min(left_coord)])
    left_coord.remove(min(left_coord))
    
    while left_coord:
        left, right = min(left_coord), min(right_coord)
        accurate_domain.append([left, right])
        left_coord.remove(left)
        right_coord.remove(right)
        
    accurate_domain.append([right_coord[0], domain[1]])
    
    return bc_dict, index_dict, accurate_domain
    
    
    
    
    
    """
    # unpacks the boundary conditions according to the order of derivatives
    u, u_x, u_xx, u_xx_in, u_xxx, u_xxx_in = [], [], [], [], [], []
    u_index, u_x_index, u_xx_index, u_xx_in_index, u_xxx_index, u_xxx_in_index = [], [], [], [], [], []

    for i in range(bc.shape[0]):
        if not np.isnan(bc[i, -4]):
            u.append([bc[i, 0], bc[i, -4]])
            u_index.append(i)
        
        if not np.isnan(bc[i, -3]):
            u_x.append([bc[i, 0], bc[i, -3]])
            u_x_index.append(i)
        
        if not np.isnan(bc[i, -2]):
            if bc[i, 0] == domain[0]:
                u_xx.append([bc[i, 0], -bc[i, -2] / EI])
                u_xx_index.append(i)
            elif bc[i, 0] == domain[1]:
                u_xx.append([bc[i, 0], bc[i, -2] / EI])
                u_xx_index.append(i)
            else:
                u_xx_in.append([bc[i, 0], bc[i, -2] / EI])
                u_xx_in_index.append(i)
        
        if not np.isnan(bc[i, -1]):
            if bc[i, 0] == domain[0]:
                u_xxx.append([bc[i, 0], bc[i, -1] / EI])
                u_xxx_index.append(i)
            elif bc[i, 0] == domain[1]:
                u_xxx.append([bc[i, 0], -bc[i, -1] / EI])
                u_xxx_index.append(i)
            else:
                u_xxx_in.append([bc[i, 0], bc[i, -1] / EI])
                u_xxx_in_index.append(i)

    # keeps the shape constant to avoid errors
    u = np.array(u) if u_index else np.zeros((1, 1))
    u_x = np.array(u_x) if u_x_index else np.zeros((1, 1))
    u_xx = np.array(u_xx) if u_xx_index else np.zeros((1, 1))
    u_xxx = np.array(u_xxx) if u_xxx_index else np.zeros((1, 1))
    u_xx_in = np.array(u_xx_in) if u_xx_in_index else np.zeros((1, 1))
    u_xxx_in = np.array(u_xxx_in) if u_xxx_in_index else np.zeros((1, 1))
    return [u, u_x, u_xx, u_xxx, u_xx_in, u_xxx_in], \
           [u_index, u_x_index, u_xx_index, u_xxx_index, u_xx_in_index, u_xxx_in_index]
    """

def assemble_boundary_set(bc_dict):
    key_lst = list(bc_dict.keys())
    length = len(key_lst)
    return np.reshape(np.array(key_lst), (length, 1))
    """
    X_u_train = bc_sets[0][:, 0:1]
    if bc_sets[1].shape[1] == 2:
        X_u_train = np.vstack((X_u_train, bc_sets[1][:, 0:1]))
    if bc_sets[2].shape[1] == 2:
        X_u_train = np.vstack((X_u_train, bc_sets[2][:, 0:1]))
    if bc_sets[3].shape[1] == 2:
        X_u_train = np.vstack((X_u_train, bc_sets[3][:, 0:1]))
    if bc_sets[4].shape[1] == 2:
        X_u_train = np.vstack((X_u_train, bc_sets[4][:, 0:1]))
    if bc_sets[5].shape[1] == 2:
        X_u_train = np.vstack((X_u_train, bc_sets[5][:, 0:1]))
    return X_u_train
    """

def assemble_training_set(X_u_train, accurate_domain, N_f):
    domain_length = [sub_domain[1] - sub_domain[0] for sub_domain in accurate_domain]
    points_per_domain = [floor(length / sum(domain_length) * N_f) for length in domain_length]
    print(points_per_domain)
    points_per_domain[-1] = N_f - sum(points_per_domain[0:-1])
    X_f_train = np.zeros((0, 1))
    for i in range(len(accurate_domain)):
        sub_domain = accurate_domain[i]
        colloc_points = sub_domain[0] + (sub_domain[1] - sub_domain[0]) * lhs(1, points_per_domain[i])
        X_f_train = np.vstack((X_f_train, colloc_points))
    return np.vstack((X_u_train, X_f_train))

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