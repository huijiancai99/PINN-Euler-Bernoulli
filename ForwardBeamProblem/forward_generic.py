import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
# from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
import utils
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN():
    
    def __init__(self, X_f, loads, bc_dict, index_dict, layers, domain, EI, loss_weights):
        
        self.domain = domain
        
        self.x_f = X_f
        self.loads = loads

        self.EI = EI

        self.bc_dict = bc_dict
        self.index_dict = index_dict
        
        self.layers = layers
        self.bc_list = [[], [], [], [], [], [], [], [], [], [], [], []]
        
        self.weights, self.biases = self.initialize_NN(layers)
        
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.x_u_l_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.x_u_r_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        
        self.u_tf = tf.placeholder(tf.float32, shape=[None, 1])  
        self.u_x_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u_xx_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u_xxx_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        self.u_xx_con_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u_xxx_con_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        self.loads_tf = tf.placeholder(tf.float32, shape=[None, self.loads.shape[1]])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        
        self.u_pred = self.net_u(self.x_u_tf)
        self.u_l_pred = self.net_u(self.x_u_l_tf)
        self.u_r_pred = self.net_u(self.x_u_r_tf)
        self.f_pred, self.grads_pred = self.net_f(self.x_f_tf)
        
        self.unpack_gradients()
        # check shapes of the self.xxx varst
        # optimize individual loss components and validate results
        # adjust coefficients of the loss terms
        loss_components = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        # residual
        loss_components[0] = loss_weights[0] * tf.reduce_mean(tf.square(self.f_pred))
        # bc of u
        loss_components[1] = loss_weights[1] * tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
        # bc of u_x
        loss_components[2] = loss_weights[2] * tf.reduce_mean(tf.square(self.u_x_tf - self.u_x_pred)) \
                             if self.bc_list[3] else 0
        # bc of u_xx
        loss_components[3] = loss_weights[3] * tf.reduce_mean(tf.square(self.u_xx_tf - self.u_xx_pred)) \
                             if self.bc_list[6] else 0
        # bc of u_xxx
        loss_components[4] = loss_weights[4] * tf.reduce_mean(tf.square(self.u_xxx_tf - self.u_xxx_pred)) \
                             if self.bc_list[9] else 0
        # bc of u (continuity)
        loss_components[5] = loss_weights[5] * tf.reduce_mean(tf.square(self.u_l_pred - self.u_r_pred)) \
                             if self.bc_list[1] else 0
        
        # bc of u_x (continuity)
        loss_components[6] = loss_weights[6] * tf.reduce_mean(tf.square(self.u_x_l_pred - self.u_x_r_pred)) \
                             if self.bc_list[4] else 0
                             
        # bc of u_xx (continuity)
        loss_components[7] = loss_weights[7] * tf.reduce_mean(tf.square(self.u_xx_l_pred - self.u_xx_r_pred - self.u_xx_con_tf)) \
                             if self.bc_list[7] else 0
        
        # bc of u_xxx (continuity)
        loss_components[8] = loss_weights[8] * tf.reduce_mean(tf.square(self.u_xxx_l_pred - self.u_xxx_r_pred - self.u_xxx_con_tf)) \
                             if self.bc_list[10] else 0
        
        print(loss_components)
        self.loss = sum(loss_components)
             
        # try combining Adam, should be put before L-BFGS-B
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol' : 1.0 * np.finfo(float).eps})
        init = tf.global_variables_initializer()
        self.sess.run(init)
    
    
    """    
    def unpack_bc(self, bc):
        # unpacks the boundary conditions according to the order of derivatives
        u, u_x, u_xx, u_xxx = [], [], [], []
        self.u_index, self.u_x_index, self.u_xx_index, self.u_xxx_index = [], [], [], []
        
        for i in range(bc.shape[0]):
            if not np.isnan(bc[i, -4]):
                u.append([bc[i, 0], bc[i, -4]])
                self.u_index.append(i)
            if not np.isnan(bc[i, -3]):
                u_x.append([bc[i, 0], bc[i, -3]])
                self.u_x_index.append(i)
            if not np.isnan(bc[i, -2]):
                if bc[i, 0] == self.lb:
                    u_xx.append([bc[i, 0], -bc[i, -2] / self.EI])
                    self.u_xx_index.append(i)
                if bc[i, 0] == self.ub:
                    u_xx.append([bc[i, 0], bc[i, -2] / self.EI])
                    self.u_xx_index.append(i)
            if not np.isnan(bc[i, -1]):
                if bc[i, 0] == self.lb:
                    u_xxx.append([bc[i, 0], bc[i, -1] / self.EI])
                    self.u_xxx_index.append(i)
                if bc[i, 0] == self.ub:
                    u_xxx.append([bc[i, 0], -bc[i, -1] / self.EI])
                    self.u_xxx_index.append(i)
        
        # keeps the shape constant to avoid errors
        self.u = np.array(u) if self.u_index else np.zeros((1, 1))
        self.u_x = np.array(u_x) if self.u_x_index else np.zeros((1, 1))
        self.u_xx = np.array(u_xx) if self.u_xx_index else np.zeros((1, 1))
        self.u_xxx = np.array(u_xxx) if self.u_xxx_index else np.zeros((1, 1))
        """    

    def unpack_gradients(self):
        
        for key in self.bc_dict.keys():
            bc = self.bc_dict[key]
            for j in range(bc.shape[0] - 1):
                self.interpret_bc(key, bc[j], j)
        
        self.x_u, self.u = np.zeros((0, 1)), np.zeros((0, 1))
        for i in range(len(self.bc_list[0])):
            key = self.x_f[self.bc_list[0][i], 0]
            self.x_u = np.vstack((self.x_u, key))
            self.u = np.vstack((self.u, self.bc_dict[key][0]))
        
        if self.bc_list[1]:
            self.x_u_l, self.x_u_r, self.u_con = np.zeros((0, 1)), np.zeros((0, 1)), np.zeros((0, 1))
            for i in range(len(self.bc_list[0])):
                l_key, r_key = self.x_f[self.bc_list[1][i], 0], self.x_f[self.bc_list[2][i], 0]
                self.x_u_l = np.vstack((self.x_u_l, l_key))
                self.x_u_r = np.vstack((self.x_u_r, r_key))
                self.u_con = np.vstack((self.u_con, self.bc_dict[l_key][0]))
        
        if self.bc_list[3]:
            self.u_x_pred = tf.transpose(tf.constant([[]]))
            self.u_x = np.zeros((0, 1))
            for i in range(len(self.bc_list[3])):
                index = self.bc_list[3][i]
                key = self.x_f[index, 0]
                self.u_x_pred = tf.concat((self.u_x_pred, self.grads_pred[0][index:index + 1, :]), axis=0)
                self.u_x = np.vstack((self.u_x, self.bc_dict[key][1]))
        
        if self.bc_list[6]:
            self.u_xx_pred = tf.transpose(tf.constant([[]]))
            self.u_xx = np.zeros((0, 1))
            for i in range(len(self.bc_list[6])):
                index = self.bc_list[6][i]
                key = self.x_f[index, 0]
                self.u_xx_pred = tf.concat((self.u_xx_pred, self.grads_pred[1][index:index + 1, :]), axis=0)
                self.u_xx = np.vstack((self.u_xx, self.bc_dict[key][2] / self.EI)) # sign adjustment
        
        if self.bc_list[9]:
            self.u_xxx_pred = tf.transpose(tf.constant([[]]))
            self.u_xxx = np.zeros((0, 1))
            for i in range(len(self.bc_list[9])):
                index = self.bc_list[9][i]
                key = self.x_f[index, 0]
                self.u_xxx_pred = tf.concat((self.u_xxx_pred, self.grads_pred[2][index:index + 1, :]), axis=0)
                self.u_xxx = np.vstack((self.u_xxx, -self.bc_dict[key][3] / self.EI)) # sign adjustment
                
        if self.bc_list[4]:
            self.u_x_l_pred = tf.transpose(tf.constant([[]]))
            self.u_x_r_pred = tf.transpose(tf.constant([[]]))
            self.u_x_con = np.zeros((0, 1))
            for i in range(len(self.bc_list[4])):
                l_index, r_index = self.bc_list[4][i], self.bc_list[5][i]
                l_key = self.x_f[l_index, 0]
                self.u_x_l_pred = tf.concat((self.u_x_l_pred, self.grads_pred[0][l_index:l_index + 1, :]), axis=0)
                self.u_x_r_pred = tf.concat((self.u_x_r_pred, self.grads_pred[0][r_index:r_index + 1, :]), axis=0)
                self.u_x_con = np.vstack((self.u_x_con, self.bc_dict[l_key][1])) # sign adjustment
                
        if self.bc_list[7]:
            self.u_xx_l_pred = tf.transpose(tf.constant([[]]))
            self.u_xx_r_pred = tf.transpose(tf.constant([[]]))
            self.u_xx_con = np.zeros((0, 1))
            for i in range(len(self.bc_list[7])):
                l_index, r_index = self.bc_list[7][i], self.bc_list[8][i]
                l_key = self.x_f[l_index, 0]
                self.u_xx_l_pred = tf.concat((self.u_xx_l_pred, self.grads_pred[1][l_index:l_index + 1, :]), axis=0)
                self.u_xx_r_pred = tf.concat((self.u_xx_r_pred, self.grads_pred[1][r_index:r_index + 1, :]), axis=0)
                self.u_xx_con = np.vstack((self.u_xx_con, self.bc_dict[l_key][2] / self.EI)) # sign adjustment
        
        if self.bc_list[10]:
            self.u_xxx_l_pred = tf.transpose(tf.constant([[]]))
            self.u_xxx_r_pred = tf.transpose(tf.constant([[]]))
            self.u_xxx_con = np.zeros((0, 1))
            for i in range(len(self.bc_list[10])):
                l_index, r_index = self.bc_list[10][i], self.bc_list[11][i]
                l_key = self.x_f[l_index, 0]
                self.u_xxx_l_pred = tf.concat((self.u_xxx_l_pred, self.grads_pred[2][l_index:l_index + 1, :]), axis=0)
                self.u_xxx_r_pred = tf.concat((self.u_xxx_r_pred, self.grads_pred[2][r_index:r_index + 1, :]), axis=0)
                self.u_xxx_con = np.vstack((self.u_xxx_con, self.bc_dict[l_key][3] / self.EI))
                
        
        """
        # unpacks the gradients for use in the loss function
        if self.u_x_index:
            self.u_x_pred = tf.transpose(tf.constant([[]]))
            for i in range(len(self.u_x_index)):
                index = self.u_x_index[i]
                self.u_x_pred = tf.concat((self.u_x_pred, self.grads_pred[0][index:index + 1, :]), axis=0)

        if self.u_xx_index:
            self.u_xx_pred = tf.transpose(tf.constant([[]]))
            for i in range(len(self.u_xx_index)):
                index = self.u_xx_index[i]
                self.u_xx_pred = tf.concat((self.u_xx_pred, self.grads_pred[1][index:index + 1, :]), axis=0)

        if self.u_xxx_index:
            self.u_xxx_pred = tf.transpose(tf.constant([[]]))
            for i in range(len(self.u_xxx_index)):
                index = self.u_xxx_index[i]
                self.u_xxx_pred = tf.concat((self.u_xxx_pred, self.grads_pred[2][index:index + 1, :]), axis=0)
        """
    
    
    def interpret_bc(self, key, bc_val, index):
        # 3 * index: boundary points
        # 3 * index + 1: left
        # 3 * index + 2: right
        
        if not np.isnan(bc_val):
            # flag to avoid duplication
            if self.bc_dict[key][-1] == 0:
                self.bc_list[3 * index].append(self.index_dict[key])
            if self.bc_dict[key][-1] == -1:
                self.bc_list[3 * index + 1].append(self.index_dict[key])
            if self.bc_dict[key][-1] == 1:
                self.bc_list[3 * index + 2].append(self.index_dict[key])
        
    
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases
    
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0 * (X - self.domain[0]) / (self.domain[1] - self.domain[0]) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_u(self, x):
        u = self.neural_net(x, self.weights, self.biases)
        return u
    
    def net_f(self, x):
        u = self.net_u(x)
        u_x = tf.gradients(u, x)[0][:, 0:1]
        u_xx = tf.gradients(u_x, x)[0][:, 0:1]
        u_xxx = tf.gradients(u_xx, x)[0][:, 0:1]
        u_xxxx = tf.gradients(u_xxx, x)[0][:, 0:1]
        f = self.EI * u_xxxx + self.loads_tf
        return f, [u_x, u_xx, u_xxx]
    
    def callback(self, loss):
        print('Loss: ', loss)
    
    def train(self):
         
        tf_dict = {self.x_f_tf: self.x_f,
                   self.x_u_tf: self.x_u,
                   self.x_u_l_tf: self.x_u_l if self.bc_list[1] else np.zeros((1, 1)),
                   self.x_u_r_tf: self.x_u_r if self.bc_list[2] else np.zeros((1, 1)),
                   self.u_tf: self.u,
                   self.u_x_tf: self.u_x if self.bc_list[3] else np.zeros((1, 1)),
                   self.u_xx_tf: self.u_xx if self.bc_list[6] else np.zeros((1, 1)),
                   self.u_xxx_tf: self.u_xxx if self.bc_list[9] else np.zeros((1, 1)),
                   self.u_xx_con_tf: self.u_xx_con if self.bc_list[7] else np.zeros((1, 1)),
                   self.u_xxx_con_tf: self.u_xxx_con if self.bc_list[10] else np.zeros((1, 1)),
                   self.loads_tf: self.loads}
        
        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)
    
    def predict(self, X_star, loads):
        
        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:,0:1]})  
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:,0:1], self.loads_tf: loads})
        #phi_star = self.sess.run(self.phi_pred, {self.x_f_tf: X_star[:, 0:1]})
               
        return u_star, f_star #, phi_star
    
if __name__ == "__main__":
    
    noise = 0.0
    
    layers = [1, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    
    data = scipy.io.loadmat('./sample_cantilever.mat')
    # x: (256, 1) vector containing coordinates from 0:6, equally spaced
    # bc: boundary condition matrices
    # bc shape: (N_u, 5) when ConstantLoad == 1
    # (N_u, 6) when ConstantLoad == 0
    # column format: coordinate, load at coordinate, u, u_x, u_xx, u_xxx
    # 0 for fixed, non-zero vals for prescribed values, nan for free for the last for columns
    EI = data['EI']
    domain = data['domain'][0]
    x = data['x']
    bc = data['BC']
    exact = np.real(data['usol'])
    
    bc_dict, index_dict = utils.unpack_bc(bc, domain, EI)
    
    X_u_train = utils.assemble_disp_bc(bc_dict, index_dict)
    boundary_set = utils.assemble_boundary_set(bc_dict)
    
    N_u = X_u_train.shape[0]
    N_f = 10000
    
    X_f_train = utils.assemble_training_set(boundary_set, domain, N_f)
    
    X_star = x

    #loads = utils.triangular_load(X_f_train, 6, 8, 1)
    #loads_star = utils.triangular_load(X_star, 6, 8, 1)
    loads = utils.uniform_load(X_f_train, 0)
    loads_star = utils.uniform_load(X_star, 0)
   
    grid_x = 1
    grid_y = 1

    loss_weights = [1, 100, 100, 100, 1e5, 1, 1, 1, 1]
    results = np.zeros((256, grid_x * grid_y))
    err_lst = np.zeros((grid_x, grid_y))
    
    for i in range(grid_x):
        # loss_weights[1] = 1 - 0.008 * i
        for j in range(grid_y):
            # loss_weights[2] = 1 - 0.08 * j
            model = PhysicsInformedNN(X_f_train, loads, bc_dict, index_dict, layers, domain, EI, loss_weights)
     
            start_time = time.time()
            model.train()
            elapsed = time.time() - start_time
            print('training time: %.4f' % (elapsed))
    
    
            u_pred, f_pred = model.predict(X_star, loads_star)
            np.savetxt("./u_pred.txt", u_pred)
            np.savetxt("./f_pred.txt", f_pred)
            # np.savetxt("./phi_pred.txt", phi_pred)
            results[:, (grid_x * i + j):(grid_x * i + j + 1)] = u_pred
    
            error_u = np.linalg.norm(exact - u_pred, 2) / np.linalg.norm(exact, 2)
            print('Error u: %e' % (error_u))
            err_lst[i, j] = error_u
    
    """
    np.savetxt("weight_experimentation4.csv", results)
    np.savetxt("err_lst4.csv", err_lst)
    """
