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
    
    def __init__(self, X_u, X_f, label_b, label_con, loads, layers, domain, EI, loss_weights):
        
        self.domain = domain
        
        self.x_u = X_u
        self.x_f = X_f
        self.label_b = label_b
        self.label_con = label_con
        
        self.loads = loads

        self.EI = EI
        
        self.layers = layers
        self.bc_list = [[], [], [], [], [], [], [], [], [], [], [], []]
        
        self.weights, self.biases = self.initialize_NN(layers)
        
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u[0].shape[1]])
        
        self.u_tf = tf.placeholder(tf.float32, shape=[None, 1])  
        self.u_x_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u_xx_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u_xxx_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        self.u_xx_con_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u_xxx_con_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        self.loads_tf = []
        self.x_f_tf = []
        for i in range(len(self.x_f)):
            self.loads_tf.append(tf.placeholder(tf.float32, shape=[None, self.loads[0].shape[1]]))
            self.x_f_tf.append(tf.placeholder(tf.float32, shape=[None, self.x_f[0].shape[1]]))
       
        
        self.u_pred = self.net_u(self.x_u_tf)

        self.f_pred, self.grads_pred = tf.transpose(tf.constant([[]])), []
        for i in range(len(self.x_f)):
            f_pred, grads_pred = self.net_f(self.x_f_tf[i], self.loads_tf[i])
            self.f_pred = tf.concat((self.f_pred, f_pred), axis=0)
            self.grads_pred.append(grads_pred)
        
        self.u_b_pred = [0, 0, 0]
        self.u_con_l_pred = [0, 0, 0, 0]
        self.u_con_r_pred = [0, 0, 0, 0]
        
        self.unpack_gradients()
        # check shapes of the self.xxx varst
        # optimize individual loss components and validate results
        # adjust coefficients of the loss terms
        loss_components = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        # residual
        loss_components[0] = loss_weights[0] * tf.reduce_mean(tf.square(self.f_pred)) # come back last to modify
        # bc of u
        loss_components[1] = loss_weights[1] * tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
        # bc of u_x
        loss_components[2] = loss_weights[2] * tf.reduce_mean(tf.square(self.u_x_tf - self.u_b_pred[0])) \
                             if self.x_u[1].shape[0] else 0
        # bc of u_xx
        loss_components[3] = loss_weights[3] * tf.reduce_mean(tf.square(self.u_xx_tf - self.u_b_pred[1])) \
                             if self.x_u[2].shape[0] else 0
        # bc of u_xxx
        loss_components[4] = loss_weights[4] * tf.reduce_mean(tf.square(self.u_xxx_tf - self.u_b_pred[2])) \
                             if self.x_u[3].shape[0] else 0
        # bc of u (continuity)
        loss_components[5] = loss_weights[5] * tf.reduce_mean(tf.square(self.u_con_l_pred[0] - self.u_con_r_pred[0])) \
                             if self.label_con[0].shape[0] else 0
        
        # bc of u_x (continuity)
        loss_components[6] = loss_weights[6] * tf.reduce_mean(tf.square(self.u_con_l_pred[1] - self.u_con_r_pred[1])) \
                             if self.label_con[1].shape[0] else 0
                             
        # bc of u_xx (continuity)
        loss_components[7] = loss_weights[7] * tf.reduce_mean(tf.square(self.u_con_l_pred[2] - self.u_con_r_pred[2] - self.u_xx_con_tf)) \
                             if self.label_con[2].shape[0] else 0
        
        # bc of u_xxx (continuity)
        loss_components[8] = loss_weights[8] * tf.reduce_mean(tf.square(self.u_con_l_pred[3] - self.u_con_r_pred[3] - self.u_xxx_con_tf)) \
                             if self.label_con[3].shape[0] else 0
        
        
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

    def unpack_gradients(self):
        # gradients on boundary
        for i in range(3):
            if self.x_u[i + 1].shape[0]:
                if self.x_u[i + 1].shape[0] == 2:
                    u_b_pred = tf.concat((self.grads_pred[0][i + 1][0:1, :], self.grads_pred[-1][i + 1][1:2, :]), axis=0)
                    print(i)
                if self.x_u[i + 1][0, 0] == self.domain[0]:
                    u_b_pred = self.grads_pred[0][i + 1][0:1, :]
                if self.x_u[i + 1][0, 0] == self.domain[1]:
                    u_b_pred = self.grads_pred[-1][i + 1][1:2, :]
                self.u_b_pred[i] = u_b_pred
        
        # gradients for continuity
        num_segments = len(self.x_f)
        for i in range(4):
            u_con_l_pred = tf.transpose(tf.constant([[]]))
            u_con_r_pred = tf.transpose(tf.constant([[]]))
            for j in range(num_segments - 1):
                u_con_l_pred = tf.concat((u_con_l_pred, self.grads_pred[j][i][1:2, 0:1]), axis=0)
                u_con_r_pred = tf.concat((u_con_r_pred, self.grads_pred[j + 1][i][0:1, 0:1]), axis=0)
            
            self.u_con_l_pred[i] = u_con_l_pred
            self.u_con_r_pred[i] = u_con_r_pred
    
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
    
    def net_f(self, x, loads):
        u = self.net_u(x)
        u_x = tf.gradients(u, x)[0][:, 0:1]
        u_xx = tf.gradients(u_x, x)[0][:, 0:1]
        u_xxx = tf.gradients(u_xx, x)[0][:, 0:1]
        u_xxxx = tf.gradients(u_xxx, x)[0][:, 0:1]
        f = self.EI * u_xxxx + loads
        return f, [u, u_x, u_xx, u_xxx]
    
    def callback(self, loss):
        print('Loss: ', loss)
    
    def train(self):
        
        tf_dict = {self.x_u_tf: self.x_u[0],
                   self.u_tf: self.label_b[0],
                   self.u_x_tf: self.label_b[1] if self.label_b[1].shape[0] else np.zeros((1, 1)),
                   self.u_xx_tf: self.label_b[2] if self.label_b[2].shape[0] else np.zeros((1, 1)),
                   self.u_xxx_tf: self.label_b[3] if self.label_b[3].shape[0] else np.zeros((1, 1)),
                   self.u_xx_con_tf: self.label_con[2] if self.label_con[0].shape[0] else np.zeros((1, 1)),
                   self.u_xxx_con_tf: self.label_con[3] if self.label_con[1].shape[0] else np.zeros((1, 1))}
        
        for i in range(len(self.x_f)):
            tf_dict[self.x_f_tf[i]] = self.x_f[i]
            tf_dict[self.loads_tf[i]] = self.loads[i]
        
        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)
    
    def predict(self, X_star, loads, X_star_con):
        feed_dict = {}
        for i in range(len(self.x_f)):
            feed_dict[self.x_f_tf[i]] = X_star[i]
            feed_dict[self.loads_tf[i]] = loads[i]
        
        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star_con})  
        f_star = self.sess.run(self.f_pred, feed_dict)
        
        pred1 = self.sess.run(self.grads_pred[0][3], {self.x_f_tf[0]: X_star[0][:, 0:1]})
        pred2 = self.sess.run(self.grads_pred[1][3], {self.x_f_tf[1]: X_star[1][:, 0:1]})
        u_xxx_star = np.vstack((pred1, pred2))
               
        return u_star, f_star, u_xxx_star
    
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
