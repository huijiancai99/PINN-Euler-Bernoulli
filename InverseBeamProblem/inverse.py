import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
# from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
import strain_generator as sg
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN():
    
    def __init__(self, X_u, X_f, curv, bc, layers, lb, ub, EI, loss_weights):
        
        self.lb = lb
        self.ub = ub
        
        #self.x_u = X_u
        
        self.x_f = X_f
        self.curv = curv

        self.EI = EI

        self.unpack_bc(bc)
        
        self.layers = layers
        
        self.weights, self.biases = self.initialize_NN(layers)
        
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1] - 1])  
        self.u_x_tf = tf.placeholder(tf.float32, shape=[None, self.u_x.shape[1] - 1])
        self.u_xx_tf = tf.placeholder(tf.float32, shape=[None, self.u_xx.shape[1] - 1])
        self.u_xxx_tf = tf.placeholder(tf.float32, shape=[None, self.u_xxx.shape[1] - 1])
        self.u_xx_sensor_tf = tf.placeholder(tf.float32, shape=[None, 1])
        #self.loads_tf = tf.placeholder(tf.float32, shape=[None, self.loads.shape[1]])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.u_pred = self.net_u(self.x_u_tf)
        self.grads_pred = self.net_f(self.x_f_tf)
        
        self.unpack_gradients()
        print(self.u_xxx[:, 1:2])

        
        # check shapes of the self.xxx varst
        # optimize individual loss components and validate results
        # adjust coefficients of the loss terms
        loss_components = [0, 0, 0, 0, 0]
        #loss_components[0] = loss_weights[0] * tf.reduce_mean(tf.square(self.f_pred))
        loss_components[0] = loss_weights[1] * tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
        loss_components[1] = loss_weights[2] * tf.reduce_mean(tf.square(self.u_x_tf - self.u_x_pred)) \
                             if self.u_x_index else 0
        loss_components[2] = loss_weights[3] * tf.reduce_mean(tf.square(self.u_xx_tf - self.u_xx_pred)) \
                             if self.u_xx_index else 0
        loss_components[3] = loss_weights[3] * tf.reduce_mean(tf.square(self.u_xx_sensor_tf - self.grads_pred[1]))
        loss_components[4] = loss_weights[4] * tf.reduce_mean(tf.square(self.u_xxx_tf + self.u_xxx_pred)) \
                             if self.u_xxx_index else 0
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
                u_xx.append([bc[i, 0], bc[i, -2] / self.EI])
                self.u_xx_index.append(i)
            if not np.isnan(bc[i, -1]):
                u_xxx.append([bc[i, 0], bc[i, -1] / self.EI])
                self.u_xxx_index.append(i)
        
        # keeps the shape constant to avoid errors
        self.u = np.array(u) if self.u_index else np.zeros((1, 1))
        self.u_x = np.array(u_x) if self.u_x_index else np.zeros((1, 1))
        self.u_xx = np.array(u_xx) if self.u_xx_index else np.zeros((1, 1))
        self.u_xxx = np.array(u_xxx) if self.u_xxx_index else np.zeros((1, 1))
            

    def unpack_gradients(self):
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
        
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
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
        # f = self.EI * u_xxxx + self.loads_tf
        return [u_x, u_xx, u_xxx]
    
    def callback(self, loss):
        print('Loss: ', loss)
    
    def train(self):
        
        
        tf_dict = {self.x_u_tf: self.u[:, 0:1], self.u_tf: self.u[:, 1:2],
                   self.x_f_tf: self.x_f, self.u_x_tf: self.u_x[:, 1:2],
                   self.u_xx_tf: self.u_xx[:, 1:2], self.u_xxx_tf: self.u_xxx[:, 1:2],
                   self.u_xx_sensor_tf: self.curv}
                   #self.loads_tf: self.loads}
            
        
        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)
        
    
    def predict(self, X_star, loads):
        
        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:,0:1]})  
        #f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:,0:1], self.loads_tf: loads})
        #phi_star = self.sess.run(self.phi_pred, {self.x_f_tf: X_star[:, 0:1]})
               
        return u_star #, f_star , phi_star
    
if __name__ == "__main__":
    
    
    
    noise = 0.0
    
    N_u = 2
    
    name = "sample_ss_cmin"
    
    data = scipy.io.loadmat('./input/' + name + '.mat')
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
    
    X_star = x
    curv_star = sg.ss_concentrated_moment_in(X_star, 100, domain, EI, 3)
    
    # domain bounds
    lb = X_star.min(0)
    ub = X_star.max(0)
    
    layers = [1, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    X_u_train = bc[:, 0:1]
    
    save_path = "./results/" + name
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    err_lst = np.zeros((0, 1))
    for N_f in [2, 5, 10, 100, 1000, 10000]:
        X_f_train = lb + (ub - lb) * lhs(1, N_f)
        X_f_train = np.vstack((X_u_train, X_f_train))
        
        curv = sg.ss_concentrated_moment_in(X_f_train, 100, domain, EI, 3)
        
        loss_weights = [100, 100, 100, 100, 100]

        model = PhysicsInformedNN(X_u_train, X_f_train, curv, bc, layers, lb, ub, EI, loss_weights)

        start_time = time.time()
        model.train()
        elapsed = time.time() - start_time
        print('training time: %.4f' % (elapsed))

        u_pred = model.predict(X_star, curv_star)
        np.savetxt(save_path + "/u_pred_" + str(N_f) + ".txt", u_pred)
        # np.savetxt("./f_pred.txt", f_pred)
        # np.savetxt("./phi_pred.txt", phi_pred)
        # results[:, (grid_x * i + j):(grid_x * i + j + 1)] = u_pred

        error_u = np.linalg.norm(exact - u_pred, 2) / np.linalg.norm(exact, 2)
        print('Error u: %e' % (error_u))
        err_lst = np.vstack((err_lst, error_u))
    np.savetxt(save_path + "/err_lst.txt", err_lst)
