import numpy as np
import tensorflow as tf

class Loss():
    '''
    This class provides the physics loss function to the network training
    '''   
    def __init__(self, model, regularization):       
        # save neural network (weights are updated during training)
        self.model = model
        
        regularization_map = {
            "no_reg": None,
            "unstable_fp": self.regularizer_unstable_fp,
            "reg_derivative": self.regularizer_derivative,
            "reg_derivative_unstable_fp": self.regularizer_derivative_unstable_fp
        }
        self.regularizer = regularization_map[regularization]
   
    @tf.function
    def initial_condition(self, X_IC, u_IC):
        '''
        Standard MSE loss for initial condition
        '''
        u_pred = self.model(X_IC)
        loss_IC = tf.reduce_mean(tf.square(u_pred - u_IC))
        
        return loss_IC
    
    
    @tf.function
    def boundary_condition(self, X_BC_top, X_BC_bottom):
        '''
        Loss function for periodic boundary conditions 
        Separate Loss for absolute value and derivative
        '''     
        # GradientTape for determining derivatives on top wall     
        with tf.GradientTape() as t:
            t.watch(X_BC_top)
            u_top = self.model(X_BC_top)
        u_d_top = t.batch_jacobian(u_top, X_BC_top) 
        # GradientTape for determining derivatives on bottom wall  
        with tf.GradientTape() as t:
            t.watch(X_BC_bottom)
            u_bottom = self.model(X_BC_bottom)
        u_d_bottom = t.batch_jacobian(u_bottom, X_BC_bottom) 
        
        # get derivatives w.r.t x
        u_x_top = u_d_top[:, 0, 0:1]
        u_x_bottom = u_d_bottom[:, 0, 0:1]
        
        # periodic loss for absolute values and derivatives
        loss_BC1 = tf.reduce_mean(tf.square(u_top - u_bottom))
        loss_BC2 = tf.reduce_mean(tf.square(u_x_top - u_x_bottom))   
        # combine both losses
        return loss_BC1 + loss_BC2
    
    
    @tf.function
    def allen_cahn(self, X_col, reg_coeff):
        '''
        Physics loss function for the Allen-Cahn Equation
        '''    
        # tape forward propergation to retrieve gradients
        with tf.GradientTape() as t:
            t.watch(X_col)
            with tf.GradientTape() as tt:
                tt.watch(X_col)
                U = self.model(X_col)
                U_d = tt.batch_jacobian(U, X_col) 
            U_dd = t.batch_jacobian(U_d, X_col)
            
        # get prediction and derivatives
        u = U[:, 0]   
        u_t = U_d[:, 0, 1]
        u_xx = U_dd[:, 0, 0, 0]
        
        # Allen-Cahn equation
        res = u_t - 0.0001*u_xx + 5*u**3 - 5*u 
        loss = tf.reduce_mean(tf.square(res))
        if self.regularizer is not None:
            loss += reg_coeff * tf.reduce_mean(self.regularizer(X_col, u_t, u_xx, u))
        return loss
    
    
    def regularizer_derivative(self, X_col, u_t, u_xx, u):
        eps = 10**0
        return tf.exp(-(u_t**2) / eps)

    def regularizer_unstable_fp(self, X_col, u_t, u_xx, u):
        loss = tf.nn.relu(0.0001 * u_xx - 15 * u**2 + 5)
        return loss
    
    def regularizer_derivative_unstable_fp(self, X_col, u_t, u_xx, u):
        return self.regularizer_derivative(X_col, u_t, u_xx, u) * \
            self.regularizer_derivative_unstable_fp(X_col, u_t, u_xx, u)