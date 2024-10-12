import numpy as np
import tensorflow as tf

class Loss():
    '''
    This class provides the physics loss function 
    '''       
     # settings read from config (set as class attributes)
    args = ['g', 'l', 'theta0', 'omega0']
    
    
    def __init__(self, model, config, regularization):
        
        # load and set class attributes from config
        for arg in self.args:
            setattr(self, arg, config[arg])
            
        # convert degrees to radians
        self.theta0 = np.radians(self.theta0)
        self.omega0 = np.radians(self.omega0)
        
        
        # save neural network (weights are updated during training)
        self.model = model
        
        regularization_map = {
            "no_reg": None,
            "unstable_fp": self.regularizer_unstable_fp,
            "reg_derivative": self.regularizer_derivative,
            "reg_derivative_unstable_fp": self.regularizer_derivative_unstable_fp
        }
        self.regularizer = regularization_map[regularization]
        
        
    def initial_condition(self):
        '''
        Determines IC loss for angle and velocity
        '''        
        t0 = tf.constant([0.])    
        with tf.GradientTape() as tape:
            tape.watch(t0)
            theta0 = self.model(t0)
        omega0 = tape.gradient(theta0, t0)
        
        # IC loss for angle
        loss_IC1 = tf.reduce_mean(tf.square(theta0 - self.theta0))
        # and velocity
        loss_IC2 = tf.reduce_mean(tf.square(omega0 - self.omega0))
        return loss_IC1 + loss_IC2
               
        
    def pendulum(self, t_col, reg_coeff):
        '''
        Determines physics loss of the pendulum's differential equation
        '''
        res_squared, theta, omega, omega_t = self.physics_loss(t_col)
       
        loss = tf.reduce_mean(res_squared)
        if self.regularizer is not None:
            loss += reg_coeff * tf.reduce_mean(self.regularizer(t_col, theta, omega_t, omega))
        return loss

    def physics_loss(self, t_col):
        """
        Physics loss

        :param t_col: colocation points
        """
        with tf.GradientTape() as t:
            t.watch(t_col)
            with tf.GradientTape() as tt:
                tt.watch(t_col)    
                theta = self.model(t_col)
            omega = tt.gradient(theta, t_col) 
        omega_t = t.gradient(omega, t_col)
        
        res = omega_t - self.g/self.l * tf.math.sin(theta)
        return tf.square(res), theta, omega, omega_t

    def regularizer_unstable_fp(self, t_col, theta, omega, omega_t):
        a = 90 - self.theta0
        loss = tf.nn.relu(tf.cos(theta + a) * self.g / self.l)
        return loss
    
    def regularizer_derivative(self, t_col, theta, omega, omega_t):
        eps = 10**-2
        loss = tf.exp(-(omega_t**2 + omega**2) / eps)
        return loss
    
    def regularizer_derivative_unstable_fp(self, t_col, theta, omega, omega_t):
        eps = 0.01
        return self.regularizer_derivative(omega_t, omega, t_col) * \
            self.regularizer_derivative_unstable_fp(omega_t, omega, t_col)