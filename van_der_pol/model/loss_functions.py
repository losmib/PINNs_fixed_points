import numpy as np
import tensorflow as tf

class Loss():
    '''
    This class provides the physics loss function 
    '''       
     # settings read from config (set as class attributes)
    args = ['mu', 'x0', 'x_t0']
    
    
    def __init__(self, model, config, regularization):
        
        # load and set class attributes from config
        for arg in self.args:
            setattr(self, arg, config[arg])
            
        # convert degrees to radians
        #self.x0 = np.radians(self.theta0).astype(np.float32)
        #self.x_t0 = np.radians(self.omega0).astype(np.float32)
        
        
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
            x0 = self.model(t0)
        x_t0 = tape.gradient(x0, t0)
        
        # IC loss for angle
        loss_IC1 = tf.reduce_mean(tf.square(x0 - self.x0))
        # and velocity
        loss_IC2 = tf.reduce_mean(tf.square(x_t0 - self.x_t0))
        return loss_IC1 + loss_IC2
               
        
    def van_der_pol(self, t_col, reg_coeff):
        '''
        Determines physics loss of the pendulum's differential equation
        '''
        res_squared, x, x_t, x_tt = self.physics_loss(t_col)
       
        loss = tf.reduce_mean(res_squared)
        if self.regularizer is not None:
            loss += reg_coeff * tf.reduce_mean(self.regularizer(t_col, x, x_t, x_tt))
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
                x = self.model(t_col)
                x_t = tt.gradient(x, t_col) 
            x_tt = t.gradient(x_t, t_col)
        
        res = x_tt - self.mu * (1 - x**2) * x_t + x
        return tf.square(res), x, x_t, x_tt

    def regularizer_unstable_fp(self, t_col, x, x_t, x_tt):
        lam1 = 0.5 * (self.mu * (1 + x**2) + tf.sqrt(tf.nn.relu(self.mu**2 + 2*self.mu**2*x**2 + self.mu**2*x**4 - 4 - 8*self.mu*x_t*x)))
        lam2 = 0.5 * (self.mu * (1 + x**2) - tf.sqrt(tf.nn.relu(self.mu**2 + 2*self.mu**2*x**2 + self.mu**2*x**4 - 4 - 8*self.mu*x_t*x)))
        return tf.nn.relu(lam1) + tf.nn.relu(lam2)

    def regularizer_derivative(self, t_col, x, x_t, x_tt):
        eps = 1
        loss = tf.exp(-(x_tt**2 + x_t**2) / eps)
        return loss
    
    def regularizer_derivative_unstable_fp(self, t_col, x, x_t, x_tt):
        eps = 1
        return self.regularizer_derivative(t_col, x, x_t, x_tt) * \
            self.regularizer_unstable_fp(t_col, x, x_t, x_tt)