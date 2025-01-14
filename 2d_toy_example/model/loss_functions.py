import tensorflow as tf


class Loss():
    '''
    This class provides the physics loss function to the network training
    '''   

    args = ['x0', 'y0']


    def __init__(self, model, config, regularization):
        
        # save neural network (weights are updated during training)
        self.model = model

        # load and set class attributes from config
        for arg in self.args:
            setattr(self, arg, config[arg])

        self.regularization_map = {
            "no_reg": None,
            "unstable_fp": self.regularizer_unstable_fp,
            "reg_derivative": self.regularizer_derivative,
            "reg_derivative_unstable_fp": self.regularizer_derivative_unstable_fp
        }
        self.regularizer = self.regularization_map[regularization]

    
    def initial_condition(self):
        '''
        Determines IC loss for angle and velocity
        '''        
        t0 = tf.constant([0.])    
        preds = self.model(t0)
        x0 = preds[0, 0]
        y0 = preds[0, 1]
        
        # IC loss for angle
        loss_IC1 = tf.reduce_mean(tf.square(x0 - self.x0))
        # and velocity
        loss_IC2 = tf.reduce_mean(tf.square(y0 - self.y0))
        return loss_IC1 + loss_IC2

        
    def toy_example(self, t_col, reg_coeff):
        '''
        Determines physics loss residuals of the differential equation
        '''
        res, x, x_t, y, y_t = self.physics_loss(t_col)
        loss = tf.reduce_mean(res) 
        # reg_coeff = tf.reduce_sum(res**4)**0.25 
        if self.regularizer is not None:
            loss += reg_coeff * tf.reduce_mean(self.regularizer(t_col, x, x_t, y, y_t))
            
        return loss 
        
    def physics_loss(self, t_col):
        with tf.GradientTape(persistent=False) as t:
            t.watch(t_col)
            with tf.GradientTape(persistent=False) as tt:
                tt.watch(t_col)
                preds = self.model(t_col)
                x = preds[:, 0]
                y = preds[:, 1]
    
            
                x_t = t.gradient(x, t_col)[:, 0]
                y_t = tt.gradient(y, t_col)[:, 0]
            
               
        res1 = x_t - x * (3 - x - 2*y)
        res2 = y_t - y * (2 - x - y)
    
        return tf.square(res1) + tf.square(res2), x, x_t, y, y_t
        
    def regularizer_unstable_fp(self, t_col, x, x_t, y, y_t):
      
        lam1 = 0.5 * (-1*(3*x + 4*y - 5) - tf.sqrt(tf.nn.relu((3*x + 4*y - 5)**2 - 4 * (2 * x**2 + 4*y**2 + 4*x*y - 7*x - 10*y + 6)) + 10**-12))
        lam2 = 0.5 * (-1*(3*x + 4*y - 5) + tf.sqrt(tf.nn.relu((3*x + 4*y - 5)**2 - 4 * (2 * x**2 + 4*y**2 + 4*x*y - 7*x - 10*y + 6)) + 10**-12))
        reg_loss = tf.nn.relu(lam1) + tf.nn.relu(lam2)
        return reg_loss
    
    def regularizer_fp(self, t_col, x, x_t, y, y_t):
        return tf.exp(-((y - 1)**2 + y**2 + (y + 1)**2))
    
    def regularizer_derivative(self, t_col, x, x_t, y, y_t):
        eps = 0.01
        return tf.exp(-(x_t**2 + y_t**2) / eps)
    
    def regularizer_derivative_unstable_fp(self, t_col, x, x_t, y, y_t):
        eps = 0.01
        return self.regularizer_derivative(t_col, x, x_t, y, y_t) * self.regularizer_unstable_fp(t_col, x, x_t, y, y_t)