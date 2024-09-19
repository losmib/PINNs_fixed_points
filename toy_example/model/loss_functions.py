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

        
    def toy_example(self, t_col, reg_coeff):
        '''
        Determines physics loss residuals of the differential equation
        '''
        # the tf-GradientTape function is used to retreive network derivatives
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t_col)
            y = self.model(t_col)
        y_t = tape.gradient(y, t_col) 
            
        res = y_t - (y - y**3)
        loss = tf.reduce_mean(tf.square(res))
                        
        if self.regularizer is not None:
            loss += reg_coeff * self.regularizer(t_col=t_col, y=y, y_t=y_t)
            
        return loss 
        
        
    def regularizer_unstable_fp(self, t_col, y, y_t):
        y = self.model(t_col)
        reg_loss = tf.reduce_mean(tf.nn.relu(1 - 3 * y**2))
        return reg_loss
    
    
    def regularizer_fp(self, t_col, y, y_t):
        return tf.reduce_mean(tf.exp(-((y - 1)**2 + y**2 + (y + 1)**2)))
    
    def regularizer_derivative(self, t_col, y, y_t):
        eps = 1
        return tf.reduce_mean(tf.exp(-(y_t**2) / eps))
    
    def regularizer_derivative_unstable_fp(self, t_col, y, y_t):
        eps = 1
        return tf.reduce_mean(tf.exp(-(y_t**2) / eps) * tf.nn.relu(1 - 3 * y**2))