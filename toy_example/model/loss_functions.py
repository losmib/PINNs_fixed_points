import tensorflow as tf


class Loss():
    '''
    This class provides the physics loss function to the network training
    '''   
    def __init__(self, model):
        
        # save neural network (weights are updated during training)
        self.model = model

        
    def toy_example(self, t_col, reg_coeff):
        '''
        Determines physics loss residuals of the differential equation
        '''
        # the tf-GradientTape function is used to retreive network derivatives
        with tf.GradientTape() as tape:
            tape.watch(t_col)
            y = self.model(t_col)
        y_t = tape.gradient(y, t_col) 
        
        res = y_t - (y - y**3)
        loss = tf.reduce_mean(tf.square(res))
        
        loss += reg_coeff * self.regularizer_fp(y)
        return loss 
        
        
    def regularizer_unstable_fp(self, t_col):
        y = self.model(t_col)
        reg_loss = tf.reduce_mean(tf.nn.relu(1 - 3 * y**2))
        return reg_loss
    
    
    def regularizer_fp(self, y):
        return tf.reduce_mean(tf.exp(-((y - 1)**2 + y**2 + (y + 1)**2)))
    
    def regularizer_derivative(self, y_t):
        eps = 10**0
        return tf.reduce_mean(tf.exp(-(y_t**2) / eps))