import numpy as np
import tensorflow as tf
from scipy.integrate import solve_ivp

from numpy.random import random

class DataLoader():
 
    # settings read from config (set as class attributes)
    args = ['seed', 'T', 'x0', 'y0', 'N_col']
    
    
    def __init__(self, config):
        
        # load and set class attributes from config
        for arg in self.args:
            setattr(self, arg, config[arg])
        # set seed for data sampling
        np.random.seed(self.seed)
        
        
    def array2tensor(self, array, exp_dim=True):
        '''
        Auxiliary function: converts numpy-array to tf-tensor
        '''         
        if exp_dim:
            array = np.expand_dims(array, axis=1)       
        return tf.convert_to_tensor(array, dtype=tf.float32)
    
                
    def t_line(self, t_delta=0.01, tensor=True): 
        '''
        Returns an equally-spaced data array for postprocessing
        and visualization of final predictions
        '''    
        t_line = np.arange(0, self.T, t_delta)  
        
        if tensor == True:
            return self.array2tensor(t_line)
        else:
            return t_line
    

    def collocation(self, N=None):
        '''
        Returns an uniformly sampled collocation data set
        '''      
        # take default data settings if N is not provided
        N = self.N_col if N == None else N       
        t_col = self.T * random(N)  
        return self.array2tensor(t_col)
    

    def diff_equations(self, t, y):
        '''
        Auxiliary function: Used in Runge-Kutta Integration
        '''  
        x, y = y[0], y[1]
        return np.array([x * (3 - x - 2 * y), y * (2 - x - y)]) 
    

    def reference(self, N_eval=None):
        '''
        Determines reference solution by using Runge-Kutta Integration
        '''          
        if N_eval is None:
            t_line = self.t_line(tensor=False)
        else:
            t_line = np.linspace(0, self.T, N_eval) 
            
        # initial conditions
        init_y = [self.x0, self.y0]

        # solve ODE
        results = solve_ivp(self.diff_equations, (0, max(t_line)), 
                            init_y, method='RK45', t_eval=t_line, 
                            rtol=1e-8)
        
        t_line = self.array2tensor(results.t)
        x = self.array2tensor(results.y[0])
        y = self.array2tensor(results.y[1])
        return t_line, x, y