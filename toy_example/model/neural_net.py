import tensorflow as tf

from pathlib import Path
import pickle

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam

from model.data_loader import DataLoader
from model.loss_functions import Loss
from model.callback import CustomCallback


class PhysicsInformedNN(Model):
    '''
    This class provides the basic Physics-Informed Neural Network
    with hard constraints for the initial condition
    '''       
    # settings read from config (set as class attributes)
    args = ['version', 'seed', 'y0',
            'N_hidden', 'N_neurons', 'activation',
            'N_epochs', 'learning_rate', 'decay_rate', 'reg_epochs', 'reg_coeff', 'reg_decay', 'freq_save']
    # default log Path
    log_path = Path('logs')
    
    
    def __init__(self, config, verbose=False): 

        # call parent constructor & build NN
        super().__init__(name='PhysicsInformedNN')    
        # load and set class attributes from config
        for arg in self.args:
            setattr(self, arg, config[arg])
        
        self.reg_epochs = int(self.reg_epochs * self.N_epochs)
        
        self.build_layers(verbose) 
        # data loader for sampling data at each training epoch
        self.data = DataLoader(config) 
        # loss functions for IC and physics
        self.loss = Loss(self)
        # callback for log recording and saving
        self.callback = CustomCallback(config) 
        # create model path to save logs
        self._path = self.log_path.joinpath(self.version)
        self._path.mkdir(parents=True, exist_ok=True)
        print('*** PINN build & initialized ***')            

        
    def build_layers(self, verbose):
        '''
        Builds nested neural network (tf.Sequential) and its layers
        The nested neural network is needed for the hard constraints)
        '''      
        # set seed for weights initialization
        tf.random.set_seed(self.seed)     
        # create nested sequential model 
        self.neural_net = Sequential(name='nested_PINN') 
        # build input layer
        self.neural_net.add(InputLayer(input_shape=(1,)))
        # build hidden layers
        for i in range(self.N_hidden):
            self.neural_net.add(Dense(units=self.N_neurons, 
                                      activation=self.activation))
        # build linear output layer
        self.neural_net.add(Dense(units=1, activation=None))
        # provide weights to outer class
        self._weights = self.neural_net.weights
        # print network summary
        if verbose:
            self.neural_net.summary() 
                     
    
    def call(self, t):
        '''
        Overwrites default call function for
        implementing hard constraints (initial condition)
        '''   
        # hyperbolic tangent distance function
        return self.y0 + tf.math.tanh(t) * self.neural_net(t)
    
    
    def train(self):                                     
        '''
        Training loop with batch gradiend-descent optimization 
        Samples training data (collocation) at each epoch
        '''                                          
        # learning rate schedule
        lr_schedule = ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=1000,
            decay_rate=self.decay_rate)                                    
        # Adam optimizer with default settings for momentum
        self.optimizer = Adam(learning_rate=lr_schedule)   
                      
        print("Training started...")
        for epoch in range(self.N_epochs):
            
            # sample collocation points
            t_col = self.data.collocation()                      
            # perform one train step
            if epoch > self.reg_epochs:
                self.reg_coeff = 0
                
            train_logs = self.train_step(t_col, self.reg_coeff)
            # provide logs to callback 
            self.callback.write_logs(train_logs, epoch)
            
            self.reg_coeff *= self.reg_decay
            
            if self.freq_save != 0:
                if (epoch % self.freq_save) == 0:
                    self.save_weights(flag=epoch)
        
        # save log
        self.callback.save_logs(self._path)
        print("### Training finished ###")
        return self.callback.log
    
    
    @tf.function
    def train_step(self, t_col, reg):
        '''
        Performs a single gradient-descent optimization step
        '''    
        # open a GradientTape to record forward/loss pass                   
        with tf.GradientTape() as tape:     
            # get physcics loss of toy example equation
            loss = self.loss.toy_example(t_col, reg)
            
        # retrieve gradients
        grads = tape.gradient(loss, self.weights)        
        # perform single GD step 
        self.optimizer.apply_gradients(zip(grads, self.weights))       
        
        # save logs for recording
        train_logs = {'loss': loss}       
        return train_logs
    
    def save_weights(self, flag=''):        
        weights_file = self.log_path.joinpath(f'model_weights/weights_{flag}.pkl')
        with open(weights_file, 'wb') as pickle_file:
            pickle.dump(self.neural_net.get_weights(), pickle_file)                    

    def load_weights(self, weights_file):
        with open(weights_file, 'rb') as pickle_file:
            weights = pickle.load(pickle_file)
        self.neural_net.set_weights(weights)