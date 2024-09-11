import tensorflow as tf

from pathlib import Path
import pickle

from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam

from model.data_loader import DataLoader
from model.loss_functions import Loss
from model.callback import CustomCallback


class PhysicsInformedNN(Sequential):
    '''
    This class provides the Physics-Informed Neural Network
    '''       
    # settings read from config (set as class attributes)
    args = ['version', 'seed',
            'N_hidden', 'N_neurons', 'activation',
            'N_epochs', 'learning_rate', 'decay_rate', 'reg_coeff',
            'reg_decay', 'reg_epochs', 'freq_save']
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
        self.loss = Loss(self, config)
        # callback for log recording and saving
        self.callback = CustomCallback(config) 
        # create model path to save logs
        self._path = self.log_path.joinpath(self.version)
        self._path.mkdir(parents=True, exist_ok=True)
        print('*** PINN build & initialized ***')  
        
 
    def build_layers(self, verbose):
        '''
        Builds the network layers
        '''         
        # set seed for weights initialization
        tf.random.set_seed(self.seed)         
        # build input layer
        self.add(InputLayer(input_shape=(1,)))
        # build hidden layers
        for i in range(self.N_hidden):
            self.add(Dense(units=self.N_neurons, 
                           activation=self.activation))
        # build linear output layer
        self.add(Dense(units=1, 
                       activation=None))
        if verbose:
            self.summary()                         

            
    def train(self):
        '''
        Training loop with batch gradiend-descent optimization 
        Samples training data (collocation, IC) at each batch iteration
        '''                 
        # learning rate schedule
        lr_schedule = ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=1000,
            decay_rate=self.decay_rate)          
        # Adam optimizer with default settings for momentum
        self.optimizer = Adam(learning_rate=lr_schedule)    
        print("Training started...")

        reg_coeff = tf.constant(self.reg_coeff, dtype=tf.float32)
        reg_decay = tf.constant(self.reg_decay, dtype=tf.float32)
        
        for epoch in range(self.N_epochs):

            t_col = self.data.collocation() 
            # perform one train step
            if epoch > self.reg_epochs:
                reg_coeff = 0
            train_logs = self.train_step(t_col, reg_coeff)
            # provide logs to callback 
            self.callback.write_logs(train_logs, epoch)
            
            reg_coeff *= reg_decay
            
            if self.freq_save != 0:
                if (epoch % self.freq_save) == 0:
                    self.save_weights(flag=epoch)

        # save log
        self.callback.save_logs(self._path)
        print("Training finished!")
        return self.callback.log

    
    @tf.function
    def train_step(self, t_col, reg_coeff):
        '''
        Performs a single SGD training step by minimizing the 
        IC and physics loss residuals using MSE
        '''      
        # open a GradientTape to record forward/loss pass                   
        with tf.GradientTape() as tape: 
            # inital condition loss
            loss_IC = self.loss.initial_condition()
            # physics loss
            loss_P = self.loss.pendulum(t_col, reg_coeff)
            # final training loss
            loss_train = loss_IC + loss_P
            
        # retrieve gradients
        grads = tape.gradient(loss_train, self.weights)        
        # perform single GD step 
        self.optimizer.apply_gradients(zip(grads, self.weights))              
        # save logs for recording
        train_logs = {'loss_train': loss_train, 'loss_P': loss_P, 'loss_IC': loss_IC}       
        return train_logs
    
    
    def omega(self, t):
        with tf.GradientTape() as tape:
            tape.watch(t)
            theta = self(t)
        omega = tape.gradient(theta, t)
        return omega

    def save_weights(self, path=""):        
        if path == "":
            weights_file = self.log_path.joinpath(f'model_weights/weights.pkl')
        else:
            weights_file = path
        with open(weights_file, 'wb') as pickle_file:
            pickle.dump(self.get_weights(), pickle_file)                    

    def load_weights(self, weights_file):
        with open(weights_file, 'rb') as pickle_file:
            weights = pickle.load(pickle_file)
        self.set_weights(weights)