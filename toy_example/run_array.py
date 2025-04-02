import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from configs.config_loader import load_config
from model.neural_net import PhysicsInformedNN
from model.plots import learning_curves, toy_example_dynamics
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


NUM_TRAINING_RUNS = 20


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, 
                        default='config/default.yaml',
                        help="Path to configuration file")
    args = parser.parse_args()
    return args


def train_and_test(config):

    PINN = PhysicsInformedNN(config, verbose=True)
   
    training_log = PINN.train()
    
    t_line = PINN.data.t_line()
    # get reference solution (analytical)
    y_true = PINN.data.reference(t_line)
    # get PINN prediction
    y_pred = PINN(t_line)
    loss = mean_squared_error(y_true, y_pred)
    loss_success = np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true) < 0.15
    return loss, loss_success


if __name__ == "__main__":

    args = parse_arguments()
    losses = []
    loss_successes = []
    # train ten unique instances
    for seed in range(NUM_TRAINING_RUNS):
        
        time_start = time()

        # Loading configuration file
        config = load_config(
            file=args.config,
            config_update={'seed': seed}
        )
        config["regularization"] = "reg_derivative"
        config["reg_decay"] = "linear"
        config["reg_coeff"] = 1.0
        config["reg_epochs"] = 0.5
        # run code with config
        loss, loss_success = train_and_test(config)
        losses.append(loss)
        loss_successes.append(loss_success)

        time_end = time()
        print(f"Finished in {time_end-time_start:.1f} seconds!")

    table_entry = pd.DataFrame({k: [v] for k, v in config.items()})
    
    table_entry["mean_loss"] = np.mean(losses)
    table_entry["loss_successes_percent"] = np.sum(loss_successes) / float(NUM_TRAINING_RUNS)
    
    dirname = "regularized_PINN/reg_derivative"

    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    table_entry.to_csv(f"{dirname}/results_vanilla_T{config['T']}-y0{config['y0']}.csv")