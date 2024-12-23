from itertools import product
from typing import Any, Dict, Iterable

from sklearn.metrics import mean_squared_error
from configs.config_loader import load_config
from model.neural_net import PhysicsInformedNN
from model.plots import learning_curves, toy_example_dynamics, loss_over_tcoll, plot_regularization
import pandas as pd
import numpy as np
import os
import re
import tensorflow as tf

NUM_TRAINING_RUNS = 20


def grid_parameters(parameters: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
        for params in product(*parameters.values()):
            yield dict(zip(parameters.keys(), params))
            


config_base = load_config('configs/default.yaml')
param_grid = {
    "T": [7.5, 10],
    "x0-y0": [#(0.01, 0.01),
                (0.1, 0.1),
               # (0.5, 0.5),
               # (1.01, 1.01),
               # (1.5, 1.5),
               # (1.5, 0.01),
               # (2.0, 1.75)],
            ],
    "network_architectures": [
        (4, 50),
    ],
    "activations": [
        "swish",
    ],
    "learning_rates": [
        0.001,
    ],
    "collocations": [
        1024, 
    ],
    "epochs": [
        25000,
    ],
    "regularization": [
        "reg_derivative_unstable_fp"
        #"unstable_fp"
    ],
    "reg_epochs": [
        25, 0.5, 0.75, 1.0
    ],
    "reg_coeff": [
       0.1, 1, 10, 100, 1000
    ],
    "reg_decay": [
        "linear"
        #None
    ]
}


results_list = []

for params in grid_parameters(param_grid):
    print(params)
    
    config = config_base
    config["activation"] = params["activations"]
    config["N_hidden"] = params["network_architectures"][0]
    config["N_neurons"] = params["network_architectures"][1]
    config["N_epochs"] = params["epochs"]
    config["regularization"] = params["regularization"]
    config["reg_epochs"] = params["reg_epochs"]
    config["reg_coeff"] = params["reg_coeff"]
    config["reg_decay"] = params["reg_decay"]
    config["learning_rate"] = params["learning_rates"]
    config["N_col"] = params["collocations"]
    config["T"] = params["T"]
    config["freq_save"] = 0
    config["x0"], config["y0"] = params["x0-y0"]
    

    dirname = f"plots/{config['regularization']}/reg_coeff_{config['reg_coeff']}/reg_epochs_{config['reg_epochs']}/T_{config['T']}/x0_{config['x0']}_y0_{config['y0']}/" + re.sub('\W+', '_', str(params))

    losses = []
    loss_successes = []
    
    fixed_points = {
        (0.0, 0.0) : 0, 
        (1.0, 1.0): 0, 
        (0.0, 2.0): 0, 
        (3.0, 0.0): 0
    } 

    for i in range(NUM_TRAINING_RUNS):
        if config["regularization"] == "no_reg" or config["reg_epochs"] == 0:
            if config["reg_coeff"] > 1:
                continue

        if not os.path.exists(f"logs/{dirname}/run_{i}"):
            os.makedirs(f"logs/{dirname}/run_{i}")
        config["version"] = f"{dirname}/run_{i}"
        
        PINN = PhysicsInformedNN(config, verbose=True)
        try:
            training_log = PINN.train()
       
        
            # get reference solution (analytical)
            t_line, x_true, y_true = PINN.data.reference()

            xy_true = np.concatenate([x_true.numpy(), y_true.numpy()], axis=1)
            # get PINN prediction
            xy_pred = PINN(t_line)
            
            loss = mean_squared_error(xy_true, xy_pred)
            loss_success = np.linalg.norm(xy_true - xy_pred) / np.linalg.norm(xy_true) < 0.15
            losses.append(loss)
            loss_successes.append(loss_success)
            
            x_last_pred = tf.reduce_mean(xy_pred[-10:, 0])
            y_last_pred = tf.reduce_mean(xy_pred[-10:, 1])
            for fp, counter in fixed_points.items():
                if np.linalg.norm(x_last_pred - fp[0]) < 0.1 and np.linalg.norm(y_last_pred - fp[1]) < 0.1:
                    fixed_points[fp] += 1
                

        except Exception as e:
            print(e)
            i -= 1
            continue

        toy_example_dynamics(PINN, path=f"logs/{dirname}/run_{i}/dynamics")
        learning_curves(training_log, path=f"logs/{dirname}/run_{i}/learning_curve")
        loss_over_tcoll(PINN, path=f"logs/{dirname}/run_{i}/loss_over_tcol")
        plot_regularization(PINN, path=f"logs/{dirname}/run_{i}/regularization_plot")
                
    table_entry = pd.DataFrame({k: [v] for k, v in params.items()})
    
    for fp, counter in fixed_points.items():
        table_entry[fp] = counter

    table_entry["mean_loss"] = np.mean(losses)
    table_entry["loss_successes_percent"] = np.sum(loss_successes) / float(NUM_TRAINING_RUNS)
    results_list.append(table_entry)
    pd.concat(results_list).to_csv("results_reg.csv")

