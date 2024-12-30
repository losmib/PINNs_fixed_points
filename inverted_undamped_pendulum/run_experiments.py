from itertools import product
from typing import Any, Dict, Iterable

from sklearn.metrics import mean_squared_error
from configs.config_loader import load_config
from model.neural_net import PhysicsInformedNN
from model.plots import learning_curves, pendulum_dynamics, loss_over_tcoll, plot_regularization
import pandas as pd
import numpy as np

import re
import os
import tensorflow as tf


NUM_TRAINING_RUNS = 20


def grid_parameters(parameters: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
        for params in product(*parameters.values()):
            yield dict(zip(parameters.keys(), params))
            


config_base = load_config('configs/default.yaml')
param_grid = {
    "l": [1],
    "T": [2.5, 5, 7.5],
    "theta0": [20, 90, 100, 270, 340],
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
    ],
    "reg_epochs": [
        0.5#0.25, 0.5, 0.75, 1.0
    ],
    "reg_coeff": [
        0.1#, 1, 10, 100, 1000
    ],
    "reg_decay": [
        "linear"
    ]
}



def run_experiments_on_params(param_grid, results_path="results.csv"):
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
        config["theta0"] = params["theta0"]    
        config["l"] = params["l"]

        dirname = f"plots/{config['regularization']}/reg_coeff_{config['reg_coeff']}/reg_epochs_{config['reg_epochs']}/l_{config['l']}/T_{config['T']}/theta0_{config['theta0']}/" + re.sub('\W+', '_', str(params))

        losses = []
        loss_successes = []
        if config["regularization"] == "no_reg" or config["reg_epochs"] == 0:
            if config["reg_coeff"] > 1:
                continue

        fixed_points = {
            0.0 : 0, 
            np.pi: 0, 
            2 * np.pi: 0, 
        }

        for i in range(NUM_TRAINING_RUNS):
            if not os.path.exists(f"logs/{dirname}/run_{i}"):
                os.makedirs(f"logs/{dirname}/run_{i}")
            config["version"] = f"{dirname}/run_{i}"
            
            PINN = PhysicsInformedNN(config, verbose=True)
            try:
                training_log = PINN.train()
            except Exception as e:
                print(e)
                i -= 1
                continue
            
            t_line, theta_true, omega_true = PINN.data.reference()
            
            # get PINN prediction
            theta_pred = PINN(t_line)
            
            loss = mean_squared_error(theta_true, theta_pred)
            loss_success = (np.linalg.norm(theta_true - theta_pred) / np.linalg.norm(theta_true)) < 0.15
            losses.append(loss)
            loss_successes.append(loss_success)
            
            pendulum_dynamics(PINN, path=f"logs/{dirname}/run_{i}/dynamics.png")
            learning_curves(training_log, path=f"logs/{dirname}/run_{i}/learning_curve.png")
            loss_over_tcoll(PINN, path=f"logs/{dirname}/run_{i}/loss_over_tcol.png")
            plot_regularization(PINN, path=f"logs/{dirname}/run_{i}/regularization_plot.png")

            theta_last_pred = tf.reduce_mean(theta_pred[-10:])
            
            for fp, counter in fixed_points.items():
                if np.linalg.norm(theta_last_pred - fp) < 0.1 and loss_success == 0:
                    fixed_points[fp] += 1
        
        table_entry = pd.DataFrame({k: [v] for k, v in params.items()})
        
        for fp, counter in fixed_points.items():
            table_entry[fp] = counter

        
        table_entry["mean_loss"] = np.mean(losses)
        table_entry["loss_successes_percent"] = np.sum(loss_successes) / float(NUM_TRAINING_RUNS)
        results_list.append(table_entry)
        pd.concat(results_list).to_csv(results_path)


run_experiments_on_params(param_grid=param_grid, results_path="results_reg_full.csv")
