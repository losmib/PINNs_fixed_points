from itertools import product
from typing import Any, Dict, Iterable

from sklearn.metrics import mean_squared_error
from configs.config_loader import load_config
from model.neural_net import PhysicsInformedNN
from model.plots import learning_curves, van_der_pol_dynamics, loss_over_tcoll, plot_regularization
import pandas as pd
import numpy as np

import re
import os


NUM_TRAINING_RUNS = 10


def grid_parameters(parameters: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
        for params in product(*parameters.values()):
            yield dict(zip(parameters.keys(), params))
            


config_base = load_config('configs/default.yaml')
param_grid = {
    "mu": [0.1, 1, 5],
    "T": [5, 7.5, 10, 12.5, 15],
    "x0": [0.001, 0.01, 0.1, 0.5],
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
        50000,
    ],
    "regularization": [
        "no_reg",
    ],
    "reg_epochs": [
        1
    ],
    "reg_coeff": [
      0
    ],
    "reg_decay": [
        None, 
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
    config["x0"] = params["x0"]    
    config["mu"] = params["mu"]

    dirname = f"plots/{config['regularization']}/reg_coeff_{config['reg_coeff']}/reg_epochs_{config['reg_epochs']}/mu_{config['mu']}/T_{config['T']}/x0_{config['x0']}/" + re.sub('\W+', '_', str(params))

    losses = []
    loss_successes = []
    if config["regularization"] is "no_reg":
        if config["reg_coeff"] > 1:
            continue
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
        
        t_line, x_true, x_t_true = PINN.data.reference()

        # get PINN prediction
        x_pred = PINN(t_line)
        
        loss = mean_squared_error(x_true, x_pred)
        loss_success = (np.linalg.norm(x_true - x_pred) / np.linalg.norm(x_true)) < 0.15
        losses.append(loss)
        loss_successes.append(loss_success)
        
        van_der_pol_dynamics(PINN, path=f"logs/{dirname}/run_{i}/dynamics")
        learning_curves(training_log, path=f"logs/{dirname}/run_{i}/learning_curve")
        loss_over_tcoll(PINN, path=f"logs/{dirname}/run_{i}/loss_over_tcol")
        plot_regularization(PINN, path=f"logs/{dirname}/run_{i}/regularization_plot")
    
    table_entry = pd.DataFrame({k: [v] for k, v in params.items()})
    
    table_entry["mean_loss"] = np.mean(losses)
    table_entry["loss_successes_percent"] = np.sum(loss_successes) / float(NUM_TRAINING_RUNS)
    results_list.append(table_entry)
    pd.concat(results_list).to_csv("results_no_reg.csv")

