from itertools import product
from typing import Any, Dict, Iterable

from sklearn.metrics import mean_squared_error
from configs.config_loader import load_config
from model.neural_net import PhysicsInformedNN
import pandas as pd
import numpy as np


NUM_TRAINING_RUNS = 20


def grid_parameters(parameters: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
        for params in product(*parameters.values()):
            yield dict(zip(parameters.keys(), params))
            


config_base = load_config('configs/default.yaml')
param_grid = {
    "T": [7.5, 10],
    "y0": [25, 100, 175],
    "network_architectures": [
        (4, 50),
    ],
    "activations": [
        "tanh",
    ],
    "learning_rates": [
        0.001,
        # 0.01,
        # 0.1
    ],
    "collocations": [
        1024, 
    ],
    "epochs": [
        30000,
    ],
    "reg_epochs": [
        0,
        0.5,
        0.95,
    ],
    "reg_coeff": [
      1  
    ],
    "reg_decay": [
        1
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
    config["reg_epochs"] = params["reg_epochs"]
    config["reg_coeff"] = params["reg_coeff"]
    config["reg_decay"] = params["reg_decay"]
    config["learning_rate"] = params["learning_rates"]
    config["N_col"] = params["collocations"]
    config["T"] = params["T"]
    config["freq_save"] = 0
    config["y0"] = params["y0"]    
    losses = []
    loss_successes = []
    for i in range(NUM_TRAINING_RUNS):
        PINN = PhysicsInformedNN(config, verbose=True)
        try:
            training_log = PINN.train()
        except Exception as e:
            print(e)
            i -= 1
            continue
        
        t_line, theta_true, omega_true = PINN.data.reference()
        theta_pred = PINN(t_line)
        # get PINN prediction
        y_pred = PINN(t_line)
        loss = mean_squared_error(theta_true, theta_pred)
        loss_success = (np.linalg.norm(theta_true - theta_pred) / np.linalg.norm(theta_true)) < 0.15
        losses.append(loss)
        loss_successes.append(loss_success)
    
    table_entry = pd.DataFrame({k: [v] for k, v in params.items()})
    
    table_entry["mean_loss"] = np.mean(losses)
    table_entry["loss_successes_percent"] = np.sum(loss_successes) / float(NUM_TRAINING_RUNS)
    results_list.append(table_entry)
    pd.concat(results_list).to_csv("results.csv")

