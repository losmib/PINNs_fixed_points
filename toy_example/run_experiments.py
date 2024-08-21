from itertools import product
from typing import Any, Dict, Iterable

from sklearn.metrics import mean_squared_error
from configs.config_loader import load_config
from model.neural_net import PhysicsInformedNN
from model.plots import learning_curves, toy_example_dynamics
import pandas as pd
import numpy as np


NUM_TRAINING_RUNS = 20


def grid_parameters(parameters: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
        for params in product(*parameters.values()):
            yield dict(zip(parameters.keys(), params))
            


config_base = load_config('configs/default.yaml')
param_grid = {
    "T": [7.5, 10],
    "network_architectures": [
        (4, 50),
        (8, 100)
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
        10000,
    ],
    "reg_percentages": [
        0,
        0.5,
        0.95,
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
    config["reg_epochs"] = int(params["epochs"] * params["reg_percentages"])
    config["learning_rate"] = params["learning_rates"]
    config["N_col"] = params["collocations"]
    config["T"] = params["T"]
    
    losses = []
    loss_successes = []
    for i in range(NUM_TRAINING_RUNS):
    
        config["y0"] = np.random.uniform(low=-0.99, high=0.99)
        PINN = PhysicsInformedNN(config, verbose=True)
        try:
            training_log = PINN.train()
        except Exception as e:
            print(e)
            i -= 1
            continue
        t_line = PINN.data.t_line()
        # get reference solution (analytical)
        y_true = PINN.data.reference(t_line)
        # get PINN prediction
        y_pred = PINN(t_line)
        loss = mean_squared_error(y_true, y_pred)
        loss_success = np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true) < 0.15
        losses.append(loss)
        loss_successes.append(loss_success)
    # path_suffix = str(params).replace(".", "").replace(",", "").replace(":", "")
    # learning_curves(training_log, f"figures/learning_curve{path_suffix}")
    # toy_example_dynamics(PINN, f"figures/dynamics_{path_suffix}")
    
    table_entry = pd.DataFrame({k: [v] for k, v in params.items()})
    
    t_line = PINN.data.t_line()
    # get reference solution (analytical)
    y_true = PINN.data.reference(t_line)
    # get PINN prediction
    y_pred = PINN(t_line)
    
    table_entry["mean_loss"] = np.mean(losses)
    table_entry["loss_successes_percent"] = np.sum(loss_successes) / float(NUM_TRAINING_RUNS)
    results_list.append(table_entry)
    pd.concat(results_list).to_csv("results_unstable_fp_reseting_optim.csv")

