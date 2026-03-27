from itertools import product
from typing import Any, Dict, Iterable
from datetime import datetime

from sklearn.metrics import mean_squared_error
from configs.config_loader import load_config
from model.neural_net import PhysicsInformedNN
import pandas as pd
import numpy as np
import os
import re
import tensorflow as tf
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os


NUM_SAMPLES = 30

config_base = load_config('configs/default.yaml')

fixed_points = {
        (0.0, 0.0) : "unstable", 
    } 
config = config_base
config["activation"] = "swish"
config["N_hidden"] = 4
config["N_neurons"] = 50
config["N_epochs"] = 25000
config["T"] = 15
config["freq_save"] = 0


x0s = np.random.normal(0, 0.5, size=NUM_SAMPLES)
x_t0s = np.random.normal(0, 0.5, size=NUM_SAMPLES)

base_dir = f"logs/models/visualization_experiments_{str(config["T"]).replace(".", "_")}"

dirname_no_reg = f"{base_dir}/no_reg"
if not os.path.exists(dirname_no_reg):
    os.makedirs(dirname_no_reg)

# dirname_reg_derivative = f"logs/models/visualization_experiments/reg_derivative"
# if not os.path.exists(dirname_reg_derivative):
#     os.makedirs(dirname_reg_derivative)

dirname_reg_derivative_unstable_fp = f"{base_dir}/reg_derivative_unstable_fp"
if not os.path.exists(dirname_reg_derivative_unstable_fp):
    os.makedirs(dirname_reg_derivative_unstable_fp)

results_list = []

for i in range(NUM_SAMPLES):
    config["x0"] = float(x0s[i])
    config["x_t0"] = float(x_t0s[i])
    
    try:
        # Without regularization
        config["regularization"] = "no_reg"
        config["reg_coeff"] = 0.0
        PINN = PhysicsInformedNN(config, verbose=True)
        training_log = PINN.train()

        
        # if not os.path.exists(dirname_no_reg):
        #     os.makedirs(dirname_no_reg)

        PINN.save_weights(path=f"{dirname_no_reg}/run_{i}.pkl")
        

        t_line, x_true, x_t_true = PINN.data.reference()

        # get PINN prediction
        x_pred = PINN(t_line)
        loss = mean_squared_error(x_true.numpy(), x_pred.numpy())
        loss_success_no_reg = (np.linalg.norm(x_true - x_pred) / np.linalg.norm(x_true)) < 0.15


        """
        # With time derivative regularization
        config["regularizer"] = "reg_derivative"
        config["reg_coeff"] = 1.0
        config["reg_epochs"] = 0.5
        config["reg_decay"] = "linear"
        PINN = PhysicsInformedNN(config, verbose=True)
        training_log = PINN.train()

        PINN.save_weights(f"{dirname_reg_derivative}/run_{i}.pkl")

        # get PINN prediction
        x_pred = PINN(t_line)
        loss = mean_squared_error(x_true.numpy(), x_pred.numpy())
        loss_success_reg_derivative = (np.linalg.norm(x_true - x_pred) / np.linalg.norm(x_true)) < 0.15
        """

        # With unstable fp regularization
        config["regularization"] = "reg_derivative_unstable_fp"
        config["reg_coeff"] = 1.0
        config["reg_epochs"] = 0.5
        config["eps"] = 0.01
        config["reg_decay"] = "linear"
        PINN = PhysicsInformedNN(config, verbose=True)
        training_log = PINN.train()
        
        # if not os.path.exists(dirname_reg_derivative_unstable_fp):
        #     os.makedirs(dirname_reg_derivative_unstable_fp)

        PINN.save_weights(path=f"{dirname_reg_derivative_unstable_fp}/run_{i}.pkl")
        
        # get PINN prediction
        x_pred = PINN(t_line)
        loss = mean_squared_error(x_true.numpy(), x_pred.numpy())
        loss_success_reg_derivative_unstable_fp = (np.linalg.norm(x_true - x_pred) / np.linalg.norm(x_true)) < 0.15

        table_entry = pd.DataFrame({"(x0, x_t0)": [(float(x0s[i]), float(x_t0s[i]))], 
                                    "success_no_reg": [loss_success_no_reg],
                                    "success_reg_derivative_unstable_fp": [loss_success_reg_derivative_unstable_fp]})
        
        results_list.append(table_entry)
        results_table = pd.concat(results_list)
        results_table.to_csv(f"visual_results_T_{str(config["T"]).replace(".", "_")}.csv")

    except Exception as e:
        print(e.with_traceback(e.__traceback__))
        exit(-1)
     