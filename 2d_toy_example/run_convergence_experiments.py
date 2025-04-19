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


NUM_SAMPLES = 200

config_base = load_config('configs/default.yaml')

config = config_base
config["activation"] = "swish"
config["N_hidden"] = 4
config["N_neurons"] = 50
config["N_epochs"] = 25000
config["T"] = 12.5
config["freq_save"] = 0


x0s = np.random.uniform(-1, 4, size=NUM_SAMPLES)
y0s = np.random.uniform(-1, 4, size=NUM_SAMPLES)


dirname_no_reg = f"logs/models/visualization_experiments/no_reg"
if not os.path.exists(dirname_no_reg):
    os.makedirs(dirname_no_reg)

dirname_reg_derivative = f"logs/models/visualization_experiments/reg_derivative"
if not os.path.exists(dirname_reg_derivative):
    os.makedirs(dirname_reg_derivative)

dirname_reg_derivative_unstable_fp = f"logs/models/visualization_experiments/reg_derivative_unstable_fp"
if not os.path.exists(dirname_reg_derivative_unstable_fp):
    os.makedirs(dirname_reg_derivative_unstable_fp)

results_list = []

for i in range(NUM_SAMPLES):
    config["x0"] = float(x0s[i])
    config["y0"] = float(y0s[i])
    
    try:
        # Without regularization
        config["regularization"] = "no_reg"
        PINN = PhysicsInformedNN(config, verbose=True)
        training_log = PINN.train()

        PINN.save_weights(f"{dirname_no_reg}/run_{i}.pkl")

        t_line, x_true, y_true = PINN.data.reference()

        xy_true = np.concatenate([x_true.numpy(), y_true.numpy()], axis=1)
            # get PINN prediction
        xy_pred = PINN(t_line)
            
        loss = mean_squared_error(xy_true, xy_pred)
        loss_success_no_reg = np.linalg.norm(xy_true - xy_pred) / np.linalg.norm(xy_true) < 0.15
        


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
        config["reg_decay"] = "linear"
        PINN = PhysicsInformedNN(config, verbose=True)
        training_log = PINN.train()
        
        PINN.save_weights(f"{dirname_reg_derivative_unstable_fp}/run_{i}.pkl")
        
        # get PINN prediction
        xy_true = np.concatenate([x_true.numpy(), y_true.numpy()], axis=1)
        # get PINN prediction
        xy_pred = PINN(t_line)
            
        loss = mean_squared_error(xy_true, xy_pred)
        loss_success_reg_derivative_unstable_fp = np.linalg.norm(xy_true - xy_pred) / np.linalg.norm(xy_true) < 0.15

        table_entry = pd.DataFrame({"(x0, y0)": [(float(x0s[i]), float(y0s[i]))], 
                                    "success_no_reg": [loss_success_no_reg],
                                    "success_reg_derivative_unstable_fp": [loss_success_reg_derivative_unstable_fp]})
        
        results_list.append(table_entry)
        results_table = pd.concat(results_list)
        results_table.to_csv("visual_results_T_12_5.csv")

    except Exception as e:
        print(e)
            
     