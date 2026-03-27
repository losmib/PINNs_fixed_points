from itertools import product
from typing import Any, Dict, Iterable
from datetime import datetime

from sklearn.metrics import mean_squared_error
from configs.config_loader import load_config
from model.neural_net import PhysicsInformedNN
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import pandas as pd
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
config["N_epochs"] = 50000
config["T"] = 15
config["freq_save"] = 0


mean = [0, 0]

cov = [[0.25, 0],
       [0, 0.25]]

samples = np.random.multivariate_normal(mean, cov, NUM_SAMPLES)

base_dir = f"logs/models/visualization_experiments_{str(config["T"]).replace(".", "_")}_50000_new"


dirname_no_reg = f"{base_dir}/no_reg"
if not os.path.exists(dirname_no_reg):
    os.makedirs(dirname_no_reg)


dirname_reg_derivative_unstable_fp = f"{base_dir}/reg_derivative_unstable_fp"
if not os.path.exists(dirname_reg_derivative_unstable_fp):
    os.makedirs(dirname_reg_derivative_unstable_fp)

results_list = []

for i in range(NUM_SAMPLES):
    config["x0"] = float(samples[i, 0])
    config["x_t0"] = float(samples[i, 1])
    
    try:
        # Without regularization
        config_no_reg = config
        config_no_reg["reg"] = "no_reg"
        config_no_reg["reg_coeff"] = 0.0
        PINN_no_reg = PhysicsInformedNN(config_no_reg, verbose=True)
        PINN_no_reg.train()
        PINN_no_reg.save_weights(path=f"{dirname_no_reg}/run_{i}.pkl")
        t_line, x_true, x_t_true = PINN_no_reg.data.reference()

        # get PINN prediction
        x_pred = PINN_no_reg(t_line)
        
        loss = mean_squared_error(x_true.numpy(), x_pred.numpy())
        loss_success_no_reg = (np.linalg.norm(x_true - x_pred) / np.linalg.norm(x_true)) < 0.15        

        config_reg = config
        config_reg["reg"] = "reg_derivative_unstable_fp"
        config_reg["reg_coeff"] = 1.0
        config_reg["reg_decay"] = "linear"
        config_reg["reg_epochs"] = 0.5
        config_reg["eps"] = 0.01
        PINN_reg = PhysicsInformedNN(config_reg, verbose=True)
        PINN_reg.train()
        PINN_reg.save_weights(path=f"{dirname_reg_derivative_unstable_fp}/run_{i}.pkl")
        t_line, x_true, x_t_true = PINN_reg.data.reference()

        # get PINN prediction
        x_pred = PINN_reg(t_line)
        
        loss = mean_squared_error(x_true.numpy(), x_pred.numpy())
        loss_success_reg = (np.linalg.norm(x_true - x_pred) / np.linalg.norm(x_true)) < 0.15

        results_list.append({"x0": config["x0"], 
                            "x_t0": config["x_t0"], 
                            "success no reg": loss_success_no_reg,
                            "success reg": loss_success_reg,
                            "path no reg": f"{dirname_no_reg}/run_{i}.pkl",
                            "path reg": f"{dirname_reg_derivative_unstable_fp}/run_{i}.pkl"
                            })
        
        # if not os.path.exists(dirname_reg_
    except Exception as e:
        print(e.with_traceback(e.__traceback__))
        exit(-1)
     
    results_table = pd.DataFrame(results_list)
    results_table.to_csv("qualitative_results_new_50000.csv")