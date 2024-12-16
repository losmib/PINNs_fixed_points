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

NUM_SAMPLES_PER_FP = 20

def grid_parameters(parameters: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
        for params in product(*parameters.values()):
            yield dict(zip(parameters.keys(), params))
            


config_base = load_config('configs/default.yaml')

fixed_points = {
        (0.0, 0.0) : "unstable", 
        (1.0, 1.0): "unstable", 
        (0.0, 2.0): "stable", 
        (3.0, 0.0): "stable"
    } 

results_list = []

    
config = config_base
config["activation"] = "swish"
config["N_hidden"] = 4
config["N_neurons"] = 50
config["N_epochs"] = 25000
config["regularization"] = "no_reg"
config["T"] = 10
config["freq_save"] = 0

dirname = f"plots/"

losses = []
loss_successes = []
    
table_list = []

for fp in fixed_points:
    for i in range(NUM_SAMPLES_PER_FP):
        config["x0"] = np.random.normal(fp[0], 1)
        config["y0"] = np.random.normal(fp[1], 1)
            
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
                
            x_last_pred = tf.reduce_mean(xy_pred[-10:, 0])
            y_last_pred = tf.reduce_mean(xy_pred[-10:, 1])

            table_entry = pd.DataFrame({"(x0, y0)": [(config["x0"], config["y0"])], "success": [loss_success]})

            for fp, counter in fixed_points.items():
                if np.linalg.norm(x_last_pred - fp[0]) < 0.1 and np.linalg.norm(y_last_pred - fp[1]) < 0.1:
                    table_entry["converged_to"] = fp
                    table_entry["stability"] = fixed_points[fp]
            
            table_list.append(table_entry)

        except Exception as e:
            print(e)

        table = pd.concat([table_list])
        table.to_csv("convergence_points.csv")
        
                
    

