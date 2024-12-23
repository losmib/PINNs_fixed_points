from itertools import product
from typing import Any, Dict, Iterable
from datetime import datetime

from sklearn.metrics import mean_squared_error
from configs.config_loader import load_config
from model.neural_net import PhysicsInformedNN
from model.plots import learning_curves, toy_example_dynamics, loss_over_tcoll, plot_regularization
import pandas as pd
import numpy as np
import os
import re
import tensorflow as tf

NUM_SAMPLES_PER_FP = 50

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
config = config_base
config["activation"] = "swish"
config["N_hidden"] = 4
config["N_neurons"] = 50
config["N_epochs"] = 25000
config["regularization"] = "no_reg"
config["T"] = 15
config["freq_save"] = 0


def run_without_reg():
    results_list = []

    dirname = f"plots/"

    losses = []
    loss_successes = []
        
    table_list = []

    for fp in fixed_points:
        x0s = np.random.normal(fp[0], 0.3, size=NUM_SAMPLES_PER_FP)
        y0s = np.random.normal(fp[1], 0.3, size=NUM_SAMPLES_PER_FP)
        print(x0s)
        print(y0s)
        for i in range(NUM_SAMPLES_PER_FP):
            config["x0"] = float(x0s[i])
            config["y0"] = float(y0s[i])
                    
            PINN = PhysicsInformedNN(config, verbose=True)
            try:
                training_log = PINN.train()
            
                # get reference solution (analytical)
                t_line, x_true, y_true = PINN.data.reference()

                xy_true = np.concatenate([x_true.numpy(), y_true.numpy()], axis=1)
                # get PINN prediction
                xy_pred = PINN(t_line)
                print("here")

                loss = mean_squared_error(xy_true, xy_pred.numpy())
                loss_success = np.linalg.norm(xy_true - xy_pred) / np.linalg.norm(xy_true) < 0.15
        
                x_last_pred = tf.reduce_mean(xy_pred[-5:, 0])
                y_last_pred = tf.reduce_mean(xy_pred[-5:, 1])
                
                closest_fp = None
                stability = None

                for fp, counter in fixed_points.items():
                    if tf.linalg.norm(x_last_pred - fp[0]) < 0.1 and tf.linalg.norm(y_last_pred - fp[1]) < 0.1:
                        closest_fp = fp
                        stability = fixed_points[fp]
                
                table_entry = pd.DataFrame({"(x0, y0)": [(config["x0"], config["y0"])], 
                                            "converged to": [(float(x_last_pred), float(y_last_pred))],
                                            "closest fp": [closest_fp],
                                            "stability": [stability],
                                            "success": [loss_success]})
                
                table_list.append(table_entry)

            except Exception as e:
                print(e)

            table = pd.concat(table_list)
            table.to_csv("convergence_points.csv")
            
                    
def repeat_with_reg(file):
    table_no_reg = pd.read_csv(file)
    table_no_reg["x0"] = table_no_reg["(x0, y0)"].apply(lambda row: eval(row)[0])
    table_no_reg["y0"] = table_no_reg["(x0, y0)"].apply(lambda row: eval(row)[1])
    config["regularization"] = "reg_derivative_unstable_fp"

    table_reg = table_no_reg
    table_reg["converged to with reg"] = ""
    table_reg["closest fp with reg"] = ""
    table_reg["stability with reg"] = ""
    for i in range(table_no_reg.shape[0]):
        config["xo"] = table_reg["x0"].iloc[i]
        config["y0"] = table_reg["y0"].iloc[i]
        config["reg_coeff"] = 1.0
        config["reg_epochs"] = 0.5
        config["reg_decay"] = "linear"


        PINN = PhysicsInformedNN(config, verbose=True)
        try:
                training_log = PINN.train()
            
                # get reference solution (analytical)
                t_line, x_true, y_true = PINN.data.reference()

                xy_true = np.concatenate([x_true.numpy(), y_true.numpy()], axis=1)
                # get PINN prediction
                xy_pred = PINN(t_line)
                print("here")

                loss = mean_squared_error(xy_true, xy_pred.numpy())
                loss_success = np.linalg.norm(xy_true - xy_pred) / np.linalg.norm(xy_true) < 0.15
        
                x_last_pred = tf.reduce_mean(xy_pred[-5:, 0])
                y_last_pred = tf.reduce_mean(xy_pred[-5:, 1])
                
                closest_fp = None
                stability = None

                for fp, counter in fixed_points.items():
                    if tf.linalg.norm(x_last_pred - fp[0]) < 0.1 and tf.linalg.norm(y_last_pred - fp[1]) < 0.1:
                        closest_fp = fp
                        stability = fixed_points[fp]
                
                table_reg["converged to with reg"].iloc[i] = str(tuple((float(x_last_pred), float(y_last_pred))))
                table_reg["closest fp with reg"].iloc[i] = str(closest_fp)
                table_reg["stability with reg"].iloc[i] = str(stability)
                

        except Exception as e:
                print(e)

            
        table_reg.to_csv("convergence_points_reg.csv")


repeat_with_reg("convergence_points.csv")