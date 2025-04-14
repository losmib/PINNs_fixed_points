from itertools import product
from typing import Any, Dict, Iterable
from datetime import datetime

from sklearn.metrics import mean_squared_error
from configs.config_loader import load_config
from model.neural_net import PhysicsInformedNN
from model.plots import learning_curves, van_der_pol_dynamics, loss_over_tcoll, plot_regularization
import pandas as pd
import numpy as np
import os
import re
import tensorflow as tf

NUM_SAMPLES_PER_FP = 300

def grid_parameters(parameters: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
        for params in product(*parameters.values()):
            yield dict(zip(parameters.keys(), params))
            


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


def run_without_reg():
    results_list = []
    config["regularization"] = "no_reg"
    
    dirname = f"convergence_plots/no_reg"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    losses = []
    loss_successes = []
        
    table_list = []

    for fp in fixed_points:
        x0s = np.random.normal(fp[0], 1.0, size=NUM_SAMPLES_PER_FP)
        x_t0s = np.random.normal(fp[1], 1.0, size=NUM_SAMPLES_PER_FP)
        print(x0s)
        print(x_t0s)
        for i in range(NUM_SAMPLES_PER_FP):
            config["x0"] = float(x0s[i])
            config["x_t0"] = float(x_t0s[i])
                    
            PINN = PhysicsInformedNN(config, verbose=True)
            try:
                training_log = PINN.train()
            
                # get reference solution (analytical)
                t_line, x_true, x_t_true = PINN.data.reference()

                xxt_true = np.concatenate([x_true.numpy(), x_t_true.numpy()], axis=1)
                # get PINN prediction
                with tf.GradientTape() as tape:
                    tape.watch(t_line)
                    x_pred = PINN(t_line)
                    x_t_pred = tape.gradient(x_pred, t_line)

                loss = mean_squared_error(x_true.numpy(), x_pred.numpy()) 
                loss_success = np.linalg.norm(x_true - x_pred) / np.linalg.norm(x_true) < 0.15
        
                x_last_pred = tf.reduce_mean(x_pred[-5:, 0])
                x_t_last_pred = tf.reduce_mean(x_t_pred[-5:])
                
                closest_fp = None
                stability = None

                for fp, counter in fixed_points.items():
                    if tf.linalg.norm(x_last_pred - fp[0]) < 0.1 and tf.linalg.norm(x_t_last_pred - fp[1]) < 0.1:
                        closest_fp = fp
                        stability = fixed_points[fp]
                
                table_entry = pd.DataFrame({"(x0, x_t0)": [(config["x0"], config["x_t0"])], 
                                            "converged to": [(float(x_last_pred), float(x_t_last_pred))],
                                            "closest fp": [closest_fp],
                                            "stability": [stability],
                                            "success": [loss_success]})
                
                table_list.append(table_entry)

                van_der_pol_dynamics(PINN, f"{dirname}/fp{str(fp)}_run{i}.png")

            except IndexError as e:
                print(e)

            table = pd.concat(table_list)
            table.to_csv("convergence_points.csv")
            
                    
def repeat_with_reg(file):
    table_no_reg = pd.read_csv(file)
    table_no_reg["x0"] = table_no_reg["(x0, x_t0)"].apply(lambda row: eval(row)[0])
    table_no_reg["x_t0"] = table_no_reg["(x0, x_t0)"].apply(lambda row: eval(row)[1])
    config["regularization"] = "reg_derivative_unstable_fp"

    table_reg = table_no_reg
    table_reg["converged to with reg"] = ""
    table_reg["closest fp with reg"] = ""
    table_reg["stability with reg"] = ""
    table_reg["success reg"] = ""

    dirname = f"convergence_plots/no_reg"

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for i in range(table_no_reg.shape[0]):
        config["x0"] = table_reg["x0"].iloc[i]
        config["x_t0"] = table_reg["x_t0"].iloc[i]
        config["reg_coeff"] = 1.0
        config["reg_epochs"] = 0.5
        config["reg_decay"] = "linear"


        PINN = PhysicsInformedNN(config, verbose=True)
        try:
                training_log = PINN.train()
            
                # get reference solution (analytical)
                t_line, x_true, x_t_true = PINN.data.reference()

                xxt_true = np.concatenate([x_true.numpy(), x_t_true.numpy()], axis=1)
                # get PINN prediction
                with tf.GradientTape() as tape:
                    tape.watch(t_line)
                    x_pred = PINN(t_line)
                    x_t_pred = tape.gradient(x_pred, t_line)

                loss = mean_squared_error(x_true.numpy(), x_pred.numpy())
                loss_success = np.linalg.norm(x_true - x_pred) / np.linalg.norm(x_true) < 0.15
        
                x_last_pred = tf.reduce_mean(x_pred[-5:, 0])
                x_t_last_pred = tf.reduce_mean(x_t_pred[-5:])
                
                closest_fp = None
                stability = None

                for fp, counter in fixed_points.items():
                    if tf.linalg.norm(x_last_pred - fp[0]) < 0.1 and tf.linalg.norm(x_t_last_pred - fp[1]) < 0.1:
                        closest_fp = fp
                        stability = fixed_points[fp]
                
                table_reg["converged to with reg"].iloc[i] = str(tuple((float(x_last_pred), float(x_t_last_pred))))
                table_reg["closest fp with reg"].iloc[i] = str(closest_fp)
                table_reg["stability with reg"].iloc[i] = str(stability)
                table_reg["success reg"].iloc[i] = loss_success

                van_der_pol_dynamics(PINN, f"{dirname}/fp{str(fp)}_run{i}.png")
                

        except Exception as e:
                print(e)

            
        table_reg.to_csv("convergence_points_reg.csv")


run_without_reg()
repeat_with_reg("convergence_points.csv")