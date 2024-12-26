from itertools import product
from typing import Any, Dict, Iterable
from datetime import datetime

from sklearn.metrics import mean_squared_error
from configs.config_loader import load_config
from model.neural_net import PhysicsInformedNN
from model.plots import learning_curves, pendulum_dynamics, loss_over_tcoll, plot_regularization
import pandas as pd
import numpy as np
import os
import re
import tensorflow as tf

NUM_SAMPLES_PER_FP = 200

def grid_parameters(parameters: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
        for params in product(*parameters.values()):
            yield dict(zip(parameters.keys(), params))
            


config_base = load_config('configs/default.yaml')

fixed_points = {
        (0.0, 0.0) : "unstable", 
        (-np.pi, 0.0) : "stable",
        (np.pi, 0.0): "stable"
    } 

config = config_base
config["activation"] = "swish"
config["N_hidden"] = 4
config["N_neurons"] = 50
config["N_epochs"] = 25000
config["T"] = 15
config["freq_save"] = 0

def run_without_reg():
    
    config["regularizer"] = "no_reg"

    results_list = []

    dirname = f"convergence_plots/no_reg"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    losses = []
    loss_successes = []
        
    table_list = []

    for fp in [(0.0, 0.0)]:
        theta0s = np.random.uniform(-np.pi, np.pi, size=NUM_SAMPLES_PER_FP)
        omega0s = np.random.normal(0.0, 1.0, size=NUM_SAMPLES_PER_FP)
        print(theta0s)
        print(omega0s)
        for i in range(NUM_SAMPLES_PER_FP):
            config["theta0"] = float(theta0s[i])
            config["omega0"] = float(omega0s[i])
                    
            PINN = PhysicsInformedNN(config, verbose=True)
            try:
                training_log = PINN.train()
            
                # get reference solution (analytical)
                t_line, theta_true, omega_true = PINN.data.reference()

                xxt_true = np.concatenate([theta_true.numpy(), omega_true.numpy()], axis=1)
                # get PINN prediction
                with tf.GradientTape() as tape:
                    tape.watch(t_line)
                    theta_pred = PINN(t_line)
                    omega_pred = tape.gradient(theta_pred, t_line)

                loss = mean_squared_error(theta_true.numpy(), theta_pred.numpy()) 
                loss_success = np.linalg.norm(theta_true - theta_pred) / np.linalg.norm(theta_true) < 0.15
        
                theta_last_pred = tf.reduce_mean(theta_pred[-5:, 0])
                omega_last_pred = tf.reduce_mean(omega_pred[-5:])
                
                closest_fp = None
                stability = None

                for fp, counter in fixed_points.items():
                    if tf.linalg.norm(theta_last_pred - fp[0]) < 0.1 and tf.linalg.norm(omega_last_pred - fp[1]) < 0.1:
                        closest_fp = fp
                        stability = fixed_points[fp]
                
                table_entry = pd.DataFrame({"(theta0, omega0)": [(config["theta0"], config["omega0"])], 
                                            "converged to": [(float(theta_last_pred), float(omega_last_pred))],
                                            "closest fp": [closest_fp],
                                            "stability": [stability],
                                            "success": [loss_success]})
                
                table_list.append(table_entry)
                pendulum_dynamics(PINN, f"{dirname}/fp{str(fp)}_run{i}.png")


            except IndexError as e:
                print(e)

            table = pd.concat(table_list)
            table.to_csv("convergence_points.csv")
            
                    
def repeat_with_reg(file):
    table_no_reg = pd.read_csv(file)
    table_no_reg["theta0"] = table_no_reg["(theta0, omega0)"].apply(lambda row: eval(row)[0])
    table_no_reg["omega0"] = table_no_reg["(theta0, omega0)"].apply(lambda row: eval(row)[1])
    config["regularizer"] = "reg_derivative_unstable_fp"

    table_reg = table_no_reg
    table_reg["converged to with reg"] = ""
    table_reg["closest fp with reg"] = ""
    table_reg["stability with reg"] = ""
    table_reg["success reg"] = ""

    dirname = f"convergence_plots/reg"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for i in range(table_no_reg.shape[0]):
        config["thetao"] = table_reg["theta0"].iloc[i]
        config["omega0"] = table_reg["omega0"].iloc[i]
        config["reg_coeff"] = 1.0
        config["reg_epochs"] = 0.5
        config["reg_decay"] = "linear"


        PINN = PhysicsInformedNN(config, verbose=True)
        try:
                training_log = PINN.train()
            
                # get reference solution (analytical)
                t_line, theta_true, omega_true = PINN.data.reference()

                xxt_true = np.concatenate([theta_true.numpy(), omega_true.numpy()], axis=1)
                # get PINN prediction
                with tf.GradientTape() as tape:
                    tape.watch(t_line)
                    theta_pred = PINN(t_line)
                    omega_pred = tape.gradient(theta_pred, t_line)

                loss = mean_squared_error(theta_true.numpy(), theta_pred.numpy())
                loss_success = np.linalg.norm(theta_true - theta_pred) / np.linalg.norm(theta_true) < 0.15
        
                theta_last_pred = tf.reduce_mean(theta_pred[-5:, 0])
                omega_last_pred = tf.reduce_mean(omega_pred[-5:])
                
                closest_fp = None
                stability = None

                for fp, counter in fixed_points.items():
                    if tf.linalg.norm(theta_last_pred - fp[0]) < 0.1 and tf.linalg.norm(omega_last_pred - fp[1]) < 0.1:
                        closest_fp = fp
                        stability = fixed_points[fp]
                
                table_reg["converged to with reg"].iloc[i] = str(tuple((float(theta_last_pred), float(omega_last_pred))))
                table_reg["closest fp with reg"].iloc[i] = str(closest_fp)
                table_reg["stability with reg"].iloc[i] = str(stability)
                table_reg["success reg"].iloc[i] = loss_success

                pendulum_dynamics(PINN, f"{dirname}/fp{str(fp)}_run{i}.png")

    
        except Exception as e:
                print(e)

            
        table_reg.to_csv("convergence_points_reg.csv")


run_without_reg()
repeat_with_reg("convergence_points.csv")