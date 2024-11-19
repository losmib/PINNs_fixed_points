import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns


def learning_curves(log, path=None):
    
    fig, ax = plt.subplots(figsize=(4, 2.5))

    # Plot loss curves
    epochs = np.arange(0, log['N_epochs'], log['freq_log'])
    ax.plot(epochs, log['loss'], lw=1)

    # Axis appearance
    ax.set_title('Learning Curves')
    ax.set_yscale('log')
    ax.grid(ls='--')
    ax.set_xlabel('Epoch')

    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()
    
    
def toy_example_dynamics(PINN, path=None):
    
    # get (equally-spaced) data points
    t_line = PINN.data.t_line()
    # get reference solution (analytical)
    y_true = PINN.data.reference(t_line)
    # get PINN prediction
    y_pred = PINN(t_line)

    fig, ax = plt.subplots(figsize=(4, 2.5))

    # include fixed point lines
    for y_fix in [-1, 1]:
        ax.axhline(y_fix, lw=1, ls='--', c='green')
    ax.axhline(0, lw=1, ls='--', c='red')

    # make plots
    ax.plot(t_line, y_true, c='blue', lw=1, label='Reference')
    ax.plot(t_line, y_pred, c='red', lw=1, ls='--', label='Prediction')

    # Axis appearance
    ax.legend()
    ax.set_ylabel(r'$y$')
    ax.set_xlabel(r'$t$')

    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()
        
    
def loss_over_tcoll(PINN, path=None):
    """
    Plots physics loss and its gradient over collocation points

    :param PINN: 
    :param path: , defaults to None
    """
    t_col = PINN.data.t_line()
    
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t_col, PINN(t_col), label="predictions")
    plt.legend()
    
    with tf.GradientTape() as tape:
        tape.watch(t_col)
        loss, _, _ = PINN.loss.physics_loss(t_col)
        loss_grad = tape.gradient(loss, t_col)
        
    plt.subplot(3, 1, 2)
    plt.plot(t_col, loss, label="physics loss")
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(t_col, loss_grad, label="physics loss gradient")
    
    plt.legend()
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()
        

def plot_regularization(PINN, path=None):
    t_col = PINN.data.t_line()
    
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t_col, PINN(t_col), label="predictions")
    plt.legend()
    
    physics_loss, y, y_t = PINN.loss.physics_loss(t_col)
    
    reg_loss = tf.zeros_like(t_col)
    if PINN.loss.regularizer is not None:
        reg_loss = PINN.loss.regularizer(t_col, y, y_t)
        
    plt.subplot(3, 1, 2)
    plt.plot(t_col, reg_loss, label="regularization loss")
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(t_col, tf.exp(-y_t**2), label="rbf distance to fixed point")
    
    plt.legend()
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()
        
    
def results_heatmaps(results):
    results_no_reg = results.loc[results["regularization"] == "no_reg"]
    table_no_reg = pd.pivot_table(results_no_reg, values="loss_successes_percent", 
                           index=["T"],
                           columns=["y0"],
                           aggfunc="mean")
    sns.heatmap(table_no_reg, vmin=0, vmax=1.0)
    plt.title("results without regulariyzation")
    plt.show()

    results["reg_decay"].fillna("no decay", inplace=True)
    table = pd.pivot_table(results, values="loss_successes_percent", 
                           index=["regularization"],
                           columns=["reg_coeff"],
                           aggfunc="mean")
    sns.heatmap(table, vmin=0, vmax=1.0)
    plt.title("results across regularization-regularization coefficient")
    plt.show()

    for regularization in results["regularization"].unique():
        if regularization == "no_reg":
            continue
        results_reg = results.loc[results["regularization"] == regularization]
        table_reg = pd.pivot_table(results_reg, values="loss_successes_percent", 
                           index=["T", "y0"],
                           columns=["reg_coeff"],
                           aggfunc="mean")
        sns.heatmap(table_reg, vmin=0, vmax=1.0)
        plt.title(f"results {regularization}")
        plt.show()

