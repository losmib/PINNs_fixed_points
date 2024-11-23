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
    # get reference solution (analytical)
    t_line, x_true, y_true = PINN.data.reference()
    # get PINN prediction
    preds = PINN(t_line)
    x_pred = preds[:, 0]
    y_pred = preds[:, 1]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 2.5))

    # include fixed point lines
    for x_fix in [0, 3]:
        axes[0].axhline(x_fix, lw=1, ls='--', c='green')
    axes[0].axhline(0, lw=1, ls='--', c='red')

    # make plots
    axes[0].plot(t_line, x_true, c='blue', lw=1, label='Reference')
    axes[0].plot(t_line, x_pred, c='red', lw=1, ls='--', label='Prediction')

    # Axis appearance
    axes[0].legend()
    axes[0].set_ylabel(r'$x$')
    axes[0].set_xlabel(r'$t$')

    # include fixed point lines
    for y_fix in [0, 2]:
        axes[1].axhline(y_fix, lw=1, ls='--', c='green')
    axes[1].axhline(0, lw=1, ls='--', c='red')

    # make plots
    axes[1].plot(t_line, y_true, c='blue', lw=1, label='Reference')
    axes[1].plot(t_line, y_pred, c='red', lw=1, ls='--', label='Prediction')

    # Axis appearance
    axes[1].legend()
    axes[1].set_ylabel(r'$y$')
    axes[1].set_xlabel(r'$t$')

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
    t_col, x_pred, y_pred = PINN.data.reference()
    
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t_col, x_pred, label="predictions x")
    plt.plot(t_col, y_pred, label="predictions y")
    plt.legend()
    
    with tf.GradientTape() as tape:
        tape.watch(t_col)
        loss, _, _, _, _ = PINN.loss.physics_loss(t_col)
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
        
    
def results_heatmap(results):
    results["reg_decay"].fillna("no decay", inplace=True)
    table = pd.pivot_table(results, values="loss_successes_percent", 
                           index=["regularization", "reg_epochs", "reg_coeff", "reg_decay"],
                           columns=["y0", "T"],
                           aggfunc="mean")
    sns.heatmap(table)
    plt.show()


def plot_comparissons(results):
    results["T-y0"] = results["T"].astype(str) + "-" + results["y0"].astype(str)
    sns.barplot(results, x="T-y0", y="loss_successes_percent", hue="regularization")
    plt.show()