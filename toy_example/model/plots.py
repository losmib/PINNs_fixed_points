import numpy as np
import matplotlib.pyplot as plt
from model.loss_functions import Loss


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
    else:
        plt.show()
        
        
def loss_collocation(PINN, t_coll, path=None):
    """
    Plots loss over the collocation points

    :param PINN: 
    :param t_coll: 
    :param path:
    """
    t_coll_sorted = np.sort(t_coll)
    loss = Loss(PINN).toy_example(t_coll_sorted)
    
    plt.plot(t_coll_sorted, loss, label="physics loss")
    plt.legend()
    plt.show()
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    
    
