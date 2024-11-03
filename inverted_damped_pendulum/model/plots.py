import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns


def learning_curves(log, path=None):
    
    fig, ax = plt.subplots(figsize=(4, 2.5))

    # Plot loss curves
    epochs = np.arange(0, log['N_epochs'], log['freq_log'])
    for loss in ['loss_IC', 'loss_P']:
        ax.plot(epochs, log[loss], lw=1, label=loss)

    # Axis appearance
    ax.set_title('Learning Curves')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(ls='--')
    ax.set_xlabel('Epoch')

    plt.tight_layout()
    
    if path == None:
        plt.show()
        plt.close()
    else:
        plt.savefig(path)
    

def pendulum_dynamics(PINN, path=None):
    
    fig, axes = plt.subplots(1, 2, figsize=(6, 2))

    ############################
    # Prediciton plot
    ############################

    # stable/unstable fixed points
    line_props = {'ls': '--', 'lw': 0.5}
    axes[0].axhline(0, c='red', **line_props)
    axes[0].axhline(np.pi, c='green', **line_props)
    axes[0].axhline(2 * np.pi, c='red', **line_props)

    # Reference data and PINN prediction
    t_line, theta_true, omega_true = PINN.data.reference()
    theta_pred = PINN(t_line)
    omega_pred = PINN.omega(t_line)

    theta_max = max(np.ceil(np.max(theta_true) / np.pi) * np.pi, 2*np.pi)
    theta_min = min(np.floor(np.min(theta_true) / np.pi) * np.pi, -2*np.pi)

    omega_max = max(np.ceil(np.max(omega_true) / np.pi) * np.pi, 2*np.pi)
    omega_min = min(np.floor(np.min(omega_true) / np.pi) * np.pi, -2*np.pi)

    # make plot
    axes[0].plot(t_line, theta_true, c='blue', lw=1, label='Reference')
    axes[0].plot(t_line, theta_pred, c='red', lw=1, ls='--', label='Prediction')

    # Axis appearance
    axes[0].set_xlabel(r'$t$')
    axes[0].set_ylabel(r'$\theta$')
    axes[0].legend(frameon=False, loc=1, ncol=2, fontsize=8)
    axes[0].set_ylim([theta_min, theta_max])

    axes[0].set_yticks([i * np.pi for i in range(int(theta_min / np.pi), int(theta_max / np.pi))])
    axes[0].set_yticklabels([str(i) + r'$\pi$' for i in range(int(theta_min / np.pi), int(theta_max / np.pi))])
    #################
    # Quiver plot (Phase Space)
    #################

    # Background arrows
    xscale, yscale, n_arrows = 1.2, 2, 10
    theta = np.linspace(theta_min, theta_max, n_arrows)
    omega = np.linspace(omega_min, omega_max, n_arrows)
    XX, YY = np.meshgrid(theta, omega)
    Y = np.vstack([XX.flatten(), YY.flatten()])
    t = np.zeros(len(Y))
    [dtheta, domega] = PINN.data.diff_equations(t, Y)
    theta, omega = XX.flatten(), YY.flatten()
    axes[1].quiver(theta, omega, dtheta, domega, color='0.5')
    # Fixed Points
    axes[1].scatter(np.pi, 0, edgecolors='r', facecolors='none')
    axes[1].scatter(-np.pi, 0, edgecolors='r', facecolors='none')
    axes[1].scatter(0, 0, edgecolors='g', facecolors='none')
    # Axis lines
    axes[1].axhline(0, lw=1, ls='--', c='black')
    axes[1].axvline(0, lw=1, ls='--', c='black')

    # Plot trajectories
    axes[1].plot(theta_true, omega_true, c='blue', lw=1, label='Reference')
    axes[1].plot(theta_pred, omega_pred, c='red', lw=1, ls='--', label='Prediction')

    # Axis appearance
    axes[1].set_xlabel(r'$\theta$')
    axes[1].set_ylabel(r'$\omega$')
    axes[1].set_xticks([i * np.pi for i in range(int(theta_min / np.pi), int(theta_max / np.pi))])
    axes[1].set_xticklabels([str(i) + r'$\pi$' for i in range(int(theta_min / np.pi), int(theta_max / np.pi))])                
    print(theta_min)
    print(theta_max)
    plt.tight_layout()
    
    if path == None:
        plt.show()
        plt.close()
    else:
        plt.savefig(path)
        
        
def loss_over_tcoll(PINN, path=None):
    """
    Plots physics loss and its gradient over collocation points

    :param PINN: 
    :param path: , defaults to None
    """
    t_line, theta_true, omega_true = PINN.data.reference()
    theta_pred = PINN(t_line)
    omega_pred = PINN.omega(t_line)
    
    plt.subplot(3, 1, 1)
    plt.plot(t_line, theta_pred, label="predictions")
    plt.legend()
    
    with tf.GradientTape() as tape:
        tape.watch(t_line)
        loss, theta, _, _ = PINN.loss.physics_loss(t_line)
        loss_grad = tape.gradient(loss, t_line)
        
    plt.subplot(3, 1, 2)
    plt.plot(t_line, loss, label="physics loss")
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(t_line, loss_grad, label="physics loss gradient")
    
    plt.legend()
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()
        
        
def plot_regularization(PINN, path=None):
    t_line, theta_true, omega_true = PINN.data.reference()
    theta_pred = PINN(t_line)
    omega_pred = PINN.omega(t_line)
    
    plt.subplot(3, 1, 1)
    plt.plot(t_line, theta_pred, label="predictions")
    plt.legend()
    
    physics_loss, theta, omega, omega_t = PINN.loss.physics_loss(t_line)
    
    reg_loss = tf.zeros_like(t_line)
    if PINN.loss.regularizer is not None:
        reg_loss = PINN.loss.regularizer(t_line, theta, omega, omega_t)
        
    plt.subplot(3, 1, 2)
    plt.plot(t_line, reg_loss, label="regularization loss")
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(t_line, tf.exp(-(omega_t**2 + omega**2)), label="rbf distance to fixed point")
    
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
                           columns=["theta0", "T"],
                           aggfunc="mean")
    sns.heatmap(table)
    plt.show()