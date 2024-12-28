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

    # make plot
    axes[0].plot(t_line, theta_true, c='blue', lw=1, label='Reference')
    axes[0].plot(t_line, theta_pred, c='red', lw=1, ls='--', label='Prediction')

    # Axis appearance
    axes[0].set_xlabel(r'$t$')
    axes[0].set_ylabel(r'$\theta$')
    axes[0].legend(frameon=False, loc=1, ncol=2, fontsize=8)
    axes[0].set_ylim([-7, 7])

    #################
    # Quiver plot (Phase Space)
    #################

    # Background arrows
    xscale, yscale, n_arrows = 1.2, 2, 10
    theta = np.linspace(-xscale*np.pi, xscale*np.pi, n_arrows)
    omega = np.linspace(-yscale*np.pi, yscale*np.pi, n_arrows)
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
    axes[1].set_xticks([-np.pi, 0, np.pi])
    axes[1].set_xticklabels([r'$\pi$', 0, r'$\pi$'])                

    plt.tight_layout()
    
    if path == None:
        plt.show()
        plt.close()
    else:
        plt.savefig(path)

def plot_regularization_over_domain(PINN, path=None):
    fig, axes = plt.subplots(1, 2, figsize=(6, 2))

    t_line, theta_true, omega_true = PINN.data.reference()
    N = 500
    eps = 0.1
    # x = np.linspace(min(0, np.min(x_true)) - eps, np.max(x_true) + eps, N)
    # y = np.linspace(min(0, np.min(y_true)) - eps, np.max(y_true) + eps, N)

    theta = np.linspace(-2*np.pi - 0.1, 2*np.pi + 0.1, t_line.shape[0]) 
    reg_loss = PINN.loss.regularizer_unstable_fp(t_col=None, theta=theta, omega=None, omega_t=None).numpy()
    reg_loss_grid = np.repeat(reg_loss.reshape(-1, 1), reg_loss.shape[0], axis=1)
    
    im1 = axes[0].imshow(reg_loss_grid, vmin=0., vmax=np.max(reg_loss), cmap=plt.cm.coolwarm, origin='lower',
               extent=[t_line.numpy().min(), t_line.numpy().max(), theta.min(), theta.max()], aspect='auto')
    fig.colorbar(im1, orientation="vertical", ax=axes[0])
    axes[0].plot(t_line, theta_true, label="reference", color="black")
    axes[0].legend(frameon=False, loc=1, ncol=2, fontsize=8)
    axes[0].set_xlabel("T")
    axes[0].set_ylabel("theta")
    
    
    # Background arrows
    xscale, yscale, n_arrows = 2.0, 2.0, 10
    theta = np.linspace(-xscale*np.pi, xscale*np.pi, n_arrows)
    omega = np.linspace(-yscale*np.pi, yscale*np.pi, n_arrows)
    XX, YY = np.meshgrid(theta, omega)
    Y = np.vstack([XX.flatten(), YY.flatten()])
    t = np.zeros(len(Y))
    [dtheta, domega] = PINN.data.diff_equations(t, Y)
    theta, omega = XX.flatten(), YY.flatten()

    loss_theta = np.linspace(-xscale*np.pi, xscale*np.pi, 500)
    loss_omega = np.linspace(-yscale*np.pi, yscale*np.pi, 500)
    loss_theta_grid, loss_omega_grid = np.meshgrid(loss_theta, loss_omega)

    reg_loss_grid = PINN.loss.regularizer_unstable_fp(t_col=None, theta=loss_theta_grid.reshape(-1, 1), omega=None, omega_t=None).numpy().reshape(loss_theta_grid.shape)
    im2 = axes[1].imshow(reg_loss_grid, vmin=0., vmax=np.max(reg_loss), cmap=plt.cm.coolwarm, origin='lower',
               extent=[theta.min(), theta.max(), omega.min(), omega.max()], aspect='auto')
    fig.colorbar(im2, orientation="vertical", ax=axes[1])

    axes[1].quiver(theta, omega, dtheta, domega, color='0.5')
    # Fixed Points
    axes[1].scatter(np.pi, 0, edgecolors='r', facecolors='none')
    axes[1].scatter(-np.pi, 0, edgecolors='r', facecolors='none')
    axes[1].scatter(0, 0, edgecolors='g', facecolors='none')
    # Axis lines
    axes[1].axhline(0, lw=1, ls='--', c='black')
    axes[1].axvline(0, lw=1, ls='--', c='black')

    # Plot trajectories
    axes[1].plot(theta_true, omega_true, c='black', lw=1, label='Reference')

    # Axis appearance
    axes[1].set_xlabel(r'$\theta$')
    axes[1].set_ylabel(r'$\omega$')
    axes[1].set_xticks([-np.pi, 0, np.pi])
    axes[1].set_xticklabels([r'$\pi$', 0, r'$\pi$'])                

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
        
        
def results_heatmaps(results):
    results_no_reg = results.loc[results["regularization"] == "no_reg"]
    table_no_reg = pd.pivot_table(results_no_reg, values="loss_successes_percent", 
                           index=["T"],
                           columns=["theta0"],
                           aggfunc="mean")
    
    sns.heatmap(table_no_reg, vmin=0, vmax=1.0)
    plt.title("results without regulariyzation")
    plt.show()

    results = results.loc[results["reg_decay"] == "linear"]

    results["reg_decay"].fillna("no decay", inplace=True)
    table = pd.pivot_table(results, values="loss_successes_percent", 
                           index=["regularization"],
                           columns=["reg_coeff", "reg_epochs"],
                           aggfunc="mean")
    sns.heatmap(table, vmin=0, vmax=1.0)
    plt.title("results across regularization-regularization coefficient")
    plt.show()

    for regularization in results["regularization"].unique():
        if regularization == "no_reg":
            continue
        results_reg = results.loc[results["regularization"] == regularization]
        table_reg = pd.pivot_table(results_reg, values="loss_successes_percent", 
                           index=["T", "theta0"],
                           columns=["reg_coeff", "reg_epochs"],
                           aggfunc="mean")
        table_reg_coeffs = pd.pivot_table(results_reg, values="loss_successes_percent", 
                           index=["reg_coeff"],
                           columns=["reg_epochs"],
                           aggfunc="mean")
        plt.subplot(1, 2, 1)
        sns.heatmap(table_reg, vmin=0, vmax=1.0)

        plt.subplot(1, 2, 2)
        sns.heatmap(table_reg_coeffs, vmin=0, vmax=1.0)
        plt.title(f"results {regularization}")
        
        plt.show()


def plot_comparissons(results):
    results["T-theta0"] = results["T"].astype(str) + "-" + results["theta0"].astype(str)
    sns.barplot(results, x="T-theta0", y="loss_successes_percent", hue="regularization")
    plt.show()


def plot_EL_loss(PINN, path=None):
    t_line, theta_true, omega_true = PINN.data.reference()
    
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            tape1.watch(t_line)  
            tape2.watch(t_line) 
            physics_loss, theta, omega, omega_t = PINN.loss.physics_loss(t_line)

            left_side = 2 * (omega_t - PINN.loss.g/PINN.loss.l * tf.math.sin(theta)) * (-PINN.loss.g/PINN.loss.l * tf.math.cos(theta))
            right_side_derivative = tape1.gradient(2 * (omega_t - PINN.loss.g/PINN.loss.l * tf.math.sin(theta)), t_line)
            right_side = tape2.gradient(right_side_derivative, t_line)
            EL_loss = (left_side + right_side)**2

    plt.figure()
    plt.plot(t_line, physics_loss, label="physics loss")
    plt.plot(t_line, EL_loss, label="Euler Lagrange loss")
    plt.legend()
    
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()