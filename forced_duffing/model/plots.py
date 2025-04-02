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
    

def forced_duffing_dynamics(PINN, path=None):
    
    fig, axes = plt.subplots(1, 2, figsize=(6, 2))

    ############################
    # Prediciton plot
    ############################

    # stable/unstable fixed points
    line_props = {'ls': '--', 'lw': 0.5}
    axes[0].axhline(0, c='red', **line_props)

    # Reference data and PINN prediction
    t_line, x_true, x_t_true = PINN.data.reference()
    x_pred = PINN(t_line)
    x_t_pred = PINN.x_t(t_line)

    plot_lim_offset = 0.02

    x_max = max(np.max(x_true) , np.max(x_pred)) + plot_lim_offset
    x_min = min(np.min(x_true), np.min(x_pred)) - plot_lim_offset

    x_t_max = max(np.max(x_t_true) , np.max(x_t_pred)) + plot_lim_offset
    x_t_min = min(np.min(x_t_true) , np.min(x_t_pred)) - plot_lim_offset

    # make plot
    axes[0].plot(t_line, x_true, c='blue', lw=1, label='Reference')
    axes[0].plot(t_line, x_pred, c='red', lw=1, ls='--', label='Prediction')

    # plot the cosine for reference
    axes[0].plot(t_line, PINN.loss.gamma * np.cos(PINN.loss.omega * t_line), c='black', lw=1, ls='--', label='Cosine')

    # Axis appearance
    axes[0].set_xlabel(f"t")
    axes[0].set_ylabel(f"x")
    axes[0].legend(frameon=False, loc=1, ncol=2, fontsize=8)
    # axes[0].set_ylim([x_min, x_max])

    # axes[0].set_yticks([i * np.pi for i in range(int(x_min / np.pi), int(x_max / np.pi))])
    # axes[0].set_yticklabels([str(i) + r'$\pi$' for i in range(int(x_min / np.pi), int(x_max / np.pi))])

    # Plot trajectories
    axes[1].plot(x_true, x_t_true, c='blue', lw=1, label='Reference')
    axes[1].plot(x_pred, x_t_pred, c='red', lw=1, ls='--', label='Prediction')

    # Axis appearance
    axes[1].set_xlabel(f"x")
    axes[1].set_ylabel(f"x_t")
    # axes[1].set_xticks([i * np.pi for i in range(int(x_min / np.pi), int(x_max / np.pi))])
    # axes[1].set_xticklabels([str(i) + r'$\pi$' for i in range(int(x_min / np.pi), int(x_max / np.pi))])                
  
    plt.tight_layout()
    
    if path == None:
        plt.show()
        plt.close()
    else:
        plt.savefig(path)
        

def regularization_over_domain(PINN, path=None):
    t_line, x_true, x_t_true = PINN.data.reference()
    x_pred = PINN(t_line)
    x_t_pred = PINN.x_t(t_line)

    plot_lim_offset = 0.02

    x_max = max(np.max(x_true), np.max(x_pred)) + plot_lim_offset
    x_min = min(np.min(x_true), np.min(x_pred)) - plot_lim_offset

    x_t_max = max(np.max(x_t_true), np.max(x_t_pred)) + plot_lim_offset
    x_t_min = min(np.min(x_t_true), np.min(x_t_pred)) - plot_lim_offset

    #################
    # Quiver plot (Phase Space) over regularization loss landscape
    #################

    plt.figure()

    # Background arrows
    xscale, yscale, n_arrows = 1.2, 2, 10
    x = np.linspace(x_min, x_max, 500)
    x_t = np.linspace(x_t_min, x_t_max, 500)
    XX, YY = np.meshgrid(x, x_t)
    grid_shape = XX.shape
    x, x_t = XX.flatten(), YY.flatten()
    
    reg_loss = PINN.loss.regularizer_unstable_fp(t_col=None, x=x, x_t=x_t, x_tt=None)
    # Plot regularization loss landscape
    # plt.contourf(x, x_t, reg_loss)
    plt.imshow(reg_loss.numpy().reshape(grid_shape), vmin=0., vmax=np.max(reg_loss), cmap=plt.cm.coolwarm, origin='lower', 
           extent=[x.min(), x.max(), x_t.min(), x_t.max()])
    plt.colorbar()

    x = np.linspace(x_min, x_max, n_arrows)
    x_t = np.linspace(x_t_min, x_t_max, n_arrows)
    XX, YY = np.meshgrid(x, x_t)
    
    Y = np.vstack([XX.flatten(), YY.flatten()])
    t = np.zeros(len(Y))
    [dx, dx_t] = PINN.data.diff_equations(t, Y)
    x, x_t = XX.flatten(), YY.flatten()

    plt.quiver(x, x_t, dx, dx_t, color='0.5')
    # Fixed Points
    plt.scatter(0, 0, edgecolors='r', facecolors='none')
    # Axis lines
    plt.axhline(0, lw=1, ls='--', c='black')
    plt.axvline(0, lw=1, ls='--', c='black')

    # Plot trajectories
    plt.plot(x_true, x_t_true, c='blue', lw=1, label='Reference')
    plt.plot(x_pred, x_t_pred, c='red', lw=1, ls='--', label='Prediction')

    # Axis appearance
    plt.xlabel(f"x")
    plt.ylabel(f"x_t")

        
def loss_over_tcoll(PINN, path=None):
    """
    Plots physics loss and its gradient over collocation points

    :param PINN: 
    :param path: , defaults to None
    """
    t_line, x_true, x_t_true = PINN.data.reference()
    x_pred = PINN(t_line)
    x_t_pred = PINN.x_t(t_line)
    
    plt.subplot(3, 1, 1)
    plt.plot(t_line, x_pred, label="predictions")
    plt.legend()
    
    with tf.GradientTape() as tape:
        tape.watch(t_line)
        loss, x, _, _ = PINN.loss.physics_loss(t_line)
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
    t_line, x_true, x_t_true = PINN.data.reference()
    x_pred = PINN(t_line)
    x_t_pred = PINN.x_t(t_line)
    
    plt.subplot(3, 1, 1)
    plt.plot(t_line, x_pred, label="predictions")
    plt.legend()
    
    physics_loss, x, x_t, x_tt = PINN.loss.physics_loss(t_line)
    
    reg_loss = tf.zeros_like(t_line)
    if PINN.loss.regularizer is not None:
        reg_loss = PINN.loss.regularizer(t_line, x, x_t, x_tt)
        
    plt.subplot(3, 1, 2)
    plt.plot(t_line, reg_loss, label="regularization loss")
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(t_line, tf.exp(-(x_tt**2 + x_t**2)), label="rbf distance to fixed point")
    
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
                           columns=["x0", "T"],
                           aggfunc="mean")
    sns.heatmap(table)
    plt.show()


def results_linear_decay(results):
    results = results.loc[results["reg_decay"] == "linear"]
    
    results["split"] = results["T"].astype(str) + "-" + results["x0"].astype(str) + "-" +results["regularization"].astype(str)

    for split in results["split"].unique():
        results_split = results.loc[results["split"] == split]
        table = pd.pivot_table(results_split, values="loss_successes_percent", 
                            index=["reg_epochs"],
                            columns=["reg_coeff"],
                            aggfunc="mean")
        sns.heatmap(table)
        plt.title(split)
        plt.show()