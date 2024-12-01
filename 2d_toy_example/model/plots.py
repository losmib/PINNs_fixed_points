from matplotlib.patches import FancyArrowPatch
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns


def arrow(x,y,ax,n):
    d = len(x)//(n+1)    
    ind = np.arange(d,len(x),d)
    for i in ind:
        ar = FancyArrowPatch((x[i-1],y[i-1]),(x[i],y[i]), 
                              arrowstyle='->', mutation_scale=20)
        ax.add_patch(ar)


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
    for x_fix in [0, 1]:
        axes[0].axhline(x_fix, lw=1, ls='--', c='red')
    axes[0].axhline(3, lw=1, ls='--', c='green')

    # make plots
    axes[0].plot(t_line, x_true, c='blue', lw=1, label='Reference')
    axes[0].plot(t_line, x_pred, c='red', lw=1, ls='--', label='Prediction')

    # Axis appearance
    axes[0].legend()
    axes[0].set_ylabel(r'$x$')
    axes[0].set_xlabel(r'$t$')

    # include fixed point lines
    for y_fix in [0, 1]:
        axes[1].axhline(y_fix, lw=1, ls='--', c='red')
    axes[1].axhline(2, lw=1, ls='--', c='green')

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


def plot_toy_example_direction(PINN, path=None):
    # get reference solution (analytical)
    t_line, x_true, y_true = PINN.data.reference()
  
    # get PINN prediction
    preds = PINN(t_line)
    x_pred = preds[:, 0]
    y_pred = preds[:, 1]

    fig, ax = plt.subplots()
    ax.plot(x_true, y_true, c='blue', lw=1, label='Reference')
    ax.plot(x_pred, y_pred, c='red', lw=1, ls='--', label='Prediction')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(ls='--')
    ax.legend()
    
    arrow(tf.squeeze(x_true), tf.squeeze(y_true), ax=ax, n=3)
    arrow(x_pred, y_pred, ax=ax, n=3)
        
    # include fixed points
    for x_fix, y_fix, col in [(0, 0, 'red'), (0, 2, 'green'), (3, 0, 'green'), (1, 1, 'red')]:
        ax.scatter(x_fix, y_fix, c=col)

    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


def plot_regularization_over_domain(PINN, path=None):
    fig, ax = plt.subplots()

    t_line, x_true, y_true = PINN.data.reference()
    N = 500
    eps = 0.1
    # x = np.linspace(min(0, np.min(x_true)) - eps, np.max(x_true) + eps, N)
    # y = np.linspace(min(0, np.min(y_true)) - eps, np.max(y_true) + eps, N)

    x = np.linspace(-1, 3, N)
    y = np.linspace(-1, 3, N)
    
    xx, yy = np.meshgrid(x, y)
    mesh_shape = xx.shape
    xx, yy = xx.reshape(-1, 1), yy.reshape(-1, 1)

    unstable_fp_reg_loss = PINN.loss.regularizer_unstable_fp(t_col=None, x=xx, x_t=None, y=yy, y_t=None)
    zz = unstable_fp_reg_loss.numpy().reshape(mesh_shape)
    xx, yy = xx.reshape(mesh_shape), yy.reshape(mesh_shape)

    # plt.contourf(xx, yy, zz)
    im = ax.imshow(zz, vmin = 0., vmax = np.max(zz), cmap=plt.cm.coolwarm, origin='lower', 
           extent=[xx.min(), xx.max(), yy.min(), yy.max()])
    plt.colorbar(im, ax=ax)
    ax.plot(x_true, y_true, c='blue', lw=1, label='Reference')
    arrow(tf.squeeze(x_true), tf.squeeze(y_true), ax=ax, n=3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
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
    t_col, x_true, y_true = PINN.data.reference()
    preds = PINN(t_col)
    x_pred = preds[:, 0]
    y_pred = preds[:, 1]

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t_col, x_pred, label="predictions x")
    plt.plot(t_col, y_pred, label="predictions y")
    plt.legend()
    
    with tf.GradientTape() as tape:
        tape.watch(t_col)
        loss, _, _, _, _ = PINN.loss.physics_loss(t_col)
        tf.print(loss.shape)
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
    
    physics_loss, x, x_t, y, y_t = PINN.loss.physics_loss(t_col)
    
    reg_loss = tf.zeros_like(t_col)
    if PINN.loss.regularizer is not None:
        reg_loss = PINN.loss.regularizer(t_col, x, x_t, y, y_t)
        
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