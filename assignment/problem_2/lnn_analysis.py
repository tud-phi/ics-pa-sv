from functools import partial
from jax import Array, vmap
from jax import numpy as jnp
from matplotlib import rcParams
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import seaborn as sns
from tqdm.notebook import tqdm  # progress bar
from typing import Dict, List

from jax_double_pendulum.dynamics import dynamical_matrices


def plot_dataset_state_distribution_polar_scatter(
    dt: float,
    dataset: Dict[str, Array],
    filepath: str = None,
):
    """
    Plot the state distribution of the training set using two polar scatter plots
    Args:
        dt: Time step of the dataset [s]
        dataset: Dictionary containing the dataset. The entries are:
            - dt_ss: array of shape (N, ) containing the time step between the current and the next state [s]
            - tau_ss: Array of shape (N, 2) containing the external generalized torques acting on the double pendulum. [Nm]
            - th_curr_ss: Array of shape (N, 2) containing the current link angles of the double pendulum. [rad]
            - th_d_curr_ss: Array of shape (N, 2) containing the current link angular velocities of the double pendulum. [rad/s]
            - th_next_ss: Array of shape (N, 2) containing the nextt link angles of the double pendulum. [rad]
            - th_d_next_ss: Array of shape (N, 2) containing the next link angular velocities of the double pendulum. [rad/s]
        filepath: Path to save the plot.
    """
    fig = plt.figure(num="Training set distribution", figsize=(9, 4.5))
    ax1 = plt.subplot(121, projection="polar")
    ax2 = plt.subplot(122, projection="polar")

    c1 = ax1.scatter(
        dataset["th_curr_ss"][:, 0],
        dataset["th_d_curr_ss"][:, 0],
        c=(dataset["th_d_next_ss"][:, 0] - dataset["th_d_curr_ss"][:, 0]) / dt,
        s=rcParams["lines.markersize"] ** 2 / 6,
        cmap="coolwarm",
    )
    fig.colorbar(c1, ax=ax1, shrink=0.6, label=r"$\ddot{\theta}_1$ [rad/s^2]")
    c2 = ax2.scatter(
        dataset["th_curr_ss"][:, 1],
        dataset["th_d_curr_ss"][:, 1],
        c=(dataset["th_d_next_ss"][:, 1] - dataset["th_d_curr_ss"][:, 1]) / dt,
        s=rcParams["lines.markersize"] ** 2 / 6,
        cmap="coolwarm",
    )
    fig.colorbar(c2, ax=ax2, shrink=0.6, label=r"$\ddot{\theta}_2$ [rad/s^2]")

    ax1.set_title(r"First link")
    ax2.set_title(r"Second link")
    ax1.set_xlabel(r"$\theta_1$ [deg]")
    ax1.text(
        0, ax1.get_rmax() / 2.0, r"$\dot{\theta}_1$ [rad/s]", ha="center", va="center"
    )
    ax2.set_xlabel(r"$\theta_2$ [deg]")
    ax2.text(
        0, ax2.get_rmax() / 2.0, r"$\dot{\theta}_2$ [rad/s]", ha="center", va="center"
    )

    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)


def plot_dataset_state_distribution_torus(
    num_simulations: int,
    dataset: Dict[str, Array],
    r1: float = 2.0,
    r2: float = 1.0,
    res: float = 40,
    color="#808080",
    filepath: str = None,
):
    """
    Plot the distribution of the training set using a torus
    Args:
        num_simulations: Number of simulations included in the dataset.
        dataset: Dictionary containing the dataset. The entries are:
            - dt_ss: array of shape (N, ) containing the time step between the current and the next state [s]
            - tau_ss: Array of shape (N, 2) containing the external generalized torques acting on the double pendulum. [Nm]
            - th_curr_ss: Array of shape (N, 2) containing the current link angles of the double pendulum. [rad]
            - th_d_curr_ss: Array of shape (N, 2) containing the current link angular velocities of the double pendulum. [rad/s]
            - th_next_ss: Array of shape (N, 2) containing the nextt link angles of the double pendulum. [rad]
            - th_d_next_ss: Array of shape (N, 2) containing the next link angular velocities of the double pendulum. [rad/s]
        r1: large radius of the torus
        r2: small radius of the torus
        res: resolution of the torus
        color: color of the torus
        filepath: Path to save the plot.
    """

    def immerse_torus(u: Array, v: Array):
        alpha = r1 + r2 * jnp.cos(v)
        return alpha * jnp.cos(u), alpha * jnp.sin(u), r2 * jnp.sin(v)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Hide the Figure name at the top of the figure
    fig.canvas.header_visible = False
    # If true then scrolling while the mouse is over the canvas will not move the entire notebook
    fig.canvas.capture_scroll = True

    def plot_curve(curve, *args, **kwargs):
        c = jnp.atleast_2d(curve)
        x, y, z = immerse_torus(c[..., 0], c[..., 1])
        ax.plot(x, y, z, *args, **kwargs)

    # plot wireframe
    u, v = jnp.meshgrid(
        jnp.linspace(-jnp.pi, jnp.pi, res),
        jnp.linspace(-jnp.pi, jnp.pi, res),
    )
    torus = immerse_torus(u, v)
    ax.plot_wireframe(*torus, lw=0.5, color=color)

    # reshape dataset entries
    reshaped_dataset = {}
    for key, value in dataset.items():
        if value.ndim == 1:
            reshaped_dataset[key] = value.reshape((num_simulations, -1))
        elif value.ndim == 2:
            reshaped_dataset[key] = value.reshape(
                (num_simulations, -1, value.shape[-1])
            )
        else:
            raise NotImplementedError

    # plot trajectories
    th_d_norm_max = jnp.max(jnp.linalg.norm(reshaped_dataset["th_d_curr_ss"], axis=-1))
    num_colors = 1000
    colors = plt.cm.coolwarm(jnp.linspace(0, 1, num_colors))
    color_indices = (
        jnp.linalg.norm(reshaped_dataset["th_d_curr_ss"], axis=-1)
        / th_d_norm_max
        * (num_colors - 1)
    ).astype(int)

    """ Simple plot with a different color for each simulation
    for sim_idx in (pbar := tqdm(range(1, num_simulations + 1))):
        plot_curve(
            reshaped_dataset["th_curr_ss"][sim_idx - 1],
            label=f"Sim. {sim_idx}"
        )
    """

    # Euclidean norm of configuration space velocities
    th_d_euclidean_norm = jnp.linalg.norm(reshaped_dataset["th_d_curr_ss"], axis=-1)
    # Create a continuous norm to map from data points to colors
    cmap = plt.cm.coolwarm
    color_norm = plt.Normalize(
        jnp.min(th_d_euclidean_norm), jnp.max(th_d_euclidean_norm)
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=color_norm)
    for sim_idx in (pbar := tqdm(range(1, num_simulations + 1))):
        # Create a set of line segments so that we can color them individually
        # The segments array for line collection
        # needs to be (numlines) x 2 (points per line) x 3 (for x, y, z)
        planar_segments = reshaped_dataset["th_curr_ss"][sim_idx - 1]
        th_curr_x, th_curr_y, th_curr_z = immerse_torus(
            reshaped_dataset["th_curr_ss"][sim_idx - 1, :, 0],
            reshaped_dataset["th_curr_ss"][sim_idx - 1, :, 1],
        )
        th_next_x, th_next_y, th_next_z = immerse_torus(
            reshaped_dataset["th_next_ss"][sim_idx - 1, :, 0],
            reshaped_dataset["th_next_ss"][sim_idx - 1, :, 1],
        )
        th_curr_xyz = jnp.stack([th_curr_x, th_curr_y, th_curr_z], axis=-1).reshape(
            -1, 1, 3
        )
        th_next_xyz = jnp.stack([th_next_x, th_next_y, th_next_z], axis=-1).reshape(
            -1, 1, 3
        )
        segments = jnp.concatenate([th_curr_xyz, th_next_xyz], axis=1)

        lc = Line3DCollection(segments, cmap=cmap, norm=color_norm)
        # Set the values used for colormapping
        lc.set_array(th_d_euclidean_norm[sim_idx - 1])
        lc.set_linewidth(2)
        line = ax.add_collection(lc)

    plt.colorbar(sm, ax=ax, shrink=0.6, label=r"$||\dot{\theta}||_2$ [rad/s]")

    size = r1 + r2
    for lim in [ax.set_xlim, ax.set_ylim, ax.set_zlim]:
        lim(-size, size)

    ax.axis("off")

    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)


def plot_dataset_generalized_torque_distribution_violin(
    rp: Dict,
    dataset: Dict[str, Array],
    filepath: str = None,
):
    """
    Plot the generalized torque distribution of the training set using a violin plot
    Args:
        dataset: Dictionary containing the dataset. The entries are:
            - dt_ss: array of shape (N, ) containing the time step between the current and the next state [s]
            - tau_ss: Array of shape (N, 2) containing the external generalized torques acting on the double pendulum. [Nm]
            - th_curr_ss: Array of shape (N, 2) containing the current link angles of the double pendulum. [rad]
            - th_d_curr_ss: Array of shape (N, 2) containing the current link angular velocities of the double pendulum. [rad/s]
            - th_next_ss: Array of shape (N, 2) containing the nextt link angles of the double pendulum. [rad]
            - th_d_next_ss: Array of shape (N, 2) containing the next link angular velocities of the double pendulum. [rad/s]
        filepath: Path to save the plot.
    """
    dynamical_matrices_vmapped = vmap(partial(dynamical_matrices, rp))
    M_ss, C_ss, G_ss = dynamical_matrices_vmapped(
        dataset["th_curr_ss"], dataset["th_d_curr_ss"]
    )
    coriolis_forces_ss = jnp.einsum("ijk,ik->ij", C_ss, dataset["th_d_curr_ss"])

    plt.figure(num="Distribution of generalized torques")
    data = {
        r"Coriolis $C(\theta, \dot{\theta}) \, \dot{\theta}$": jnp.linalg.norm(
            coriolis_forces_ss, axis=-1
        ),
        r"Gravity $G(\theta)$": jnp.linalg.norm(G_ss, axis=-1),
        r"External $\tau$": jnp.linalg.norm(dataset["tau_ss"], axis=-1),
    }
    ax = sns.violinplot(data=data, density_norm="count", legend=False)
    plt.ylabel(r"$\ell_2$ norm of torques [Nm]")
    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)


def plot_lnn_training_convergence(
    val_loss_history: Array,
    train_metrics_history: List[Dict[str, Array]],
    val_metrics_history: List[Dict[str, Array]],
    show: bool = True,
    filepath: str = None,
):
    """
    Plot the convergence of the Lagrangian neural network training
    Args:
        val_loss_history: Array of validation losses for each epoch.
        train_metrics_history: List of dictionaries containing the training metrics for each epoch.
        val_metrics_history: List of dictionaries containing the validation metrics for each epoch.
        show: Boolean whether to show the plot or not.
        filepath: Path to save the plot.
    """
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        num="Lagrangian neural network training convergence",
        figsize=(9, 3.5),
    )

    # history of learning rates
    lr_history = jnp.array(
        [metrics["lr_mass_matrix_nn"] for metrics in train_metrics_history]
    )

    axes[0].plot(
        jnp.arange(lr_history.shape[0]),
        lr_history,
        linewidth=2,
    )
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Learning rate")

    axes[1].plot(
        jnp.arange(val_loss_history.shape[0]),
        val_loss_history,
        linewidth=2,
    )

    axes[1].set_yscale("log")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel(
        r"Validation loss: MSE of $\dot{\theta}_{k+1}$ $[\frac{\mathrm{rad}^2}{\mathrm{s}^2}]$"
    )

    for ax in axes.flatten():
        ax.grid(True)

    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)

    if show:
        plt.show()
    else:
        plt.close()


def plot_link_angular_acceleration_prediction_error(
    batch: Dict[str, Array],
    preds: Dict[str, Array],
    filepath: str = None,
):
    """
    Plot the prediction error of the (continuous) link angular accelerations
    Args:
        batch: dictionary of batch data
        preds: dictionary of predicted data
        filepath: path to save the plot
    """
    fig = plt.figure(num="Joint accelerations prediction error", figsize=(9, 4.5))
    ax1 = plt.subplot(121, projection="polar")
    ax2 = plt.subplot(122, projection="polar")

    compute_th_dd_fn = vmap(
        lambda dt, th_d_curr, th_d_next: (th_d_next - th_d_curr) / dt
    )
    th_dd_pred = compute_th_dd_fn(
        batch["dt_ss"], batch["th_d_curr_ss"], preds["th_d_next_ss"]
    )
    th_dd_target = compute_th_dd_fn(
        batch["dt_ss"], batch["th_d_curr_ss"], batch["th_d_next_ss"]
    )

    c1 = ax1.scatter(
        batch["th_curr_ss"][:, 0],
        batch["th_d_curr_ss"][:, 0],
        c=jnp.abs((th_dd_pred - th_dd_target))[:, 0],
        s=rcParams["lines.markersize"] ** 2 / 6,
        cmap="Reds",
    )
    fig.colorbar(
        c1,
        ax=ax1,
        shrink=0.6,
        label=r"$|\hat{\ddot{\theta}}_1 - \ddot{\theta}_1|$ [rad/s^2]",
    )
    c2 = ax2.scatter(
        batch["th_curr_ss"][:, 1],
        batch["th_d_curr_ss"][:, 1],
        c=jnp.abs((th_dd_pred - th_dd_target))[:, 1],
        s=rcParams["lines.markersize"] ** 2 / 6,
        cmap="Reds",
    )
    fig.colorbar(
        c2,
        ax=ax2,
        shrink=0.6,
        label=r"$|\hat{\ddot{\theta}}_2 - \ddot{\theta}_2|$ [rad/s^2]",
    )

    ax1.set_title(r"First link")
    ax2.set_title(r"Second link")
    ax1.set_xlabel(r"$\theta_1$ [deg]")
    ax1.text(
        0, ax1.get_rmax() / 2.0, r"$\dot{\theta}_1$ [rad/s]", ha="center", va="center"
    )
    ax2.set_xlabel(r"$\theta_2$ [deg]")
    ax2.text(
        0, ax2.get_rmax() / 2.0, r"$\dot{\theta}_2$ [rad/s]", ha="center", va="center"
    )

    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
    plt.show()


def plot_rollout_learned_dynamics_configuration_space(
    nominal_sim_ts: Dict[str, Array],
    learned_sim_ts: Dict[str, Array],
    filepath: str = None,
):
    """
    Plot the rollout of the learned dynamics in configuration space
    Args:
        nominal_sim_ts: dictionary of nominal simulation data
        learned_sim_ts: dictionary of learned simulation data
        filepath: path to save the plot
    """
    fig, axes = plt.subplots(
        3, 2, num="Rollout of learned dynamics: configuration-space", figsize=(10, 5)
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # plot th1
    axes[0, 0].plot(
        learned_sim_ts["t_ts"],
        learned_sim_ts["th_ts"][:, 0],
        color=colors[0],
        linewidth=3,
        label=r"$\hat{\theta}_1$",
    )
    axes[0, 0].plot(
        nominal_sim_ts["t_ts"],
        nominal_sim_ts["th_ts"][:, 0],
        color="black",
        linestyle=":",
        linewidth=2,
        label=r"$\theta_1$",
    )
    axes[0, 0].legend(loc="upper right")

    # plot th2
    axes[1, 0].plot(
        learned_sim_ts["t_ts"],
        learned_sim_ts["th_ts"][:, 1],
        color=colors[1],
        linewidth=3,
        label=r"$\hat{\theta}_2$",
    )
    axes[1, 0].plot(
        nominal_sim_ts["t_ts"],
        nominal_sim_ts["th_ts"][:, 1],
        color="black",
        linestyle=":",
        linewidth=2,
        label=r"$\theta_2$",
    )
    axes[1, 0].legend(loc="upper right")

    # plot error of th
    axes[2, 0].plot(
        learned_sim_ts["t_ts"],
        nominal_sim_ts["th_ts"][:, 0] - learned_sim_ts["th_ts"][:, 0],
        color=colors[0],
        linewidth=2,
        label=r"$\theta_1$",
    )
    axes[2, 0].plot(
        learned_sim_ts["t_ts"],
        nominal_sim_ts["th_ts"][:, 1] - learned_sim_ts["th_ts"][:, 1],
        color=colors[1],
        linewidth=2,
        label=r"$\theta_2$",
    )
    axes[2, 0].legend(loc="upper right")

    # set axis labels for first column
    axes[-1, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel(r"$\theta_1$ [rad]")
    axes[1, 0].set_ylabel(r"$\theta_2$ [rad]")
    axes[2, 0].set_ylabel(r"$\theta$ error [rad]")

    # plot th_d1
    axes[0, 1].plot(
        learned_sim_ts["t_ts"],
        learned_sim_ts["th_d_ts"][:, 0],
        color=colors[0],
        linewidth=3,
        label=r"$\hat{\dot{\theta}}_1$",
    )
    axes[0, 1].plot(
        nominal_sim_ts["t_ts"],
        nominal_sim_ts["th_d_ts"][:, 0],
        color="black",
        linestyle=":",
        linewidth=2,
        label=r"$\dot{\theta}_1$",
    )
    axes[0, 1].legend(loc="upper right")

    # plot th_d2
    axes[1, 1].plot(
        learned_sim_ts["t_ts"],
        learned_sim_ts["th_d_ts"][:, 1],
        color=colors[1],
        linewidth=3,
        label=r"$\hat{\dot{\theta}}_2$",
    )
    axes[1, 1].plot(
        nominal_sim_ts["t_ts"],
        nominal_sim_ts["th_d_ts"][:, 1],
        color="black",
        linestyle=":",
        linewidth=2,
        label=r"$\dot{\theta}_2$",
    )
    axes[1, 1].legend(loc="upper right")

    # plot error of th
    axes[2, 1].plot(
        learned_sim_ts["t_ts"],
        nominal_sim_ts["th_d_ts"][:, 0] - learned_sim_ts["th_d_ts"][:, 0],
        color=colors[0],
        linewidth=2,
        label=r"$\dot{\theta}_1$",
    )
    axes[2, 1].plot(
        learned_sim_ts["t_ts"],
        nominal_sim_ts["th_d_ts"][:, 1] - learned_sim_ts["th_d_ts"][:, 1],
        color=colors[1],
        linewidth=2,
        label=r"$\dot{\theta}_2$",
    )
    axes[2, 1].legend(loc="upper right")

    # set axis labels for second column
    axes[-1, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel(r"$\dot{\theta}_1$ [rad / s]")
    axes[1, 1].set_ylabel(r"$\dot{\theta}_2$ [rad / s]")
    axes[2, 1].set_ylabel(r"$\dot{\theta}$ error [rad / s]")

    for ax in axes.flatten():
        ax.set_xlim(nominal_sim_ts["t_ts"][0], nominal_sim_ts["t_ts"][-1])
        ax.grid()

    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
    plt.show()


def plot_rollout_learned_dynamics_operational_space(
    nominal_sim_ts: Dict[str, Array],
    learned_sim_ts: Dict[str, Array],
    filepath: str = None,
):
    """
    Plots the rollout of the learned dynamics in operational space.
    Args:
        nominal_sim_ts: Dictionary containing the time series of the nominal simulation.
        learned_sim_ts: Dictionary containing the time series of the learned simulation.
        filepath: Path to save the plot to.
    """
    fig, axes = plt.subplots(
        3, 2, num="Rollout of learned dynamics: operational-space", figsize=(10, 5)
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # plot x1
    axes[0, 0].plot(
        learned_sim_ts["t_ts"],
        learned_sim_ts["x_ts"][:, 0],
        color=colors[0],
        linewidth=3,
        label=r"$\hat{x}_1$",
    )
    axes[0, 0].plot(
        nominal_sim_ts["t_ts"],
        nominal_sim_ts["x_ts"][:, 0],
        color="black",
        linestyle=":",
        linewidth=2,
        label=r"$x_1$",
    )
    axes[0, 0].legend(loc="upper right")

    # plot x2
    axes[1, 0].plot(
        learned_sim_ts["t_ts"],
        learned_sim_ts["x_ts"][:, 1],
        color=colors[1],
        linewidth=3,
        label=r"$\hat{x}_2$",
    )
    axes[1, 0].plot(
        nominal_sim_ts["t_ts"],
        nominal_sim_ts["x_ts"][:, 1],
        color="black",
        linestyle=":",
        linewidth=2,
        label=r"$x_2$",
    )
    axes[1, 0].legend(loc="upper right")

    # plot error
    axes[2, 0].plot(
        learned_sim_ts["t_ts"],
        nominal_sim_ts["x_ts"][:, 0] - learned_sim_ts["x_ts"][:, 0],
        color=colors[0],
        linewidth=2,
        label=r"$x_1$",
    )
    axes[2, 0].plot(
        learned_sim_ts["t_ts"],
        nominal_sim_ts["x_ts"][:, 1] - learned_sim_ts["x_ts"][:, 1],
        color=colors[1],
        linewidth=2,
        label=r"$x_2$",
    )
    axes[2, 0].legend(loc="upper right")

    # set axis labels for first column
    axes[-1, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel(r"$x_1$ [m]")
    axes[1, 0].set_ylabel(r"$x_2$ [m]")
    axes[2, 0].set_ylabel(r"$x$ error [m]")

    # plot x_d1
    axes[0, 1].plot(
        learned_sim_ts["t_ts"],
        learned_sim_ts["x_d_ts"][:, 0],
        color=colors[0],
        linewidth=3,
        label=r"$\hat{\dot{x}}_1$",
    )
    axes[0, 1].plot(
        nominal_sim_ts["t_ts"],
        nominal_sim_ts["x_d_ts"][:, 0],
        color="black",
        linestyle=":",
        linewidth=2,
        label=r"$\dot{x}_1$",
    )
    axes[0, 1].legend(loc="upper right")

    # plot x_d2
    axes[1, 1].plot(
        learned_sim_ts["t_ts"],
        learned_sim_ts["x_d_ts"][:, 1],
        color=colors[1],
        linewidth=3,
        label=r"$\hat{\dot{x}}_2$",
    )
    axes[1, 1].plot(
        nominal_sim_ts["t_ts"],
        nominal_sim_ts["x_d_ts"][:, 1],
        color="black",
        linestyle=":",
        linewidth=2,
        label=r"$\dot{x}_2$",
    )
    axes[1, 1].legend(loc="upper right")

    # plot xd error
    axes[2, 1].plot(
        learned_sim_ts["t_ts"],
        nominal_sim_ts["x_d_ts"][:, 0] - learned_sim_ts["x_d_ts"][:, 0],
        color=colors[0],
        linewidth=2,
        label=r"$\dot{x}_1$",
    )
    axes[2, 1].plot(
        learned_sim_ts["t_ts"],
        nominal_sim_ts["x_d_ts"][:, 1] - learned_sim_ts["x_d_ts"][:, 1],
        color=colors[1],
        linewidth=2,
        label=r"$\dot{x}_2$",
    )
    axes[2, 1].legend(loc="upper right")

    # set axis labels for second column
    axes[-1, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel(r"$\dot{x}_1$ [m / s]")
    axes[1, 1].set_ylabel(r"$\dot{x}_2$ [m / s]")
    axes[2, 1].set_ylabel(r"$\dot{x}$ error [m / s]")

    for ax in axes.flatten():
        ax.set_xlim(nominal_sim_ts["t_ts"][0], nominal_sim_ts["t_ts"][-1])
        ax.grid()

    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
    plt.show()
