from jax import numpy as jnp
import matplotlib.pyplot as plt
from typing import Dict, Tuple


def plot_configuration_space_trajectory_following(
    traj_ts: Dict[str, jnp.ndarray],
    sim_ts: Dict[str, jnp.ndarray],
    filepath: str = None,
):
    fig, axes = plt.subplots(
        3, 2, num="Configuration-space trajectory following", figsize=(10, 5)
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # plot th1
    axes[0, 0].plot(
        sim_ts["t_ts"],
        sim_ts["th_ts"][:, 0],
        color=colors[0],
        linewidth=3,
        label=r"$\theta_1$",
    )
    if "th_est_ts" in sim_ts:
        axes[0, 0].plot(
            sim_ts["t_ts"],
            sim_ts["th_est_ts"][:, 0],
            color=colors[0],
            linestyle="--",
            linewidth=2,
            label=r"$\hat{\theta}_1$",
        )
    axes[0, 0].plot(
        traj_ts["t_ts"],
        traj_ts["th_ts"][:, 0],
        color="black",
        linestyle=":",
        linewidth=2,
        label=r"$\theta_1^\mathrm{d}$",
    )
    axes[0, 0].legend(loc="upper right")

    # plot th2
    axes[1, 0].plot(
        sim_ts["t_ts"],
        sim_ts["th_ts"][:, 1],
        color=colors[1],
        linewidth=3,
        label=r"$\theta_2$",
    )
    if "th_est_ts" in sim_ts:
        axes[1, 0].plot(
            sim_ts["t_ts"],
            sim_ts["th_est_ts"][:, 1],
            color=colors[1],
            linestyle="--",
            linewidth=2,
            label=r"$\hat{\theta}_2$",
        )
    axes[1, 0].plot(
        traj_ts["t_ts"],
        traj_ts["th_ts"][:, 1],
        color="black",
        linestyle=":",
        linewidth=2,
        label=r"$\theta_2^\mathrm{d}$",
    )
    axes[1, 0].legend(loc="upper right")

    # plot error of th
    axes[2, 0].plot(
        sim_ts["t_ts"],
        traj_ts["th_ts"][:, 0] - sim_ts["th_ts"][:, 0],
        color=colors[0],
        linewidth=2,
        label=r"$\theta_1^\mathrm{d} - \theta_1$",
    )
    if "th_est_ts" in sim_ts:
        axes[2, 0].plot(
            sim_ts["t_ts"],
            sim_ts["th_ts"][:, 0] - sim_ts["th_est_ts"][:, 0],
            color=colors[0],
            linestyle="--",
            linewidth=2,
            label=r"$\theta_1 - \hat{\theta}_1$",
        )
    axes[2, 0].plot(
        sim_ts["t_ts"],
        traj_ts["th_ts"][:, 1] - sim_ts["th_ts"][:, 1],
        color=colors[1],
        linewidth=2,
        label=r"$\theta_2^\mathrm{d} - \theta_2$",
    )
    if "th_est_ts" in sim_ts:
        axes[2, 0].plot(
            sim_ts["t_ts"],
            sim_ts["th_ts"][:, 1] - sim_ts["th_est_ts"][:, 1],
            color=colors[1],
            linestyle="--",
            linewidth=2,
            label=r"$\theta_2 - \hat{\theta}_2$",
        )
    axes[2, 0].legend(loc="upper right")

    # set axis labels for first column
    axes[-1, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel(r"$\theta_1$ [rad]")
    axes[1, 0].set_ylabel(r"$\theta_2$ [rad]")
    axes[2, 0].set_ylabel(r"$\theta$ error [rad]")

    # plot th_d1
    axes[0, 1].plot(
        sim_ts["t_ts"],
        sim_ts["th_d_ts"][:, 0],
        color=colors[0],
        linewidth=3,
        label=r"$\dot{\theta}_1$",
    )
    axes[0, 1].plot(
        traj_ts["t_ts"],
        traj_ts["th_d_ts"][:, 0],
        color="black",
        linestyle=":",
        linewidth=2,
        label=r"$\dot{\theta}_1^\mathrm{d}$",
    )
    axes[0, 1].legend(loc="upper right")

    # plot th_d2
    axes[1, 1].plot(
        sim_ts["t_ts"],
        sim_ts["th_d_ts"][:, 1],
        color=colors[1],
        linewidth=3,
        label=r"$\dot{\theta}_2$",
    )
    axes[1, 1].plot(
        traj_ts["t_ts"],
        traj_ts["th_d_ts"][:, 1],
        color="black",
        linestyle=":",
        linewidth=2,
        label=r"$\dot{\theta}_2^\mathrm{d}$",
    )
    axes[1, 1].legend(loc="upper right")

    # plot error of th_d
    axes[2, 1].plot(
        sim_ts["t_ts"],
        traj_ts["th_d_ts"][:, 0] - sim_ts["th_d_ts"][:, 0],
        color=colors[0],
        linewidth=2,
        label=r"$\dot{\theta}_1$",
    )
    axes[2, 1].plot(
        sim_ts["t_ts"],
        traj_ts["th_d_ts"][:, 1] - sim_ts["th_d_ts"][:, 1],
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
        ax.set_xlim(traj_ts["t_ts"][0], traj_ts["t_ts"][-1])
        ax.grid()

    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
    plt.show()


def plot_operational_space_trajectory_following(
    traj_ts: Dict[str, jnp.ndarray],
    sim_ts: Dict[str, jnp.ndarray],
    filepath: str = None,
):
    fig, axes = plt.subplots(
        3, 2, num="Operational-space trajectory following", figsize=(10, 5)
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # plot x1
    axes[0, 0].plot(
        sim_ts["t_ts"],
        sim_ts["x_ts"][:, 0],
        color=colors[0],
        linewidth=3,
        label=r"$x_1$",
    )
    if "x_est_ts" in sim_ts:
        axes[0, 0].plot(
            sim_ts["t_ts"],
            sim_ts["x_est_ts"][:, 0],
            color=colors[0],
            linestyle="--",
            linewidth=2,
            label=r"$\hat{x}_1$",
        )
    axes[0, 0].plot(
        traj_ts["t_ts"],
        traj_ts["x_ts"][:, 0],
        color="black",
        linestyle=":",
        linewidth=2,
        label=r"$x_1^\mathrm{d}$",
    )
    axes[0, 0].legend(loc="upper right")

    # plot x2
    axes[1, 0].plot(
        sim_ts["t_ts"],
        sim_ts["x_ts"][:, 1],
        color=colors[1],
        linewidth=3,
        label=r"$x_2$",
    )
    if "x_est_ts" in sim_ts:
        axes[1, 0].plot(
            sim_ts["t_ts"],
            sim_ts["x_est_ts"][:, 1],
            color=colors[1],
            linestyle="--",
            linewidth=2,
            label=r"$\hat{x}_2$",
        )
    axes[1, 0].plot(
        traj_ts["t_ts"],
        traj_ts["x_ts"][:, 1],
        color="black",
        linestyle=":",
        linewidth=2,
        label=r"$x_2^\mathrm{d}$",
    )
    axes[1, 0].legend(loc="upper right")

    # plot error
    axes[2, 0].plot(
        sim_ts["t_ts"],
        traj_ts["x_ts"][:, 0] - sim_ts["x_ts"][:, 0],
        color=colors[0],
        linewidth=2,
        label=r"$x^\mathrm{d}_1 - x_1$",
    )
    if "x_est_ts" in sim_ts:
        axes[2, 0].plot(
            sim_ts["t_ts"],
            sim_ts["x_ts"][:, 0] - sim_ts["x_est_ts"][:, 0],
            color=colors[0],
            linestyle="--",
            linewidth=2,
            label=r"$x_1 - \hat{x}_1$",
        )
    axes[2, 0].plot(
        sim_ts["t_ts"],
        traj_ts["x_ts"][:, 1] - sim_ts["x_ts"][:, 1],
        color=colors[1],
        linewidth=2,
        label=r"$x^\mathrm{d}_2 - x_2$",
    )
    if "x_est_ts" in sim_ts:
        axes[2, 0].plot(
            sim_ts["t_ts"],
            sim_ts["x_ts"][:, 1] - sim_ts["x_est_ts"][:, 1],
            color=colors[1],
            linestyle="--",
            linewidth=2,
            label=r"$x_2 - \hat{x}_2$",
        )
    axes[2, 0].legend(loc="upper right")

    # set axis labels for first column
    axes[-1, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel(r"$x_1$ [m]")
    axes[1, 0].set_ylabel(r"$x_2$ [m]")
    axes[2, 0].set_ylabel(r"$x$ error [m]")

    # plot x_d1
    axes[0, 1].plot(
        sim_ts["t_ts"],
        sim_ts["x_d_ts"][:, 0],
        color=colors[0],
        linewidth=3,
        label=r"$\dot{x}_1$",
    )
    axes[0, 1].plot(
        traj_ts["t_ts"],
        traj_ts["x_d_ts"][:, 0],
        color="black",
        linestyle=":",
        linewidth=2,
        label=r"$\dot{x}_1^\mathrm{d}$",
    )
    axes[0, 1].legend(loc="upper right")

    # plot x_d2
    axes[1, 1].plot(
        sim_ts["t_ts"],
        sim_ts["x_d_ts"][:, 1],
        color=colors[1],
        linewidth=3,
        label=r"$\dot{x}_2$",
    )
    axes[1, 1].plot(
        traj_ts["t_ts"],
        traj_ts["x_d_ts"][:, 1],
        color="black",
        linestyle=":",
        linewidth=2,
        label=r"$\dot{x}_2^\mathrm{d}$",
    )
    axes[1, 1].legend(loc="upper right")

    # plot xd error
    axes[2, 1].plot(
        sim_ts["t_ts"],
        traj_ts["x_d_ts"][:, 0] - sim_ts["x_d_ts"][:, 0],
        color=colors[0],
        linewidth=2,
        label=r"$\dot{x}_1$",
    )
    axes[2, 1].plot(
        sim_ts["t_ts"],
        traj_ts["x_d_ts"][:, 1] - sim_ts["x_d_ts"][:, 1],
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
        ax.set_xlim(traj_ts["t_ts"][0], traj_ts["t_ts"][-1])
        ax.grid()

    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
    plt.show()
