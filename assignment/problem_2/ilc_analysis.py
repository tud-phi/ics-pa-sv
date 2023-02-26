import numpy as np
from jax import numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import animation
from pathlib import Path
from typing import Dict, Tuple

from jax_double_pendulum.utils import normalize_link_angles


def compute_configuration_space_rmse_ilc_its(
    traj_ts: Dict[str, jnp.ndarray],
    ilc_its: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes the root mean squared error between the desired and actual trajectory in configuration space

    Args:
        traj_ts: A dictionary containing the desired trajectory states.
        ilc_its: A dictionary tracking the system states and inputs across ILC iterations and time steps.

    Returns:
        rmse_th_its: The root mean squared error of the link angles. Shape: (num_its, 2)
        rmse_th_d_its: The root mean squared error of the link angular velocities. Shape: (num_its, 2)
        rmse_th_dd_its: The root mean squared error of the link angular accelerations. Shape: (num_its, 2)
    """
    th_des_its = jnp.repeat(
        jnp.expand_dims(traj_ts["th_ts"], axis=0),
        repeats=ilc_its["it_its"].shape[0],
        axis=0,
    )
    th_d_des_its = jnp.repeat(
        jnp.expand_dims(traj_ts["th_d_ts"], axis=0),
        repeats=ilc_its["it_its"].shape[0],
        axis=0,
    )
    th_dd_des_its = jnp.repeat(
        jnp.expand_dims(traj_ts["th_dd_ts"], axis=0),
        repeats=ilc_its["it_its"].shape[0],
        axis=0,
    )

    def compute_rmse_its(err_its: jnp.ndarray) -> jnp.ndarray:
        # mean over time
        return jnp.sqrt(jnp.mean(err_its**2, axis=1))

    # error mapped to [-pi, pi]
    th_err = normalize_link_angles(th_des_its - ilc_its["th_its"])

    # compute RMSEs
    rmse_th_its = compute_rmse_its(th_err)
    rmse_th_d_its = compute_rmse_its(th_d_des_its - ilc_its["th_d_its"])
    rmse_th_dd_its = compute_rmse_its(th_dd_des_its - ilc_its["th_dd_its"])

    return rmse_th_its, rmse_th_d_its, rmse_th_dd_its


def plot_configuration_space_ilc_convergence(
    traj_ts: Dict[str, jnp.ndarray],
    ilc_its: Dict[str, jnp.ndarray],
    show: bool = True,
    filepath: str = None,
) -> None:
    """
    Plot the convergence of the configuration-space states in the ILC framework.

    Args:
        traj_ts: A dictionary containing the desired trajectory states.
        ilc_its: Dictionary of ILC iterations.
        show: boolean flag to decide whether show the animation
        filepath: Path to save the plot to.
    """
    fig, ax = plt.subplots(
        1, 1, num="Configuration-space ILC convergence", figsize=(5.5, 3.5)
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # compute RMSEs
    (
        rmse_th_its,
        rmse_th_d_its,
        rmse_th_dd_its,
    ) = compute_configuration_space_rmse_ilc_its(traj_ts, ilc_its)

    # plot RMSEs
    ax.plot(
        ilc_its["it_its"],
        rmse_th_its[:, 0],
        linewidth=2,
        color=colors[0],
        label=r"RMSE $\theta_1$",
    )
    ax.plot(
        ilc_its["it_its"],
        rmse_th_its[:, 1],
        linewidth=2,
        color=colors[1],
        label=r"RMSE $\theta_2$",
    )

    # set axis labels
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"RMSE $\theta$ [rad]")

    ax.set_ylim(bottom=0.0, top=None)
    ax.legend(loc="upper right")
    ax.grid()

    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)

    if show:
        plt.show()
    else:
        plt.close()


def animate_configuration_space_trajectory_following_plot(
    traj_ts: Dict[str, jnp.ndarray],
    ilc_its: Dict[str, jnp.ndarray],
    max_num_animated_its: int = 100,
    show: bool = True,
    filepath: str = None,
) -> animation.FuncAnimation:
    """
    Animate the configuration-space sequence through the ILC iterations.

    Args:
        traj_ts: A dictionary containing the desired trajectory states.
        ilc_its: Dictionary of ILC iterations.
        max_num_animated_its: Maximum number of iterations being animated. All other iterations in between are skipped.
        show: boolean flag to decide whether show the animation
        filepath: Path to save the plot to.
    """
    fig, axes = plt.subplots(
        3,
        2,
        num="Animation ILC Configuration-space trajectory following",
        figsize=(10, 5),
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    num_its = ilc_its["t_its"].shape[0]
    # it_text = plt.text(0.5, 1.1, "")
    it_text = fig.suptitle("")

    # plot th1
    (line_th1,) = axes[0, 0].plot(
        [],
        [],
        color=colors[0],
        linewidth=3,
        label=r"$\theta_1$",
    )
    axes[0, 0].plot(
        traj_ts["t_ts"],
        traj_ts["th_ts"][:, 0],
        color="black",
        linestyle=":",
        linewidth=2,
        label=r"$\theta_1^\mathrm{d}$",
    )

    # plot th2
    (line_th2,) = axes[1, 0].plot(
        [],
        [],
        color=colors[1],
        linewidth=3,
        label=r"$\theta_2$",
    )
    axes[1, 0].plot(
        traj_ts["t_ts"],
        traj_ts["th_ts"][:, 1],
        color="black",
        linestyle=":",
        linewidth=2,
        label=r"$\theta_2^\mathrm{d}$",
    )

    # plot error of th
    (line_error_th1,) = axes[2, 0].plot(
        [],
        [],
        color=colors[0],
        linewidth=2,
        label=r"$\theta_1$",
    )
    (line_error_th2,) = axes[2, 0].plot(
        [],
        [],
        color=colors[1],
        linewidth=2,
        label=r"$\theta_2$",
    )
    axes[2, 0].set_ylim(
        [-jnp.max(jnp.abs(traj_ts["th_ts"])), jnp.max(jnp.abs(traj_ts["th_ts"]))]
    )

    # set axis labels for first column
    axes[-1, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel(r"$\theta_1$ [rad]")
    axes[1, 0].set_ylabel(r"$\theta_2$ [rad]")
    axes[2, 0].set_ylabel(r"$\theta$ error [rad]")

    # plot th_d1
    (line_th_d1,) = axes[0, 1].plot(
        [],
        [],
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

    # plot th_d2
    (line_th_d2,) = axes[1, 1].plot(
        [],
        [],
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

    # plot error of th_d
    (line_error_th_d1,) = axes[2, 1].plot(
        [],
        [],
        color=colors[0],
        linewidth=2,
        label=r"$\dot{\theta}_1$",
    )
    (line_error_th_d2,) = axes[2, 1].plot(
        [],
        [],
        color=colors[1],
        linewidth=2,
        label=r"$\dot{\theta}_2$",
    )
    axes[2, 1].set_ylim(
        [-jnp.max(jnp.abs(traj_ts["th_d_ts"])), jnp.max(jnp.abs(traj_ts["th_d_ts"]))]
    )

    # set axis labels for second column
    axes[-1, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel(r"$\dot{\theta}_1$ [rad / s]")
    axes[1, 1].set_ylabel(r"$\dot{\theta}_2$ [rad / s]")
    axes[2, 1].set_ylabel(r"$\dot{\theta}$ error [rad / s]")

    for ax in axes.flatten():
        ax.set_xlim(traj_ts["t_ts"][0], traj_ts["t_ts"][-1])
        ax.grid()
        ax.legend(loc="upper left")

    plt.tight_layout()

    def _init():
        it_text.set_text("")
        return (it_text,)

    def _animate(it: int):
        it_text.set_text(f"Iteration {it + 1} / {num_its}")

        line_th1.set_data(ilc_its["t_its"][it], ilc_its["th_its"][it, :, 0])
        line_th2.set_data(ilc_its["t_its"][it], ilc_its["th_its"][it, :, 1])
        line_error_th1.set_data(
            ilc_its["t_its"][it], traj_ts["th_ts"][:, 0] - ilc_its["th_its"][it, :, 0]
        )
        line_error_th2.set_data(
            ilc_its["t_its"][it], traj_ts["th_ts"][:, 1] - ilc_its["th_its"][it, :, 1]
        )

        line_th_d1.set_data(ilc_its["t_its"][it], ilc_its["th_d_its"][it, :, 0])
        line_th_d2.set_data(ilc_its["t_its"][it], ilc_its["th_d_its"][it, :, 1])
        line_error_th_d1.set_data(
            ilc_its["t_its"][it],
            traj_ts["th_d_ts"][:, 0] - ilc_its["th_d_its"][it, :, 0],
        )
        line_error_th_d2.set_data(
            ilc_its["t_its"][it],
            traj_ts["th_d_ts"][:, 1] - ilc_its["th_d_its"][it, :, 1],
        )

        return (
            line_th1,
            line_th2,
            line_error_th1,
            line_error_th2,
            line_th_d1,
            line_th_d2,
            line_error_th_d1,
            line_error_th_d2,
        )

    ani = animation.FuncAnimation(
        fig=fig,
        func=_animate,
        init_func=_init,
        frames=np.linspace(
            start=0,
            stop=num_its - 1,
            num=min(max_num_animated_its, num_its),
            endpoint=True,
            dtype=int,
        ).tolist(),
        interval=200,
        blit=False,
    )

    if filepath is not None:
        ani.save(filepath)
        fig.savefig(str(Path(filepath).with_suffix(".pdf")))

    if show:
        plt.show()
    else:
        plt.close()

    return ani


def animate_operational_space_trajectory_following_plot(
    traj_ts: Dict[str, jnp.ndarray],
    ilc_its: Dict[str, jnp.ndarray],
    max_num_animated_its: int = 100,
    show: bool = True,
    filepath: str = None,
) -> animation.FuncAnimation:
    """
    Animate the operational-space sequence through the ILC iterations.

    Args:
        traj_ts: A dictionary containing the desired trajectory states.
        ilc_its: Dictionary of ILC iterations.
        max_num_animated_its: Maximum number of iterations being animated. All other iterations in between are skipped.
        show: boolean flag to decide whether show the animation
        filepath: Path to save the plot to.
    """
    fig, axes = plt.subplots(
        3,
        2,
        num="Animation ILC Operational-space trajectory following",
        figsize=(10, 5),
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    num_its = ilc_its["t_its"].shape[0]
    it_text = fig.suptitle("")

    # plot x1
    (line_x1,) = axes[0, 0].plot(
        [],
        [],
        color=colors[0],
        linewidth=3,
        label=r"$x_1$",
    )
    axes[0, 0].plot(
        traj_ts["t_ts"],
        traj_ts["x_ts"][:, 0],
        color="black",
        linestyle=":",
        linewidth=2,
        label=r"$x_1^\mathrm{d}$",
    )

    # plot x2
    (line_x2,) = axes[1, 0].plot(
        [],
        [],
        color=colors[1],
        linewidth=3,
        label=r"$x_2$",
    )
    axes[1, 0].plot(
        traj_ts["t_ts"],
        traj_ts["x_ts"][:, 1],
        color="black",
        linestyle=":",
        linewidth=2,
        label=r"$x_2^\mathrm{d}$",
    )

    # plot error of x
    (line_error_x1,) = axes[2, 0].plot(
        [],
        [],
        color=colors[0],
        linewidth=2,
        label=r"$x_1$",
    )
    (line_error_x2,) = axes[2, 0].plot(
        [],
        [],
        color=colors[1],
        linewidth=2,
        label=r"$x_2$",
    )
    axes[2, 0].set_ylim(
        [-jnp.max(jnp.abs(traj_ts["x_ts"])), jnp.max(jnp.abs(traj_ts["x_ts"]))]
    )

    # set axis labels for first column
    axes[-1, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel(r"$x_1$ [m]")
    axes[1, 0].set_ylabel(r"$x_2$ [m]")
    axes[2, 0].set_ylabel(r"$x$ error [m]")

    # plot x_d1
    (line_x_d1,) = axes[0, 1].plot(
        [],
        [],
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

    # plot x_d2
    (line_x_d2,) = axes[1, 1].plot(
        [],
        [],
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

    # plot error of x_d
    (line_error_x_d1,) = axes[2, 1].plot(
        [],
        [],
        color=colors[0],
        linewidth=2,
        label=r"$\dot{x}_1$",
    )
    (line_error_x_d2,) = axes[2, 1].plot(
        [],
        [],
        color=colors[1],
        linewidth=2,
        label=r"$\dot{x}_2$",
    )
    axes[2, 1].set_ylim(
        [-jnp.max(jnp.abs(traj_ts["x_d_ts"])), jnp.max(jnp.abs(traj_ts["x_d_ts"]))]
    )

    # set axis labels for second column
    axes[-1, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel(r"$\dot{x}_1$ [m / s]")
    axes[1, 1].set_ylabel(r"$\dot{x}_2$ [m / s]")
    axes[2, 1].set_ylabel(r"$\dot{x}$ error [m / s]")

    for ax in axes.flatten():
        ax.set_xlim(traj_ts["t_ts"][0], traj_ts["t_ts"][-1])
        ax.grid()
        ax.legend(loc="upper left")

    plt.tight_layout()

    def _init():
        it_text.set_text("")
        return (it_text,)

    def _animate(it: int):
        it_text.set_text(f"Iteration {it + 1} / {num_its}")

        line_x1.set_data(ilc_its["t_its"][it], ilc_its["x_its"][it, :, 0])
        line_x2.set_data(ilc_its["t_its"][it], ilc_its["x_its"][it, :, 1])
        line_error_x1.set_data(
            ilc_its["t_its"][it], traj_ts["x_ts"][:, 0] - ilc_its["x_its"][it, :, 0]
        )
        line_error_x2.set_data(
            ilc_its["t_its"][it], traj_ts["x_ts"][:, 1] - ilc_its["x_its"][it, :, 1]
        )

        line_x_d1.set_data(ilc_its["t_its"][it], ilc_its["x_d_its"][it, :, 0])
        line_x_d2.set_data(ilc_its["t_its"][it], ilc_its["x_d_its"][it, :, 1])
        line_error_x_d1.set_data(
            ilc_its["t_its"][it], traj_ts["x_d_ts"][:, 0] - ilc_its["x_d_its"][it, :, 0]
        )
        line_error_x_d2.set_data(
            ilc_its["t_its"][it], traj_ts["x_d_ts"][:, 1] - ilc_its["x_d_its"][it, :, 1]
        )

        return (
            line_x1,
            line_x2,
            line_error_x1,
            line_error_x2,
            line_x_d1,
            line_x_d2,
            line_error_x_d1,
            line_error_x_d2,
        )

    ani = animation.FuncAnimation(
        fig=fig,
        func=_animate,
        init_func=_init,
        frames=np.linspace(
            start=0,
            stop=num_its - 1,
            num=min(max_num_animated_its, num_its),
            endpoint=True,
            dtype=int,
        ).tolist(),
        interval=200,
        blit=False,
    )

    if filepath is not None:
        ani.save(filepath)
        fig.savefig(str(Path(filepath).with_suffix(".pdf")))

    if show:
        plt.show()
    else:
        plt.close()

    return ani


def animate_actuation_plot(
    traj_ts: Dict[str, jnp.ndarray],
    ilc_its: Dict[str, jnp.ndarray],
    max_num_animated_its: int = 100,
    show: bool = True,
    filepath: str = None,
) -> animation.FuncAnimation:
    """
    Animate the actuation sequence through the ILC iterations.

    Args:
        traj_ts: A dictionary containing the desired trajectory states.
        ilc_its: Dictionary of ILC iterations.
        max_num_animated_its: Maximum number of iterations being animated. All other iterations in between are skipped.
        show: boolean flag to decide whether show the animation
        filepath: Path to save the plot to.
    """
    fig, axes = plt.subplots(2, 1, num="Animation ILC actuation", figsize=(6, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    num_its = ilc_its["t_its"].shape[0]

    (line_u1,) = axes[0].plot(
        [],
        [],
        color=colors[0],
        linewidth=2,
        label=r"ILC action $u_1$",
    )
    (line_tau_fb1,) = axes[0].plot(
        [],
        [],
        color=colors[1],
        linewidth=2,
        label=r"Feedback action $\tau_{\mathrm{fb},1}$",
    )
    (line_tau1,) = axes[0].plot(
        [],
        [],
        color=colors[2],
        linewidth=3,
        label=r"Applied torque $\tau_1$",
    )
    (line_tau_ref1,) = axes[0].plot(
        traj_ts["t_ts"][:-1],
        ilc_its["tau_eq_ts"][:-1, 0],
        linestyle=":",
        linewidth=2,
        color="black",
        label=r"Equilibrium torque $\bar{\tau}_1$",
    )
    ax0_ylim_bottom = jnp.min(
        jnp.stack(
            [
                ilc_its["u_its"][:, :-1, 0].min(),
                ilc_its["tau_eq_ts"][:-1, 0].min(),
                ilc_its["tau_fb_its"][:, :-1, 0].min(),
                ilc_its["tau_its"][:, :-1, 0].min(),
            ]
        )
    )
    ax0_ylim_top = jnp.max(
        jnp.stack(
            [
                ilc_its["u_its"][:, :-1, 0].max(),
                ilc_its["tau_eq_ts"][:-1, 0].max(),
                ilc_its["tau_fb_its"][:, :-1, 0].max(),
                ilc_its["tau_its"][:, :-1, 0].max(),
            ]
        )
    )
    axes[0].set_ylim(ax0_ylim_bottom, ax0_ylim_top)

    (line_u2,) = axes[1].plot(
        [],
        [],
        color=colors[0],
        linewidth=2,
        label=r"ILC action $u_2$",
    )
    (line_tau_fb2,) = axes[1].plot(
        [],
        [],
        color=colors[1],
        linewidth=2,
        label=r"Feedback action $\tau_{\mathrm{fb},2}$",
    )
    (line_tau2,) = axes[1].plot(
        [],
        [],
        color=colors[2],
        linewidth=3,
        label=r"Applied torque $\tau_2$",
    )
    (line_tau_ref2,) = axes[1].plot(
        traj_ts["t_ts"][:-1],
        ilc_its["tau_eq_ts"][:-1, 1],
        linestyle=":",
        linewidth=2,
        color="black",
        label=r"Equilibrium torque $\bar{\tau}_2$",
    )
    ax1_ylim_bottom = jnp.min(
        jnp.stack(
            [
                ilc_its["u_its"][:, :-1, 1].min(),
                ilc_its["tau_eq_ts"][:-1, 1].min(),
                ilc_its["tau_fb_its"][:, :-1, 1].min(),
                ilc_its["tau_its"][:, :-1, 1].min(),
            ]
        )
    )
    ax1_ylim_top = jnp.max(
        jnp.stack(
            [
                ilc_its["u_its"][:, :-1, 1].max(),
                ilc_its["tau_eq_ts"][:-1, 1].max(),
                ilc_its["tau_fb_its"][:, :-1, 1].max(),
                ilc_its["tau_its"][:, :-1, 1].max(),
            ]
        )
    )
    axes[1].set_ylim(ax1_ylim_bottom, ax1_ylim_top)

    it_text1 = axes[0].text(0.025, 0.05, "", transform=axes[0].transAxes)
    it_text2 = axes[1].text(0.025, 0.05, "", transform=axes[1].transAxes)

    axes[1].set_xlabel(r"Time [s]")
    axes[0].set_ylabel(r"Torque $\tau_1$ [Nm]")
    axes[1].set_ylabel(r"Torque $\tau_2$ [Nm]")

    for ax in axes:
        ax.set_xlim(traj_ts["t_ts"][0], traj_ts["t_ts"][-1])
        ax.legend(loc="upper left")
        ax.grid()

    plt.tight_layout()

    def _init():
        it_text1.set_text("")
        it_text2.set_text("")
        return (it_text1, it_text2)

    def _animate(it: int):
        it_text1.set_text(f"Iteration {it + 1} / {num_its}")
        it_text2.set_text(f"Iteration {it + 1} / {num_its}")

        line_u1.set_data(ilc_its["t_its"][it, :-1], ilc_its["u_its"][it, :-1, 0])
        line_tau_fb1.set_data(
            ilc_its["t_its"][it, :-1], ilc_its["tau_fb_its"][it, :-1, 0]
        )
        line_tau1.set_data(ilc_its["t_its"][it, :-1], ilc_its["tau_its"][it, :-1, 0])
        line_tau_ref1.set_data(
            traj_ts["t_ts"][:-1],
            ilc_its["tau_eq_ts"][:-1, 0],
        )
        line_u2.set_data(ilc_its["t_its"][it, :-1], ilc_its["u_its"][it, :-1, 1])
        line_tau_fb2.set_data(
            ilc_its["t_its"][it, :-1], ilc_its["tau_fb_its"][it, :-1, 1]
        )
        line_tau2.set_data(ilc_its["t_its"][it, :-1], ilc_its["tau_its"][it, :-1, 1])
        line_tau_ref2.set_data(
            traj_ts["t_ts"][:-1],
            ilc_its["tau_eq_ts"][:-1, 1],
        )

        return (
            it_text1,
            it_text2,
            line_u1,
            line_tau_fb1,
            line_tau1,
            line_tau_ref1,
            line_u2,
            line_tau_fb2,
            line_tau2,
            line_tau_ref2,
        )

    ani = animation.FuncAnimation(
        fig=fig,
        func=_animate,
        init_func=_init,
        frames=np.linspace(
            start=0,
            stop=num_its - 1,
            num=min(max_num_animated_its, num_its),
            endpoint=True,
            dtype=int,
        ).tolist(),
        interval=200,
        blit=True,
    )

    if filepath is not None:
        ani.save(filepath)
        fig.savefig(str(Path(filepath).with_suffix(".pdf")))

    if show:
        plt.show()
    else:
        plt.close()

    return ani
