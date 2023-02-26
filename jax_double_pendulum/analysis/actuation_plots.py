from jax import numpy as jnp
import matplotlib.pyplot as plt
from typing import Dict, Tuple


def plot_actuation(
    sim_ts: Dict[str, jnp.ndarray],
    filepath: str = None,
):
    fig, axes = plt.subplots(2, 1, num="Actuation", figsize=(6, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    (line_tau_ff1,) = axes[0].plot(
        sim_ts["t_ts"][:-1],
        sim_ts["tau_ff_ts"][:-1, 0],
        color=colors[0],
        linewidth=2,
        label=r"Feedforward action $\tau_{\mathrm{ff},1}$",
    )
    (line_tau_fb1,) = axes[0].plot(
        sim_ts["t_ts"][:-1],
        sim_ts["tau_fb_ts"][:-1, 0],
        color=colors[1],
        linewidth=2,
        label=r"Feedback action $\tau_{\mathrm{fb},1}$",
    )
    (line_tau1,) = axes[0].plot(
        sim_ts["t_ts"][:-1],
        sim_ts["tau_ts"][:-1, 0],
        color=colors[2],
        linewidth=3,
        label=r"Total torque $\tau_1$",
    )

    (line_tau_ff2,) = axes[1].plot(
        sim_ts["t_ts"][:-1],
        sim_ts["tau_ff_ts"][:-1, 1],
        color=colors[0],
        linewidth=2,
        label=r"Feedforward action $\tau_{\mathrm{ff},2}$",
    )
    (line_tau_fb2,) = axes[1].plot(
        sim_ts["t_ts"][:-1],
        sim_ts["tau_fb_ts"][:-1, 1],
        color=colors[1],
        linewidth=2,
        label=r"Feedback action $\tau_{\mathrm{fb},2}$",
    )
    (line_tau2,) = axes[1].plot(
        sim_ts["t_ts"][:-1],
        sim_ts["tau_ts"][:-1, 1],
        color=colors[2],
        linewidth=3,
        label=r"Total torque $\tau_2$",
    )

    axes[1].set_xlabel(r"Time [s]")
    axes[0].set_ylabel(r"Torque $\tau_1$ [Nm]")
    axes[1].set_ylabel(r"Torque $\tau_2$ [Nm]")

    for ax in axes:
        ax.set_xlim(sim_ts["t_ts"][0], sim_ts["t_ts"][-1])
        ax.legend(loc="upper left")
        ax.grid()

    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
    plt.show()
