from jax import numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict


def animate_robot(
    rp: Dict[str, float],
    sim_ts: Dict[str, jnp.ndarray],
    sim_hat_ts: Dict[str, jnp.ndarray] = None,
    traj_ts: Dict[str, jnp.ndarray] = None,
    show_reference: bool = True,
    show_ee_traj: bool = True,
    show_robot: bool = True,
    step_skip: int = 1,
    alpha_hat: float = 0.5,
    show: bool = True,
    filepath: str = None,
) -> animation.FuncAnimation:
    """
    Animates the robot
    :param rp: Robot parameters dictionary
    :param sim_ts: Dictionary of simulation data
    :param sim_hat_ts: Dictionary of simulation data of some other simulation, rendered with transparency
    :param traj_ts: Dictionary of trajectory data
    :param show_reference: boolean flag to show the reference trajectory
    :param show_ee_traj: boolean flag to show the end-effector trajectory
    :param show_robot: boolean flag to render the robot links and joints
    :param step_skip: number of simulation steps to skip between animation frames.
        This is useful for speeding up the computation of the animation.
    :param alpha_hat: transparency of the other simulation
    :param show: boolean flag to show the animation
    :param filepath: path of the file where the animation is saved
    """
    sim_dt = jnp.round(jnp.mean(sim_ts["t_ts"][1:] - sim_ts["t_ts"][:-1]), 3)
    assert type(step_skip) == int, "step_skip must be an integer bigger than 1"
    assert step_skip > 1, "step_skip must be an integer bigger than 1"
    num_frames = sim_ts["t_ts"].shape[0]
    interval = step_skip * sim_dt * 1000

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig = plt.figure(num="Robot animation")
    l_sum = rp["l1"] + rp["l2"]
    ax = fig.add_subplot(
        111, autoscale_on=False, xlim=(-l_sum, l_sum), ylim=(-l_sum, l_sum)
    )
    ax.set_aspect("equal")
    plt.grid(True)
    plt.xlabel("$x$ [m]")
    plt.ylabel("$y$ [m]")

    time_template = "Time: %.3fs [s]"
    time_text = ax.text(0.65, 0.9, "", transform=ax.transAxes)

    line_ee_traj, line_ee_pos = None, None
    if show_ee_traj and sim_ts is not None:
        (line_ee_traj,) = ax.plot(
            [], [], linewidth=3, color=colors[0], label="End-effector trajectory"
        )
        if show_robot is False:
            (line_ee_pos,) = ax.plot(
                [],
                [],
                marker=".",
                markersize=10,
                color=colors[0],
                # label="End-effector position",
            )

    line_ee_traj_hat, line_ee_pos_hat = None, None
    if show_ee_traj and sim_hat_ts is not None:
        (line_ee_traj_hat,) = ax.plot(
            [],
            [],
            linewidth=3,
            color=colors[0],
            alpha=alpha_hat,
            label="Est. end-effector traj.",
        )
        if show_robot is False:
            (line_ee_pos_hat,) = ax.plot(
                [],
                [],
                marker=".",
                markersize=10,
                color=colors[0],
                alpha=alpha_hat,
            )

    line_ref_traj, line_ref_pos = None, None
    if show_reference and traj_ts is not None:
        (line_ref_traj,) = ax.plot(
            [],
            [],
            linestyle=":",
            linewidth=2,
            color="black",
            label="Reference trajectory",
        )
        (line_ref_pos,) = ax.plot(
            [],
            [],
            marker=".",
            markersize=10,
            color="black",
            # label="Reference position",
        )

    line_robot = None
    if show_robot and sim_ts is not None:
        (line_robot,) = ax.plot(
            [],
            [],
            linewidth=3.5,
            linestyle="-",
            marker="o",
            markersize=8,
            color=colors[1],
            label="Robot state",
        )

    line_robot_hat = None
    if show_robot and sim_hat_ts is not None:
        (line_robot_hat,) = ax.plot(
            [],
            [],
            linewidth=3.5,
            linestyle="-",
            marker="o",
            markersize=8,
            color=colors[1],
            alpha=alpha_hat,
            label="Estimated robot state",
        )

    def _init():
        time_text.set_text("")
        return (time_text,)

    def _animate(time_idx: int):
        time_text.set_text(time_template % (time_idx * sim_dt))
        return_list = [time_text]

        if line_ee_traj is not None:
            line_ee_traj.set_data(
                sim_ts["x_ts"][:time_idx, 0], sim_ts["x_ts"][:time_idx, 1]
            )
            return_list.append(line_ee_traj)
        if line_ee_pos is not None:
            line_ee_pos.set_data(
                sim_ts["x_ts"][time_idx, 0], sim_ts["x_ts"][time_idx, 1]
            )
            return_list.append(line_ee_pos)
        if line_ee_traj_hat is not None:
            line_ee_traj_hat.set_data(
                sim_hat_ts["x_ts"][:time_idx, 0], sim_hat_ts["x_ts"][:time_idx, 1]
            )
            return_list.append(line_ee_traj_hat)
        if line_ee_pos_hat is not None:
            line_ee_pos_hat.set_data(
                sim_hat_ts["x_ts"][time_idx, 0], sim_hat_ts["x_ts"][time_idx, 1]
            )
            return_list.append(line_ee_pos_hat)

        if line_ref_traj is not None:
            line_ref_traj.set_data(
                traj_ts["x_ts"][:time_idx, 0], traj_ts["x_ts"][:time_idx, 1]
            )
            return_list.append(line_ref_traj)
        if line_ref_pos is not None:
            line_ref_pos.set_data(
                traj_ts["x_ts"][time_idx, 0], traj_ts["x_ts"][time_idx, 1]
            )
            return_list.append(line_ref_pos)

        if line_robot is not None:
            robot_pos = jnp.stack(
                [
                    jnp.array([0, 0]),
                    sim_ts["x_eb_ts"][time_idx, :],
                    sim_ts["x_ts"][time_idx, :],
                ]
            )
            line_robot.set_data(robot_pos[:, 0], robot_pos[:, 1])
            return_list.append(line_robot)

        if line_robot_hat is not None:
            robot_hat_pos = jnp.stack(
                [
                    jnp.array([0, 0]),
                    sim_hat_ts["x_eb_ts"][time_idx, :],
                    sim_hat_ts["x_ts"][time_idx, :],
                ]
            )
            line_robot_hat.set_data(robot_hat_pos[:, 0], robot_hat_pos[:, 1])
            return_list.append(line_robot_hat)

        if time_idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles[:], labels=labels[:], loc="upper left")

        return tuple(return_list)

    ani = animation.FuncAnimation(
        fig=fig,
        func=_animate,
        init_func=_init,
        frames=range(0, num_frames, step_skip),
        interval=interval,
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
