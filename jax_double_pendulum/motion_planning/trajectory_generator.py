from jax import jit, lax
from jax import numpy as jnp
from typing import Callable, Dict, Tuple

from ..kinematics.inverse_kinematics import inverse_kinematics


@jit
def generate_ellipse_trajectory(
    rp: Dict,
    t_ts: jnp.ndarray,
    omega: float,
    rx: float,
    ry: float,
    ell_angle: float,
    x0: float,
    y0: float,
) -> Dict[str, jnp.ndarray]:
    """
    Generates a trajectory for the end-effector of the double pendulum to follow an ellipse
    :param rp: dictionary of robot parameters
    :param t_ts: time steps of the trajectory [s] of shape (num_time_steps, )
    :param omega: Angular velocity of the trajectory [rad/s]
    :param rx: radius of the ellipse in x-direction [m]
    :param ry: radius of the ellipse in y-direction [m]
    :param ell_angle: angle of inclination of ellipse
    :param x0: center of ellipse in x-direction [m]
    :param y0: center of ellipse in y-direction [m]
    :return x_ts: End-effector position in task-space [m] of shape (num_time_steps, 2)
    :return x_d_ts: End-effector velocity in task-space [m/s] of shape (num_time_steps, 2)
    :return x_dd_ts: End-effector acceleration in task-space [m/s^2] of shape (num_time_steps, 2)
    """
    num_time_steps = t_ts.shape[0]

    # some short notations
    c_w = jnp.cos(omega * t_ts)
    s_w = jnp.sin(omega * t_ts)
    c_ell = jnp.cos(ell_angle)
    s_ell = jnp.sin(ell_angle)

    x_ts = jnp.array(
        [
            x0 + rx * c_w * c_ell - ry * s_w * s_ell,
            y0 + rx * c_w * s_ell + ry * s_w * c_ell,
        ]
    ).transpose()  # transpose to shape (num_time_steps, 2)

    x_d_ts = jnp.array(
        [
            omega * (-rx * s_w * c_ell - ry * c_w * s_ell),
            omega * (-rx * s_w * s_ell + ry * c_w * c_ell),
        ]
    ).transpose()  # transpose to shape (num_time_steps, 2)

    x_dd_ts = jnp.array(
        [
            omega**2 * (-rx * c_w * c_ell + ry * s_w * s_ell),
            omega**2 * (-rx * c_w * s_ell - ry * s_w * c_ell),
        ]
    ).transpose()  # transpose to shape (num_time_steps, 2)

    for_init_val = dict(
        rp=rp,
        th=jnp.array([0.0, 0.0]),
        th_d=jnp.array([0.0, 0.0]),
        th_dd=jnp.array([0.0, 0.0]),
        th_ts=jnp.zeros((num_time_steps, 2)),
        th_d_ts=jnp.zeros((num_time_steps, 2)),
        th_dd_ts=jnp.zeros((num_time_steps, 2)),
        x_ts=x_ts,
        x_d_ts=x_d_ts,
        x_dd_ts=x_dd_ts,
    )
    for_output = lax.fori_loop(
        lower=0,
        upper=num_time_steps,
        body_fun=_for_loop_body_fun,
        init_val=for_init_val,
    )

    traj_ts = dict(
        t_ts=t_ts,
        th_ts=for_output["th_ts"],
        th_d_ts=for_output["th_d_ts"],
        th_dd_ts=for_output["th_dd_ts"],
        x_ts=x_ts,
        x_d_ts=x_d_ts,
        x_dd_ts=x_dd_ts,
    )

    return traj_ts


@jit
def _for_loop_body_fun(i, val: Dict):
    val["th"], val["th_d"], val["th_dd"] = inverse_kinematics(
        val["rp"], val["th"], val["x_ts"][i], val["x_d_ts"][i], val["x_dd_ts"][i]
    )

    val["th_ts"] = val["th_ts"].at[i].set(val["th"])
    val["th_d_ts"] = val["th_d_ts"].at[i].set(val["th_d"])
    val["th_dd_ts"] = val["th_dd_ts"].at[i].set(val["th_dd"])

    return val
