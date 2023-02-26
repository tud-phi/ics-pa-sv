from jax import jit
from jax import numpy as jnp
from typing import Dict, Tuple

from ..utils import normalize_link_angles


@jit
def compute_configuration_space_rmse(
    traj_ts: Dict[str, jnp.ndarray], sim_ts: Dict[str, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes the root mean squared error between the desired and actual trajectory in configuration space

    Args:
        traj_ts: A dictionary containing the desired trajectory states.
        sim_ts: A dictionary containing the actual trajectory states.

    Returns:
        rmse_th: The root mean squared error of the link angles. Shape: (2,)
        rmse_th_d: The root mean squared error of the link angular velocities. Shape: (2,)
        rmse_th_dd: The root mean squared error of the link angular accelerations. Shape: (2,)
    """
    # error mapped to [-pi, pi]
    th_err = normalize_link_angles(traj_ts["th_ts"] - sim_ts["th_ts"])

    rmse_th = jnp.sqrt(jnp.mean(th_err**2, axis=0))
    rmse_th_d = jnp.sqrt(
        jnp.mean((traj_ts["th_d_ts"] - sim_ts["th_d_ts"]) ** 2, axis=0)
    )
    rmse_th_dd = jnp.sqrt(
        jnp.mean((traj_ts["th_dd_ts"] - sim_ts["th_dd_ts"]) ** 2, axis=0)
    )

    return rmse_th, rmse_th_d, rmse_th_dd


@jit
def compute_operational_space_rmse(
    traj_ts: Dict[str, jnp.ndarray], sim_ts: Dict[str, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes the root mean squared error between the desired and actual trajectory in task space

    Args:
        traj_ts: A dictionary containing the desired trajectory states.
        sim_ts: A dictionary containing the actual trajectory states.

    Returns:
        rmse_x: The root mean squared error of the end-effector position. Shape: (2,)
        rmse_x_d: The root mean squared error of the end-effector velocity. Shape: (2,)
        rmse_x_dd: The root mean squared error of the end-effector acceleration. Shape: (2,)

    """
    rmse_x = jnp.sqrt(jnp.mean((traj_ts["x_ts"] - sim_ts["x_ts"]) ** 2, axis=0))
    rmse_x_d = jnp.sqrt(jnp.mean((traj_ts["x_d_ts"] - sim_ts["x_d_ts"]) ** 2, axis=0))
    rmse_x_dd = jnp.sqrt(
        jnp.mean((traj_ts["x_dd_ts"] - sim_ts["x_dd_ts"]) ** 2, axis=0)
    )

    return rmse_x, rmse_x_d, rmse_x_dd
