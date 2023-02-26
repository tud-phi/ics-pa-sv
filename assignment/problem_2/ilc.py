from functools import partial
from jax import jit, lax, vmap
import jax.numpy as jnp
from typing import Callable, Dict, Tuple

from jax_double_pendulum.dynamics import discrete_forward_dynamics
from jax_double_pendulum.robot_simulation import _simulate_robot

# import feedback controller from controllers.ipynb
from ipynb.fs.full.controllers import ctrl_fb_pd


def init_ilc_its(
    num_iterations: int, traj_ts: Dict[str, jnp.ndarray]
) -> Dict[str, jnp.ndarray]:
    """
    Initialize the dictionary to track the system states and inputs across ILC iterations and time
    Args:
        num_iterations: number of iterations of the Q-ILC algorithm
        traj_ts: dictionary of time series of trajectories
    Returns:
        ilc_its: dictionary to track states across iterations and time steps
    """
    num_its = num_iterations
    ilc_its = {
        "it_its": jnp.arange(num_its),
        "t_its": jnp.zeros((num_its,) + traj_ts["t_ts"].shape),
        "th_its": jnp.zeros((num_its,) + traj_ts["th_ts"].shape),
        "th_d_its": jnp.zeros((num_its,) + traj_ts["th_d_ts"].shape),
        "th_dd_its": jnp.zeros((num_its,) + traj_ts["th_dd_ts"].shape),
        "x_its": jnp.zeros((num_its,) + traj_ts["x_ts"].shape),
        "x_d_its": jnp.zeros((num_its,) + traj_ts["x_d_ts"].shape),
        "x_dd_its": jnp.zeros((num_its,) + traj_ts["x_dd_ts"].shape),
        "u_its": jnp.zeros((num_its,) + traj_ts["th_ts"].shape),
        "tau_fb_its": jnp.zeros((num_its,) + traj_ts["th_ts"].shape),
        "tau_ilc_its": jnp.zeros((num_its,) + traj_ts["th_ts"].shape),
        "tau_its": jnp.zeros((num_its,) + traj_ts["th_ts"].shape),
    }
    return ilc_its


@jit
def apply_ilc_control_action_to_system(
    rp: dict,
    traj_ts: Dict[str, jnp.ndarray],
    th_0: jnp.ndarray,
    th_d_0: jnp.ndarray,
    it: int,
    ilc_its: Dict[str, jnp.ndarray],
    tau_ilc_ts: jnp.ndarray,
    kp_fb: jnp.ndarray = jnp.zeros((2, 2)),
    kd_fb: jnp.ndarray = jnp.zeros((2, 2)),
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """
    Apply the control action to the system and update the state and input time series
    Args:
        rp: Dictionary of robot parameters
        traj_ts: dictionary of time series of trajectories
        th_0: initial link angles of shape (2,)
        th_d_0: initial link angular velocities of shape (2,)
        it: current iteration of the ILC algorithm
        ilc_its: dictionary to track states across iterations and time steps
        tau_ilc_ts: time series of link torques computed by the ILC algorithm applied to the system
        kp_fb: proportional gains of the parallel feedback controller of shape (2, 2)
        kd_fb: derivative gains of the parallel feedback controller of shape (2, 2)

    Returns:
        sim_ts: dictionary of time series of current system evolution
        ilc_its: updated dictionary to track states across iterations and time steps

    """
    sim_ts = _simulate_robot(
        rp=rp,
        t_ts=traj_ts["t_ts"],
        discrete_forward_dynamics_fn=partial(discrete_forward_dynamics, rp),
        th_0=th_0,
        th_d_0=th_d_0,
        tau_ext_ts=tau_ilc_ts,
        th_des_ts=traj_ts["th_ts"],
        th_d_des_ts=traj_ts["th_d_ts"],
        th_dd_des_ts=traj_ts["th_dd_ts"],
        ctrl_ff=lambda th, th_d, th_des, th_d_des, th_dd_des: jnp.zeros((2,)),
        ctrl_fb=partial(ctrl_fb_pd, kp=kp_fb, kd=kd_fb),
    )

    # write simulation results to the dictionary
    ilc_its["t_its"] = ilc_its["t_its"].at[it].set(sim_ts["t_ts"])
    ilc_its["th_its"] = ilc_its["th_its"].at[it].set(sim_ts["th_ts"])
    ilc_its["th_d_its"] = ilc_its["th_d_its"].at[it].set(sim_ts["th_d_ts"])
    ilc_its["th_dd_its"] = ilc_its["th_dd_its"].at[it].set(sim_ts["th_dd_ts"])
    ilc_its["x_its"] = ilc_its["x_its"].at[it].set(sim_ts["x_ts"])
    ilc_its["x_d_its"] = ilc_its["x_d_its"].at[it].set(sim_ts["x_d_ts"])
    ilc_its["x_dd_its"] = ilc_its["x_dd_its"].at[it].set(sim_ts["x_dd_ts"])
    ilc_its["tau_ilc_its"] = ilc_its["tau_ilc_its"].at[it].set(tau_ilc_ts)
    ilc_its["tau_fb_its"] = ilc_its["tau_fb_its"].at[it].set(sim_ts["tau_fb_ts"])
    ilc_its["tau_its"] = ilc_its["tau_its"].at[it].set(sim_ts["tau_ts"])

    return sim_ts, ilc_its
