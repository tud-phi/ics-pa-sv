from functools import partial
from jax import debug, jit, lax
from jax import numpy as jnp
from typing import Callable, Dict, Tuple

from .dynamics import discrete_forward_dynamics
from .kinematics.forward_kinematics import extended_forward_kinematics


def simulate_robot(
    rp: dict,
    t_ts: jnp.ndarray,
    discrete_forward_dynamics_fn: Callable = None,
    th_0: jnp.ndarray = jnp.array([0.0, 0.0]),
    th_d_0: jnp.ndarray = jnp.array([0.0, 0.0]),
    tau_ext_ts: jnp.ndarray = None,
    th_des_ts: jnp.ndarray = None,
    th_d_des_ts: jnp.ndarray = None,
    th_dd_des_ts: jnp.ndarray = None,
    ctrl_ff: Callable = lambda th, th_d, th_des, th_d_des, th_dd_des: jnp.zeros((2,)),
    ctrl_fb: Callable = lambda th, th_d, th_des, th_d_des: jnp.zeros((2,)),
) -> Dict[str, jnp.ndarray]:
    """
    Simulates the double pendulum robot.

    Args:
        rp: dictionary of robot parameters
        t_ts: time steps of the trajectory [s] of shape (num_time_steps, )
        discrete_forward_dynamics_fn: function that computes the discrete forward dynamics.
            Given the time step dt, the current link state (th, th_d), and the link torque tau,
            it needs to return the next link state (th, th_d) and the link angular acceleration th_dd.
            It must have the signature: discrete_forward_dynamics_fn(dt, th, th_d, tau) -> (th_next, th_d_next, th_dd)
        th_0: initial link angles of shape (2, )
        th_d_0: initial link angular velocities of shape (2, )
        tau_ext_ts: link torques of shape (num_time_steps, 2)
            which are applied to the system in addition to the evaluated feedforward and feedback torques
        th_des_ts: desired link angles of shape (num_time_steps, 2)
        th_d_des_ts: desired link angular velocities of shape (num_time_steps, 2)
        th_dd_des_ts: desired link angular accelerations of shape (num_time_steps, 2)
        ctrl_ff: callable computing the feed-forward control torques. Needs to return jax.array of shape (2, )
            It must have the signature: ctrl_ff(th, th_d, th_des, th_d_des, th_dd_des) -> tau_ff
        ctrl_fb: callable computing the feed-back control torques. Needs to return jax.array of shape (2, )
            It must have the signature: ctrl_fb(th, th_d, th_des, th_d_des) -> tau_fb

    Returns:
        sim_ts: dictionary of states and other time series data of simulation
    """
    if discrete_forward_dynamics_fn is None:
        discrete_forward_dynamics_fn = partial(discrete_forward_dynamics, rp)

    if tau_ext_ts is None:
        tau_ext_ts = jnp.zeros((t_ts.shape[0], 2))

    if th_des_ts is None:
        th_des_ts = jnp.zeros((t_ts.shape[0], 2))

    if th_d_des_ts is None:
        th_d_des_ts = jnp.zeros((t_ts.shape[0], 2))

    if th_dd_des_ts is None:
        th_dd_des_ts = jnp.zeros((t_ts.shape[0], 2))

    return _simulate_robot(
        rp=rp,
        t_ts=t_ts,
        discrete_forward_dynamics_fn=discrete_forward_dynamics_fn,
        th_0=th_0,
        th_d_0=th_d_0,
        tau_ext_ts=tau_ext_ts,
        th_des_ts=th_des_ts,
        th_d_des_ts=th_d_des_ts,
        th_dd_des_ts=th_dd_des_ts,
        ctrl_ff=ctrl_ff,
        ctrl_fb=ctrl_fb,
    )


@partial(
    jit,
    static_argnums=(2, 9, 10),
    static_argnames=("discrete_forward_dynamics_fn", "ctrl_ff", "ctrl_fb"),
)
def _simulate_robot(
    rp: dict,
    t_ts: jnp.ndarray,
    discrete_forward_dynamics_fn: Callable,
    th_0: jnp.ndarray,
    th_d_0: jnp.ndarray,
    tau_ext_ts: jnp.ndarray,
    th_des_ts: jnp.ndarray,
    th_d_des_ts: jnp.ndarray,
    th_dd_des_ts: jnp.ndarray,
    ctrl_ff: Callable,
    ctrl_fb: Callable,
) -> Dict[str, jnp.ndarray]:
    """
    Simulates the double pendulum robot.

    Args:
        rp: dictionary of robot parameters
        t_ts: time steps of the trajectory [s] of shape (num_time_steps, )
        discrete_forward_dynamics_fn: function that computes the discrete forward dynamics.
            Given the time step dt, the current link state (th, th_d), and the link torque tau,
            it needs to return the next link state (th, th_d) and the link angular acceleration th_dd.
            It must have the signature: discrete_forward_dynamics_fn(dt, th, th_d, tau) -> (th_next, th_d_next, th_dd)
        th_0: initial link angles of shape (2, )
        th_d_0: initial link angular velocities of shape (2, )
        tau_ext_ts: link torques of shape (num_time_steps, 2)
            which are applied to the system in addition to the evaluated feedforward and feedback torques
        th_des_ts: desired link angles of shape (num_time_steps, 2)
        th_d_des_ts: desired link angular velocities of shape (num_time_steps, 2)
        th_dd_des_ts: desired link angular accelerations of shape (num_time_steps, 2)
        ctrl_ff: callable computing the feed-forward control torques. Needs to return jax.array of shape (2, )
            It must have the signature: ctrl_ff(th, th_d, th_des, th_d_des, th_dd_des) -> tau_ff
        ctrl_fb: callable computing the feed-back control torques. Needs to return jax.array of shape (2, )
            It must have the signature: ctrl_fb(th, th_d, th_des, th_d_des) -> tau_fb

    Returns:
        sim_ts: dictionary of states and other time series data of simulation
    """

    num_time_steps = t_ts.shape[0]

    @jit
    def _for_loop_body_fun(_time_idx, _sim_ts: Dict):
        _dt = t_ts[_time_idx] - t_ts[_time_idx - 1]
        (
            _th,
            _th_d,
            _th_dd,
            _x,
            _x_d,
            _x_dd,
            _x_eb,
            _tau,
            _tau_ff,
            _tau_fb,
        ) = simulation_iteration(
            rp,
            _dt,
            discrete_forward_dynamics_fn,
            _sim_ts["th_ts"][_time_idx - 1],
            _sim_ts["th_d_ts"][_time_idx - 1],
            tau_ext_ts[_time_idx - 1],
            th_des_ts[_time_idx],
            th_d_des_ts[_time_idx],
            th_dd_des_ts[_time_idx],
            ctrl_ff,
            ctrl_fb,
        )
        _sim_ts["th_ts"] = _sim_ts["th_ts"].at[_time_idx].set(_th)
        _sim_ts["th_d_ts"] = _sim_ts["th_d_ts"].at[_time_idx].set(_th_d)
        _sim_ts["th_dd_ts"] = _sim_ts["th_dd_ts"].at[_time_idx].set(_th_dd)
        _sim_ts["x_ts"] = _sim_ts["x_ts"].at[_time_idx].set(_x)
        _sim_ts["x_d_ts"] = _sim_ts["x_d_ts"].at[_time_idx].set(_x_d)
        _sim_ts["x_dd_ts"] = _sim_ts["x_dd_ts"].at[_time_idx].set(_x_dd)
        _sim_ts["x_eb_ts"] = _sim_ts["x_eb_ts"].at[_time_idx].set(_x_eb)
        _sim_ts["tau_ts"] = _sim_ts["tau_ts"].at[_time_idx - 1].set(_tau)
        _sim_ts["tau_ff_ts"] = _sim_ts["tau_ff_ts"].at[_time_idx - 1].set(_tau_ff)
        _sim_ts["tau_fb_ts"] = _sim_ts["tau_fb_ts"].at[_time_idx - 1].set(_tau_fb)

        return _sim_ts

    # initialize diagnostic dictionary of system states over the trajectory
    sim_ts = dict(
        t_ts=t_ts,
        th_ts=jnp.zeros((num_time_steps, 2)),
        th_d_ts=jnp.zeros((num_time_steps, 2)),
        th_dd_ts=jnp.zeros((num_time_steps, 2)),
        x_ts=jnp.zeros((num_time_steps, 2)),
        x_d_ts=jnp.zeros((num_time_steps, 2)),
        x_dd_ts=jnp.zeros((num_time_steps, 2)),
        x_eb_ts=jnp.zeros((num_time_steps, 2)),
        tau_ts=jnp.zeros((num_time_steps, 2)),
        tau_ff_ts=jnp.zeros((num_time_steps, 2)),
        tau_fb_ts=jnp.zeros((num_time_steps, 2)),
    )

    # evaluate quantities at initial state
    sim_ts["th_ts"] = sim_ts["th_ts"].at[0].set(th_0)
    sim_ts["th_d_ts"] = sim_ts["th_d_ts"].at[0].set(th_d_0)
    x_0, x_d_0, x_dd_0, x_eb_0 = extended_forward_kinematics(
        rp, th_0, th_d_0, th_dd=jnp.zeros((2,))
    )
    sim_ts["x_ts"] = sim_ts["x_ts"].at[0].set(x_0)
    sim_ts["x_d_ts"] = sim_ts["x_d_ts"].at[0].set(x_d_0)
    sim_ts["x_dd_ts"] = sim_ts["x_dd_ts"].at[0].set(x_dd_0)
    sim_ts["x_eb_ts"] = sim_ts["x_eb_ts"].at[0].set(x_eb_0)

    sim_ts = lax.fori_loop(
        lower=1, upper=num_time_steps, body_fun=_for_loop_body_fun, init_val=sim_ts
    )

    return sim_ts


@partial(
    jit,
    static_argnums=(2, 9, 10),
    static_argnames=("discrete_forward_dynamics_fn", "ctrl_ff", "ctrl_fb"),
)
def simulation_iteration(
    rp: dict,
    dt: jnp.ndarray,
    discrete_forward_dynamics_fn: Callable,
    th: jnp.ndarray,
    th_d: jnp.ndarray,
    tau_ext: jnp.ndarray,
    th_des: jnp.ndarray,
    th_d_des: jnp.ndarray,
    th_dd_des: jnp.ndarray,
    ctrl_ff: Callable,
    ctrl_fb: Callable,
) -> Tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """
    Simulates the double pendulum robot for one time step
    Args:
        rp: dictionary of robot parameters
        dt: time step between the current and the next state [s]
        discrete_forward_dynamics_fn: function that computes the discrete forward dynamics.
            Given the time step dt, the current link state (th, th_d), and the link torque tau,
            it needs to return the next link state (th, th_d) and the link angular acceleration th_dd.
            It must have the signature: discrete_forward_dynamics_fn(dt, th, th_d, tau) -> (th_next, th_d_next, th_dd)
        th: current link angles of shape (2, )
        th_d: current link angular velocities of shape (2, )
        tau_ext: link torques of shape (2, )
            which are applied to the system in addition to the evaluated feedforward and feedback torques
        th_des: desired link angles of shape (2, )
        th_d_des: desired link angular velocities of shape (2, )
        th_dd_des: desired link angular accelerations of shape (2, )
        ctrl_ff: callable computing the feed-forward control torques. Needs to return jax.array of shape (2, )
            It must have the signature: ctrl_ff(th, th_d, th_des, th_d_des, th_dd_des) -> tau_ff
        ctrl_fb: callable computing the feed-back control torques. Needs to return jax.array of shape (2, )
            It must have the signature: ctrl_fb(th, th_d, th_des, th_d_des) -> tau_fb

    Returns:
        th_next: next link angles of shape (2, )
        th_d_next: next link angular velocities of shape (2, )
        th_dd: link angular accelerations of shape (2, )
        x_next: next end-effector position of shape (2, )
        x_d_next: next end-effector velocity of shape (2, )
        x_dd: end-effector acceleration of shape (2, )
        x_eb: next position of elbow joint of shape (2, )
        tau: total external torque applied to the links of shape (2, )
        tau_ff: torque computed by feed-forward controller of shape (2, )
        tau_fb: torque computed by feedback controller of shape (2, )
    """
    # evaluate feedforward and feedback controllers
    tau_ff = ctrl_ff(th, th_d, th_des, th_d_des, th_dd_des)
    tau_fb = ctrl_fb(th, th_d, th_des, th_d_des)
    tau = tau_ext + tau_ff + tau_fb

    # evaluate the dynamics
    th_next, th_d_next, th_dd = discrete_forward_dynamics_fn(dt, th, th_d, tau)

    # evaluate forward kinematics
    x_next, x_d_next, x_dd, x_eb_next = extended_forward_kinematics(
        rp, th_next, th_d_next, th_dd
    )

    return (
        th_next,
        th_d_next,
        th_dd,
        x_next,
        x_d_next,
        x_dd,
        x_eb_next,
        tau,
        tau_ff,
        tau_fb,
    )
