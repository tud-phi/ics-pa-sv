from functools import partial
from jax import Array, debug, jit, lax
from jax import numpy as jnp
from typing import Callable, Dict, Optional, Tuple

from .dynamics import discrete_forward_dynamics
from .kinematics.forward_kinematics import extended_forward_kinematics


def simulate_robot(
    rp: dict,
    t_ts: Array,
    discrete_forward_dynamics_fn: Optional[Callable] = None,
    th_0: Array = jnp.array([0.0, 0.0]),
    th_d_0: Array = jnp.array([0.0, 0.0]),
    tau_ext_ts: Optional[Array] = None,
    th_des_ts: Optional[Array] = None,
    th_d_des_ts: Optional[Array] = None,
    th_dd_des_ts: Optional[Array] = None,
    ctrl_ff: Callable = lambda th, th_d, th_des, th_d_des, th_dd_des: jnp.zeros((2,)),
    ctrl_fb: Callable = lambda th, th_d, th_des, th_d_des: jnp.zeros((2,)),
    jit_compile: bool = True,
) -> Dict[str, Array]:
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
        jit_compile: whether to jit compile the simulation. Default is True.
    Returns:
        sim_ts: dictionary of states and other time series data of simulation
    """
    if discrete_forward_dynamics_fn is None:
        discrete_forward_dynamics_fn = jit(partial(discrete_forward_dynamics, rp))

    if tau_ext_ts is None:
        tau_ext_ts = jnp.zeros((t_ts.shape[0], 2))

    if th_des_ts is None:
        th_des_ts = jnp.zeros((t_ts.shape[0], 2))

    if th_d_des_ts is None:
        th_d_des_ts = jnp.zeros((t_ts.shape[0], 2))

    if th_dd_des_ts is None:
        th_dd_des_ts = jnp.zeros((t_ts.shape[0], 2))

    if jit_compile:
        return _simulate_robot_jitted(
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
    else:
        return _simulate_robot_not_jitted(
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


def _simulate_robot_jitted(
    rp: dict,
    t_ts: Array,
    discrete_forward_dynamics_fn: Callable,
    th_0: Array,
    th_d_0: Array,
    tau_ext_ts: Array,
    th_des_ts: Array,
    th_d_des_ts: Array,
    th_dd_des_ts: Array,
    ctrl_ff: Callable,
    ctrl_fb: Callable,
) -> Dict[str, Array]:
    """
    Simulates the double pendulum robot while jitting everything because we rely on the lax.scan function.

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
    # compute time step
    dt = t_ts[1] - t_ts[0]

    step_simulator_fn = partial(
        step_simulator,
        rp,
        discrete_forward_dynamics_fn,
        ctrl_ff,
        ctrl_fb,
    )

    input_ts = dict(
        t_ts=t_ts,
        tau_ext_ts=tau_ext_ts,
        th_des_ts=th_des_ts,
        th_d_des_ts=th_d_des_ts,
        th_dd_des_ts=th_dd_des_ts,
    )

    carry = dict(
        t=t_ts[0] - dt,
        th=th_0,
        th_d=th_d_0,
    )

    carry, sim_ts = lax.scan(step_simulator_fn, carry, input_ts)

    return sim_ts


def _simulate_robot_not_jitted(
    rp: dict,
    t_ts: Array,
    discrete_forward_dynamics_fn: Callable,
    th_0: Array,
    th_d_0: Array,
    tau_ext_ts: Array,
    th_des_ts: Array,
    th_d_des_ts: Array,
    th_dd_des_ts: Array,
    ctrl_ff: Callable,
    ctrl_fb: Callable,
) -> Dict[str, Array]:
    """
    Simulates the double pendulum robot with a Python implementation of the scan function.

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
    dt = t_ts[1] - t_ts[0]

    step_simulator_fn = partial(
        step_simulator,
        rp,
        discrete_forward_dynamics_fn,
        ctrl_ff,
        ctrl_fb,
    )

    input_ts = dict(
        t_ts=t_ts,
        tau_ext_ts=tau_ext_ts,
        th_des_ts=th_des_ts,
        th_d_des_ts=th_d_des_ts,
        th_dd_des_ts=th_dd_des_ts,
    )

    carry = dict(
        t=t_ts[0] - dt,
        th=th_0,
        th_d=th_d_0,
    )

    _sim_ts = []
    for time_idx in range(num_time_steps):
        input = {k: v[time_idx] for k, v in input_ts.items()}

        carry, step_data = step_simulator_fn(carry, input)
        _sim_ts.append(step_data)

    sim_ts = {k: jnp.stack([step_data[k] for step_data in _sim_ts]) for k in _sim_ts[0]}

    return sim_ts


def step_simulator(
    rp: dict,
    discrete_forward_dynamics_fn: Callable,
    ctrl_ff: Callable,
    ctrl_fb: Callable,
    carry: Dict[str, Array],
    input: Dict[str, Array],
) -> Tuple[Dict[str, Array], Dict[str, Array]]:
    """
    Simulates the double pendulum robot for one time step
    Args:
        rp: dictionary of robot parameters
        discrete_forward_dynamics_fn: function that computes the discrete forward dynamics.
            Given the time step dt, the current link state (th, th_d), and the link torque tau,
            it needs to return the next link state (th, th_d) and the link angular acceleration th_dd.
            It must have the signature: discrete_forward_dynamics_fn(dt, th, th_d, tau) -> (th_next, th_d_next, th_dd)
        ctrl_ff: callable computing the feed-forward control torques. Needs to return jax.array of shape (2, )
            It must have the signature: ctrl_ff(th, th_d, th_des, th_d_des, th_dd_des) -> tau_ff
        ctrl_fb: callable computing the feed-back control torques. Needs to return jax.array of shape (2, )
            It must have the signature: ctrl_fb(th, th_d, th_des, th_d_des) -> tau_fb
        carry: dictionary of the simulator state containing:
            t: current world time [s]
            th: link angles at the next time step of shape (2, )
            th_d: link angular velocities at the next time step of shape (2, )
        input: dictionary of the simulator input containing:
            t_ts: current world time [s]
            tau_ext_ts: link torques of shape (2, )
                which are applied to the system in addition to the evaluated feedforward and feedback torques
            th_des_ts: desired link angles of shape (2, )
            th_d_des_ts: desired link angular velocities of shape (2, )
            th_dd_des_ts: desired link angular accelerations of shape (2, )

    Returns:
        carry: dictionary of the simulator state containing:
            t: current world time [s]
            th: link angles at the next time step of shape (2, )
            th_d: link angular velocities at the next time step of shape (2, )
        step_data: dictionary of the simulator data of the current time step containing:
            th: link angles at the next time step of shape (2, )
            th_d: link angular velocities at the next time step of shape (2, )
            th_dd: link angular accelerations of shape (2, )
            x: end-effector position at the next time step of shape (2, )
            x_d: next end-effector velocity of shape (2, )
            x_dd: end-effector acceleration of shape (2, )
            x_eb: next position of elbow joint of shape (2, )
            tau: total external torque applied to the links of shape (2, )
            tau_ff: torque computed by feed-forward controller of shape (2, )
            tau_fb: torque computed by feedback controller of shape (2, )
    """
    # extract the current state from the carry dictionary
    t_curr = carry["t"]
    th_curr = carry["th"]
    th_d_curr = carry["th_d"]

    # compute time step
    dt = input["t_ts"] - t_curr

    # evaluate feedforward and feedback controllers
    tau_ff = ctrl_ff(
        th_curr,
        th_d_curr,
        input["th_des_ts"],
        input["th_d_des_ts"],
        input["th_dd_des_ts"],
    )
    tau_fb = ctrl_fb(th_curr, th_d_curr, input["th_des_ts"], input["th_d_des_ts"])

    # compute total torque
    tau = input["tau_ext_ts"] + tau_ff + tau_fb

    # evaluate the dynamics
    th_next, th_d_next, th_dd = discrete_forward_dynamics_fn(
        dt, th_curr, th_d_curr, tau
    )

    # evaluate forward kinematics at the current time step
    x_current, x_d_current, x_dd, x_eb_current = extended_forward_kinematics(
        rp, th_curr, th_d_curr, th_dd
    )

    # save the current state and the state transition data
    step_data = dict(
        t_ts=carry["t"],
        th_ts=th_curr,
        th_d_ts=th_d_curr,
        th_dd_ts=th_dd,
        x_ts=x_current,
        x_d_ts=x_d_current,
        x_dd_ts=x_dd,
        x_eb_ts=x_eb_current,
        tau_ts=tau,
        tau_ff_ts=tau_ff,
        tau_fb_ts=tau_fb,
    )

    # update the carry array
    carry = dict(t=input["t_ts"], th=th_next, th_d=th_d_next)

    return carry, step_data
