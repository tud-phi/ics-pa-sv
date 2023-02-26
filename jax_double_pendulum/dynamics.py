from functools import partial
from jax import jit
from jax import numpy as jnp
from typing import Tuple

from jax_double_pendulum.integrators import euler_step, rk4_step


@jit
def dynamical_matrices(
    rp: dict, th: jnp.ndarray, th_d: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes the components of the Equations of Motion (EoM), which also referred to as RBD matrices.
    The EoM are formulated as:
        M @ th_dd + C @ th_d + G = tau
    where tau is the configuration-space torque vector.
    Args:
        rp: dictionary of robot parameters
        th: link angles of shape (2, )
        th_d: link angular velocities of shape (2, )
    Returns:
        M: inertial matrix of shape (2, 2)
        C: coriolis and centrifugal matrix of shape (2, 2)
        G: gravitational matrix of shape (2, )
    """
    c1 = jnp.cos(th[0])
    c2 = jnp.cos(th[1])
    c21 = jnp.cos(th[1] - th[0])
    s21 = jnp.sin(th[1] - th[0])

    # Dynamic matrices
    M = jnp.array(
        [
            [
                rp["j1"] + rp["m1"] * rp["lc1"] ** 2 + rp["m2"] * rp["l1"] ** 2,
                rp["m2"] * rp["l1"] * rp["lc2"] * c21,
            ],
            [
                rp["m2"] * rp["l1"] * rp["lc2"] * c21,
                rp["j2"] + rp["m2"] * rp["lc2"] ** 2,
            ],
        ]
    )
    C = jnp.array(
        [
            [0, -rp["m2"] * rp["l1"] * rp["lc2"] * s21 * th_d[1]],
            [rp["m2"] * rp["l1"] * rp["lc2"] * s21 * th_d[0], 0],
        ]
    )
    G = jnp.array(
        [
            (rp["m1"] * rp["lc1"] + rp["m2"] * rp["l1"]) * rp["g"] * c1,
            rp["m2"] * rp["g"] * rp["lc2"] * c2,
        ]
    )

    return M, C, G


@jit
def continuous_forward_dynamics(
    rp: dict,
    th: jnp.ndarray,
    th_d: jnp.ndarray,
    tau: jnp.ndarray = jnp.zeros((2,)),
) -> jnp.ndarray:
    """
    Compute the continuous forward dynamics of the system
    Args:
        rp: dictionary of robot parameters
        th: link angles of double pendulum of shape (2, )
        th_d: link angular velocities of double pendulum of shape (2, )
        tau: link torques of double pendulum of shape (2, )
    Returns:
        th_dd: link angular accelerations of double pendulum of shape (2, )
    """
    M, C, G = dynamical_matrices(rp, th, th_d)

    # link angular accelerations
    th_dd = jnp.linalg.inv(M) @ (tau - C @ th_d - G)

    return th_dd


@jit
def continuous_inverse_dynamics(
    rp: dict,
    th: jnp.ndarray,
    th_d: jnp.ndarray,
    th_dd: jnp.ndarray = jnp.zeros((2,)),
) -> jnp.ndarray:
    """
    Compute the continuous inverse dynamics of the system
    Args:
        rp: dictionary of robot parameters
        th: link angles of double pendulum of shape (2, )
        th_d: link angular velocities of double pendulum of shape (2, )
        th_dd: link angular accelerations of double pendulum of shape (2, )
    Returns:
        tau: link torques of double pendulum of shape (2, )
    """
    M, C, G = dynamical_matrices(rp, th, th_d)

    # link torques
    tau = M @ th_dd + C @ th_d + G

    return tau


def continuous_state_space_dynamics(
    rp: dict,
    x: jnp.ndarray,
    tau: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the continuous forward dynamics of the system in state-space representation
    Args:
        rp: dictionary of robot parameters
        x: system state of shape (4, ) consisting of the link angles and velocities
        tau: link torques of shape (2, )
    Returns:
        dx_dt: time derivative of the system state of shape (4, )
        y: system output of shape (2, ) consisting of the link angles
    """
    # split the state into link angles and velocities
    th, th_d = jnp.split(x, 2)

    # compute the link angular accelerations
    th_dd = continuous_forward_dynamics(rp, th, th_d, tau)

    # time derivative of state
    dx_dt = jnp.concatenate([th_d, th_dd])

    # the system output are the two link angles
    y = th

    return dx_dt, y


@jit
def discrete_forward_dynamics(
    rp: dict,
    dt: jnp.ndarray,
    th_curr: jnp.ndarray,
    th_d_curr: jnp.ndarray,
    tau: jnp.ndarray = jnp.zeros((2,)),
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute the discrete forward dynamics of the system
    Args:
        rp: dictionary of robot parameters
        dt: time step between the current and the next state [s]
        th_curr: current link angles of double pendulum of shape (2, )
        th_d_curr: current link angular velocities of double pendulum of shape (2, )
        tau: link torques of double pendulum of shape (2, )
    Returns:
        th_next: link angles at the next time step of double pendulum of shape (2, )
        th_d_next: link angular velocities at the next time step of double pendulum of shape (2, )
        th_dd: link angular accelerations between current and next time step of double pendulum of shape (2, )
    """
    th_dd = continuous_forward_dynamics(rp, th_curr, th_d_curr, tau)

    x_next = rk4_step(
        # generate `ode_fun`, which conforms to the signature ode_fun(x) -> dx_dt
        ode_fun=lambda _x: partial(continuous_state_space_dynamics, rp, tau=tau)(_x)[0],
        x=jnp.concatenate([th_curr, th_d_curr]),  # concatenate the current state
        dt=dt,  # time step
    )
    th_next, th_d_next = jnp.split(x_next, 2)

    return th_next, th_d_next, th_dd


def continuous_linear_state_space_representation(
    rp: dict,
    th_eq: jnp.ndarray,
    th_d_eq: jnp.ndarray = jnp.zeros((2,)),
    tau_eq: jnp.ndarray = jnp.zeros((2,)),
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Linearize the system about the specified state (th, th_d) and input tau
    and return the linearized system in state space representation

    Args:
        rp: dictionary of robot parameters
        th_eq: equilibrium link angles of double pendulum of shape (2, )
        th_d_eq: equilibrium link angular velocities of double pendulum of shape (2, )
        tau_eq: equilibrium link torques of double pendulum of shape (2, )

    Returns:
        A: state transition matrix of shape (4, 4)
        B: input matrix of shape (4, 2)
        C: output matrix of shape (2, 4)
        D: feed-through matrix of shape (2, 2)
    """
    th1, th2 = th_eq[0], th_eq[1]
    th_d1, th_d2 = th_d_eq[0], th_d_eq[1]
    tau1, tau2 = tau_eq[0], tau_eq[1]
    l1, l2 = rp["l1"], rp["l2"]
    lc1, lc2 = rp["lc1"], rp["lc2"]
    m1, m2 = rp["m1"], rp["m2"]
    j1, j2 = rp["j1"], rp["j2"]
    g = rp["g"]

    A = jnp.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [
                (
                    2
                    * l1**2
                    * lc2**2
                    * m2**2
                    * (
                        l1
                        * lc2
                        * m2
                        * (
                            -g * lc2 * m2 * jnp.cos(th2)
                            + l1 * lc2 * m2 * th_d1**2 * jnp.sin(th1 - th2)
                            + tau2
                        )
                        * jnp.cos(th1 - th2)
                        + (j2 + lc2**2 * m2)
                        * (
                            g * (l1 * m2 + lc1 * m1) * jnp.cos(th1)
                            + l1 * lc2 * m2 * th_d2**2 * jnp.sin(th1 - th2)
                            - tau1
                        )
                    )
                    * jnp.sin(th1 - th2)
                    * jnp.cos(th1 - th2)
                    + (
                        -(l1**2)
                        * lc2**2
                        * m2**2
                        * th_d1**2
                        * jnp.cos(th1 - th2) ** 2
                        + l1
                        * lc2
                        * m2
                        * (
                            -g * lc2 * m2 * jnp.cos(th2)
                            + l1 * lc2 * m2 * th_d1**2 * jnp.sin(th1 - th2)
                            + tau2
                        )
                        * jnp.sin(th1 - th2)
                        + (j2 + lc2**2 * m2)
                        * (
                            g * (l1 * m2 + lc1 * m1) * jnp.sin(th1)
                            - l1 * lc2 * m2 * th_d2**2 * jnp.cos(th1 - th2)
                        )
                    )
                    * (
                        j1 * j2
                        + j1 * lc2**2 * m2
                        + j2 * l1**2 * m2
                        + j2 * lc1**2 * m1
                        - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                        + l1**2 * lc2**2 * m2**2
                        + lc1**2 * lc2**2 * m1 * m2
                    )
                )
                / (
                    j1 * j2
                    + j1 * lc2**2 * m2
                    + j2 * l1**2 * m2
                    + j2 * lc1**2 * m1
                    - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                    + l1**2 * lc2**2 * m2**2
                    + lc1**2 * lc2**2 * m1 * m2
                )
                ** 2,
                l1
                * lc2
                * m2
                * (
                    -2
                    * l1
                    * lc2
                    * m2
                    * (
                        l1
                        * lc2
                        * m2
                        * (
                            -g * lc2 * m2 * jnp.cos(th2)
                            + l1 * lc2 * m2 * th_d1**2 * jnp.sin(th1 - th2)
                            + tau2
                        )
                        * jnp.cos(th1 - th2)
                        + (j2 + lc2**2 * m2)
                        * (
                            g * (l1 * m2 + lc1 * m1) * jnp.cos(th1)
                            + l1 * lc2 * m2 * th_d2**2 * jnp.sin(th1 - th2)
                            - tau1
                        )
                    )
                    * jnp.sin(th1 - th2)
                    * jnp.cos(th1 - th2)
                    + (
                        -lc2
                        * m2
                        * (g * jnp.sin(th2) - l1 * th_d1**2 * jnp.cos(th1 - th2))
                        * jnp.cos(th1 - th2)
                        + th_d2**2 * (j2 + lc2**2 * m2) * jnp.cos(th1 - th2)
                        - (
                            -g * lc2 * m2 * jnp.cos(th2)
                            + l1 * lc2 * m2 * th_d1**2 * jnp.sin(th1 - th2)
                            + tau2
                        )
                        * jnp.sin(th1 - th2)
                    )
                    * (
                        j1 * j2
                        + j1 * lc2**2 * m2
                        + j2 * l1**2 * m2
                        + j2 * lc1**2 * m1
                        - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                        + l1**2 * lc2**2 * m2**2
                        + lc1**2 * lc2**2 * m1 * m2
                    )
                )
                / (
                    j1 * j2
                    + j1 * lc2**2 * m2
                    + j2 * l1**2 * m2
                    + j2 * lc1**2 * m1
                    - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                    + l1**2 * lc2**2 * m2**2
                    + lc1**2 * lc2**2 * m1 * m2
                )
                ** 2,
                -2
                * l1**2
                * lc2**2
                * m2**2
                * th_d1
                * jnp.sin(2 * th1 - 2 * th2)
                / (
                    2 * j1 * j2
                    + 2 * j1 * lc2**2 * m2
                    + 2 * j2 * l1**2 * m2
                    + 2 * j2 * lc1**2 * m1
                    - l1**2 * lc2**2 * m2**2 * jnp.cos(2 * th1 - 2 * th2)
                    + l1**2 * lc2**2 * m2**2
                    + 2 * lc1**2 * lc2**2 * m1 * m2
                ),
                -2
                * l1
                * lc2
                * m2
                * th_d2
                * (j2 + lc2**2 * m2)
                * jnp.sin(th1 - th2)
                / (
                    j1 * j2
                    + j1 * lc2**2 * m2
                    + j2 * l1**2 * m2
                    + j2 * lc1**2 * m1
                    - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                    + l1**2 * lc2**2 * m2**2
                    + lc1**2 * lc2**2 * m1 * m2
                ),
            ],
            [
                l1
                * lc2
                * m2
                * (
                    -2
                    * l1
                    * lc2
                    * m2
                    * (
                        l1
                        * lc2
                        * m2
                        * (
                            g * (l1 * m2 + lc1 * m1) * jnp.cos(th1)
                            + l1 * lc2 * m2 * th_d2**2 * jnp.sin(th1 - th2)
                            - tau1
                        )
                        * jnp.cos(th1 - th2)
                        + (j1 + l1**2 * m2 + lc1**2 * m1)
                        * (
                            -g * lc2 * m2 * jnp.cos(th2)
                            + l1 * lc2 * m2 * th_d1**2 * jnp.sin(th1 - th2)
                            + tau2
                        )
                    )
                    * jnp.sin(th1 - th2)
                    * jnp.cos(th1 - th2)
                    + (
                        th_d1**2
                        * (j1 + l1**2 * m2 + lc1**2 * m1)
                        * jnp.cos(th1 - th2)
                        - (
                            g * (l1 * m2 + lc1 * m1) * jnp.sin(th1)
                            - l1 * lc2 * m2 * th_d2**2 * jnp.cos(th1 - th2)
                        )
                        * jnp.cos(th1 - th2)
                        - (
                            g * (l1 * m2 + lc1 * m1) * jnp.cos(th1)
                            + l1 * lc2 * m2 * th_d2**2 * jnp.sin(th1 - th2)
                            - tau1
                        )
                        * jnp.sin(th1 - th2)
                    )
                    * (
                        j1 * j2
                        + j1 * lc2**2 * m2
                        + j2 * l1**2 * m2
                        + j2 * lc1**2 * m1
                        - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                        + l1**2 * lc2**2 * m2**2
                        + lc1**2 * lc2**2 * m1 * m2
                    )
                )
                / (
                    j1 * j2
                    + j1 * lc2**2 * m2
                    + j2 * l1**2 * m2
                    + j2 * lc1**2 * m1
                    - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                    + l1**2 * lc2**2 * m2**2
                    + lc1**2 * lc2**2 * m1 * m2
                )
                ** 2,
                lc2
                * m2
                * (
                    2
                    * l1**2
                    * lc2
                    * m2
                    * (
                        l1
                        * lc2
                        * m2
                        * (
                            g * (l1 * m2 + lc1 * m1) * jnp.cos(th1)
                            + l1 * lc2 * m2 * th_d2**2 * jnp.sin(th1 - th2)
                            - tau1
                        )
                        * jnp.cos(th1 - th2)
                        + (j1 + l1**2 * m2 + lc1**2 * m1)
                        * (
                            -g * lc2 * m2 * jnp.cos(th2)
                            + l1 * lc2 * m2 * th_d1**2 * jnp.sin(th1 - th2)
                            + tau2
                        )
                    )
                    * jnp.sin(th1 - th2)
                    * jnp.cos(th1 - th2)
                    + (
                        -(l1**2) * lc2 * m2 * th_d2**2 * jnp.cos(th1 - th2) ** 2
                        + l1
                        * (
                            g * (l1 * m2 + lc1 * m1) * jnp.cos(th1)
                            + l1 * lc2 * m2 * th_d2**2 * jnp.sin(th1 - th2)
                            - tau1
                        )
                        * jnp.sin(th1 - th2)
                        + (g * jnp.sin(th2) - l1 * th_d1**2 * jnp.cos(th1 - th2))
                        * (j1 + l1**2 * m2 + lc1**2 * m1)
                    )
                    * (
                        j1 * j2
                        + j1 * lc2**2 * m2
                        + j2 * l1**2 * m2
                        + j2 * lc1**2 * m1
                        - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                        + l1**2 * lc2**2 * m2**2
                        + lc1**2 * lc2**2 * m1 * m2
                    )
                )
                / (
                    j1 * j2
                    + j1 * lc2**2 * m2
                    + j2 * l1**2 * m2
                    + j2 * lc1**2 * m1
                    - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                    + l1**2 * lc2**2 * m2**2
                    + lc1**2 * lc2**2 * m1 * m2
                )
                ** 2,
                2
                * l1
                * lc2
                * m2
                * th_d1
                * (j1 + l1**2 * m2 + lc1**2 * m1)
                * jnp.sin(th1 - th2)
                / (
                    j1 * j2
                    + j1 * lc2**2 * m2
                    + j2 * l1**2 * m2
                    + j2 * lc1**2 * m1
                    - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                    + l1**2 * lc2**2 * m2**2
                    + lc1**2 * lc2**2 * m1 * m2
                ),
                2
                * l1**2
                * lc2**2
                * m2**2
                * th_d2
                * jnp.sin(2 * th1 - 2 * th2)
                / (
                    2 * j1 * j2
                    + 2 * j1 * lc2**2 * m2
                    + 2 * j2 * l1**2 * m2
                    + 2 * j2 * lc1**2 * m1
                    - l1**2 * lc2**2 * m2**2 * jnp.cos(2 * th1 - 2 * th2)
                    + l1**2 * lc2**2 * m2**2
                    + 2 * lc1**2 * lc2**2 * m1 * m2
                ),
            ],
        ]
    )

    B = jnp.array(
        [
            [0, 0],
            [0, 0],
            [
                (j2 + lc2**2 * m2)
                / (
                    j1 * j2
                    + j1 * lc2**2 * m2
                    + j2 * l1**2 * m2
                    + j2 * lc1**2 * m1
                    - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                    + l1**2 * lc2**2 * m2**2
                    + lc1**2 * lc2**2 * m1 * m2
                ),
                -l1
                * lc2
                * m2
                * jnp.cos(th1 - th2)
                / (
                    j1 * j2
                    + j1 * lc2**2 * m2
                    + j2 * l1**2 * m2
                    + j2 * lc1**2 * m1
                    - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                    + l1**2 * lc2**2 * m2**2
                    + lc1**2 * lc2**2 * m1 * m2
                ),
            ],
            [
                -l1
                * lc2
                * m2
                * jnp.cos(th1 - th2)
                / (
                    j1 * j2
                    + j1 * lc2**2 * m2
                    + j2 * l1**2 * m2
                    + j2 * lc1**2 * m1
                    - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                    + l1**2 * lc2**2 * m2**2
                    + lc1**2 * lc2**2 * m1 * m2
                ),
                (j1 + l1**2 * m2 + lc1**2 * m1)
                / (
                    j1 * j2
                    + j1 * lc2**2 * m2
                    + j2 * l1**2 * m2
                    + j2 * lc1**2 * m1
                    - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                    + l1**2 * lc2**2 * m2**2
                    + lc1**2 * lc2**2 * m1 * m2
                ),
            ],
        ]
    )
    C = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    D = jnp.array([[0.0, 0.0], [0.0, 0.0]])

    return A, B, C, D


def continuous_closed_loop_linear_state_space_representation(
    rp: dict,
    th_eq: jnp.ndarray,
    th_d_eq: jnp.ndarray,
    tau_eq: jnp.ndarray,
    th_des: jnp.ndarray,
    th_d_des: jnp.ndarray,
    kp: jnp.ndarray = jnp.zeros((2, 2)),
    kd: jnp.ndarray = jnp.zeros((2, 2)),
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Linearize the system about the specified state (th, th_d) and input tau
    and return the linearized system in state space representation

    Args:
        rp: dictionary of robot parameters
        th_eq: link angles of double pendulum of shape (2, )
        th_d_eq: link angular velocities of double pendulum of shape (2, )
        tau_eq: link torques of double pendulum of shape (2, )

    Returns:
        A: state transition matrix of shape (4, 4)
        B: input matrix of shape (4, 2)
        C: output matrix of shape (2, 4)
        D: feed-through matrix of shape (2, 2)
    """
    th1, th2 = th_eq[0], th_eq[1]
    th_d1, th_d2 = th_d_eq[0], th_d_eq[1]
    tau1, tau2 = tau_eq[0], tau_eq[1]
    l1, l2 = rp["l1"], rp["l2"]
    lc1, lc2 = rp["lc1"], rp["lc2"]
    m1, m2 = rp["m1"], rp["m2"]
    j1, j2 = rp["j1"], rp["j2"]
    g = rp["g"]
    th_des1, th_des2 = th_des[0], th_des[1]
    th_d_des1, th_d_des2 = th_d_des[0], th_d_des[1]
    kp1, kp2 = kp[0, 0], kp[1, 1]
    kd1, kd2 = kd[0, 0], kd[1, 1]

    A = jnp.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [
                (
                    2
                    * l1**2
                    * lc2**2
                    * m2**2
                    * (
                        -l1
                        * lc2
                        * m2
                        * (
                            g * lc2 * m2 * jnp.cos(th2)
                            + kd2 * (th_d2 - th_d_des2)
                            + kp2 * (th2 - th_des2)
                            - l1 * lc2 * m2 * th_d1**2 * jnp.sin(th1 - th2)
                            - tau2
                        )
                        * jnp.cos(th1 - th2)
                        + (j2 + lc2**2 * m2)
                        * (
                            g * (l1 * m2 + lc1 * m1) * jnp.cos(th1)
                            + kd1 * (th_d1 - th_d_des1)
                            + kp1 * (th1 - th_des1)
                            + l1 * lc2 * m2 * th_d2**2 * jnp.sin(th1 - th2)
                            - tau1
                        )
                    )
                    * jnp.sin(th1 - th2)
                    * jnp.cos(th1 - th2)
                    - (
                        l1**2
                        * lc2**2
                        * m2**2
                        * th_d1**2
                        * jnp.cos(th1 - th2) ** 2
                        + l1
                        * lc2
                        * m2
                        * (
                            g * lc2 * m2 * jnp.cos(th2)
                            + kd2 * (th_d2 - th_d_des2)
                            + kp2 * (th2 - th_des2)
                            - l1 * lc2 * m2 * th_d1**2 * jnp.sin(th1 - th2)
                            - tau2
                        )
                        * jnp.sin(th1 - th2)
                        + (j2 + lc2**2 * m2)
                        * (
                            -g * (l1 * m2 + lc1 * m1) * jnp.sin(th1)
                            + kp1
                            + l1 * lc2 * m2 * th_d2**2 * jnp.cos(th1 - th2)
                        )
                    )
                    * (
                        j1 * j2
                        + j1 * lc2**2 * m2
                        + j2 * l1**2 * m2
                        + j2 * lc1**2 * m1
                        - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                        + l1**2 * lc2**2 * m2**2
                        + lc1**2 * lc2**2 * m1 * m2
                    )
                )
                / (
                    j1 * j2
                    + j1 * lc2**2 * m2
                    + j2 * l1**2 * m2
                    + j2 * lc1**2 * m1
                    - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                    + l1**2 * lc2**2 * m2**2
                    + lc1**2 * lc2**2 * m1 * m2
                )
                ** 2,
                l1
                * lc2
                * m2
                * (
                    2
                    * l1
                    * lc2
                    * m2
                    * (
                        l1
                        * lc2
                        * m2
                        * (
                            g * lc2 * m2 * jnp.cos(th2)
                            + kd2 * (th_d2 - th_d_des2)
                            + kp2 * (th2 - th_des2)
                            - l1 * lc2 * m2 * th_d1**2 * jnp.sin(th1 - th2)
                            - tau2
                        )
                        * jnp.cos(th1 - th2)
                        - (j2 + lc2**2 * m2)
                        * (
                            g * (l1 * m2 + lc1 * m1) * jnp.cos(th1)
                            + kd1 * (th_d1 - th_d_des1)
                            + kp1 * (th1 - th_des1)
                            + l1 * lc2 * m2 * th_d2**2 * jnp.sin(th1 - th2)
                            - tau1
                        )
                    )
                    * jnp.sin(th1 - th2)
                    * jnp.cos(th1 - th2)
                    + (
                        th_d2**2 * (j2 + lc2**2 * m2) * jnp.cos(th1 - th2)
                        + (
                            -g * lc2 * m2 * jnp.sin(th2)
                            + kp2
                            + l1 * lc2 * m2 * th_d1**2 * jnp.cos(th1 - th2)
                        )
                        * jnp.cos(th1 - th2)
                        + (
                            g * lc2 * m2 * jnp.cos(th2)
                            + kd2 * (th_d2 - th_d_des2)
                            + kp2 * (th2 - th_des2)
                            - l1 * lc2 * m2 * th_d1**2 * jnp.sin(th1 - th2)
                            - tau2
                        )
                        * jnp.sin(th1 - th2)
                    )
                    * (
                        j1 * j2
                        + j1 * lc2**2 * m2
                        + j2 * l1**2 * m2
                        + j2 * lc1**2 * m1
                        - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                        + l1**2 * lc2**2 * m2**2
                        + lc1**2 * lc2**2 * m1 * m2
                    )
                )
                / (
                    j1 * j2
                    + j1 * lc2**2 * m2
                    + j2 * l1**2 * m2
                    + j2 * lc1**2 * m1
                    - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                    + l1**2 * lc2**2 * m2**2
                    + lc1**2 * lc2**2 * m1 * m2
                )
                ** 2,
                (
                    -j2 * kd1
                    - kd1 * lc2**2 * m2
                    - l1**2 * lc2**2 * m2**2 * th_d1 * jnp.sin(2 * th1 - 2 * th2)
                )
                / (
                    j1 * j2
                    + j1 * lc2**2 * m2
                    + j2 * l1**2 * m2
                    + j2 * lc1**2 * m1
                    - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                    + l1**2 * lc2**2 * m2**2
                    + lc1**2 * lc2**2 * m1 * m2
                ),
                l1
                * lc2
                * m2
                * (
                    kd2 * jnp.cos(th1 - th2)
                    - 2 * th_d2 * (j2 + lc2**2 * m2) * jnp.sin(th1 - th2)
                )
                / (
                    j1 * j2
                    + j1 * lc2**2 * m2
                    + j2 * l1**2 * m2
                    + j2 * lc1**2 * m1
                    - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                    + l1**2 * lc2**2 * m2**2
                    + lc1**2 * lc2**2 * m1 * m2
                ),
            ],
            [
                l1
                * lc2
                * m2
                * (
                    2
                    * l1
                    * lc2
                    * m2
                    * (
                        -l1
                        * lc2
                        * m2
                        * (
                            g * (l1 * m2 + lc1 * m1) * jnp.cos(th1)
                            + kd1 * (th_d1 - th_d_des1)
                            + kp1 * (th1 - th_des1)
                            + l1 * lc2 * m2 * th_d2**2 * jnp.sin(th1 - th2)
                            - tau1
                        )
                        * jnp.cos(th1 - th2)
                        + (j1 + l1**2 * m2 + lc1**2 * m1)
                        * (
                            g * lc2 * m2 * jnp.cos(th2)
                            + kd2 * (th_d2 - th_d_des2)
                            + kp2 * (th2 - th_des2)
                            - l1 * lc2 * m2 * th_d1**2 * jnp.sin(th1 - th2)
                            - tau2
                        )
                    )
                    * jnp.sin(th1 - th2)
                    * jnp.cos(th1 - th2)
                    + (
                        th_d1**2
                        * (j1 + l1**2 * m2 + lc1**2 * m1)
                        * jnp.cos(th1 - th2)
                        + (
                            -g * (l1 * m2 + lc1 * m1) * jnp.sin(th1)
                            + kp1
                            + l1 * lc2 * m2 * th_d2**2 * jnp.cos(th1 - th2)
                        )
                        * jnp.cos(th1 - th2)
                        - (
                            g * (l1 * m2 + lc1 * m1) * jnp.cos(th1)
                            + kd1 * (th_d1 - th_d_des1)
                            + kp1 * (th1 - th_des1)
                            + l1 * lc2 * m2 * th_d2**2 * jnp.sin(th1 - th2)
                            - tau1
                        )
                        * jnp.sin(th1 - th2)
                    )
                    * (
                        j1 * j2
                        + j1 * lc2**2 * m2
                        + j2 * l1**2 * m2
                        + j2 * lc1**2 * m1
                        - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                        + l1**2 * lc2**2 * m2**2
                        + lc1**2 * lc2**2 * m1 * m2
                    )
                )
                / (
                    j1 * j2
                    + j1 * lc2**2 * m2
                    + j2 * l1**2 * m2
                    + j2 * lc1**2 * m1
                    - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                    + l1**2 * lc2**2 * m2**2
                    + lc1**2 * lc2**2 * m1 * m2
                )
                ** 2,
                (
                    2
                    * l1**2
                    * lc2**2
                    * m2**2
                    * (
                        l1
                        * lc2
                        * m2
                        * (
                            g * (l1 * m2 + lc1 * m1) * jnp.cos(th1)
                            + kd1 * (th_d1 - th_d_des1)
                            + kp1 * (th1 - th_des1)
                            + l1 * lc2 * m2 * th_d2**2 * jnp.sin(th1 - th2)
                            - tau1
                        )
                        * jnp.cos(th1 - th2)
                        - (j1 + l1**2 * m2 + lc1**2 * m1)
                        * (
                            g * lc2 * m2 * jnp.cos(th2)
                            + kd2 * (th_d2 - th_d_des2)
                            + kp2 * (th2 - th_des2)
                            - l1 * lc2 * m2 * th_d1**2 * jnp.sin(th1 - th2)
                            - tau2
                        )
                    )
                    * jnp.sin(th1 - th2)
                    * jnp.cos(th1 - th2)
                    + (
                        -(l1**2)
                        * lc2**2
                        * m2**2
                        * th_d2**2
                        * jnp.cos(th1 - th2) ** 2
                        + l1
                        * lc2
                        * m2
                        * (
                            g * (l1 * m2 + lc1 * m1) * jnp.cos(th1)
                            + kd1 * (th_d1 - th_d_des1)
                            + kp1 * (th1 - th_des1)
                            + l1 * lc2 * m2 * th_d2**2 * jnp.sin(th1 - th2)
                            - tau1
                        )
                        * jnp.sin(th1 - th2)
                        - (j1 + l1**2 * m2 + lc1**2 * m1)
                        * (
                            -g * lc2 * m2 * jnp.sin(th2)
                            + kp2
                            + l1 * lc2 * m2 * th_d1**2 * jnp.cos(th1 - th2)
                        )
                    )
                    * (
                        j1 * j2
                        + j1 * lc2**2 * m2
                        + j2 * l1**2 * m2
                        + j2 * lc1**2 * m1
                        - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                        + l1**2 * lc2**2 * m2**2
                        + lc1**2 * lc2**2 * m1 * m2
                    )
                )
                / (
                    j1 * j2
                    + j1 * lc2**2 * m2
                    + j2 * l1**2 * m2
                    + j2 * lc1**2 * m1
                    - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                    + l1**2 * lc2**2 * m2**2
                    + lc1**2 * lc2**2 * m1 * m2
                )
                ** 2,
                l1
                * lc2
                * m2
                * (
                    kd1 * jnp.cos(th1 - th2)
                    + 2
                    * th_d1
                    * (j1 + l1**2 * m2 + lc1**2 * m1)
                    * jnp.sin(th1 - th2)
                )
                / (
                    j1 * j2
                    + j1 * lc2**2 * m2
                    + j2 * l1**2 * m2
                    + j2 * lc1**2 * m1
                    - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                    + l1**2 * lc2**2 * m2**2
                    + lc1**2 * lc2**2 * m1 * m2
                ),
                (
                    -kd2 * (j1 + l1**2 * m2 + lc1**2 * m1)
                    + l1**2 * lc2**2 * m2**2 * th_d2 * jnp.sin(2 * th1 - 2 * th2)
                )
                / (
                    j1 * j2
                    + j1 * lc2**2 * m2
                    + j2 * l1**2 * m2
                    + j2 * lc1**2 * m1
                    - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                    + l1**2 * lc2**2 * m2**2
                    + lc1**2 * lc2**2 * m1 * m2
                ),
            ],
        ]
    )

    B = jnp.array(
        [
            [0, 0],
            [0, 0],
            [
                (j2 + lc2**2 * m2)
                / (
                    j1 * j2
                    + j1 * lc2**2 * m2
                    + j2 * l1**2 * m2
                    + j2 * lc1**2 * m1
                    - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                    + l1**2 * lc2**2 * m2**2
                    + lc1**2 * lc2**2 * m1 * m2
                ),
                -l1
                * lc2
                * m2
                * jnp.cos(th1 - th2)
                / (
                    j1 * j2
                    + j1 * lc2**2 * m2
                    + j2 * l1**2 * m2
                    + j2 * lc1**2 * m1
                    - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                    + l1**2 * lc2**2 * m2**2
                    + lc1**2 * lc2**2 * m1 * m2
                ),
            ],
            [
                -l1
                * lc2
                * m2
                * jnp.cos(th1 - th2)
                / (
                    j1 * j2
                    + j1 * lc2**2 * m2
                    + j2 * l1**2 * m2
                    + j2 * lc1**2 * m1
                    - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                    + l1**2 * lc2**2 * m2**2
                    + lc1**2 * lc2**2 * m1 * m2
                ),
                (j1 + l1**2 * m2 + lc1**2 * m1)
                / (
                    j1 * j2
                    + j1 * lc2**2 * m2
                    + j2 * l1**2 * m2
                    + j2 * lc1**2 * m1
                    - l1**2 * lc2**2 * m2**2 * jnp.cos(th1 - th2) ** 2
                    + l1**2 * lc2**2 * m2**2
                    + lc1**2 * lc2**2 * m1 * m2
                ),
            ],
        ]
    )
    C = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    D = jnp.array([[0.0, 0.0], [0.0, 0.0]])

    return A, B, C, D
