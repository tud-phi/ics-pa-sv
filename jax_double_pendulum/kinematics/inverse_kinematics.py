from jax import debug, jit, lax
from jax import numpy as jnp
from typing import Tuple

from .jacobian import jacobian
from .jacobian_dot import jacobian_dot
from ..utils import normalize_link_angles


@jit
def inverse_kinematics(
    rp: dict,
    th: jnp.ndarray,
    x_des: jnp.ndarray,
    x_d_des: jnp.ndarray,
    x_dd_des: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes the inverse kinematics of the double pendulum
        - Follow a pre-computed trajectory (x_des^T, x_d_des^T, x_dd_des^T)^T as closely as possible
        - As robot state is not unique (th = a or th = a + 2 * pi is the same in task-space),
            compute the closest value to the current theta
    :param rp: dictionary of robot parameters
    :param th: current link angles of shape (2, )
    :param x_des: desired end-effector position in Cartesian space of shape (2, )
    :param x_d_des: desired end-effector velocity in Cartesian space of shape (2, )
    :param x_dd_des: desired end-effector acceleration in Cartesian space of shape (2, )
    :return th: link angles of shape (2, )
    :return th_d: link angular velocities of shape (2, )
    :return th_dd: link angular accelerations of shape (2, )
    """
    A = (
        x_des[0] ** 2
        + x_des[1] ** 2
        + rp["l1"] ** 2
        - rp["l2"] ** 2
        + 2 * rp["l1"] * x_des[0]
    )
    B = -4 * rp["l1"] * x_des[1]
    C = (
        x_des[0] ** 2
        + x_des[1] ** 2
        + rp["l1"] ** 2
        - rp["l2"] ** 2
        - 2 * rp["l1"] * x_des[0]
    )

    # save th as current link angles
    th_curr = th
    # catch Cartesian coordinates that are not reachable by the end-effector
    th = lax.select(
        (B**2 - 4 * A * C) >= 0,  # pred
        _compute_theta(rp, th, x_des, A, B, C),  # on_true
        th,  # on_false
    )

    # difference of computed next th to the current th mapped to [-pi, pi]
    th_diff = normalize_link_angles(th - th_curr)
    th = th_curr + th_diff

    J = jacobian(rp, th)
    # checker whether the Jacobian is singular
    # the matrix reciprocal condition is the inverse of the condition number
    # this is provided by the rcond function in Matlab
    # https://www.thecodingforums.com/threads/rcond-in-numpy.696616/
    th_d = lax.select(
        1 / jnp.linalg.cond(J) > 0,  # pred
        jnp.linalg.solve(J, x_d_des),  # on_true
        jnp.zeros((2,)),  # on_false
    )
    th_dd = lax.select(
        1 / jnp.linalg.cond(J) > 0,  # pred
        jnp.linalg.solve(J, x_dd_des - jacobian_dot(rp, th, th_d) @ th_d),  # on_true
        jnp.zeros((2,)),  # on_false
    )

    return th, th_d, th_dd


@jit
def _compute_theta(
    _rp: dict,
    _th: jnp.ndarray,
    _x_des: jnp.ndarray,
    _A: jnp.ndarray,
    _B: jnp.ndarray,
    _C: jnp.ndarray,
) -> jnp.ndarray:
    th1 = 2 * jnp.arctan2(-_B + jnp.sqrt(_B**2 - 4 * _A * _C), 2 * _A)
    th2 = jnp.arctan2(
        _x_des[1] - _rp["l1"] * jnp.sin(th1), _x_des[0] - _rp["l1"] * jnp.cos(th1)
    )

    th = jnp.array([th1, th2])
    return th
