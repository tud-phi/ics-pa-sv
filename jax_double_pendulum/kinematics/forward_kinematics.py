from functools import partial
from jax import Array, jit
from jax import numpy as jnp
from typing import Tuple

from .jacobian import jacobian
from .jacobian_dot import jacobian_dot


@jit
def forward_kinematics(rp: dict, th: Array) -> Tuple[Array, Array]:
    """
    Computes the xy positions of the tip of the links
    Args:
        rp: dictionary of robot parameters
        th: link angles of shape (2, )
    Returns:
        x_eb: position of the elbow of shape (2, )
        x: position of the end effector of shape (2, )
    """
    # Compute the elbow position
    x_eb = jnp.array(
        [
            rp["l1"] * jnp.cos(th[0]),
            rp["l1"] * jnp.sin(th[0]),
        ]
    )

    # compute the end effector position
    x = x_eb + jnp.array(
        [
            rp["l2"] * jnp.cos(th[1]),
            rp["l2"] * jnp.sin(th[1]),
        ]
    )

    return x_eb, x


@jit
def extended_forward_kinematics(
    rp: dict, th: Array, th_d: Array, th_dd: Array
) -> Tuple[Array, Array, Array, Array]:
    """
    Computes the extended forward kinematics of the double pendulum.
    This includes the position, velocity and acceleration of the end effector.
    Args:
        rp: dictionary of robot parameters
        th: link angles of shape (2, )
        th_d: link angular velocities of shape (2, )
        th_dd: link angular accelerations of shape (2, )
    Returns:
        x: position of the end effector of shape (2, )
        x_d: velocity of the end effector of shape (2, )
        x_dd: acceleration of the end effector of shape (2, )
    """

    x_eb, x = forward_kinematics(rp, th)

    J = jacobian(rp, th)
    # velocity in Cartesian space
    x_d = J @ th_d

    J_d = jacobian_dot(rp, th, th_d)
    # acceleration in Cartesian space
    x_dd = J @ th_dd + J_d @ th_d

    return x, x_d, x_dd, x_eb
