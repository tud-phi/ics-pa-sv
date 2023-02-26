from jax import jit, vmap
from jax import numpy as jnp


@jit
def jacobian(rp: dict, th: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the positional Jacobian of the end-effector with respect to the link angles
    :param rp: dictionary of robot parameters
    :param th: link angles of shape (2, )
    :return J: positional Jacobian of end-effector of shape (2, 2)
    """
    J = jnp.array(
        [
            [-rp["l1"] * jnp.sin(th[0]), -rp["l2"] * jnp.sin(th[1])],
            [rp["l1"] * jnp.cos(th[0]), rp["l2"] * jnp.cos(th[1])],
        ]
    )

    return J
