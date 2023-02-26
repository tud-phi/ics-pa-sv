from jax import lax
import jax.numpy as jnp


def normalize_link_angles(th: jnp.ndarray) -> jnp.ndarray:
    """
    Normalize the link angles `th` to the interval [-pi, pi].
    """
    th_norm = jnp.mod(th + jnp.pi, 2 * jnp.pi) - jnp.pi
    return th_norm


def damped_pinv(A: jnp.ndarray, damping: float = 0.0) -> jnp.ndarray:
    """
    Computes the damped pseudo-inverse of the matrix A.
    Args:
        A: The matrix to be pseudo-inverted of shape (m, n).
        damping: The damping factor lambda.
    Returns:
        A_pinv: The pseudo-inverse of A of shape (n, m).
    """
    A_pinv = lax.select(
        A.shape[0] >= A.shape[1],
        on_true=jnp.linalg.inv(A.T @ A + damping**2 * jnp.eye(A.shape[1]))
        @ A.T,  # left pseudo-inverse
        on_false=A.T
        @ jnp.linalg.inv(
            A @ A.T + damping**2 * jnp.eye(A.shape[0])
        ),  # right pseudo-inverse
    )

    return A_pinv
