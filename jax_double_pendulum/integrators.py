from functools import partial
from jax import jit
import jax.numpy as jnp
from typing import Callable


@partial(
    jit,
    static_argnums=(0,),
    static_argnames=("ode_fun",),
)
def euler_step(ode_fun: Callable, x: jnp.ndarray, dt: jnp.ndarray) -> jnp.ndarray:
    """
    Euler's integration of the state `x` with the derivative `dx_dt` and time step `dt`.
    Args:
        ode_fun: function that computes the time derivative of the state `x`.
            Needs to conform to the signature: ode_fun(x) -> dx_dt
        x: current state of shape (num_state_dims, )
        dt: time step of shape ( ) [s]
    Returns:
        x_next: next state of shape (num_state_dims, )
    """
    dx = ode_fun(x)
    return x + dt * dx


@partial(
    jit,
    static_argnums=(0,),
    static_argnames=("ode_fun",),
)
def rk4_step(ode_fun: Callable, x: jnp.ndarray, dt: jnp.ndarray) -> jnp.ndarray:
    """
    Runge-Kutta / RK4 integration of the state `x` with the derivative `dx_dt` and time step `dt`.
    Args:
        ode_fun: function that computes the time derivative of the state `x`.
            Needs to conform to the signature: ode_fun(x) -> dx_dt
        x: current state of shape (num_state_dims, )
        dt: time step of shape ( ) [s]
    Returns:
        x_next: next state of shape (num_state_dims, )
    """
    k1 = ode_fun(x)
    k2 = ode_fun(x + dt * k1 / 2)
    k3 = ode_fun(x + dt * k2 / 2)
    k4 = ode_fun(x + dt * k3)

    x_next = x + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * dt
    return x_next
