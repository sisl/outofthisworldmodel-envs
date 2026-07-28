from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

StateFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


class Integrator:
    """
    Simple ODE integrator utilities for JAX-based dynamics.

    We integrate x' = f(x, u) with fixed time step dt.
    """

    def __init__(self, dt: float):
        self.dt = float(dt)

    def euler(self, f: StateFn, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Explicit Euler step.
        """
        dx = f(x, u)
        return x + self.dt * dx

    def rk4(self, f: StateFn, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Classic Runge-Kutta 4 integrator.
        """
        h = self.dt
        k1 = f(x, u)
        k2 = f(x + 0.5 * h * k1, u)
        k3 = f(x + 0.5 * h * k2, u)
        k4 = f(x + h * k3, u)
        return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
