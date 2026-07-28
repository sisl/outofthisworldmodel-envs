import jax.numpy as jnp
import numpy as np

from owm_envs.core.integrator import Integrator


def test_rk4_matches_exponential_decay():
    # x' = -x  =>  x(t) = x0 * exp(-t). RK4 is 4th order, so error over 1s is tiny.
    def f(x, u):
        del u
        return -x

    dt = 0.01
    integ = Integrator(dt)
    x = jnp.array([1.0], dtype=jnp.float32)
    u = jnp.zeros((1,), dtype=jnp.float32)
    for _ in range(100):
        x = integ.rk4(f, x, u)
    assert np.isclose(float(x[0]), float(np.exp(-1.0)), atol=1e-5)


def test_rk4_constant_derivative_is_exact():
    # x' = 2 (constant) => exact linear growth, RK4 must be exact here.
    def f(x, u):
        del x
        return u

    integ = Integrator(0.1)
    x = jnp.array([0.0], dtype=jnp.float32)
    u = jnp.array([2.0], dtype=jnp.float32)
    for _ in range(10):
        x = integ.rk4(f, x, u)
    assert np.isclose(float(x[0]), 2.0, atol=1e-6)


def test_rk4_is_more_accurate_than_euler():
    def f(x, u):
        del u
        return -x

    integ = Integrator(0.1)
    x_rk4 = jnp.array([1.0], dtype=jnp.float32)
    x_euler = jnp.array([1.0], dtype=jnp.float32)
    u = jnp.zeros((1,), dtype=jnp.float32)
    for _ in range(10):
        x_rk4 = integ.rk4(f, x_rk4, u)
        x_euler = integ.euler(f, x_euler, u)
    truth = float(np.exp(-1.0))
    assert abs(float(x_rk4[0]) - truth) < abs(float(x_euler[0]) - truth)
