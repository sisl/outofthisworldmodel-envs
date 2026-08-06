import importlib.resources
import math

import jax
import jax.numpy as jnp
import numpy as np
from astrojax.constants import GM_EARTH, J2_EARTH, R_EARTH

from owm_envs.envs.common.zonal_gravity import (
    J3_EARTH,
    J4_EARTH,
    J5_EARTH,
    J6_EARTH,
    accel_zonal_gravity,
)

_J = {2: J2_EARTH, 3: J3_EARTH, 4: J4_EARTH, 5: J5_EARTH, 6: J6_EARTH}

_LEGENDRE = {
    2: lambda s: 0.5 * (3 * s**2 - 1),
    3: lambda s: 0.5 * (5 * s**3 - 3 * s),
    4: lambda s: 0.125 * (35 * s**4 - 30 * s**2 + 3),
    5: lambda s: 0.125 * (63 * s**5 - 70 * s**3 + 15 * s),
    6: lambda s: 0.0625 * (231 * s**6 - 315 * s**4 + 105 * s**2 - 5),
}


def _potential(r, n_max):
    rn = jnp.linalg.norm(r)
    s = r[2] / rn
    u = 1.0
    for n in range(2, n_max + 1):
        u = u - _J[n] * (R_EARTH / rn) ** n * _LEGENDRE[n](s)
    return -GM_EARTH / rn * u


POSITIONS = [
    jnp.array([R_EARTH + 420e3, 0.0, 0.0], jnp.float64),
    jnp.array([0.0, 0.0, R_EARTH + 420e3], jnp.float64),
    jnp.array([4.5e6, -3.2e6, 4.1e6], jnp.float64),
]


def test_matches_potential_gradient_all_degrees():
    for n_max in range(2, 7):
        grad = jax.grad(lambda r, n=n_max: _potential(r, n))
        for r in POSITIONS:
            a_ref = -np.asarray(grad(r))
            a = np.asarray(accel_zonal_gravity(r, n_max))
            np.testing.assert_allclose(a, a_ref, rtol=1e-10)


def test_each_degree_in_isolation():
    # The cumulative test above checks point mass + J2..Jn together, so a
    # small error in a high degree sits under the far larger lower terms.
    # Difference out everything below n and compare against the gradient of
    # that single potential term, which puts each degree on its own footing.
    for n in range(2, 7):
        term = jax.grad(
            lambda r, m=n: GM_EARTH / jnp.linalg.norm(r)
            * _J[m] * (R_EARTH / jnp.linalg.norm(r)) ** m
            * _LEGENDRE[m](r[2] / jnp.linalg.norm(r))
        )
        for r in POSITIONS:
            a_ref = -np.asarray(term(r))
            a = np.asarray(accel_zonal_gravity(r, n)) - np.asarray(
                accel_zonal_gravity(r, n - 1)
            )
            # atol is the cancellation floor: the difference extracts a term
            # of order 1e-6 m/s^2 from two totals of order 10 m/s^2, whose
            # f64 ulp is ~2e-15. rtol still binds on the larger components.
            np.testing.assert_allclose(a, a_ref, rtol=1e-9, atol=1e-14)


def test_zonal_constants_match_egm2008():
    # The gradient tests read the same J_n the implementation does, so they
    # validate the closed forms but not the constants. Recover J_n from the
    # EGM2008 coefficients astrojax ships: J_n = -C_n0 * sqrt(2n + 1).
    path = importlib.resources.files("astrojax.data.gravity_models") / "EGM2008_360.gfc"
    c_n0 = {}
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) > 3 and f[0] == "gfc" and f[2] == "0" and f[1].isdigit():
            if 3 <= int(f[1]) <= 6:
                c_n0[int(f[1])] = float(f[3])
    assert set(c_n0) == {3, 4, 5, 6}
    for n, c in c_n0.items():
        assert _J[n] == -c * math.sqrt(2 * n + 1), n


def test_degree_below_two_is_point_mass():
    r = POSITIONS[2]
    rn = np.linalg.norm(np.asarray(r))
    pm = -GM_EARTH * np.asarray(r) / rn**3
    for n_max in (0, 1):
        np.testing.assert_allclose(
            np.asarray(accel_zonal_gravity(r, n_max)), pm, rtol=1e-14
        )


def test_j2_magnitude_at_iss_altitude():
    r = POSITIONS[0]
    a2 = np.asarray(accel_zonal_gravity(r, 2))
    pm = np.asarray(accel_zonal_gravity(r, 0))
    ratio = np.linalg.norm(a2 - pm) / np.linalg.norm(pm)
    assert 1.0e-3 < ratio < 2.0e-3  # ~1.5 * J2 * (Re/r)^2 ~= 1.4e-3


def test_output_is_float64():
    a = accel_zonal_gravity(POSITIONS[2], 6)
    assert a.dtype == jnp.float64


def test_jits_with_static_n_max():
    f = jax.jit(accel_zonal_gravity, static_argnums=1)
    np.testing.assert_allclose(
        np.asarray(f(POSITIONS[2], 6)),
        np.asarray(accel_zonal_gravity(POSITIONS[2], 6)),
        rtol=1e-14,
    )
