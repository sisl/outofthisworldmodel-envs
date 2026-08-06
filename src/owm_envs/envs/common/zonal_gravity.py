"""Closed-form zonal-harmonic gravity, J2..J6, in ECI.

Port of brahe's accel_earth_zonal_gravity (src/orbit_dynamics/gravity.rs at
8da92e6f) to JAX. astrojax offers only full spherical harmonics, which need
a gravity-model file and an EOP-driven ECI->ECEF rotation -- too heavy for a
jitted environment step; the zonal-only closed forms depend on position
alone. `n_max` is a static Python int, branched at trace time.

J2 comes from astrojax.constants; J3..J6 are the EGM2008 values brahe uses
(J_n = -C_n0 * sqrt(2n + 1), Pavlis et al. 2012), reproduced here because
astrojax does not define them. astrojax's J2 is the JGM-3/EGM96 zero-tide
value, which differs from EGM2008's by 9e-6 relative -- 1e-8 of total
gravity, far below the model error of truncating at degree 6.

The point-mass term is written out rather than taken from astrojax's
accel_point_mass: that function casts its inputs to astrojax's own
module-wide dtype (astrojax.config, float32 by default and independent of
jax_enable_x64), which would cap this function at ~5e-8 relative accuracy.
"""

from __future__ import annotations

import jax.numpy as jnp
from astrojax.constants import GM_EARTH, J2_EARTH, R_EARTH

J3_EARTH = -2.5324105185677225e-06
J4_EARTH = -1.6198975999169731e-06
J5_EARTH = -0.22775359073083618e-06
J6_EARTH = 0.5406665762838132e-06


def accel_zonal_gravity(r_eci: jnp.ndarray, n_max: int) -> jnp.ndarray:
    """Total gravitational acceleration (point mass + zonals through n_max)."""
    r_eci = jnp.asarray(r_eci)
    i, j, k = r_eci[0], r_eci[1], r_eci[2]
    r = jnp.linalg.norm(r_eci)
    k_r = k / r
    k_r2 = k_r * k_r
    k_r4 = k_r2 * k_r2
    k_r6 = k_r4 * k_r2

    accel = -GM_EARTH * r_eci / r**3
    if n_max < 2:
        return accel

    ax = jnp.zeros((), r_eci.dtype)
    ay = jnp.zeros((), r_eci.dtype)
    az = jnp.zeros((), r_eci.dtype)

    j2_coeff = (-3.0 * J2_EARTH * GM_EARTH * R_EARTH**2) / (2.0 * r**5)
    ax += j2_coeff * (1.0 - 5.0 * k_r2) * i
    ay += j2_coeff * (1.0 - 5.0 * k_r2) * j
    az += j2_coeff * (3.0 - 5.0 * k_r2) * k

    if n_max >= 3:
        j3_coeff = (-5.0 * J3_EARTH * GM_EARTH * R_EARTH**3) / (2.0 * r**7)
        ax += j3_coeff * (3.0 * k - 7.0 * k * k_r2) * i
        ay += j3_coeff * (3.0 * k - 7.0 * k * k_r2) * j
        az += j3_coeff * (6.0 * k**2 - 7.0 * k**2 * k_r2 - (3.0 / 5.0) * r**2)

    if n_max >= 4:
        j4_coeff = (15.0 * J4_EARTH * GM_EARTH * R_EARTH**4) / (8.0 * r**7)
        ax += j4_coeff * (1.0 - 14.0 * k_r2 + 21.0 * k_r4) * i
        ay += j4_coeff * (1.0 - 14.0 * k_r2 + 21.0 * k_r4) * j
        az += j4_coeff * (5.0 - (70.0 / 3.0) * k_r2 + 21.0 * k_r4) * k

    if n_max >= 5:
        re5 = R_EARTH**5
        j5_coeff = (3.0 * J5_EARTH * GM_EARTH * re5) / (8.0 * r**9)
        ax += j5_coeff * (35.0 - 210.0 * k_r2 + 231.0 * k_r4) * i * k
        ay += j5_coeff * (35.0 - 210.0 * k_r2 + 231.0 * k_r4) * j * k
        az += j5_coeff * (105.0 - 315.0 * k_r2 + 231.0 * k_r4) * k**2 - (
            15.0 * J5_EARTH * GM_EARTH * re5
        ) / (8.0 * r**7)

    if n_max >= 6:
        j6_coeff = (-J6_EARTH * GM_EARTH * R_EARTH**6) / (16.0 * r**9)
        j6_xy = 35.0 - 945.0 * k_r2 + 3465.0 * k_r4 - 3003.0 * k_r6
        j6_z = 245.0 - 2205.0 * k_r2 + 4851.0 * k_r4 - 3003.0 * k_r6
        ax += j6_coeff * j6_xy * i
        ay += j6_coeff * j6_xy * j
        az += j6_coeff * j6_z * k

    return accel + jnp.stack([ax, ay, az])
