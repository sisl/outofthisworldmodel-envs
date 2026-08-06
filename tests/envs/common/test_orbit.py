import jax.numpy as jnp
import numpy as np
from astrojax.constants import GM_EARTH

from owm_envs.envs.common.orbit import (
    RTN_FROM_WORLD,
    OrbitConfig,
    ReferenceOrbit,
    illumination,
    moon_vector_world,
    sun_direction_world,
)


def test_rtn_from_world_is_rotation():
    m = RTN_FROM_WORLD
    np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(m), 1.0)


def test_mean_motion_matches_kepler():
    ref = ReferenceOrbit(OrbitConfig())
    n = float(np.sqrt(GM_EARTH / 6_795_000.0**3))
    assert np.isclose(ref.mean_motion, n, rtol=1e-12)


def test_chief_state_radius_and_period():
    ref = ReferenceOrbit(OrbitConfig())
    x0 = np.asarray(ref.chief_state_eci(0.0))
    assert abs(np.linalg.norm(x0[:3]) - 6_795_000.0 * (1 - 0.0005)) < 5e3
    period = 2 * np.pi / ref.mean_motion
    x1 = np.asarray(ref.chief_state_eci(period))
    # astrojax's Keplerian conversion returns its own (f32) dtype budget --
    # see the module docstring's error budget note.
    np.testing.assert_allclose(x1, x0, atol=5.0)  # closes after one period


def test_world_from_eci_maps_radial_to_plus_z():
    ref = ReferenceOrbit(OrbitConfig())
    x = ref.chief_state_eci(137.0)
    R = np.asarray(ref.world_from_eci(x))
    r_hat = np.asarray(x[:3]) / np.linalg.norm(np.asarray(x[:3]))
    # chief_state_eci is astrojax-dtype (f32); atol matches the module's
    # ~1e-7 rad direction budget rather than f64 machine precision.
    np.testing.assert_allclose(R @ r_hat, [0.0, 0.0, 1.0], atol=1e-7)


def test_illumination_extremes():
    ref = ReferenceOrbit(OrbitConfig())
    e = ref.epoch0
    from astrojax.orbit_dynamics.third_body import sun_position

    s = sun_position(e)
    s_hat = np.asarray(s) / np.linalg.norm(np.asarray(s))
    r = 6_795_000.0
    assert float(illumination(jnp.asarray(s_hat * r), e)) == 1.0  # sun side
    assert float(illumination(jnp.asarray(-s_hat * r), e)) == 0.0  # umbra


def test_sun_and_moon_world_vectors():
    ref = ReferenceOrbit(OrbitConfig())
    x = ref.chief_state_eci(0.0)
    s = np.asarray(sun_direction_world(x, ref.epoch0))
    assert np.isclose(np.linalg.norm(s), 1.0)
    m = np.asarray(moon_vector_world(x, ref.epoch0))
    assert 3.4e8 < np.linalg.norm(m) < 4.2e8  # real lunar distance range
