import jax.numpy as jnp
import numpy as np
import pytest
from astrojax import config as astrojax_config
from astrojax.constants import GM_EARTH
from astrojax.coordinates.keplerian import state_koe_to_eci
from astrojax.orbit_dynamics.third_body import moon_position, sun_position

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


def _raan_of(state: np.ndarray) -> float:
    """Right ascension of the ascending node read back out of an ECI state.
    The node lies along z_hat x h with h = r x v, i.e. at (-h_y, h_x, 0)."""
    h = np.cross(state[:3], state[3:6])
    return float(np.arctan2(h[0], -h[1]))


def test_chief_state_radius_and_period():
    """One period closes the in-plane geometry -- same radius, same speed --
    but no longer the whole state: J2 has regressed the node by
    raan_dot * period, some 0.32 deg, in the meantime. So assert the two
    halves separately. A whole-state closure would now only pass with a
    tolerance wide enough to swallow a drift of the wrong sign, and a wrong
    sign is exactly what this must keep catching."""
    ref = ReferenceOrbit(OrbitConfig())
    x0 = np.asarray(ref.chief_state_eci(0.0), dtype=np.float64)
    assert abs(np.linalg.norm(x0[:3]) - 6_795_000.0 * (1 - 0.0005)) < 5e3
    period = 2 * np.pi / ref.mean_motion
    x1 = np.asarray(ref.chief_state_eci(period), dtype=np.float64)

    # In plane, the orbit still closes. The tolerances are astrojax's
    # Keplerian conversion at its own (f32) dtype budget -- see the module
    # docstring's error budget note -- and sit far below the ~3.4e4 m the
    # node regression alone displaces the chief by over one period, so
    # neither doubles as a licence for the drift to leak in here.
    assert abs(np.linalg.norm(x1[:3]) - np.linalg.norm(x0[:3])) < 5.0
    assert abs(np.linalg.norm(x1[3:]) - np.linalg.norm(x0[3:])) < 5e-3

    # Out of plane, it does not. Reading the node back out of the states
    # pins the drift actually applied to the elements rather than merely the
    # rate that was computed, and it is signed: a prograde orbit regresses,
    # so a positive value here is the sign error.
    raan_dot, _, _ = ref.secular_rates
    assert _raan_of(x1) - _raan_of(x0) == pytest.approx(raan_dot * period, rel=1e-3)
    assert _raan_of(x1) < _raan_of(x0)


def test_chief_advances_prograde_by_a_quarter_turn():
    """The chief actually moves, and moves the right way round. A quarter
    period advances the argument of latitude by ~90 degrees (the orbit is
    near-circular, e=5e-4, so true and mean anomaly agree to ~0.06 deg), and
    the new position lies ahead along the initial velocity. Together these
    kill both a frozen chief (angle 0) and a retrograde one (negative dot)."""
    ref = ReferenceOrbit(OrbitConfig())
    period = 2 * np.pi / ref.mean_motion
    x0 = np.asarray(ref.chief_state_eci(0.0), dtype=np.float64)
    xq = np.asarray(ref.chief_state_eci(period / 4.0), dtype=np.float64)
    cos_sweep = np.dot(x0[:3], xq[:3]) / (
        np.linalg.norm(x0[:3]) * np.linalg.norm(xq[:3])
    )
    assert abs(np.degrees(np.arccos(np.clip(cos_sweep, -1.0, 1.0))) - 90.0) < 1.0
    assert np.dot(xq[:3], x0[3:]) > 0.0  # prograde: ahead along v(0)


def test_world_from_eci_maps_radial_to_plus_z():
    ref = ReferenceOrbit(OrbitConfig())
    x = ref.chief_state_eci(137.0)
    R = np.asarray(ref.world_from_eci(x))
    r_hat = np.asarray(x[:3]) / np.linalg.norm(np.asarray(x[:3]))
    # chief_state_eci is astrojax-dtype (f32); atol matches the module's
    # ~1e-7 rad direction budget rather than f64 machine precision.
    np.testing.assert_allclose(R @ r_hat, [0.0, 0.0, 1.0], atol=1e-7)


def test_public_surface_is_uniformly_astrojax_dtype():
    """Nothing here may advertise f64 for an f32-information value -- the
    world rotation used to inherit f64 from the numpy RTN_FROM_WORLD, and
    sun/moon inherit it from astrojax's ephemerides, which evaluate at the
    configured dtype but close with an `Rx` built from a Python-float
    obliquity that promotes the result. See the module docstring's dtype
    policy."""
    dt = astrojax_config.get_dtype()
    ref = ReferenceOrbit(OrbitConfig())
    x = ref.chief_state_eci(0.0)
    assert x.dtype == dt
    assert ref.world_from_eci(x).dtype == dt
    assert sun_direction_world(x, ref.epoch0).dtype == dt
    assert moon_vector_world(x, ref.epoch0).dtype == dt
    assert illumination(x[:3], ref.epoch0).dtype == dt


def test_illumination_extremes():
    ref = ReferenceOrbit(OrbitConfig())
    e = ref.epoch0
    s = sun_position(e)
    s_hat = np.asarray(s) / np.linalg.norm(np.asarray(s))
    r = 6_795_000.0
    assert float(illumination(jnp.asarray(s_hat * r), e)) == 1.0  # sun side
    assert float(illumination(jnp.asarray(-s_hat * r), e)) == 0.0  # umbra


def test_illumination_falls_smoothly_across_the_penumbra():
    """The rendered lighting reads this value straight into a light intensity,
    so a penumbra crossing has to be monotone and inside [0, 1] at the step the
    frames are rendered at -- not merely correct on average. The lens-area form
    evaluated at f32 leaves O(1) of error here, in both directions past the
    bounds, and strobes rather than fades."""
    ref = ReferenceOrbit(OrbitConfig())
    e = ref.epoch0
    sun = np.asarray(sun_position(e))
    s_hat = sun / np.linalg.norm(sun)
    # A circular chief orbit in a plane containing the sun direction, walked
    # into the shadow at the 0.05 s step the videos are rendered at.
    radius = 6_795_000.0
    across = np.cross(s_hat, [0.0, 0.0, 1.0])
    across /= np.linalg.norm(across)

    def at(theta: float) -> float:
        r = radius * (np.cos(theta) * s_hat + np.sin(theta) * across)
        return float(illumination(jnp.asarray(r, dtype=jnp.float64), e))

    # Bisect onto the shadow entry edge, then sample 400 render steps through it.
    lo, hi = np.pi / 2.0, np.pi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if at(mid) > 0.999 else (lo, mid)
    step = ref.mean_motion * 0.05
    values = np.array([at(lo - 20 * step + k * step) for k in range(400)])

    assert values.min() >= 0.0 and values.max() <= 1.0
    assert values[0] > 0.999 and values[-1] < 0.001, "sweep must cross the whole penumbra"
    assert np.all(np.diff(values) <= 1e-9), "illumination must fall monotonically"
    partial = (values > 0.0) & (values < 1.0)
    assert np.abs(np.diff(values))[partial[:-1]].max() < 0.02


def test_sun_and_moon_world_vectors():
    ref = ReferenceOrbit(OrbitConfig())
    x = ref.chief_state_eci(0.0)
    s = np.asarray(sun_direction_world(x, ref.epoch0))
    assert np.isclose(np.linalg.norm(s), 1.0)
    m = np.asarray(moon_vector_world(x, ref.epoch0))
    assert 3.4e8 < np.linalg.norm(m) < 4.2e8  # real lunar distance range


def test_moon_is_geocentric_but_sun_is_chief_relative():
    """The two helpers deliberately differ in origin: the moon vector is
    geocentric, to be hung off the Earth-center scene node, while the sun
    direction is taken at the chief, where it becomes a light direction. Pin
    both, so neither drifts toward the other's convention."""
    ref = ReferenceOrbit(OrbitConfig())
    x = ref.chief_state_eci(0.0)
    rot = np.asarray(ref.world_from_eci(x), dtype=np.float64)
    r_chief = np.asarray(x[:3], dtype=np.float64)
    moon_eci = np.asarray(moon_position(ref.epoch0), dtype=np.float64)
    sun_eci = np.asarray(sun_position(ref.epoch0), dtype=np.float64)

    moon = moon_vector_world(x, ref.epoch0)
    assert moon.dtype == astrojax_config.get_dtype()

    # Independent physical check that the origin is the Earth's centre and
    # not the chief: a rotation preserves length, so a geocentric vector must
    # come back with the geocentric distance. Subtracting the chief would
    # move it by up to |r_chief| = 6.8e6 m, ~1.8% of the lunar distance and
    # thousands of times the 1e3 m tolerance below (which only has to cover
    # the f32 narrowing, ~32 m at 3.8e8 m).
    moon_len = np.linalg.norm(np.asarray(moon, np.float64))
    assert abs(moon_len - np.linalg.norm(moon_eci)) < 1e3

    # ...and exactly, to pin that nothing but the rotation and the dtype
    # narrowing sits between moon_position and the result.
    np.testing.assert_array_equal(
        np.asarray(moon),
        np.asarray((ref.world_from_eci(x) @ moon_position(ref.epoch0)).astype(moon.dtype)),
    )

    # The sun, by contrast, IS translated to the chief. Check it against the
    # independently computed chief-relative direction, not merely against
    # "differs from geocentric by something small" -- a partial or wrong
    # translation would pass that.
    s_chief = np.asarray(sun_direction_world(x, ref.epoch0), dtype=np.float64)
    s_chief = s_chief / np.linalg.norm(s_chief)
    expected = rot @ ((sun_eci - r_chief) / np.linalg.norm(sun_eci - r_chief))
    expected = expected / np.linalg.norm(expected)
    assert np.arccos(np.clip(np.dot(s_chief, expected), -1.0, 1.0)) < 1e-6

    # The parallax against the geocentric direction is what that translation
    # buys: ~6.8e6 m at 1 AU is ~4.5e-5 rad (measured 3.5e-5 here), small but
    # strictly nonzero, so the sun helper cannot silently become geocentric.
    s_geo = rot @ (sun_eci / np.linalg.norm(sun_eci))
    s_geo = s_geo / np.linalg.norm(s_geo)
    parallax = np.arccos(np.clip(np.dot(s_chief, s_geo), -1.0, 1.0))
    assert 1e-5 < parallax < 1e-4


DAY_S = 86_400.0


def test_secular_rates_match_the_iss_reference():
    ref = ReferenceOrbit(OrbitConfig())
    raan_dot, argp_dot, m_dot = ref.secular_rates
    assert np.rad2deg(raan_dot) * DAY_S == pytest.approx(-4.955, abs=0.01)
    assert np.rad2deg(argp_dot) * DAY_S == pytest.approx(3.695, abs=0.01)
    assert np.rad2deg(m_dot - ref.mean_motion) * DAY_S == pytest.approx(0.620, abs=0.01)


def test_raan_regresses_over_a_week():
    # The seven-day epoch window exists to give episodes different orbit
    # planes. Without secular drift every one of them starts in the same one.
    ref = ReferenceOrbit(OrbitConfig())
    raan_dot, _, _ = ref.secular_rates
    assert np.rad2deg(raan_dot * 7.0 * DAY_S) == pytest.approx(-34.68, abs=0.1)


def test_drift_is_zero_at_the_epoch():
    # Every rate multiplies t, so epoch zero must be untouched.
    ref = ReferenceOrbit(OrbitConfig())
    drifted = np.asarray(ref.chief_state_eci(0.0), dtype=np.float64)
    two_body = np.asarray(state_koe_to_eci(ref._elements0), dtype=np.float64)
    np.testing.assert_allclose(drifted, two_body, rtol=0, atol=1e-6)


def test_the_orbit_plane_moves_over_a_week():
    ref = ReferenceOrbit(OrbitConfig())
    week = 7.0 * DAY_S

    def normal(t):
        x = np.asarray(ref.chief_state_eci(t), dtype=np.float64)
        n = np.cross(x[:3], x[3:6])
        return n / np.linalg.norm(n)

    angle = np.arccos(np.clip(np.dot(normal(0.0), normal(week)), -1.0, 1.0))
    # RAAN regression tilts the normal about the pole; the swept angle is
    # sin(i) * delta_raan to first order.
    expected = np.deg2rad(34.68) * np.sin(np.deg2rad(OrbitConfig().inc_deg))
    assert np.rad2deg(angle) == pytest.approx(np.rad2deg(expected), rel=0.05)
