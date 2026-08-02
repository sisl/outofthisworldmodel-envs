import numpy as np
import pytest
from pydantic import ValidationError

from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss.orbit import RTN_FROM_WORLD, OrbitConfig, ReferenceOrbit


def test_default_config_is_disabled():
    cfg = OrbitConfig()
    assert cfg.enabled is False
    assert ISSConfig().orbit == OrbitConfig()


def test_mean_motion_matches_the_iss_reference_value():
    # n = sqrt(mu / a^3) for the default ISS-like elements.
    orbit = ReferenceOrbit(OrbitConfig())
    assert orbit.mean_motion == pytest.approx(1.1266e-3, rel=1e-3)


def test_rtn_from_world_is_orthonormal_and_right_handed():
    m = RTN_FROM_WORLD
    np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
    r, t, n = m[0], m[1], m[2]
    np.testing.assert_allclose(np.cross(r, t), n, atol=1e-12)
    assert np.linalg.det(m) == pytest.approx(1.0)


def test_chief_radius_stays_within_apsis_bounds():
    cfg = OrbitConfig()
    orbit = ReferenceOrbit(cfg)
    period = 2 * np.pi / orbit.mean_motion
    r_min = cfg.sma_m * (1 - cfg.ecc)
    r_max = cfg.sma_m * (1 + cfg.ecc)
    for t in np.linspace(0.0, period, 17):
        r = np.linalg.norm(orbit.chief_eci_state(t)[:3])
        assert r_min - 1.0 <= r <= r_max + 1.0
        assert r == pytest.approx(cfg.sma_m, rel=2 * cfg.ecc)


def test_chief_state_is_periodic():
    orbit = ReferenceOrbit(OrbitConfig())
    period = 2 * np.pi / orbit.mean_motion
    state0 = orbit.chief_eci_state(0.0)
    state_after_one_period = orbit.chief_eci_state(period)
    # float32 two-body propagation over ~5574 s of mean anomaly; position
    # components are O(1e6) m, so a few metres of drift is expected roundoff.
    np.testing.assert_allclose(state_after_one_period, state0, atol=10.0)


def test_sun_direction_is_unit_norm():
    orbit = ReferenceOrbit(OrbitConfig())
    for t in (0.0, 1234.5, 5000.0):
        s = orbit.sun_direction_world(t)
        assert s.shape == (3,)
        assert np.linalg.norm(s) == pytest.approx(1.0, rel=1e-5)


def test_sun_direction_varies_across_the_orbit():
    # The sun's ECI direction is ~fixed over one orbit, but the LVLH/world
    # frame rotates with the chief, so the world-frame sun direction sweeps.
    orbit = ReferenceOrbit(OrbitConfig())
    period = 2 * np.pi / orbit.mean_motion
    s0 = orbit.sun_direction_world(0.0)
    s_half = orbit.sun_direction_world(period / 2.0)
    assert np.dot(s0, s_half) < 0.999


def test_epoch_offset_range_validator_rejects_bad_ranges():
    with pytest.raises(ValidationError):
        OrbitConfig(epoch_offset_range_s=(10.0, 5.0))
    with pytest.raises(ValidationError):
        OrbitConfig(epoch_offset_range_s=(-1.0, 5.0))
    OrbitConfig(epoch_offset_range_s=(0.0, 0.0))  # does not raise


def test_config_roundtrips_through_yaml(tmp_path):
    original = OrbitConfig(enabled=True, epoch_offset_range_s=(0.0, 90.0))
    path = tmp_path / "orbit.yaml"
    original.to_yaml(path)
    assert OrbitConfig.from_yaml(path) == original


def test_iss_config_roundtrip_carries_orbit(tmp_path):
    original = ISSConfig(orbit=OrbitConfig(enabled=True, sma_m=6_800_000.0))
    path = tmp_path / "run_config.yaml"
    original.to_yaml(path)
    loaded = ISSConfig.from_yaml(path)
    assert loaded == original
    assert loaded.orbit.enabled is True
