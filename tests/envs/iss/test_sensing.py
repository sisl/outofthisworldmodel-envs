import jax
import jax.numpy as jnp
import numpy as np
import pytest

from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss.sensing import PRESETS, SensorNoiseConfig, apply_sensor_noise


def _state(pos=(60.0, -30.0, 20.0)):
    q = np.array([0.7071068, 0.0, 0.7071068, 0.0], dtype=np.float32)
    return jnp.asarray(
        np.concatenate([np.array(pos, np.float32), np.array([0.1, -0.2, 0.3], np.float32),
                        q, np.array([0.01, 0.02, -0.03], np.float32)]))


def test_disabled_noise_is_the_identity():
    s = _state()
    out = apply_sensor_noise(s, jax.random.PRNGKey(0), SensorNoiseConfig())
    np.testing.assert_array_equal(np.asarray(out), np.asarray(s))


def test_noise_is_deterministic_per_key():
    noise = PRESETS["cooperative"]
    a = apply_sensor_noise(_state(), jax.random.PRNGKey(7), noise)
    b = apply_sensor_noise(_state(), jax.random.PRNGKey(7), noise)
    c = apply_sensor_noise(_state(), jax.random.PRNGKey(8), noise)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a), np.asarray(c))


def test_measurement_quaternion_stays_unit_and_same_hemisphere():
    noise = PRESETS["cooperative"]
    keys = jax.random.split(jax.random.PRNGKey(0), 256)
    out = jax.vmap(lambda k: apply_sensor_noise(_state(), k, noise))(keys)
    q = np.asarray(out[:, 6:10])
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-5)
    q_true = np.asarray(_state()[6:10])
    assert np.all(q @ q_true > 0.0)  # same hemisphere as the true attitude


def test_position_noise_scales_with_range_for_noncooperative():
    noise = PRESETS["noncooperative"]
    keys = jax.random.split(jax.random.PRNGKey(1), 4096)

    def pos_err(pos):
        s = _state(pos)
        out = jax.vmap(lambda k: apply_sensor_noise(s, k, noise))(keys)
        return np.asarray(out[:, :3] - s[None, :3]).std()

    near, far = pos_err((10.0, 0.0, 0.0)), pos_err((100.0, 0.0, 0.0))
    # sigma = frac * range: 1 m at 100 m vs 0.1 m at 10 m
    assert far / near == pytest.approx(10.0, rel=0.15)
    assert far == pytest.approx(1.0, rel=0.15)


def test_component_sigmas_match_the_cooperative_preset():
    noise = PRESETS["cooperative"]
    s = _state()
    keys = jax.random.split(jax.random.PRNGKey(2), 4096)
    out = np.asarray(jax.vmap(lambda k: apply_sensor_noise(s, k, noise))(keys))
    assert (out[:, 0:3] - np.asarray(s[0:3])).std() == pytest.approx(0.05, rel=0.15)
    assert (out[:, 3:6] - np.asarray(s[3:6])).std() == pytest.approx(0.002, rel=0.15)
    assert (out[:, 10:13] - np.asarray(s[10:13])).std() == pytest.approx(1e-5, rel=0.15)


def test_iss_config_carries_sensor_noise_default_off(tmp_path):
    cfg = ISSConfig()
    assert cfg.sensor_noise.enabled is False
    path = tmp_path / "cfg.yaml"
    ISSConfig(sensor_noise=PRESETS["noncooperative"]).to_yaml(path)
    assert ISSConfig.from_yaml(path).sensor_noise == PRESETS["noncooperative"]
