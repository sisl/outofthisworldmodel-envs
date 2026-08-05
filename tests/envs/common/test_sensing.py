import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError

from owm_envs.core.quaternion import quat_conjugate, quat_multiply
from owm_envs.envs.common.sensing import PRESETS, SensorNoiseConfig, apply_sensor_noise
from owm_envs.envs.iss.config import ISSConfig


def _state(pos=(60.0, -30.0, 20.0)):
    q = np.array([0.7071068, 0.0, 0.7071068, 0.0], dtype=np.float32)
    return jnp.asarray(
        np.concatenate([np.array(pos, np.float32), np.array([0.1, -0.2, 0.3], np.float32),
                        q, np.array([0.01, 0.02, -0.03], np.float32)]))


def _samples(noise, n, pos=(60.0, -30.0, 20.0), seed=0):
    """n independent measurements of a fixed true state."""
    s = _state(pos)
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    return s, np.asarray(jax.vmap(lambda k: apply_sensor_noise(s, k, noise))(keys))


def _rotation_angle_rad(q_true: jnp.ndarray, q_meas: jnp.ndarray) -> np.ndarray:
    """Total rotation angle (rad) between batches of true/measured quaternions.

    Uses atan2(|sin(theta/2)|, |cos(theta/2)|) rather than arccos: for the
    ~1e-5 rad angles these noise sigmas produce, cos(theta/2) rounds to
    exactly 1.0 in float32 and arccos loses the signal entirely, while atan2
    stays accurate down to the sine term's own precision.
    """
    def _angle(qt, qm):
        q_err = quat_multiply(quat_conjugate(qt), qm)
        sin_half = jnp.linalg.norm(q_err[1:4])
        cos_half = jnp.abs(q_err[0])
        return 2.0 * jnp.arctan2(sin_half, cos_half)

    return np.asarray(jax.vmap(_angle)(q_true, q_meas))


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


def test_scalar_sigma_is_the_rms_of_the_total_error():
    """A scalar sigma is the RMS of the Euclidean-norm error, not a per-axis
    std -- so the sampled norm of the error, not its per-component std,
    should match the preset value."""
    noise = PRESETS["cooperative"]
    s, out = _samples(noise, 4096, seed=2)

    def total_rms(true_slice, meas_slice):
        err = meas_slice - np.asarray(true_slice)
        return np.sqrt(np.mean(np.sum(err**2, axis=1)))

    assert total_rms(s[0:3], out[:, 0:3]) == pytest.approx(0.05, rel=0.1)
    assert total_rms(s[3:6], out[:, 3:6]) == pytest.approx(0.002, rel=0.1)
    assert total_rms(s[10:13], out[:, 10:13]) == pytest.approx(1e-5, rel=0.1)


def test_scalar_sigma_is_the_rms_of_the_total_rotation_angle():
    noise = PRESETS["cooperative"]
    s, out = _samples(noise, 4096, seed=3)
    q_true = jnp.broadcast_to(s[6:10], out[:, 6:10].shape)
    angle = _rotation_angle_rad(q_true, jnp.asarray(out[:, 6:10]))
    assert np.sqrt(np.mean(angle**2)) == pytest.approx(5e-5, rel=0.1)


def test_vector_sigma_gives_explicit_per_axis_stds():
    noise = SensorNoiseConfig(enabled=True, sigma_pos_m=(0.03, 0.06, 0.09))
    s, out = _samples(noise, 4096, seed=4)
    err = out[:, 0:3] - np.asarray(s[0:3])
    np.testing.assert_allclose(err.std(axis=0), [0.03, 0.06, 0.09], rtol=0.1)


def test_position_noise_scales_with_range_for_noncooperative():
    noise = PRESETS["noncooperative"]

    def total_rms(range_m):
        s, out = _samples(noise, 4096, pos=(range_m, 0.0, 0.0), seed=1)
        err = out[:, 0:3] - np.asarray(s[0:3])
        return np.sqrt(np.mean(np.sum(err**2, axis=1)))

    near, far = total_rms(10.0), total_rms(100.0)
    # total-RMS = frac * range: 1 m at 100 m vs 0.1 m at 10 m.
    assert far / near == pytest.approx(10.0, rel=0.15)
    assert far == pytest.approx(1.0, rel=0.1)


def test_negative_scalar_sigma_is_rejected():
    with pytest.raises(ValidationError):
        SensorNoiseConfig(sigma_pos_m=-0.1)


def test_negative_component_in_tuple_sigma_is_rejected():
    with pytest.raises(ValidationError):
        SensorNoiseConfig(sigma_att_rad=(0.1, -0.2, 0.3))


def test_two_component_tuple_sigma_is_rejected():
    with pytest.raises(ValidationError):
        SensorNoiseConfig(sigma_vel_m_s=(0.1, 0.2))


def test_iss_config_carries_sensor_noise_default_off(tmp_path):
    cfg = ISSConfig()
    assert cfg.sensor_noise.enabled is False
    path = tmp_path / "cfg.yaml"
    ISSConfig(sensor_noise=PRESETS["noncooperative"]).to_yaml(path)
    assert ISSConfig.from_yaml(path).sensor_noise == PRESETS["noncooperative"]


def test_sensor_noise_config_yaml_round_trips_a_vector_valued_field(tmp_path):
    noise = SensorNoiseConfig(enabled=True, sigma_att_rad=(1e-5, 2e-5, 3e-5))
    path = tmp_path / "noise.yaml"
    noise.to_yaml(path)
    assert SensorNoiseConfig.from_yaml(path) == noise
