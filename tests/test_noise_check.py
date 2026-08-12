"""Residual math behind scripts/check_sensor_noise.py.

The scripts directory is not a package; the script is imported by path so the
math it exits non-zero on is covered by the suite rather than only by eye.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astrojax.constants import GM_EARTH

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_sensor_noise import expected_sigmas, report_split, residual_stats  # noqa: E402
from run_view import observed_views, state_views  # noqa: E402

from owm_envs.envs import ENV_REGISTRY  # noqa: E402
from owm_envs.envs.common.sensing import PRESETS, apply_sensor_noise  # noqa: E402
from owm_envs.envs.iss.config import ISSConfig  # noqa: E402
from owm_envs.envs.iss_numerical.config import NUM_LAYOUT, NumericalConfig  # noqa: E402
from owm_envs.envs.iss_numerical.dynamics import chaser_state_from_view  # noqa: E402
from owm_envs.envs.iss_numerical.observe import make_observe  # noqa: E402


def _identity_truth(n: int) -> np.ndarray:
    truth = np.zeros((n, 13), dtype=np.float64)
    truth[:, 0] = 50.0
    truth[:, 6] = 1.0
    return truth


def test_residuals_recover_known_sigmas():
    rng = np.random.default_rng(0)
    n = 200_000
    truth = _identity_truth(n)
    obs = truth.copy()
    obs[:, 0:3] += rng.normal(0, 0.05 / np.sqrt(3), (n, 3))   # 0.05 m total-RMS
    obs[:, 3:6] += rng.normal(0, 0.002 / np.sqrt(3), (n, 3))
    stats = residual_stats(obs, truth)
    assert abs(stats["pos_rms_m"] - 0.05) < 0.001
    assert abs(stats["vel_rms_m_s"] - 0.002) < 0.0001
    assert stats["att_rms_rad"] < 1e-9   # no attitude noise injected


def test_rate_residual_is_the_last_three_dims():
    rng = np.random.default_rng(1)
    n = 200_000
    truth = _identity_truth(n)
    obs = truth.copy()
    obs[:, 10:13] += rng.normal(0, 1e-5 / np.sqrt(3), (n, 3))
    stats = residual_stats(obs, truth)
    assert abs(stats["rate_rms_rad_s"] - 1e-5) < 5e-7
    assert stats["pos_rms_m"] == 0.0


def test_attitude_residual_is_rotation_angle():
    theta = 0.01
    truth = np.zeros((1, 13)); truth[0, 6] = 1.0
    obs = truth.copy()
    obs[0, 6] = np.cos(theta / 2); obs[0, 7] = np.sin(theta / 2)
    stats = residual_stats(obs, truth)
    assert abs(stats["att_rms_rad"] - theta) < 1e-9


def test_attitude_residual_ignores_quaternion_sign():
    """q and -q are the same rotation; a residual that read them as opposite
    would report pi radians of attitude error on a noise-free run."""
    theta = 0.01
    truth = np.zeros((1, 13)); truth[0, 6] = 1.0
    obs = truth.copy()
    obs[0, 6] = -np.cos(theta / 2); obs[0, 7] = -np.sin(theta / 2)
    stats = residual_stats(obs, truth)
    assert abs(stats["att_rms_rad"] - theta) < 1e-9


def test_attitude_residual_survives_float32_storage():
    """Datasets store quaternions as float32 and the cooperative preset's
    attitude sigma is 5e-5 rad, whose cosine is 1 - 3e-10 -- far inside
    float32's spacing at 1.0. A residual read as arccos of the dot product
    recovers only that quantization (on a real split it overstates the RMS
    by 13x); the angle has to survive the storage precision.
    """
    rng = np.random.default_rng(2)
    n = 20_000
    sigma = 5e-5

    quat = rng.normal(size=(n, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    rotvec = rng.normal(0, sigma / np.sqrt(3), (n, 3))
    angle = np.linalg.norm(rotvec, axis=1, keepdims=True)
    delta = np.concatenate([np.cos(angle / 2), np.sin(angle / 2) * rotvec / angle], axis=1)
    # Hamilton product quat (x) delta, matching sensing.apply_sensor_noise.
    w = quat[:, 0:1] * delta[:, 0:1] - np.sum(quat[:, 1:] * delta[:, 1:], axis=1, keepdims=True)
    v = (quat[:, 0:1] * delta[:, 1:] + delta[:, 0:1] * quat[:, 1:]
         + np.cross(quat[:, 1:], delta[:, 1:]))

    truth = np.zeros((n, 13), dtype=np.float32)
    truth[:, 6:10] = quat
    obs = truth.copy()
    obs[:, 6:10] = np.concatenate([w, v], axis=1)

    stats = residual_stats(obs, truth)
    assert abs(stats["att_rms_rad"] - sigma) < 0.02 * sigma


def test_pos_error_by_range_tracks_range_proportional_noise():
    rng = np.random.default_rng(3)
    n = 60_000
    truth = np.zeros((n, 13), dtype=np.float64)
    truth[:, 0] = rng.uniform(0.0, 100.0, n)
    truth[:, 6] = 1.0
    frac = 0.01
    sigma = frac * truth[:, 0:1] / np.sqrt(3)
    obs = truth.copy()
    obs[:, 0:3] += sigma * rng.normal(size=(n, 3))

    rows = residual_stats(obs, truth)["pos_err_by_range"]
    assert len(rows) == 5
    for row in rows:
        expected = frac * row["range_rms_m"]
        assert abs(row["pos_rms_m"] - expected) < 0.05 * expected
    assert rows[-1]["pos_rms_m"] > 5 * rows[0]["pos_rms_m"]


def test_pos_error_by_range_covers_the_widest_range():
    """The farthest samples belong to a bin: an exclusive upper edge on the
    last bin silently drops exactly the rows the range term matters most for.
    """
    rng = np.random.default_rng(4)
    n = 10_000
    truth = np.zeros((n, 13), dtype=np.float64)
    truth[:, 0] = 100.0                       # every row sits on the top edge
    truth[:, 6] = 1.0
    obs = truth.copy()
    obs[:, 0:3] += rng.normal(0, 1.0 / np.sqrt(3), (n, 3))

    rows = residual_stats(obs, truth)["pos_err_by_range"]
    assert len(rows) == 1
    assert rows[0]["count"] == n
    assert abs(rows[0]["pos_rms_m"] - 1.0) < 0.05


def test_residual_stats_uses_the_first_13_observation_dims():
    """A goal-error run's observation is 25-dim against a 13-dim state."""
    truth = _identity_truth(10)
    obs = np.concatenate([truth, np.arange(120.0).reshape(10, 12)], axis=1)
    stats = residual_stats(obs, truth)
    assert stats["pos_rms_m"] == 0.0
    assert stats["att_rms_rad"] == 0.0


def test_residuals_recover_numerical_env_noise_through_the_relative_view():
    """iss-numerical draws its noise on the raw 21D ECI state and records the
    RELATIVE view of the measurement; the residual between the recorded view
    and the true state's view must still carry the configured sigmas, which is
    the identity the whole check rests on for that env.
    """
    spec = ENV_REGISTRY["iss-numerical"]
    cfg = NumericalConfig(sensor_noise=PRESETS["noncooperative"])

    view = jnp.asarray([60.0, -80.0, 0.0, 0.05, 0.0, 0.0,
                        1.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0], jnp.float64)
    sma = 6.795e6
    chief = jnp.asarray([sma, 0.0, 0.0, 0.0, float(np.sqrt(GM_EARTH / sma)), 0.0],
                        jnp.float64)
    epoch = jnp.asarray([2460000.5, 0.0], jnp.float64)
    state = jnp.concatenate([epoch, chief, chaser_state_from_view(chief, view)])
    range_m = float(jnp.linalg.norm(view[0:3]))   # 100 m

    n = 20_000
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    noisy = jax.vmap(lambda k: apply_sensor_noise(
        state, k, cfg.sensor_noise, layout=NUM_LAYOUT, range_m=range_m
    ))(keys)
    obs = np.asarray(jax.vmap(make_observe(cfg))(noisy))
    truth = np.tile(np.asarray(state), (n, 1))

    stats = residual_stats(observed_views(spec, cfg, obs), state_views(spec, truth))
    expected = expected_sigmas(cfg, range_rms=range_m)
    assert stats["pos_rms_m"] == pytest.approx(expected["pos_rms_m"], rel=0.02)
    assert stats["vel_rms_m_s"] == pytest.approx(expected["vel_rms_m_s"], rel=0.02)
    assert stats["att_rms_rad"] == pytest.approx(expected["att_rms_rad"], rel=0.02)
    assert stats["rate_rms_rad_s"] == pytest.approx(expected["rate_rms_rad_s"], rel=0.02)


def test_expected_sigmas_reads_the_preset():
    cfg = ISSConfig(sensor_noise=PRESETS["cooperative"])
    expected = expected_sigmas(cfg, range_rms=100.0)
    assert expected["pos_rms_m"] == pytest.approx(0.05)
    assert expected["vel_rms_m_s"] == pytest.approx(0.002)
    assert expected["att_rms_rad"] == pytest.approx(5e-5)
    assert expected["rate_rms_rad_s"] == pytest.approx(1e-5)


def test_expected_sigmas_combines_the_range_term_in_quadrature():
    cfg = ISSConfig(sensor_noise=PRESETS["noncooperative"].model_copy(
        update={"sigma_pos_m": 0.3}
    ))
    expected = expected_sigmas(cfg, range_rms=40.0)
    assert expected["pos_rms_m"] == pytest.approx(np.hypot(0.3, 0.4))


def test_expected_sigmas_is_zero_when_noise_is_disabled():
    """A config can carry sigmas with enabled=False; nothing is applied then."""
    cfg = ISSConfig(sensor_noise=PRESETS["cooperative"].model_copy(
        update={"enabled": False}
    ))
    assert set(expected_sigmas(cfg, range_rms=100.0).values()) == {0.0}


def _stats(pos=0.05, vel=0.002, att=5e-5, rate=1e-5, by_range=()):
    """A residual_stats-shaped dict, defaulting to the cooperative sigmas."""
    return {
        "pos_rms_m": pos, "vel_rms_m_s": vel, "att_rms_rad": att,
        "rate_rms_rad_s": rate, "pos_err_by_range": list(by_range),
        "range_rms_m": 100.0, "count": 1000,
    }


def _coop() -> ISSConfig:
    return ISSConfig(sensor_noise=PRESETS["cooperative"])


def test_report_split_passes_a_run_that_matches_its_config():
    assert report_split("train", _stats(), _coop(), 0.1) is True


@pytest.mark.parametrize(
    ("channel", "measured"),
    [("pos", 0.06), ("vel", 0.0024), ("att", 6e-5), ("rate", 1.2e-5)],
)
def test_report_split_fails_any_channel_past_the_tolerance(channel, measured):
    """Every channel has to reach the verdict -- one dropped from the fold
    would let a whole broken measurement channel through the gate."""
    assert report_split("train", _stats(**{channel: measured}), _coop(), 0.1) is False


def test_report_split_accepts_a_deviation_inside_the_tolerance():
    assert report_split("train", _stats(pos=0.054), _coop(), 0.1) is True
    assert report_split("train", _stats(pos=0.056), _coop(), 0.1) is False


def test_report_split_holds_a_no_noise_run_to_exact_zeros():
    cfg = ISSConfig(sensor_noise=PRESETS["off"])
    assert report_split("train", _stats(0.0, 0.0, 0.0, 0.0), cfg, 0.1) is True
    # No tolerance covers a channel nothing was injected on, however tiny.
    assert report_split("train", _stats(1e-12, 0.0, 0.0, 0.0), cfg, 1.0) is False


def test_report_split_fails_on_a_range_bin_alone():
    """The aggregate can sit on its expectation while the range profile is
    wrong, which is the whole reason the bins are computed."""
    cfg = ISSConfig(sensor_noise=PRESETS["noncooperative"])
    bins = [{"range_lo": 0.0, "range_hi": 50.0, "range_rms_m": 40.0,
             "count": 500, "pos_rms_m": 0.4}]
    stats = _stats(pos=1.0, vel=0.03, by_range=bins)   # 1% of the 100 m range RMS
    assert report_split("train", stats, cfg, 0.1) is True
    bins[0]["pos_rms_m"] = 0.8
    assert report_split("train", stats, cfg, 0.1) is False


def test_expected_sigmas_norms_a_per_axis_sigma():
    """A 3-tuple sigma is per-axis; its total-RMS is the norm."""
    cfg = ISSConfig(sensor_noise=PRESETS["cooperative"].model_copy(
        update={"sigma_vel_m_s": (0.1, 0.2, 0.2)}
    ))
    assert expected_sigmas(cfg, range_rms=0.0)["vel_rms_m_s"] == pytest.approx(0.3)
