import numpy as np

from owm_envs.drivers.scan_driver import ScanDriver
from owm_envs.drivers.types import RolloutSpec
from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss.policies import PolicyConfig
from owm_envs.envs.iss.sensing import PRESETS


def test_scan_driver_records_noisy_observations_but_true_dynamics():
    cfg = ISSConfig(max_steps=12, sensor_noise=PRESETS["cooperative"])
    clean = ScanDriver(cfg=ISSConfig(max_steps=12), policy_cfg=PolicyConfig(type="dock"), num_envs=2)
    noisy = ScanDriver(cfg=cfg, policy_cfg=PolicyConfig(type="dock"), num_envs=2)
    spec = RolloutSpec(num_episodes=2, max_steps=12, seed=0)
    b_clean, b_noisy = clean.generate(spec), noisy.generate(spec)
    # observe="state" (default): the flown trajectory is identical, only the
    # recorded observations are corrupted.
    np.testing.assert_array_equal(b_clean.actions, b_noisy.actions)
    assert not np.array_equal(b_clean.observations, b_noisy.observations)
    err = np.abs(b_noisy.observations[:, :, 0:3] - b_clean.observations[:, :, 0:3])
    assert err[b_noisy.observations[:, :, 0:3] != 0].mean() < 1.0  # noise-sized, not dynamics-sized


def test_scan_driver_measurement_policy_flies_differently():
    cfg = ISSConfig(max_steps=12, sensor_noise=PRESETS["noncooperative"])
    state_pol = ScanDriver(cfg=cfg, policy_cfg=PolicyConfig(type="dock", observe="state"), num_envs=2)
    meas_pol = ScanDriver(cfg=cfg, policy_cfg=PolicyConfig(type="dock", observe="measurement"), num_envs=2)
    spec = RolloutSpec(num_episodes=2, max_steps=12, seed=0)
    a, b = state_pol.generate(spec), meas_pol.generate(spec)
    assert not np.array_equal(a.actions, b.actions)


def test_scan_driver_noise_off_is_bit_identical_to_before():
    cfg = ISSConfig(max_steps=12)
    drv = ScanDriver(cfg=cfg, policy_cfg=PolicyConfig(type="dock"), num_envs=2)
    spec = RolloutSpec(num_episodes=2, max_steps=12, seed=3)
    a, b = drv.generate(spec), drv.generate(spec)
    np.testing.assert_array_equal(a.observations, b.observations)
