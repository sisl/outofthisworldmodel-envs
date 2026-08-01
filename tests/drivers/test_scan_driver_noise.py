import numpy as np

from owm_envs.drivers.scan_driver import ScanDriver
from owm_envs.drivers.types import RolloutSpec
from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss.policies import PolicyConfig
from owm_envs.envs.iss.sensing import PRESETS


def test_scan_driver_records_noisy_observations_but_true_dynamics():
    cfg = ISSConfig(max_steps=12, sensor_noise=PRESETS["cooperative"])
    clean = ScanDriver(cfg=ISSConfig(max_steps=12), policy_cfg=PolicyConfig(type="dock"), num_envs=2)
    noisy = ScanDriver(cfg=cfg, policy_cfg=PolicyConfig(type="dock", observe="state"), num_envs=2)
    spec = RolloutSpec(num_episodes=2, max_steps=12, seed=0)
    b_clean, b_noisy = clean.generate(spec), noisy.generate(spec)
    # observe="state" (explicit): the flown trajectory is identical, only
    # the recorded observations are corrupted.
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


def test_state_policy_trajectories_shared_for_stochastic_policies_across_autoreset():
    # random policy consumes act_key; 4 episodes over 2 lanes forces autoreset.
    clean_cfg = ISSConfig(max_steps=10)
    noisy_cfg = ISSConfig(max_steps=10, sensor_noise=PRESETS["cooperative"])
    spec = RolloutSpec(num_episodes=4, max_steps=10, seed=0)
    clean = ScanDriver(cfg=clean_cfg, policy_cfg=PolicyConfig(type="random"), num_envs=2).generate(spec)
    noisy = ScanDriver(
        cfg=noisy_cfg, policy_cfg=PolicyConfig(type="random", observe="state"), num_envs=2
    ).generate(spec)
    np.testing.assert_array_equal(clean.actions, noisy.actions)
    np.testing.assert_array_equal(clean.lengths, noisy.lengths)
    assert not np.array_equal(clean.observations, noisy.observations)


def test_default_observe_is_measurement_and_reacts_to_noise():
    # The default loop is realistic: the policy consumes the same noisy
    # measurement the dataset records, so recorded action-outcome pairs
    # carry the uncertainty of acting on an observed rather than true state.
    assert PolicyConfig().observe == "measurement"
    cfg_noisy = ISSConfig(max_steps=10, sensor_noise=PRESETS["noncooperative"])
    cfg_clean = ISSConfig(max_steps=10)
    spec = RolloutSpec(num_episodes=2, max_steps=10, seed=0)
    noisy = ScanDriver(cfg=cfg_noisy, policy_cfg=PolicyConfig(type="dock"), num_envs=2).generate(spec)
    clean = ScanDriver(cfg=cfg_clean, policy_cfg=PolicyConfig(type="dock"), num_envs=2).generate(spec)
    assert not np.array_equal(noisy.actions, clean.actions)
