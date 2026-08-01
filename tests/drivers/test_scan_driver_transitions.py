import numpy as np

from owm_envs.drivers.scan_driver import ScanDriver
from owm_envs.drivers.types import RolloutSpec
from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss.policies import PolicyConfig


def make_driver(num_envs=2, cfg=None):
    cfg = cfg or ISSConfig(max_steps=10)
    return ScanDriver(cfg=cfg, policy_cfg=PolicyConfig(type="dock"), num_envs=num_envs)


def test_scan_transitions_target_is_met_with_whole_episodes():
    driver = make_driver(num_envs=2, cfg=ISSConfig(max_steps=10))
    batch = driver.generate(RolloutSpec(max_steps=10, seed=0, min_transitions=50))
    assert batch.total_transitions >= 50
    # first-crossing: dropping the last episode must dip below the target
    assert batch.total_transitions - (int(batch.lengths[-1]) - 1) < 50


def test_scan_transitions_mode_is_deterministic():
    def batch():
        driver = make_driver(num_envs=2, cfg=ISSConfig(max_steps=10))
        return driver.generate(RolloutSpec(max_steps=10, seed=0, min_transitions=50))

    a, b = batch(), batch()
    np.testing.assert_array_equal(a.observations, b.observations)
    np.testing.assert_array_equal(a.actions, b.actions)
    np.testing.assert_array_equal(a.lengths, b.lengths)


def test_scan_transitions_mode_spans_multiple_chunks():
    # num_envs=2, max_steps=10 -> one chunk yields at most 2 lanes x 10 steps
    # = ~20-ish transitions, so a target of 60 forces >= 3 chunks.
    cfg = ISSConfig(max_steps=10)
    driver = make_driver(num_envs=2, cfg=cfg)
    spec = RolloutSpec(max_steps=10, seed=0, min_transitions=60)

    a = driver.generate(spec)
    a.validate()
    assert a.total_transitions >= 60
    assert np.all(a.lengths <= 11)  # max_steps + 1 observations

    b = make_driver(num_envs=2, cfg=cfg).generate(spec)
    np.testing.assert_array_equal(a.observations, b.observations)
    np.testing.assert_array_equal(a.actions, b.actions)
    np.testing.assert_array_equal(a.lengths, b.lengths)
