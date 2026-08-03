import jax.numpy as jnp
import numpy as np

from owm_envs.drivers.scan_driver import ScanDriver
from owm_envs.drivers.types import RolloutSpec
from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss.goal import dock_goal_error
from owm_envs.envs.iss.policies import PolicyConfig


def test_scan_driver_records_goal_error_block():
    cfg = ISSConfig(max_steps=10, observation={"goal_error": True})
    drv = ScanDriver(cfg=cfg, policy_cfg=PolicyConfig(type="dock"), num_envs=2)
    batch = drv.generate(RolloutSpec(num_episodes=2, max_steps=10, seed=0))
    assert batch.observations.shape[-1] == 25
    row = batch.observations[0, 0]
    np.testing.assert_allclose(
        row[13:], np.asarray(dock_goal_error(jnp.asarray(row[:13]), cfg)), atol=1e-5
    )


def test_scan_driver_goal_block_is_policy_aware_for_union():
    cfg = ISSConfig(max_steps=10, observation={"goal_error": True})
    drv = ScanDriver(cfg=cfg, policy_cfg=PolicyConfig(type="union"), num_envs=4)
    batch = drv.generate(RolloutSpec(num_episodes=8, max_steps=10, seed=0))
    ids = batch.policy_ids
    assert 0 in ids  # need at least one random episode at this seed; bump episodes if not
    for e in range(batch.num_episodes):
        if ids[e] == 0:
            np.testing.assert_array_equal(
                batch.observations[e, : batch.lengths[e], 13:], 0.0
            )


def test_scan_driver_goal_off_is_13_dim():
    cfg = ISSConfig(max_steps=10)
    drv = ScanDriver(cfg=cfg, policy_cfg=PolicyConfig(type="dock"), num_envs=2)
    batch = drv.generate(RolloutSpec(num_episodes=2, max_steps=10, seed=0))
    assert batch.observations.shape[-1] == 13
