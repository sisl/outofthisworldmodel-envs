import jax.numpy as jnp
import numpy as np
import pytest

from owm_envs.drivers.scan_driver import ScanDriver
from owm_envs.drivers.types import RolloutSpec
from owm_envs.envs.common.config import dock_target
from owm_envs.envs.common.docking_ports import dock_targets
from owm_envs.envs.common.goal import dock_goal_error
from owm_envs.envs.common.policies import DockParams, PolicyConfig
from owm_envs.envs.iss.config import ISSConfig


def test_scan_driver_records_goal_error_block():
    cfg = ISSConfig(max_steps=10, observation={"goal_error": True})
    drv = ScanDriver(cfg=cfg, policy_cfg=PolicyConfig(type="dock"), num_envs=2)
    batch = drv.generate(RolloutSpec(num_episodes=2, max_steps=10, seed=0))
    assert batch.observations.shape[-1] == 25
    row = batch.observations[0, 0]
    np.testing.assert_allclose(
        row[13:], np.asarray(dock_goal_error(jnp.asarray(row[:13]), dock_target(cfg))), atol=1e-5
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


@pytest.mark.parametrize("kind", ["dock", "union"])
def test_recorded_dock_target_is_the_one_the_goal_block_used(kind):
    # The recorded row is only worth anything if it is the pose the episode
    # actually flew to, so check it against the goal-error block rather than
    # against the extras it was resolved from.
    cfg = ISSConfig(max_steps=10, observation={"goal_error": True})
    policy_cfg = PolicyConfig(type=kind, dock=DockParams(ports=("all",)))
    drv = ScanDriver(cfg=cfg, policy_cfg=policy_cfg, num_envs=4)
    batch = drv.generate(RolloutSpec(num_episodes=8, max_steps=10, seed=0))

    assert batch.dock_targets is not None
    assert batch.dock_targets.shape == (8, 7)

    table = dock_targets(("all",))
    for episode in range(batch.num_episodes):
        recorded = batch.dock_targets[episode]
        # Every recorded row is one of the configured ports' poses -- the
        # index the episode drew, without storing that index.
        assert np.isclose(table, recorded, atol=1e-5).all(axis=1).sum() == 1

        if batch.policy_ids is not None and batch.policy_ids[episode] != 2:
            continue  # union episode that did not run the dock branch
        row = batch.observations[episode, 0]
        expected = dock_goal_error(jnp.asarray(row[:13]), jnp.asarray(recorded))
        np.testing.assert_allclose(row[13:], np.asarray(expected), atol=1e-5)


def test_no_port_set_records_the_config_dock_row():
    cfg = ISSConfig(max_steps=10, observation={"goal_error": True})
    drv = ScanDriver(cfg=cfg, policy_cfg=PolicyConfig(type="dock"), num_envs=2)
    batch = drv.generate(RolloutSpec(num_episodes=2, max_steps=10, seed=0))
    assert batch.dock_targets is not None
    for episode in range(batch.num_episodes):
        np.testing.assert_allclose(batch.dock_targets[episode], dock_target(cfg), atol=1e-6)


def test_scan_driver_goal_off_is_13_dim():
    cfg = ISSConfig(max_steps=10)
    drv = ScanDriver(cfg=cfg, policy_cfg=PolicyConfig(type="dock"), num_envs=2)
    batch = drv.generate(RolloutSpec(num_episodes=2, max_steps=10, seed=0))
    assert batch.observations.shape[-1] == 13
