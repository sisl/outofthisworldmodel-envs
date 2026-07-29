"""The guard against the two drivers silently diverging.

seamstress has exactly this bug: its done-logic is implemented twice, once in
JAX and once in numpy (environment_parallel.py:114-121 vs :221-226), free to
drift. This port has two rollout implementations for real performance reasons,
so it must prove they agree.

Equivalence is tested with the `dock` policy, which is deterministic and ignores
its PRNG key, so the comparison isolates rollout mechanics -- stepping,
termination detection, episode segmentation, padding -- from PRNG plumbing.
Matching two independent PRNG pipelines bitwise is a tar pit and is not the
property that matters.
"""

import numpy as np
import pytest

from owm_envs.drivers.scan_driver import ScanDriver
from owm_envs.drivers.types import RolloutSpec
from owm_envs.drivers.vector_env_driver import VectorEnvDriver
from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss.policies import PolicyConfig
from owm_envs.envs.iss.vector_env import ISSVectorEnv

DETERMINISTIC = PolicyConfig(type="dock")


def drivers_for(cfg, num_envs=2):
    from owm_envs.envs.iss.policy_source import IssPolicySource

    vec = VectorEnvDriver(
        env_factory=lambda: ISSVectorEnv(num_envs=num_envs, cfg=cfg),
        policy_source=IssPolicySource(cfg, DETERMINISTIC),
    )
    scan = ScanDriver(cfg=cfg, policy_cfg=DETERMINISTIC, num_envs=num_envs)
    return vec, scan


def test_both_drivers_agree_on_free_flight_trajectories():
    cfg = ISSConfig(collision_boxes_path=None, dock_enabled=False)
    vec, scan = drivers_for(cfg)
    spec = RolloutSpec(num_episodes=2, max_steps=25, seed=0)

    a = vec.generate(spec)
    b = scan.generate(spec)

    np.testing.assert_array_equal(a.lengths, b.lengths)
    np.testing.assert_array_equal(a.terminated, b.terminated)
    np.testing.assert_array_equal(a.truncated, b.truncated)
    np.testing.assert_allclose(a.observations, b.observations, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(a.actions, b.actions, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(a.rewards, b.rewards, rtol=1e-3, atol=1e-2)


def test_both_drivers_agree_when_episodes_terminate_on_collision():
    cfg = ISSConfig(
        collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [400.0, 400.0, 400.0]}],
        dock_enabled=False,
    )
    vec, scan = drivers_for(cfg)
    spec = RolloutSpec(num_episodes=2, max_steps=50, seed=0)

    a = vec.generate(spec)
    b = scan.generate(spec)

    np.testing.assert_array_equal(a.lengths, b.lengths)
    np.testing.assert_array_equal(a.terminated, b.terminated)


def test_both_drivers_agree_on_episode_length_distribution_for_a_stochastic_policy():
    # Random actions cannot match trajectory-for-trajectory across two PRNG
    # pipelines, but a systematic difference in termination or segmentation
    # logic would still show up as a different length distribution.
    cfg = ISSConfig(collision_boxes_path=None, dock_enabled=False)
    stochastic = PolicyConfig(type="random")
    from owm_envs.envs.iss.policy_source import IssPolicySource

    vec = VectorEnvDriver(
        env_factory=lambda: ISSVectorEnv(num_envs=4, cfg=cfg),
        policy_source=IssPolicySource(cfg, stochastic),
    )
    scan = ScanDriver(cfg=cfg, policy_cfg=stochastic, num_envs=4)
    spec = RolloutSpec(num_episodes=8, max_steps=20, seed=3)

    a = vec.generate(spec)
    b = scan.generate(spec)

    # Free flight with no termination: every episode must truncate at max_steps
    # in BOTH drivers. 20 steps -> 21 observations.
    assert np.all(a.lengths == 21)
    assert np.all(b.lengths == 21)
    assert np.all(a.truncated) and np.all(b.truncated)


def test_both_drivers_produce_batches_that_validate():
    cfg = ISSConfig(collision_boxes_path=None, dock_enabled=False)
    vec, scan = drivers_for(cfg)
    spec = RolloutSpec(num_episodes=2, max_steps=15, seed=1)
    vec.generate(spec).validate()
    scan.generate(spec).validate()
