import numpy as np
import pytest

from owm_envs.drivers.scan_driver import ScanDriver, supports_fused_rollout
from owm_envs.drivers.types import RolloutSpec
from owm_envs.envs.iss.config import DockConfig, ISSConfig, PhysicsConfig
from owm_envs.envs.iss.dynamics import ISSDynamics
from owm_envs.envs.iss.policies import PolicyConfig

FREE_FLIGHT_PHYSICS = dict(collision_boxes_path=None)
FREE_FLIGHT_DOCK = dict(enabled=False)


def make_driver(num_envs=2, policy_type="dock", physics=None, dock=None):
    cfg = ISSConfig(
        physics=PhysicsConfig(**{**FREE_FLIGHT_PHYSICS, **(physics or {})}),
        dock=DockConfig(**{**FREE_FLIGHT_DOCK, **(dock or {})}),
    )
    return ScanDriver(cfg=cfg, policy_cfg=PolicyConfig(type=policy_type), num_envs=num_envs)


def test_iss_dynamics_advertises_fused_rollout_support():
    assert supports_fused_rollout(ISSDynamics(ISSConfig())) is True


def test_an_opaque_backend_does_not_advertise_support():
    class OpaqueBackend:
        supports_fused_rollout = False

    assert supports_fused_rollout(OpaqueBackend()) is False


def test_generates_the_requested_number_of_episodes():
    batch = make_driver().generate(RolloutSpec(num_episodes=4, max_steps=20, seed=0))
    batch.validate()
    assert batch.num_episodes == 4


def test_output_shapes_and_dtypes():
    batch = make_driver().generate(RolloutSpec(num_episodes=3, max_steps=15, seed=0))
    assert batch.observations.shape[0] == 3
    assert batch.observations.shape[2] == 13
    assert batch.actions.shape[2] == 6
    assert batch.observations.dtype == np.float32
    assert batch.lengths.dtype == np.int32


def test_free_flight_episodes_truncate_at_max_steps():
    # 12 steps -> 13 observations, matching VectorEnvDriver's convention.
    batch = make_driver().generate(RolloutSpec(num_episodes=2, max_steps=12, seed=0))
    assert np.all(batch.lengths == 13)
    assert np.all(batch.truncated)
    assert not np.any(batch.terminated)


def test_collision_terminates_episodes_early():
    driver = make_driver(
        physics=dict(
            collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [400.0, 400.0, 400.0]}]
        ),
        dock=dict(enabled=False),
    )
    batch = driver.generate(RolloutSpec(num_episodes=2, max_steps=50, seed=0))
    batch.validate()
    assert np.all(batch.terminated)
    assert np.all(batch.lengths < 50)


def test_is_deterministic_in_the_seed():
    a = make_driver().generate(RolloutSpec(num_episodes=3, max_steps=10, seed=7))
    b = make_driver().generate(RolloutSpec(num_episodes=3, max_steps=10, seed=7))
    np.testing.assert_array_equal(a.observations, b.observations)


def test_union_policy_records_policy_ids():
    batch = make_driver(policy_type="union").generate(
        RolloutSpec(num_episodes=8, max_steps=10, seed=0)
    )
    assert batch.policy_ids is not None
    assert set(np.unique(batch.policy_ids)).issubset({0, 1, 2})


def test_rejects_a_non_positive_episode_count():
    with pytest.raises(ValueError):
        make_driver().generate(RolloutSpec(num_episodes=0, max_steps=10, seed=0))
