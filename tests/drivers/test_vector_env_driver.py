import numpy as np
import pytest

from owm_envs.drivers.types import RolloutSpec
from owm_envs.drivers.vector_env_driver import VectorEnvDriver
from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss.policies import PolicyConfig
from owm_envs.envs.iss.vector_env import ISSVectorEnv

FREE_FLIGHT = dict(collision_boxes_path=None, dock_enabled=False)


def make_driver(num_envs=2, policy_type="dock", **cfg_kwargs):
    cfg = ISSConfig(**{**FREE_FLIGHT, **cfg_kwargs})
    policy_cfg = PolicyConfig(type=policy_type)
    return VectorEnvDriver(
        env_factory=lambda: ISSVectorEnv(num_envs=num_envs, cfg=cfg),
        cfg=cfg,
        policy_cfg=policy_cfg,
    )


def test_generates_the_requested_number_of_episodes():
    batch = make_driver().generate(RolloutSpec(num_episodes=4, max_steps=20, seed=0))
    batch.validate()
    assert batch.num_episodes == 4


def test_output_shapes_and_dtypes():
    # Each episode stores N + 1 observations (seed state plus each post-step
    # state, including the terminal one) against N + 1 actions (N real, one
    # zero pad), so a max_steps=15 horizon yields length-16 episodes.
    batch = make_driver().generate(RolloutSpec(num_episodes=3, max_steps=15, seed=0))
    assert batch.observations.shape == (3, 16, 13)
    assert batch.actions.shape == (3, 16, 6)
    assert batch.rewards.shape == (3, 16)
    assert batch.observations.dtype == np.float32
    assert batch.actions.dtype == np.float32
    assert batch.lengths.dtype == np.int32


def test_free_flight_episodes_run_to_max_steps_and_truncate():
    # 12 real steps plus the seed observation is a length-13 episode.
    batch = make_driver().generate(RolloutSpec(num_episodes=2, max_steps=12, seed=0))
    assert np.all(batch.lengths == 13)
    assert np.all(batch.truncated)
    assert not np.any(batch.terminated)


def test_collision_terminates_episodes_early():
    driver = make_driver(
        collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [400.0, 400.0, 400.0]}],
        dock_enabled=False,
    )
    batch = driver.generate(RolloutSpec(num_episodes=2, max_steps=50, seed=0))
    batch.validate()
    assert np.all(batch.terminated)
    assert not np.any(batch.truncated)
    assert np.all(batch.lengths < 50)


def test_terminal_state_is_stored_with_a_padded_final_action():
    # The terminal (collision) observation must be captured -- it's the
    # event a world model needs to learn -- with a zero action padding the
    # final slot so observations and actions stay equal length.
    driver = make_driver(
        collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [400.0, 400.0, 400.0]}],
        dock_enabled=False,
    )
    batch = driver.generate(RolloutSpec(num_episodes=2, max_steps=50, seed=0))
    batch.validate()
    assert np.all(batch.terminated)
    for i, length in enumerate(batch.lengths):
        initial = batch.observations[i, 0]
        terminal = batch.observations[i, length - 1]
        assert not np.allclose(terminal, 0.0)
        assert not np.allclose(terminal, initial)
        np.testing.assert_array_equal(batch.actions[i, length - 1], 0.0)


def test_is_deterministic_in_the_seed():
    a = make_driver().generate(RolloutSpec(num_episodes=3, max_steps=10, seed=7))
    b = make_driver().generate(RolloutSpec(num_episodes=3, max_steps=10, seed=7))
    c = make_driver().generate(RolloutSpec(num_episodes=3, max_steps=10, seed=8))
    np.testing.assert_array_equal(a.observations, b.observations)
    assert not np.allclose(a.observations, c.observations)


def test_padding_past_episode_length_is_zero():
    driver = make_driver(
        collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [400.0, 400.0, 400.0]}],
        dock_enabled=False,
    )
    batch = driver.generate(RolloutSpec(num_episodes=2, max_steps=40, seed=0))
    for i, length in enumerate(batch.lengths):
        assert np.all(batch.observations[i, length:] == 0.0)
        assert np.all(batch.actions[i, length:] == 0.0)


def test_union_policy_records_policy_ids():
    batch = make_driver(policy_type="union").generate(
        RolloutSpec(num_episodes=8, max_steps=10, seed=0)
    )
    assert batch.policy_ids is not None
    assert batch.policy_ids.shape == (8,)
    assert set(np.unique(batch.policy_ids)).issubset({0, 1, 2})


def test_non_mixture_policy_leaves_policy_ids_none():
    batch = make_driver(policy_type="dock").generate(RolloutSpec(num_episodes=2, max_steps=10, seed=0))
    assert batch.policy_ids is None


def test_actions_stay_within_the_control_limits():
    cfg = ISSConfig(**FREE_FLIGHT)
    batch = make_driver().generate(RolloutSpec(num_episodes=2, max_steps=10, seed=0))
    assert np.all(np.abs(batch.actions[..., 0:3]) <= cfg.control_limit_force_n + 1e-3)
    assert np.all(np.abs(batch.actions[..., 3:6]) <= cfg.control_limit_torque_nm + 1e-3)


def test_rejects_a_non_positive_episode_count():
    with pytest.raises(ValueError):
        make_driver().generate(RolloutSpec(num_episodes=0, max_steps=10, seed=0))
