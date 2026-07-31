import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import owm_envs.envs  # noqa: F401  -- triggers registration
from owm_envs.envs.iss.config import DockConfig, ISSConfig, PhysicsConfig
from owm_envs.envs.iss.env import ISSEnv


def test_passes_the_gymnasium_env_checker():
    check_env(ISSEnv(), skip_render_check=True)


def test_registered_id_constructs():
    env = gym.make("ISS-Docking-v0")
    assert env is not None
    env.close()


def test_observation_is_13d_float32():
    env = ISSEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (13,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)


def test_action_space_matches_configured_limits():
    cfg = ISSConfig()
    env = ISSEnv(cfg)
    assert env.action_space.shape == (6,)
    np.testing.assert_allclose(env.action_space.high[0:3], cfg.control.limit_force_n)
    np.testing.assert_allclose(env.action_space.high[3:6], cfg.control.limit_torque_nm)


def test_reset_is_reproducible_with_the_same_seed():
    a, _ = ISSEnv().reset(seed=42)
    b, _ = ISSEnv().reset(seed=42)
    c, _ = ISSEnv().reset(seed=43)
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, c)


def test_info_always_reports_success_and_collision():
    env = ISSEnv()
    _, info = env.reset(seed=0)
    assert info["success"] is False and info["collision"] is False
    _, _, _, _, info = env.step(env.action_space.sample())
    assert isinstance(info["success"], bool)
    assert isinstance(info["collision"], bool)


def test_truncates_at_max_steps_without_terminating():
    # No collision boxes and docking off => the episode can only ever truncate.
    env = ISSEnv(ISSConfig(
        max_steps=10, physics=PhysicsConfig(collision_boxes_path=None), dock=DockConfig(enabled=False)
    ))
    env.reset(seed=0)
    zero = np.zeros(6, dtype=np.float32)
    for _ in range(9):
        _, _, terminated, truncated, _ = env.step(zero)
        assert not terminated and not truncated
    _, _, terminated, truncated, _ = env.step(zero)
    assert truncated is True
    assert terminated is False


def test_collision_terminates_and_reports_in_info():
    env = ISSEnv(ISSConfig(
        max_steps=100,
        # A box covering the whole start sphere guarantees an immediate hit.
        physics=PhysicsConfig(
            collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [400.0, 400.0, 400.0]}]
        ),
        dock=DockConfig(enabled=False),
    ))
    env.reset(seed=0)
    _, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
    assert terminated is True
    assert truncated is False
    assert info["collision"] is True
    assert info["success"] is False
    assert reward < -1000.0


def test_docking_terminates_and_reports_success():
    env = ISSEnv(ISSConfig(
        max_steps=100,
        physics=PhysicsConfig(collision_boxes_path=None),
        # Dock target at the start sphere radius, so reset lands essentially on it.
        # Attitude/rate gates off: this test isolates distance/speed, and reset's
        # nose-at-ISS attitude does not generally match dock's docking-port
        # orientation or sit still.
        dock=DockConfig(
            enabled=True, max_distance_m=200.0, max_velocity_m_s=10.0,
            max_attitude_error_deg=None, max_body_rate_rad_s=None,
        ),
    ))
    env.reset(seed=0)
    _, _, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
    assert terminated is True
    assert truncated is False
    assert info["success"] is True
    assert info["collision"] is False


def test_actions_are_clipped_to_the_action_space():
    cfg = ISSConfig(physics=PhysicsConfig(collision_boxes_path=None), dock=DockConfig(enabled=False))
    huge = np.full(6, 1e9, dtype=np.float32)
    clamped = np.concatenate(
        [np.full(3, cfg.control.limit_force_n), np.full(3, cfg.control.limit_torque_nm)]
    ).astype(np.float32)

    env_huge = ISSEnv(cfg)
    env_huge.reset(seed=0)
    obs_huge, _, _, _, _ = env_huge.step(huge)

    env_clamped = ISSEnv(cfg)
    env_clamped.reset(seed=0)
    obs_clamped, _, _, _, _ = env_clamped.step(clamped)

    np.testing.assert_array_equal(obs_huge, obs_clamped)


def test_step_before_reset_raises():
    env = ISSEnv()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(6, dtype=np.float32))
