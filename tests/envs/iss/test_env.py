import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import owm_envs.envs  # noqa: F401  -- triggers registration
from owm_envs.envs.iss.config import DockConfig, ISSConfig, PhysicsConfig
from owm_envs.envs.iss.env import ISSEnv
from owm_envs.envs.iss.sensing import PRESETS


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


def test_render_fps_tracks_the_configured_timestep():
    # One frame per step, so render_fps is the simulation rate. A fixed value
    # here would misstate the playback speed for any dt but one.
    assert ISSEnv().metadata["render_fps"] == round(1.0 / ISSConfig().dt)
    assert ISSEnv(ISSConfig(dt=0.01)).metadata["render_fps"] == 100

    # Per-instance, so one env's dt does not leak into the next.
    assert ISSEnv(ISSConfig(dt=0.02)).metadata["render_fps"] == 50


@pytest.mark.parametrize("dt", [2.0, 2.5, 10.0])
def test_render_fps_stays_a_usable_rate_for_coarse_timesteps(dt):
    # 1/dt rounds to zero at any dt >= 2 -- including exactly 2.0, where the
    # tie 0.5 rounds to even -- and zero is not a frame rate anything can
    # divide by or feed to an encoder. These dt values are unrealistic for
    # this environment but ISSConfig accepts them.
    assert ISSEnv(ISSConfig(dt=dt)).metadata["render_fps"] == 1


def test_noisy_env_observation_differs_from_true_state_and_info_carries_state():
    cfg = ISSConfig(sensor_noise=PRESETS["cooperative"])
    env = ISSEnv(cfg)
    obs, info = env.reset(seed=3)
    assert "state" in info and info["state"].shape == (13,)
    assert not np.array_equal(obs, info["state"])
    obs2, _, _, _, info2 = env.step(np.zeros(6, dtype=np.float32))
    assert not np.array_equal(obs2, info2["state"])


def test_noiseless_env_info_state_equals_observation():
    env = ISSEnv()
    obs, info = env.reset(seed=3)
    np.testing.assert_array_equal(obs, info["state"])


def test_noisy_env_reset_is_reproducible_per_seed():
    cfg = ISSConfig(sensor_noise=PRESETS["cooperative"])
    a, _ = ISSEnv(cfg).reset(seed=11)
    b, _ = ISSEnv(cfg).reset(seed=11)
    np.testing.assert_array_equal(a, b)
