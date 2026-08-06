import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import owm_envs.envs  # noqa: F401  -- triggers registration
from owm_envs.envs.common.config import DockConfig, PhysicsConfig, dock_target
from owm_envs.envs.common.goal import dock_goal_error
from owm_envs.envs.common.sensing import PRESETS
from owm_envs.envs.iss_hcw.config import HCW_LAYOUT, HCWConfig
from owm_envs.envs.iss_hcw.env import HCWEnv

FREE_FLIGHT = dict(physics=PhysicsConfig(collision_boxes_path=None), dock=DockConfig(enabled=False))


def test_passes_the_gymnasium_env_checker():
    check_env(HCWEnv(), skip_render_check=True)


def test_env_checker_accepts_goal_error_observations():
    check_env(HCWEnv(HCWConfig(observation={"goal_error": True})), skip_render_check=True)


def test_registered_id_constructs():
    env = gym.make("ISS-HCW-Docking-v0")
    assert env is not None
    env.close()


def test_observation_is_15d_float32():
    env = HCWEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (15,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)


def test_observation_space_bounds_mirror_iss_view_with_an_epoch_prefix():
    space = HCWEnv().observation_space
    # jd: unbounded above, floored at 0.
    assert space.low[0] == 0.0
    assert space.high[0] == np.inf
    # sec-of-day: bounded to [0, 86400].
    assert space.low[1] == 0.0
    assert space.high[1] == 86400.0
    # The 13D view -- pos/vel unbounded, quaternion in [-1, 1], omega unbounded.
    np.testing.assert_array_equal(space.low[2:8], -np.inf)
    np.testing.assert_array_equal(space.high[2:8], np.inf)
    np.testing.assert_array_equal(space.low[8:12], -1.0)
    np.testing.assert_array_equal(space.high[8:12], 1.0)
    np.testing.assert_array_equal(space.low[12:15], -np.inf)
    np.testing.assert_array_equal(space.high[12:15], np.inf)


def test_action_space_matches_configured_limits():
    cfg = HCWConfig()
    env = HCWEnv(cfg)
    assert env.action_space.shape == (6,)
    np.testing.assert_allclose(env.action_space.high[0:3], cfg.control.limit_force_n)
    np.testing.assert_allclose(env.action_space.high[3:6], cfg.control.limit_torque_nm)


def test_reset_is_reproducible_with_the_same_seed():
    a, _ = HCWEnv().reset(seed=42)
    b, _ = HCWEnv().reset(seed=42)
    c, _ = HCWEnv().reset(seed=43)
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, c)


def test_truncates_at_max_steps_without_terminating():
    env = HCWEnv(HCWConfig(max_steps=10, **FREE_FLIGHT))
    env.reset(seed=0)
    zero = np.zeros(6, dtype=np.float32)
    for _ in range(9):
        _, _, terminated, truncated, _ = env.step(zero)
        assert not terminated and not truncated
    _, _, terminated, truncated, _ = env.step(zero)
    assert truncated is True
    assert terminated is False


def test_step_before_reset_raises():
    env = HCWEnv()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(6, dtype=np.float32))


def test_noiseless_env_info_state_equals_observation():
    env = HCWEnv()
    obs, info = env.reset(seed=3)
    assert info["state"].shape == (15,)
    np.testing.assert_array_equal(obs, info["state"])


def test_noise_leaves_the_epoch_exact_while_position_differs():
    # `apply_sensor_noise` is documented to pass the epoch slice through
    # untouched; this is the env-level guard that the wiring (layout=
    # HCW_LAYOUT) actually reaches that behaviour end to end.
    cfg = HCWConfig(sensor_noise=PRESETS["cooperative"])
    env = HCWEnv(cfg)
    obs, info = env.reset(seed=3)
    np.testing.assert_array_equal(obs[0:2], info["state"][0:2])
    assert not np.allclose(obs[2:5], info["state"][2:5])

    obs2, _, _, _, info2 = env.step(np.zeros(6, dtype=np.float32))
    np.testing.assert_array_equal(obs2[0:2], info2["state"][0:2])
    assert not np.allclose(obs2[2:5], info2["state"][2:5])


def test_noisy_env_reset_is_reproducible_per_seed():
    cfg = HCWConfig(sensor_noise=PRESETS["cooperative"])
    a, _ = HCWEnv(cfg).reset(seed=11)
    b, _ = HCWEnv(cfg).reset(seed=11)
    np.testing.assert_array_equal(a, b)


def test_goal_error_observation_is_27_dim_and_matches_dock_goal_error_of_the_view():
    cfg = HCWConfig(observation={"goal_error": True})
    env = HCWEnv(cfg)
    assert env.observation_space.shape == (27,)
    obs, info = env.reset(seed=2)
    assert obs.shape == (27,)
    assert info["state"].shape == (15,)

    view = HCW_LAYOUT.slice_view(jnp.asarray(obs[:15]))
    expected = dock_goal_error(view, jnp.asarray(dock_target(cfg)))
    np.testing.assert_allclose(obs[15:], np.asarray(expected), atol=1e-6)


def test_goal_block_uses_the_measured_state_when_noisy():
    cfg = HCWConfig(observation={"goal_error": True}, sensor_noise=PRESETS["cooperative"])
    obs, info = HCWEnv(cfg).reset(seed=2)

    view = HCW_LAYOUT.slice_view(jnp.asarray(obs[:15]))
    expected = dock_goal_error(view, jnp.asarray(dock_target(cfg)))
    np.testing.assert_allclose(obs[15:], np.asarray(expected), atol=1e-6)
    assert not np.allclose(obs[2:15], info["state"][2:15])


@pytest.mark.parametrize(
    "cfg",
    [
        HCWConfig(),
        HCWConfig(sensor_noise=PRESETS["cooperative"]),
        HCWConfig(observation={"goal_error": True}),
    ],
    ids=["noiseless", "noisy", "goal_error"],
)
def test_the_carried_state_stays_float64(cfg):
    # The adapter narrows to float32 on the copies `_obs` and `_true_state`
    # hand out, never on the state it carries forward. The iss mirror does the
    # opposite at envs/iss/env.py:156 -- `jnp.asarray(self._state,
    # jnp.float32)` -- and copying that line across, or otherwise writing a
    # narrowed value back, would round the epoch prefix every step and
    # resurrect the 290 s/orbit drift dynamics.py documents. Nothing else here
    # would notice: the driver-equivalence test compares recorded epochs with
    # ~4.3 s of slack, which is twenty times the drift a short rollout shows.
    env = HCWEnv(cfg)
    env.reset(seed=0)
    assert env._state.dtype == jnp.float64
    for _ in range(5):
        env.step(np.zeros(6, dtype=np.float32))
        assert env._state.dtype == jnp.float64


def test_render_without_a_render_mode_returns_none():
    env = HCWEnv()
    env.reset(seed=0)
    assert env.render() is None


def test_render_before_reset_raises():
    env = HCWEnv(render_mode="rgb_array")
    with pytest.raises(RuntimeError, match="reset"):
        env.render()


def test_unknown_render_mode_raises():
    with pytest.raises(ValueError, match="render_mode"):
        HCWEnv(render_mode="ascii")
