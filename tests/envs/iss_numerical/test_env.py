import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import owm_envs.envs  # noqa: F401  -- triggers registration
from owm_envs.envs.common.config import DockConfig, PhysicsConfig, dock_target
from owm_envs.envs.common.goal import dock_goal_error
from owm_envs.envs.common.sensing import PRESETS
from owm_envs.envs.iss_numerical.config import OBS_MODE_DIM, NumericalConfig
from owm_envs.envs.iss_numerical.dynamics import relative_view
from owm_envs.envs.iss_numerical.env import NumericalEnv

FREE_FLIGHT = dict(physics=PhysicsConfig(collision_boxes_path=None), dock=DockConfig(enabled=False))


def test_passes_the_gymnasium_env_checker():
    check_env(NumericalEnv(), skip_render_check=True)


def test_env_checker_accepts_goal_error_observations():
    check_env(
        NumericalEnv(NumericalConfig(observation={"goal_error": True})), skip_render_check=True
    )


def test_registered_id_constructs():
    env = gym.make("ISS-Numerical-Docking-v0")
    assert env is not None
    env.close()


@pytest.mark.parametrize("mode", ["absolute", "chaser_absolute", "chief_absolute", "relative"])
def test_observation_width_follows_the_mode(mode):
    cfg = NumericalConfig(observation={"mode": mode})
    env = NumericalEnv(cfg)
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_MODE_DIM[mode],)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)


@pytest.mark.parametrize("mode", ["absolute", "chaser_absolute", "chief_absolute", "relative"])
def test_observation_width_follows_the_mode_with_goal_error(mode):
    cfg = NumericalConfig(observation={"mode": mode, "goal_error": True})
    env = NumericalEnv(cfg)
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_MODE_DIM[mode] + 12,)
    assert env.observation_space.shape == (OBS_MODE_DIM[mode] + 12,)


def test_action_space_matches_configured_limits():
    cfg = NumericalConfig()
    env = NumericalEnv(cfg)
    assert env.action_space.shape == (6,)
    np.testing.assert_allclose(env.action_space.high[0:3], cfg.control.limit_force_n)
    np.testing.assert_allclose(env.action_space.high[3:6], cfg.control.limit_torque_nm)


def test_reset_is_reproducible_with_the_same_seed():
    a, _ = NumericalEnv().reset(seed=42)
    b, _ = NumericalEnv().reset(seed=42)
    c, _ = NumericalEnv().reset(seed=43)
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, c)


def test_truncates_at_max_steps_without_terminating():
    env = NumericalEnv(NumericalConfig(max_steps=10, **FREE_FLIGHT))
    env.reset(seed=0)
    zero = np.zeros(6, dtype=np.float32)
    for _ in range(9):
        _, _, terminated, truncated, _ = env.step(zero)
        assert not terminated and not truncated
    _, _, terminated, truncated, _ = env.step(zero)
    assert truncated is True
    assert terminated is False


def test_step_before_reset_raises():
    env = NumericalEnv()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(6, dtype=np.float32))


def test_info_carries_true_and_measured_state_both_21d():
    env = NumericalEnv()
    obs, info = env.reset(seed=3)
    assert info["state"].shape == (21,)
    assert info["state"].dtype == np.float32
    assert info["measured_state"].shape == (21,)
    assert info["measured_state"].dtype == np.float32


def test_noiseless_env_observation_in_absolute_mode_equals_state():
    # mode="absolute" makes make_observe the identity, so with noise off the
    # observation, the measured state and the true state all agree exactly.
    cfg = NumericalConfig(observation={"mode": "absolute"})
    env = NumericalEnv(cfg)
    obs, info = env.reset(seed=3)
    np.testing.assert_array_equal(obs, info["state"])
    np.testing.assert_array_equal(obs, info["measured_state"])


def test_noise_leaves_epoch_and_chief_untouched_while_chaser_slices_differ():
    # `apply_sensor_noise` is documented to pass any slice outside
    # pos/vel/quat/omega through untouched; this is the env-level guard that
    # the wiring (layout=NUM_LAYOUT) actually reaches that behaviour end to
    # end. Checked on `measured_state`, which is always the raw 21D layout
    # regardless of `observation.mode`.
    cfg = NumericalConfig(sensor_noise=PRESETS["cooperative"])
    env = NumericalEnv(cfg)
    obs, info = env.reset(seed=3)
    measured, true = info["measured_state"], info["state"]

    np.testing.assert_array_equal(measured[0:2], true[0:2])  # epoch
    np.testing.assert_array_equal(measured[2:8], true[2:8])  # chief
    assert not np.allclose(measured[8:11], true[8:11])  # chaser position

    obs2, _, _, _, info2 = env.step(np.zeros(6, dtype=np.float32))
    measured2, true2 = info2["measured_state"], info2["state"]
    np.testing.assert_array_equal(measured2[0:2], true2[0:2])
    np.testing.assert_array_equal(measured2[2:8], true2[2:8])
    assert not np.allclose(measured2[8:11], true2[8:11])


def test_noncooperative_range_noise_is_metre_scale_not_kilometre_scale():
    # NUM_LAYOUT.pos is the chaser's ABSOLUTE ECI position (~6.8e6 m); if
    # `sigma_pos_frac_of_range` scaled off that instead of the relative range
    # a real sensor would report, the noncooperative preset's 1% would inject
    # ~68 km of position noise at a 100 m standoff. It must not.
    cfg = NumericalConfig(
        sensor_noise=PRESETS["noncooperative"],
        orbit={"start_radius_range_m": (100.0, 100.0)},
    )
    env = NumericalEnv(cfg)
    deltas = []
    for seed in range(20):
        _, info = env.reset(seed=seed)
        delta = np.linalg.norm(
            info["measured_state"][8:11] - info["state"][8:11]
        )
        deltas.append(delta)
    deltas = np.array(deltas)
    # 1% of range at 100 m -> ~1 m total-RMS; generously bounded well under a
    # kilometre and confirmed non-vacuous (actually noisy).
    assert deltas.max() < 50.0
    assert deltas.mean() > 0.01


def test_noisy_env_reset_is_reproducible_per_seed():
    cfg = NumericalConfig(sensor_noise=PRESETS["cooperative"])
    a, _ = NumericalEnv(cfg).reset(seed=11)
    b, _ = NumericalEnv(cfg).reset(seed=11)
    np.testing.assert_array_equal(a, b)


def test_goal_error_block_matches_dock_goal_error_of_the_relative_view():
    cfg = NumericalConfig(observation={"mode": "relative", "goal_error": True})
    env = NumericalEnv(cfg)
    obs, info = env.reset(seed=2)
    assert obs.shape == (27,)

    # mode="relative" observation is [epoch(2), relative_view(13)], so the
    # view sits at columns 2:15.
    view = jnp.asarray(obs[2:15])
    expected = dock_goal_error(view, jnp.asarray(dock_target(cfg)))
    np.testing.assert_allclose(obs[15:], np.asarray(expected), atol=1e-6)


def test_goal_block_uses_the_measured_state_when_noisy():
    cfg = NumericalConfig(
        observation={"mode": "relative", "goal_error": True},
        sensor_noise=PRESETS["cooperative"],
    )
    obs, info = NumericalEnv(cfg).reset(seed=2)

    # Internal consistency: the appended block matches the SAME (noisy)
    # relative view the observation itself reports -- built off `obs` alone,
    # never re-deriving `relative_view` from a float32-narrowed measured
    # state, which would reintroduce the ~1 m ECI-subtraction budget
    # `dynamics.py` documents and swamp this comparison.
    view = jnp.asarray(obs[2:15])
    expected = dock_goal_error(view, jnp.asarray(dock_target(cfg)))
    np.testing.assert_allclose(obs[15:], np.asarray(expected), atol=1e-5)

    # And the noise actually reached the recorded view: it differs from the
    # noiseless relative view of the true state.
    true_view = np.asarray(relative_view(jnp.asarray(info["state"])), dtype=np.float64)
    assert not np.allclose(np.asarray(obs[2:15], dtype=np.float64), true_view, atol=1e-3)


@pytest.mark.parametrize(
    "cfg",
    [
        NumericalConfig(),
        NumericalConfig(sensor_noise=PRESETS["cooperative"]),
        NumericalConfig(observation={"goal_error": True}),
    ],
    ids=["noiseless", "noisy", "goal_error"],
)
def test_the_carried_state_stays_float64(cfg):
    # The adapter narrows to float32 on the copies `_observation_and_measured`
    # and `_true_state` hand out, never on the state it carries forward --
    # writing a narrowed value back would round the epoch prefix (and the
    # ~6.8e6 m ECI positions) every step, resurrecting the drift `dynamics.py`
    # documents.
    env = NumericalEnv(cfg)
    env.reset(seed=0)
    assert env._state.dtype == jnp.float64
    for _ in range(5):
        env.step(np.zeros(6, dtype=np.float32))
        assert env._state.dtype == jnp.float64


def test_render_without_a_render_mode_returns_none():
    env = NumericalEnv()
    env.reset(seed=0)
    assert env.render() is None


def test_render_before_reset_raises():
    env = NumericalEnv(render_mode="rgb_array")
    with pytest.raises(RuntimeError, match="reset"):
        env.render()


def test_unknown_render_mode_raises():
    with pytest.raises(ValueError, match="render_mode"):
        NumericalEnv(render_mode="ascii")


def test_requires_zero_linear_damping():
    with pytest.raises(ValueError, match="linear_damping"):
        NumericalEnv(NumericalConfig(physics=PhysicsConfig(linear_damping=1.0)))
