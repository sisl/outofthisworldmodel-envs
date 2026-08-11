import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import owm_envs.envs  # noqa: F401  -- triggers registration
from owm_envs.envs.common.config import DockConfig, PhysicsConfig, dock_target
from owm_envs.envs.common.docking_ports import PORTS_BY_NAME, port_pose
from owm_envs.envs.common.goal import dock_goal_error
from owm_envs.envs.common.sensing import PRESETS
from owm_envs.envs.iss_hcw.config import HCW_LAYOUT, HCWConfig
from owm_envs.envs.iss_hcw.env import HCWEnv
from owm_envs.envs.iss_hcw.vector_env import HCWVectorEnv

FREE_FLIGHT = dict(physics=PhysicsConfig(collision_boxes_path=None), dock=DockConfig(enabled=False))


def test_passes_the_gymnasium_env_checker():
    check_env(HCWEnv(), skip_render_check=True)


def test_env_checker_accepts_goal_error_observations():
    check_env(HCWEnv(HCWConfig(observation={"goal_error": True})), skip_render_check=True)


def test_registered_id_constructs():
    env = gym.make("ISS-HCW-Docking-v1")
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


def _port_row(name: str) -> np.ndarray:
    position, quaternion = port_pose(PORTS_BY_NAME[name])
    return np.concatenate([position, quaternion]).astype(np.float32)


def test_a_port_set_draws_per_episode_and_governs_the_goal():
    cfg = HCWConfig(dock=DockConfig(ports=("all",)))
    env = HCWEnv(cfg)
    seen = set()
    for seed in range(16):
        _, info = env.reset(seed=seed)
        seen.add(info["dock_port"])
        np.testing.assert_allclose(
            info["goal_pose"], _port_row(info["dock_port"]), atol=1e-6
        )
    assert len(seen) > 1  # a draw, not a constant


def test_reset_options_target_a_named_port_like_iss():
    cfg = HCWConfig(dock=DockConfig(ports=("harmony_fwd_pma2", "zvezda_aft")))
    env = HCWEnv(cfg)
    _, info = env.reset(seed=0, options={"dock_port": "zvezda_aft"})
    assert info["dock_port"] == "zvezda_aft" and info["dock_port_index"] == 1
    with pytest.raises(ValueError, match="unknown reset option"):
        env.reset(seed=0, options={"dock_prt": "zvezda_aft"})


def test_reset_options_pose_override_reaches_gate_reward_and_telemetry():
    # Zero-radius start puts the chaser at the origin; drive the true state
    # onto the override pose and every consumer must agree it arrived.
    pose = np.array([5.0, -10.0, 2.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    cfg = HCWConfig(
        max_steps=100,
        physics=PhysicsConfig(collision_boxes_path=None),
        orbit={"start_radius_range_m": (0.0, 0.0)},
    )
    env = HCWEnv(cfg)
    _, info = env.reset(seed=0, options={"dock_pose": pose})
    np.testing.assert_allclose(info["goal_pose"], pose, atol=1e-6)
    assert "dock_port" not in info
    state = np.array(env._state)
    state[HCW_LAYOUT.pos] = pose[0:3]
    state[HCW_LAYOUT.vel] = 0.0
    state[HCW_LAYOUT.quat] = pose[3:7]
    state[HCW_LAYOUT.omega] = 0.0
    env._state = jnp.asarray(state, dtype=jnp.float64)
    for label, value in env._goal_error_true().items():
        assert value == pytest.approx(0.0, abs=1e-5), label
    # The dock gate itself must score against the override: one zero-thrust
    # step from the pose ends the episode docked, and the reward carries no
    # collision spike. Without the override this pose is nowhere near
    # cfg.dock, so success here proves the threading, not the default.
    _, reward, terminated, _, step_info = env.step(np.zeros(6, dtype=np.float32))
    assert step_info["success"] and terminated
    assert reward > -1_000.0


def test_a_naked_reset_forgets_the_previous_override():
    cfg = HCWConfig(dock=DockConfig(ports=("all",)))
    env = HCWEnv(cfg)
    env.reset(seed=11, options={"dock_port": "rassvet_nadir"})
    _, after = env.reset(seed=11)
    _, fresh = HCWEnv(cfg).reset(seed=11)
    assert after["dock_port"] == fresh["dock_port"]
    np.testing.assert_array_equal(after["goal_pose"], fresh["goal_pose"])


def test_vector_env_draws_ports_per_lane_and_takes_options():
    venv = HCWVectorEnv(8, HCWConfig(dock=DockConfig(ports=("all",))))
    _, info = venv.reset(seed=0)
    assert len(set(info["dock_port"])) > 1  # a per-lane draw, not one shared
    for lane, name in enumerate(info["dock_port"]):
        np.testing.assert_allclose(info["goal_pose"][lane], _port_row(name), atol=1e-6)
    _, info = venv.reset(seed=1, options={"dock_port": "zvezda_aft"})
    assert set(info["dock_port"]) == {"zvezda_aft"}
    pose = np.array([5.0, -10.0, 2.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _, info = venv.reset(seed=2, options={"dock_pose": pose})
    np.testing.assert_array_equal(info["goal_pose"], np.tile(pose, (8, 1)))
    assert "dock_port" not in info


def test_vector_lane_goal_blocks_and_step_follow_the_lanes_own_ports():
    cfg = HCWConfig(observation={"goal_error": True}, dock=DockConfig(ports=("all",)))
    venv = HCWVectorEnv(4, cfg)
    obs, info = venv.reset(seed=2)
    # And through the per-lane step path, not only at reset.
    obs, rewards, _, _, info = venv.step(np.zeros((4, 6), dtype=np.float32))
    assert np.all(np.isfinite(rewards))
    for lane in range(4):
        view = HCW_LAYOUT.slice_view(jnp.asarray(obs[lane, :15]))
        expected = dock_goal_error(view, jnp.asarray(info["goal_pose"][lane]))
        np.testing.assert_allclose(obs[lane, 15:], np.asarray(expected), atol=1e-6)


def test_vector_no_ports_reset_consumes_exactly_the_documented_key_stream():
    # The executable spec of the no-ports RNG discipline (see the iss vector
    # test of the same name): PRNGKey(seed), the NOISE_STREAM fold_in, ONE
    # split for the reset keys, nothing else -- port support must not add a
    # single draw to a no-ports run.
    seed, n = 11, 4
    venv = HCWVectorEnv(n, HCWConfig())
    obs, _ = venv.reset(seed=seed)
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    expected = venv._batched_reset(jax.random.split(subkey, n))
    np.testing.assert_array_equal(obs, np.asarray(expected, dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(venv._key), np.asarray(key))


def test_vector_autoreset_redraws_only_the_lane_that_reset():
    cfg = HCWConfig(
        max_steps=10_000,
        max_range_m=200.0,
        physics=PhysicsConfig(collision_boxes_path=None),
        orbit={"start_radius_range_m": (100.0, 100.0)},
        dock=DockConfig(ports=("all",)),
    )
    venv = HCWVectorEnv(2, cfg)
    _, info = venv.reset(seed=5)
    lane1_port = info["dock_port_index"][1]
    zero = np.zeros((2, 6), dtype=np.float32)

    lane0_history = [info["dock_port_index"][0]]
    for _ in range(8):
        states = np.array(venv._states)
        states[0, HCW_LAYOUT.pos] = (500.0, 0.0, 0.0)
        venv._states = jnp.asarray(states, dtype=jnp.float64)
        _, _, terminations, _, _ = venv.step(zero)
        assert bool(terminations[0]) and not bool(terminations[1])
        _, _, _, _, info = venv.step(zero)  # lane 0 autoresets here
        assert info["dock_port_index"][1] == lane1_port
        lane0_history.append(info["dock_port_index"][0])
    assert len(set(lane0_history)) > 1


def test_a_port_set_leaves_the_seeded_initial_state_alone():
    # The port is drawn after the dynamics seed, so the same seed places the
    # chaser identically with and without ports -- adding a port set does not
    # reshuffle the trajectories a run would otherwise have produced.
    plain = HCWEnv(HCWConfig())
    ported = HCWEnv(HCWConfig(dock=DockConfig(ports=("all",))))
    for seed in (0, 1, 42):
        _, plain_info = plain.reset(seed=seed)
        _, ported_info = ported.reset(seed=seed)
        np.testing.assert_array_equal(plain_info["state"], ported_info["state"])
