import jax
import numpy as np
import pytest

import jax.numpy as jnp

from owm_envs.envs.common.config import DockConfig, PhysicsConfig
from owm_envs.envs.common.goal import dock_goal_error
from owm_envs.envs.common.sensing import PRESETS
from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss.env import ISSEnv
from owm_envs.envs.iss.vector_env import ISSVectorEnv

FREE_FLIGHT = dict(physics=PhysicsConfig(collision_boxes_path=None), dock=DockConfig(enabled=False))


def test_spaces_are_batched_correctly():
    env = ISSVectorEnv(num_envs=4)
    assert env.num_envs == 4
    assert env.single_observation_space.shape == (13,)
    assert env.single_action_space.shape == (6,)
    assert env.observation_space.shape == (4, 13)
    assert env.action_space.shape == (4, 6)


def test_reset_returns_batched_observations():
    env = ISSVectorEnv(num_envs=8)
    obs, info = env.reset(seed=0)
    assert obs.shape == (8, 13)
    assert obs.dtype == np.float32
    assert info["success"].shape == (8,)
    assert info["collision"].shape == (8,)


def test_reset_gives_each_sub_env_a_different_state():
    obs, _ = ISSVectorEnv(num_envs=4).reset(seed=0)
    assert not np.allclose(obs[0], obs[1])
    assert not np.allclose(obs[1], obs[2])


def test_reset_is_reproducible_with_the_same_seed():
    a, _ = ISSVectorEnv(num_envs=4).reset(seed=11)
    b, _ = ISSVectorEnv(num_envs=4).reset(seed=11)
    np.testing.assert_allclose(a, b)


def test_step_returns_the_five_tuple_batched():
    env = ISSVectorEnv(num_envs=4, cfg=ISSConfig(**FREE_FLIGHT))
    env.reset(seed=0)
    obs, rewards, terminations, truncations, infos = env.step(np.zeros((4, 6), dtype=np.float32))
    assert obs.shape == (4, 13)
    assert rewards.shape == (4,)
    assert terminations.shape == (4,) and terminations.dtype == bool
    assert truncations.shape == (4,) and truncations.dtype == bool
    assert infos["success"].shape == (4,)


def test_batched_step_matches_independent_single_envs():
    """The vector env must be numerically identical to N single envs.

    This is the guard against the batched path silently diverging from the
    single-env path -- a real risk whenever done-logic is implemented twice
    and the two copies can drift apart.
    """
    cfg = ISSConfig(**FREE_FLIGHT)
    n = 4
    vec = ISSVectorEnv(num_envs=n, cfg=cfg)
    vec_obs, _ = vec.reset(seed=5)

    singles = []
    for i in range(n):
        single = ISSEnv(cfg)
        # Drive each single env from the vector env's own initial state so the
        # comparison isolates the step maths from the reset seeding scheme.
        single.reset(seed=0)
        single._state = vec._states[i]
        singles.append(single)

    rng = np.random.default_rng(0)
    for _ in range(25):
        actions = rng.uniform(-100.0, 100.0, size=(n, 6)).astype(np.float32)
        vec_obs, vec_rewards, _, _, _ = vec.step(actions)
        for i, single in enumerate(singles):
            s_obs, s_reward, _, _, _ = single.step(actions[i])
            np.testing.assert_allclose(vec_obs[i], s_obs, rtol=1e-5, atol=1e-5)
            assert np.isclose(vec_rewards[i], s_reward, rtol=1e-4)


def test_truncation_is_flagged_at_max_steps():
    env = ISSVectorEnv(num_envs=2, cfg=ISSConfig(max_steps=5, **FREE_FLIGHT))
    env.reset(seed=0)
    zero = np.zeros((2, 6), dtype=np.float32)
    for _ in range(4):
        _, _, terminations, truncations, _ = env.step(zero)
        assert not truncations.any() and not terminations.any()
    _, _, terminations, truncations, _ = env.step(zero)
    assert truncations.all()
    assert not terminations.any()


def test_terminated_sub_env_autoresets_on_the_next_step():
    env = ISSVectorEnv(num_envs=2, cfg=ISSConfig(
        max_steps=100,
        physics=PhysicsConfig(
            collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [400.0, 400.0, 400.0]}],
            start_radius_range_m=(100.0, 100.0),
        ),
        dock=DockConfig(enabled=False),
    ))
    env.reset(seed=0)
    zero = np.zeros((2, 6), dtype=np.float32)

    _, _, terminations, _, infos = env.step(zero)
    assert terminations.all()
    assert infos["collision"].all()

    # Next-step autoreset: the following step reports the fresh episode, not a
    # repeat termination, and pays no reward for the transition.
    _, rewards, terminations, truncations, _ = env.step(zero)
    assert not terminations.any()
    assert not truncations.any()
    np.testing.assert_allclose(rewards, np.zeros(2), atol=1e-6)


def test_leaving_the_domain_terminates_every_lane_and_autoresets():
    # max_range_m below the start sphere puts reset itself out of bounds, so
    # every lane escapes on its first step regardless of where it flies.
    env = ISSVectorEnv(num_envs=2, cfg=ISSConfig(
        max_steps=100,
        max_range_m=50.0,
        physics=PhysicsConfig(collision_boxes_path=None, start_radius_range_m=(100.0, 100.0)),
        dock=DockConfig(enabled=False),
    ))
    obs, infos = env.reset(seed=0)
    assert not infos["escaped"].any()
    zero = np.zeros((2, 6), dtype=np.float32)

    _, _, terminations, truncations, infos = env.step(zero)
    assert terminations.all()
    assert not truncations.any()
    assert infos["escaped"].all()
    assert not infos["collision"].any() and not infos["success"].any()

    # Autoreset clears the flag along with the termination it caused.
    _, _, terminations, _, infos = env.step(zero)
    assert not terminations.any()
    assert not infos["escaped"].any()


def test_autoreset_returns_pure_reset_observation_not_a_stepped_one():
    """On an autoreset step, the returned observation must be the pure reset
    state, not one physics step forward under the caller's action. A zero
    action cannot distinguish the two cases -- a resting reset state barely
    moves under zero force -- so this drives the autoreset step with a large
    nonzero action instead.
    """
    cfg = ISSConfig(
        max_steps=100,
        physics=PhysicsConfig(
            collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [400.0, 400.0, 400.0]}],
            start_radius_range_m=(100.0, 100.0),
        ),
        dock=DockConfig(enabled=False),
    )
    env = ISSVectorEnv(num_envs=2, cfg=cfg)
    env.reset(seed=0)
    zero = np.zeros((2, 6), dtype=np.float32)

    _, _, terminations, _, _ = env.step(zero)
    assert terminations.all()

    large = np.tile(
        np.array(
            [cfg.control.limit_force_n] * 3 + [cfg.control.limit_torque_nm] * 3,
            dtype=np.float32,
        ),
        (2, 1),
    )
    obs, rewards, terminations, truncations, _ = env.step(large)

    assert not terminations.any() and not truncations.any()
    np.testing.assert_allclose(rewards, np.zeros(2), atol=1e-6)
    # A pure reset state is at rest, on the start sphere. If the autoreset
    # lane had instead been stepped once with this large force, velocity and
    # angular velocity would be far from zero.
    np.testing.assert_array_equal(obs[:, 3:6], 0.0)
    np.testing.assert_array_equal(obs[:, 10:13], 0.0)
    np.testing.assert_allclose(
        np.linalg.norm(obs[:, 0:3], axis=1), cfg.physics.start_radius_range_m[0], rtol=1e-5
    )


def test_truncation_timing_restarts_after_autoreset():
    # Each lane's step index must reset to 0 whenever that lane resets,
    # including via autoreset, not just on the initial reset() call. With
    # max_steps=2, a lane must truncate exactly two steps after each reset --
    # a step index left over from the previous episode would instead
    # truncate the new episode early (one step in, not two).
    env = ISSVectorEnv(num_envs=2, cfg=ISSConfig(max_steps=2, **FREE_FLIGHT))
    env.reset(seed=0)
    zero = np.zeros((2, 6), dtype=np.float32)

    # Episode 1: not yet truncated after one step, truncated after two.
    _, _, terminations, truncations, _ = env.step(zero)
    assert not truncations.any() and not terminations.any()
    _, _, terminations, truncations, _ = env.step(zero)
    assert truncations.all() and not terminations.any()

    # Autoreset step: reports the fresh episode with a neutral transition;
    # this is the step where the step index must be reset to 0.
    _, _, terminations, truncations, _ = env.step(zero)
    assert not truncations.any() and not terminations.any()

    # Episode 2, one real step in: must NOT truncate yet. A step index carried
    # over from episode 1 would instead fire truncation here.
    _, _, terminations, truncations, _ = env.step(zero)
    assert not truncations.any() and not terminations.any()

    # Episode 2, two real steps in: truncates right on schedule.
    _, _, terminations, truncations, _ = env.step(zero)
    assert truncations.all() and not terminations.any()


def test_autoreset_mode_is_declared_in_metadata():
    env = ISSVectorEnv(num_envs=2)
    assert "autoreset_mode" in env.metadata


def test_render_fps_tracks_the_configured_timestep():
    # Same rule as ISSEnv: one frame per step, so render_fps is 1/dt.
    assert ISSVectorEnv(num_envs=2).metadata["render_fps"] == round(1.0 / ISSConfig().dt)
    assert ISSVectorEnv(num_envs=2, cfg=ISSConfig(dt=0.01)).metadata["render_fps"] == 100
    # Overriding it must not disturb the autoreset declaration alongside it.
    assert "autoreset_mode" in ISSVectorEnv(num_envs=2, cfg=ISSConfig(dt=0.01)).metadata


def test_integer_seed_path_is_unchanged():
    # The integer-seed reset path must derive per-env keys via
    # jax.random.split on a single PRNGKey; other reproducibility tests
    # depend on this exact derivation.
    env = ISSVectorEnv(num_envs=4, cfg=ISSConfig())
    obs, _ = env.reset(seed=11)

    expected_key, expected_subkey = jax.random.split(jax.random.PRNGKey(11))
    expected_states = env._batched_reset(jax.random.split(expected_subkey, 4))
    np.testing.assert_allclose(obs, np.asarray(expected_states, dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(env._key), np.asarray(expected_key))


def test_per_env_seed_list_produces_correctly_shaped_reproducible_states():
    env = ISSVectorEnv(num_envs=3, cfg=ISSConfig())
    obs, info = env.reset(seed=[1, 2, 3])
    assert obs.shape == (3, 13)
    assert obs.dtype == np.float32
    assert info["success"].shape == (3,)

    obs2, _ = ISSVectorEnv(num_envs=3, cfg=ISSConfig()).reset(seed=[1, 2, 3])
    np.testing.assert_allclose(obs, obs2)


def test_per_env_seed_list_gives_different_lanes_different_states_for_different_seeds():
    obs, _ = ISSVectorEnv(num_envs=2, cfg=ISSConfig()).reset(seed=[1, 2])
    assert not np.allclose(obs[0], obs[1])


def test_per_env_seed_list_wrong_length_raises():
    env = ISSVectorEnv(num_envs=3, cfg=ISSConfig())
    with pytest.raises(ValueError, match="3"):
        env.reset(seed=[1, 2])


def test_unseeded_resets_differ_across_instances():
    # Gymnasium's unseeded-reset contract requires reset() without a seed to
    # be non-deterministic across instances, so the persistent key must not
    # be initialised to a constant.
    a, _ = ISSVectorEnv(num_envs=4).reset()
    b, _ = ISSVectorEnv(num_envs=4).reset()
    assert not np.allclose(a, b)


def test_noisy_vector_env_observations_differ_from_true_states():
    cfg = ISSConfig(sensor_noise=PRESETS["cooperative"])
    env = ISSVectorEnv(num_envs=3, cfg=cfg)
    obs, info = env.reset(seed=5)
    assert info["state"].shape == (3, 13)
    assert not np.array_equal(obs, info["state"])
    obs2, _, _, _, info2 = env.step(np.zeros((3, 6), dtype=np.float32))
    assert not np.array_equal(obs2, info2["state"])


def test_noiseless_vector_env_info_state_equals_observation():
    env = ISSVectorEnv(num_envs=2)
    obs, info = env.reset(seed=5)
    np.testing.assert_array_equal(obs, info["state"])


def test_vector_goal_error_observations_are_25_dim():
    cfg = ISSConfig(observation={"goal_error": True})
    env = ISSVectorEnv(num_envs=2, cfg=cfg)
    assert env.single_observation_space.shape == (25,)
    obs, info = env.reset(seed=4)
    assert obs.shape == (2, 25)
    assert info["state"].shape == (2, 13)


def test_true_state_survives_autoreset_with_noise():
    # Sensor noise must not perturb the dynamics key stream: the true state
    # produced by an autoreset must be identical whether or not noise is
    # enabled, since the noise draws in _obs() must come from a side stream
    # that never consumes from self._key.
    clean_cfg = ISSConfig(max_steps=3)
    noisy_cfg = ISSConfig(max_steps=3, sensor_noise=PRESETS["cooperative"])

    def state_after_autoreset(cfg):
        env = ISSVectorEnv(num_envs=2, cfg=cfg)
        env.reset(seed=0)
        zero = np.zeros((2, 6), dtype=np.float32)
        info = None
        for _ in range(4):  # 3 steps truncate; the 4th is the autoreset step
            _, _, _, _, info = env.step(zero)
        return info["state"]

    np.testing.assert_array_equal(
        state_after_autoreset(clean_cfg), state_after_autoreset(noisy_cfg)
    )


def test_each_lane_draws_its_own_port_from_the_configured_set():
    venv = ISSVectorEnv(8, ISSConfig(dock=DockConfig(ports=("all",))))
    _, info = venv.reset(seed=0)
    assert info["dock_port_index"].shape == (8,)
    assert info["goal_pose"].shape == (8, 7)
    # Eight lanes over eight ports: a shared draw would name one port.
    assert len(set(info["dock_port"])) > 1
    # Deterministic per seed.
    _, again = ISSVectorEnv(8, ISSConfig(dock=DockConfig(ports=("all",)))).reset(seed=0)
    np.testing.assert_array_equal(info["dock_port_index"], again["dock_port_index"])


def test_autoreset_lanes_redraw_their_ports():
    cfg = ISSConfig(max_steps=2, dock=DockConfig(ports=("all",)))
    venv = ISSVectorEnv(8, cfg)
    _, info = venv.reset(seed=3)
    zero = np.zeros((8, 6), dtype=np.float32)
    seen = {tuple(info["dock_port_index"])}
    for _ in range(12):  # four truncation/autoreset cycles
        _, _, _, _, info = venv.step(zero)
        seen.add(tuple(info["dock_port_index"]))
    # Redraws happened: the lane-port assignment did not stay frozen across
    # autoresets. (Eight lanes over eight ports make a frozen coincidence
    # across four redraw cycles astronomically unlikely.)
    assert len(seen) > 1


def test_reset_options_narrow_the_lane_menu():
    cfg = ISSConfig(dock=DockConfig(ports=("all",)))
    venv = ISSVectorEnv(6, cfg)
    _, info = venv.reset(seed=0, options={"dock_port": "zvezda_aft"})
    assert set(info["dock_port"]) == {"zvezda_aft"}
    subset = ("poisk_zenith", "rassvet_nadir")
    _, info = venv.reset(seed=1, options={"dock_port": subset})
    assert set(info["dock_port"]) <= set(subset)
    with pytest.raises(ValueError, match="unknown dock_port"):
        venv.reset(seed=0, options={"dock_port": "not_a_port"})
    with pytest.raises(ValueError, match="unknown reset option"):
        venv.reset(seed=0, options={"dock_prt": "zvezda_aft"})


def test_reset_options_pose_is_shared_by_every_lane():
    venv = ISSVectorEnv(3, ISSConfig())
    pose = np.array([5.0, -10.0, 2.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _, info = venv.reset(seed=0, options={"dock_pose": pose})
    np.testing.assert_array_equal(info["goal_pose"], np.tile(pose, (3, 1)))
    # A pose is not a named port, so no lane claims one.
    assert "dock_port" not in info
    _, naked = venv.reset(seed=0)
    assert "dock_port" not in naked  # no ports configured, override forgotten
    assert not np.allclose(naked["goal_pose"][0], pose)


def test_lane_goal_blocks_measure_against_the_lanes_own_ports():
    cfg = ISSConfig(observation={"goal_error": True}, dock=DockConfig(ports=("all",)))
    venv = ISSVectorEnv(6, cfg)
    obs, info = venv.reset(seed=2)
    for lane in range(6):
        expected = dock_goal_error(
            jnp.asarray(obs[lane, :13]), jnp.asarray(info["goal_pose"][lane])
        )
        np.testing.assert_allclose(obs[lane, 13:], np.asarray(expected), atol=1e-6)
    # And the per-lane step path: rewards and flags come from each lane's own
    # target, exercised end to end rather than only at reset.
    obs, rewards, _, _, info = venv.step(np.zeros((6, 6), dtype=np.float32))
    assert np.all(np.isfinite(rewards))
    for lane in range(6):
        expected = dock_goal_error(
            jnp.asarray(obs[lane, :13]), jnp.asarray(info["goal_pose"][lane])
        )
        np.testing.assert_allclose(obs[lane, 13:], np.asarray(expected), atol=1e-6)


def test_a_no_ports_reset_consumes_exactly_the_documented_key_stream():
    """The executable spec of the no-ports RNG discipline.

    Published no-ports datasets are byte-identical replays of this exact
    sequence: PRNGKey(seed), a NOISE_STREAM fold_in, ONE split for the reset
    keys, and nothing else. Port support must not add so much as one split to
    a run that configures no ports -- this pins both the produced states and
    the carried key, so any unconditional extra draw fails loudly.
    """
    seed, n = 11, 4
    venv = ISSVectorEnv(n, ISSConfig())
    obs, _ = venv.reset(seed=seed)

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    # The env's own compiled reset, driven by independently derived keys:
    # what is pinned is the KEY SEQUENCE, not the reset arithmetic.
    expected_states = venv._batched_reset(jax.random.split(subkey, n))
    np.testing.assert_array_equal(obs, np.asarray(expected_states, dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(venv._key), np.asarray(key))


def test_autoreset_redraws_only_the_lane_that_reset():
    # Lane 0 is pushed out of the domain by hand; lane 1 flies on. The
    # autoreset must redraw lane 0's port from the menu while lane 1 keeps
    # the one it drew at reset -- the masking that keeps a lane's goal its
    # own for the length of its episode.
    cfg = ISSConfig(
        max_steps=10_000,
        max_range_m=200.0,
        physics=PhysicsConfig(collision_boxes_path=None, start_radius_range_m=(100.0, 100.0)),
        dock=DockConfig(ports=("all",)),
    )
    venv = ISSVectorEnv(2, cfg)
    _, info = venv.reset(seed=5)
    lane1_port = info["dock_port_index"][1]
    zero = np.zeros((2, 6), dtype=np.float32)

    lane0_history = [info["dock_port_index"][0]]
    for _ in range(8):
        # Escape lane 0 only: place it outside max_range_m.
        states = np.array(venv._states)
        states[0, 0:3] = (500.0, 0.0, 0.0)
        venv._states = jnp.asarray(states)
        _, _, terminations, _, _ = venv.step(zero)
        assert bool(terminations[0]) and not bool(terminations[1])
        _, _, _, _, info = venv.step(zero)  # lane 0 autoresets here
        assert info["dock_port_index"][1] == lane1_port
        lane0_history.append(info["dock_port_index"][0])

    # Eight redraws over eight ports: a frozen lane-0 assignment would mean
    # the redraw is not happening at all.
    assert len(set(lane0_history)) > 1
