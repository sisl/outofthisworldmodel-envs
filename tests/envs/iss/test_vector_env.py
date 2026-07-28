import numpy as np
import pytest

from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss.env import ISSEnv
from owm_envs.envs.iss.vector_env import ISSVectorEnv

FREE_FLIGHT = dict(collision_boxes_path=None, dock_enabled=False)


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
    single-env path -- the failure mode seamstress has, where done-logic is
    implemented twice.
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
        collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [400.0, 400.0, 400.0]}],
        dock_enabled=False,
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


def test_autoreset_mode_is_declared_in_metadata():
    env = ISSVectorEnv(num_envs=2)
    assert "autoreset_mode" in env.metadata
