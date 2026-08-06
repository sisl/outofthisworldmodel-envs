import numpy as np

from owm_envs.envs.common.config import DockConfig, PhysicsConfig
from owm_envs.envs.iss_hcw.config import HCWConfig
from owm_envs.envs.iss_hcw.vector_env import HCWVectorEnv

FREE_FLIGHT = dict(physics=PhysicsConfig(collision_boxes_path=None), dock=DockConfig(enabled=False))


def test_spaces_are_batched_correctly():
    env = HCWVectorEnv(num_envs=3)
    assert env.num_envs == 3
    assert env.single_observation_space.shape == (15,)
    assert env.single_action_space.shape == (6,)
    assert env.observation_space.shape == (3, 15)
    assert env.action_space.shape == (3, 6)


def test_reset_returns_batched_observations():
    env = HCWVectorEnv(num_envs=3)
    obs, info = env.reset(seed=0)
    assert obs.shape == (3, 15)
    assert obs.dtype == np.float32
    assert info["success"].shape == (3,)
    assert info["collision"].shape == (3,)
    assert info["state"].shape == (3, 15)


def test_step_returns_the_five_tuple_batched():
    env = HCWVectorEnv(num_envs=3, cfg=HCWConfig(**FREE_FLIGHT))
    env.reset(seed=0)
    obs, rewards, terminations, truncations, infos = env.step(np.zeros((3, 6), dtype=np.float32))
    assert obs.shape == (3, 15)
    assert rewards.shape == (3,)
    assert terminations.shape == (3,) and terminations.dtype == bool
    assert truncations.shape == (3,) and truncations.dtype == bool
    assert infos["success"].shape == (3,)


def test_autoreset_mode_is_declared_in_metadata():
    env = HCWVectorEnv(num_envs=3)
    assert "autoreset_mode" in env.metadata


def test_terminated_sub_env_autoresets_on_the_next_step():
    # A box covering the whole start sphere guarantees an immediate collision
    # in every lane, so the next step must report the NEXT_STEP autoreset
    # contract: no termination, no truncation, and zero reward.
    env = HCWVectorEnv(num_envs=3, cfg=HCWConfig(
        max_steps=100,
        physics=PhysicsConfig(
            collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [400.0, 400.0, 400.0]}],
        ),
        dock=DockConfig(enabled=False),
    ))
    env.reset(seed=0)
    zero = np.zeros((3, 6), dtype=np.float32)

    _, _, terminations, _, infos = env.step(zero)
    assert terminations.all()
    assert infos["collision"].all()

    _, rewards, terminations, truncations, _ = env.step(zero)
    assert not terminations.any()
    assert not truncations.any()
    np.testing.assert_allclose(rewards, np.zeros(3), atol=1e-6)
