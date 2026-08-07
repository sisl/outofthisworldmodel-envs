import jax.numpy as jnp
import numpy as np
import pytest

from owm_envs.envs.common.config import DockConfig, PhysicsConfig
from owm_envs.envs.common.sensing import PRESETS
from owm_envs.envs.iss_numerical.config import OBS_MODE_DIM, NumericalConfig
from owm_envs.envs.iss_numerical.vector_env import NumericalVectorEnv

FREE_FLIGHT = dict(physics=PhysicsConfig(collision_boxes_path=None), dock=DockConfig(enabled=False))


def test_spaces_are_batched_correctly():
    env = NumericalVectorEnv(num_envs=3)
    assert env.num_envs == 3
    assert env.single_observation_space.shape == (OBS_MODE_DIM["relative"],)
    assert env.single_action_space.shape == (6,)
    assert env.observation_space.shape == (3, OBS_MODE_DIM["relative"])
    assert env.action_space.shape == (3, 6)


@pytest.mark.parametrize("mode", ["absolute", "chaser_absolute", "chief_absolute", "relative"])
def test_observation_width_follows_the_mode(mode):
    env = NumericalVectorEnv(num_envs=2, cfg=NumericalConfig(observation={"mode": mode}))
    obs, info = env.reset(seed=0)
    assert obs.shape == (2, OBS_MODE_DIM[mode])
    assert obs.dtype == np.float32


def test_reset_returns_batched_observations_and_both_state_channels():
    env = NumericalVectorEnv(num_envs=3)
    obs, info = env.reset(seed=0)
    assert obs.shape == (3, OBS_MODE_DIM["relative"])
    assert obs.dtype == np.float32
    assert info["success"].shape == (3,)
    assert info["collision"].shape == (3,)
    assert info["state"].shape == (3, 21)
    assert info["measured_state"].shape == (3, 21)


def test_step_returns_the_five_tuple_batched():
    env = NumericalVectorEnv(num_envs=3, cfg=NumericalConfig(**FREE_FLIGHT))
    env.reset(seed=0)
    obs, rewards, terminations, truncations, infos = env.step(np.zeros((3, 6), dtype=np.float32))
    assert obs.shape == (3, OBS_MODE_DIM["relative"])
    assert rewards.shape == (3,)
    assert terminations.shape == (3,) and terminations.dtype == bool
    assert truncations.shape == (3,) and truncations.dtype == bool
    assert infos["success"].shape == (3,)
    assert infos["state"].shape == (3, 21)
    assert infos["measured_state"].shape == (3, 21)


def test_autoreset_mode_is_declared_in_metadata():
    env = NumericalVectorEnv(num_envs=3)
    assert "autoreset_mode" in env.metadata


def test_noise_leaves_epoch_and_chief_untouched_while_chaser_slices_differ():
    cfg = NumericalConfig(sensor_noise=PRESETS["cooperative"])
    env = NumericalVectorEnv(num_envs=4, cfg=cfg)
    _, info = env.reset(seed=3)
    measured, true = info["measured_state"], info["state"]

    np.testing.assert_array_equal(measured[:, 0:2], true[:, 0:2])
    np.testing.assert_array_equal(measured[:, 2:8], true[:, 2:8])
    assert not np.allclose(measured[:, 8:11], true[:, 8:11])


@pytest.mark.parametrize(
    "cfg",
    [
        NumericalConfig(),
        NumericalConfig(sensor_noise=PRESETS["cooperative"]),
        NumericalConfig(observation={"goal_error": True}),
    ],
    ids=["noiseless", "noisy", "goal_error"],
)
def test_the_carried_states_stay_float64(cfg):
    env = NumericalVectorEnv(num_envs=3, cfg=cfg)
    env.reset(seed=0)
    assert env._states.dtype == jnp.float64
    for _ in range(5):
        env.step(np.zeros((3, 6), dtype=np.float32))
        assert env._states.dtype == jnp.float64


def test_terminated_sub_env_autoresets_on_the_next_step():
    # A box covering the whole start sphere guarantees an immediate collision
    # in every lane, so the next step must report the NEXT_STEP autoreset
    # contract: no termination, no truncation, and zero reward.
    env = NumericalVectorEnv(num_envs=3, cfg=NumericalConfig(
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
