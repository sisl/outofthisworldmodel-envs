import gymnasium as gym
import jax.numpy as jnp
import numpy as np

from owm_envs.envs import ENV_REGISTRY


def test_registry_has_iss():
    spec = ENV_REGISTRY["iss"]
    assert spec.gym_id == "ISS-Docking-v0"
    assert spec.layout.state_dim == 13


def test_spec_pieces_compose():
    spec = ENV_REGISTRY["iss"]
    cfg = spec.config_cls()
    dynamics = spec.make_dynamics(cfg)
    assert dynamics.state_dim == spec.layout.state_dim
    state = jnp.arange(13.0)
    np.testing.assert_array_equal(np.asarray(spec.view(state)), np.arange(13.0))
    venv = spec.make_vector_env(2, cfg)
    assert venv.num_envs == 2
    venv.close()


def test_gym_ids_make():
    for spec in ENV_REGISTRY.values():
        env = gym.make(spec.gym_id)
        env.close()
