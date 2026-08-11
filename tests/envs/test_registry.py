import gymnasium as gym
import jax.numpy as jnp
import numpy as np

from owm_envs.envs import ENV_REGISTRY


def test_registry_has_iss():
    spec = ENV_REGISTRY["iss"]
    assert spec.gym_id == "ISS-Docking-v0"
    assert spec.layout.state_dim == 13


def test_registry_has_iss_hcw():
    spec = ENV_REGISTRY["iss-hcw"]
    assert spec.gym_id == "ISS-HCW-Docking-v0"
    assert spec.layout.state_dim == 15
    assert spec.make_dynamics(spec.config_cls()).state_dim == 15


def test_only_the_envs_the_renderer_can_pose_are_renderable():
    # The video path reads recorded true_state rows as iss-layout poses, so
    # this flag is what the CLI's --render guard reads instead of comparing
    # names. iss-hcw flips to True with the RenderInputs seam, not before.
    assert ENV_REGISTRY["iss"].renderable is True
    assert ENV_REGISTRY["iss-hcw"].renderable is False


def test_registry_keys_match_their_spec_name():
    assert all(k == s.name for k, s in ENV_REGISTRY.items())


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
