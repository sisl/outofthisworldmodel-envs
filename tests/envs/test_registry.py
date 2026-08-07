import pickle

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


def test_every_registered_env_is_renderable():
    # The CLI's --render guard reads this flag instead of comparing names.
    # iss-hcw flipped to True once its own render adapter landed, alongside
    # iss's.
    assert ENV_REGISTRY["iss"].renderable is True
    assert ENV_REGISTRY["iss-hcw"].renderable is True


def test_iss_render_adapter_poses_a_frame_from_a_view_row():
    spec = ENV_REGISTRY["iss"]
    adapter = spec.make_render_adapter(spec.config_cls())
    state = np.arange(13.0)
    inputs = adapter(state, None)
    np.testing.assert_array_equal(inputs.position_world, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(inputs.quaternion_bw, [6.0, 7.0, 8.0, 9.0])
    assert inputs.lighting is None


def test_iss_hcw_render_adapter_poses_a_frame_from_the_view_slice():
    spec = ENV_REGISTRY["iss-hcw"]
    adapter = spec.make_render_adapter(spec.config_cls())
    state = np.zeros(15, dtype=np.float64)
    state[2:15] = np.arange(13.0)
    inputs = adapter(state, None)
    np.testing.assert_array_equal(inputs.position_world, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(inputs.quaternion_bw, [6.0, 7.0, 8.0, 9.0])
    assert inputs.lighting is not None


def test_iss_render_adapter_factory_and_adapter_are_picklable():
    # Render worker processes rebuild adapters from (env_name, cfg) via
    # ENV_REGISTRY, so both the factory and the callable it returns have to
    # survive a pickle round trip -- which rules out lambdas and closures.
    spec = ENV_REGISTRY["iss"]
    factory = pickle.loads(pickle.dumps(spec.make_render_adapter))
    adapter = pickle.loads(pickle.dumps(factory(spec.config_cls())))
    state = np.arange(13.0)
    np.testing.assert_array_equal(adapter(state, None).position_world, [0.0, 1.0, 2.0])


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
