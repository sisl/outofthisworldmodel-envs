import pickle

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from owm_envs.envs import ENV_REGISTRY


def test_registry_has_iss():
    spec = ENV_REGISTRY["iss"]
    assert spec.gym_id == "ISS-Docking-v1"
    assert spec.layout.state_dim == 13


def test_registry_has_iss_hcw():
    spec = ENV_REGISTRY["iss-hcw"]
    assert spec.gym_id == "ISS-HCW-Docking-v1"
    assert spec.layout.state_dim == 15
    assert spec.make_dynamics(spec.config_cls()).state_dim == 15


def test_registry_has_iss_numerical():
    spec = ENV_REGISTRY["iss-numerical"]
    assert spec.gym_id == "ISS-Numerical-Docking-v1"
    assert spec.layout.state_dim == 21
    assert spec.make_dynamics(spec.config_cls()).state_dim == 21
    assert spec.make_observe is not None


def test_every_registered_env_is_renderable():
    # The CLI's --render guard reads this flag instead of comparing names.
    # iss-hcw flipped to True once its own render adapter landed, alongside
    # iss's, and iss-numerical the same way.
    assert ENV_REGISTRY["iss"].renderable is True
    assert ENV_REGISTRY["iss-hcw"].renderable is True
    assert ENV_REGISTRY["iss-numerical"].renderable is True


def test_renderable_means_having_a_render_adapter():
    """The flag and the adapter are one fact recorded twice, so they have to
    agree. A renderable env with no adapter passes the CLI's guard and then
    fails inside a render worker as a BrokenProcessPool, hours into a run --
    the exact failure the flag exists to turn into a usage error. An adapter
    with the flag off is the smaller mistake of rendering nobody can ask for.
    """
    assert all(
        (spec.make_render_adapter is not None) == spec.renderable
        for spec in ENV_REGISTRY.values()
    )


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


def test_iss_numerical_render_adapter_poses_a_frame_from_the_derived_view():
    # Unlike iss and iss-hcw, the raw slices are not the view -- the adapter
    # has to call `relative_view` itself -- so this only pins that a frame
    # comes out with lighting attached, not any particular numbers; the
    # derivation itself is `test_view.py`'s job and `test_render_adapter.py`
    # exercises this adapter's own lighting/pose wiring directly.
    spec = ENV_REGISTRY["iss-numerical"]
    cfg = spec.config_cls()
    dynamics = spec.make_dynamics(cfg)
    state = np.asarray(dynamics.reset(jax.random.PRNGKey(0)), dtype=np.float32)
    adapter = spec.make_render_adapter(cfg)
    inputs = adapter(state, None)
    assert inputs.position_world.shape == (3,)
    assert inputs.quaternion_bw.shape == (4,)
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


def test_iss_numerical_render_adapter_factory_and_adapter_are_picklable():
    spec = ENV_REGISTRY["iss-numerical"]
    cfg = spec.config_cls()
    dynamics = spec.make_dynamics(cfg)
    state = np.asarray(dynamics.reset(jax.random.PRNGKey(0)), dtype=np.float32)

    factory = pickle.loads(pickle.dumps(spec.make_render_adapter))
    adapter = pickle.loads(pickle.dumps(factory(cfg)))
    inputs = adapter(state, None)
    assert inputs.position_world.shape == (3,)
    assert inputs.lighting is not None


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


def test_gym_ids_are_v1():
    assert {spec.gym_id for spec in ENV_REGISTRY.values()} == {
        "ISS-Docking-v1",
        "ISS-HCW-Docking-v1",
        "ISS-Numerical-Docking-v1",
    }


def test_v0_ids_are_gone():
    # The reward changed meaning; a v0 that still resolves would be a lie
    # about what it produces.
    import gymnasium

    for retired in ("ISS-Docking-v0", "ISS-HCW-Docking-v0", "ISS-Numerical-Docking-v0"):
        with pytest.raises(gymnasium.error.DeprecatedEnv):
            gymnasium.make(retired)
