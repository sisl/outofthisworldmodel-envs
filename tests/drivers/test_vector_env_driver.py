import subprocess
import sys

import numpy as np
import pytest

from owm_envs.drivers.types import RolloutSpec
from owm_envs.drivers.vector_env_driver import VectorEnvDriver
from owm_envs.envs.iss.config import DockConfig, ISSConfig, PhysicsConfig
from owm_envs.envs.iss.policies import PolicyConfig
from owm_envs.envs.iss.policy_source import IssPolicySource
from owm_envs.envs.iss.vector_env import ISSVectorEnv

FREE_FLIGHT_PHYSICS = dict(collision_boxes_path=None)
FREE_FLIGHT_DOCK = dict(enabled=False)


def free_flight_cfg(physics=None, dock=None) -> ISSConfig:
    return ISSConfig(
        physics=PhysicsConfig(**{**FREE_FLIGHT_PHYSICS, **(physics or {})}),
        dock=DockConfig(**{**FREE_FLIGHT_DOCK, **(dock or {})}),
    )


def make_driver(num_envs=2, policy_type="dock", physics=None, dock=None):
    cfg = free_flight_cfg(physics=physics, dock=dock)
    policy_cfg = PolicyConfig(type=policy_type)
    return VectorEnvDriver(
        env_factory=lambda: ISSVectorEnv(num_envs=num_envs, cfg=cfg),
        policy_source=IssPolicySource(cfg, policy_cfg),
    )


class _ConstantPolicySource:
    """A trivial PolicySource with no JAX and no ISS types anywhere in it --
    the extensibility claim the driver seam exists to make real."""

    records_policy_ids = False

    def __init__(self, action: np.ndarray):
        self._action = np.asarray(action, dtype=np.float32)

    def new_episode(self, seed: int):
        del seed
        return None

    def act(self, observation, episode_state, step):
        del observation, episode_state, step
        return self._action.copy()

    def policy_id(self, episode_state):
        del episode_state
        return 0


def test_generates_the_requested_number_of_episodes():
    batch = make_driver().generate(RolloutSpec(num_episodes=4, max_steps=20, seed=0))
    batch.validate()
    assert batch.num_episodes == 4


def test_output_shapes_and_dtypes():
    # Each episode stores N + 1 observations (seed state plus each post-step
    # state, including the terminal one) against N + 1 actions (N real, one
    # zero pad), so a max_steps=15 horizon yields length-16 episodes.
    batch = make_driver().generate(RolloutSpec(num_episodes=3, max_steps=15, seed=0))
    assert batch.observations.shape == (3, 16, 13)
    assert batch.actions.shape == (3, 16, 6)
    assert batch.rewards.shape == (3, 16)
    assert batch.observations.dtype == np.float32
    assert batch.actions.dtype == np.float32
    assert batch.lengths.dtype == np.int32


def test_free_flight_episodes_run_to_max_steps_and_truncate():
    # 12 real steps plus the seed observation is a length-13 episode.
    batch = make_driver().generate(RolloutSpec(num_episodes=2, max_steps=12, seed=0))
    assert np.all(batch.lengths == 13)
    assert np.all(batch.truncated)
    assert not np.any(batch.terminated)


def test_collision_terminates_episodes_early():
    driver = make_driver(
        physics=dict(
            collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [400.0, 400.0, 400.0]}]
        ),
        dock=dict(enabled=False),
    )
    batch = driver.generate(RolloutSpec(num_episodes=2, max_steps=50, seed=0))
    batch.validate()
    assert np.all(batch.terminated)
    assert not np.any(batch.truncated)
    assert np.all(batch.lengths < 50)


def test_terminal_state_is_stored_with_a_padded_final_action():
    # The terminal (collision) observation must be captured -- it's the
    # event a world model needs to learn -- with a zero action padding the
    # final slot so observations and actions stay equal length.
    driver = make_driver(
        physics=dict(
            collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [400.0, 400.0, 400.0]}]
        ),
        dock=dict(enabled=False),
    )
    batch = driver.generate(RolloutSpec(num_episodes=2, max_steps=50, seed=0))
    batch.validate()
    assert np.all(batch.terminated)
    for i, length in enumerate(batch.lengths):
        initial = batch.observations[i, 0]
        terminal = batch.observations[i, length - 1]
        assert not np.allclose(terminal, 0.0)
        assert not np.allclose(terminal, initial)
        np.testing.assert_array_equal(batch.actions[i, length - 1], 0.0)


def test_is_deterministic_in_the_seed():
    a = make_driver().generate(RolloutSpec(num_episodes=3, max_steps=10, seed=7))
    b = make_driver().generate(RolloutSpec(num_episodes=3, max_steps=10, seed=7))
    c = make_driver().generate(RolloutSpec(num_episodes=3, max_steps=10, seed=8))
    np.testing.assert_array_equal(a.observations, b.observations)
    assert not np.allclose(a.observations, c.observations)


def test_padding_past_episode_length_is_zero():
    driver = make_driver(
        physics=dict(
            collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [400.0, 400.0, 400.0]}]
        ),
        dock=dict(enabled=False),
    )
    batch = driver.generate(RolloutSpec(num_episodes=2, max_steps=40, seed=0))
    for i, length in enumerate(batch.lengths):
        assert np.all(batch.observations[i, length:] == 0.0)
        assert np.all(batch.actions[i, length:] == 0.0)


def test_union_policy_records_policy_ids():
    batch = make_driver(policy_type="union").generate(
        RolloutSpec(num_episodes=8, max_steps=10, seed=0)
    )
    assert batch.policy_ids is not None
    assert batch.policy_ids.shape == (8,)
    assert set(np.unique(batch.policy_ids)).issubset({0, 1, 2})


def test_non_mixture_policy_leaves_policy_ids_none():
    batch = make_driver(policy_type="dock").generate(RolloutSpec(num_episodes=2, max_steps=10, seed=0))
    assert batch.policy_ids is None


def test_actions_stay_within_the_control_limits():
    cfg = free_flight_cfg()
    batch = make_driver().generate(RolloutSpec(num_episodes=2, max_steps=10, seed=0))
    assert np.all(np.abs(batch.actions[..., 0:3]) <= cfg.control.limit_force_n + 1e-3)
    assert np.all(np.abs(batch.actions[..., 3:6]) <= cfg.control.limit_torque_nm + 1e-3)


def test_rejects_a_non_positive_episode_count():
    with pytest.raises(ValueError):
        make_driver().generate(RolloutSpec(num_episodes=0, max_steps=10, seed=0))


def test_module_does_not_import_jax():
    # The whole point of the driver seam: a future Basilisk or brahe backend
    # must not have to bring JAX along just to get dataset generation. Check
    # in a subprocess so this is a real assertion about what importing the
    # module does, not just about what happens to already be loaded here.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import owm_envs.drivers.vector_env_driver; print('jax' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False", result.stderr


def test_driver_works_with_a_non_iss_non_jax_policy_source():
    cfg = free_flight_cfg()
    action = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    driver = VectorEnvDriver(
        env_factory=lambda: ISSVectorEnv(num_envs=2, cfg=cfg),
        policy_source=_ConstantPolicySource(action),
    )
    batch = driver.generate(RolloutSpec(num_episodes=2, max_steps=5, seed=0))
    batch.validate()
    assert batch.num_episodes == 2
    assert batch.policy_ids is None
    for i, length in enumerate(batch.lengths):
        # Every real action (all but the final zero-padded slot) is the
        # constant action the fake policy source always returns.
        np.testing.assert_array_equal(
            batch.actions[i, : length - 1], np.tile(action, (length - 1, 1))
        )
