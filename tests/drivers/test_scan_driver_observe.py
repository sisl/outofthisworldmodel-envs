"""The scan driver's `observe` hook: what the rollout records versus what it
integrates.

Only `EnvSpec.make_observe` reshapes the recorded observation. `true_state`
still carries the env's full state, and the policies and reward still read the
canonical view of it, so narrowing an observation cannot narrow the dynamics.

iss-numerical is not in the registry yet (it lands with its adapters), so the
spec is assembled here from the pieces that exist.
"""

from dataclasses import replace

import numpy as np
import pytest

from owm_envs.drivers.scan_driver import ScanDriver
from owm_envs.drivers.types import RolloutSpec
from owm_envs.envs import EnvSpec
from owm_envs.envs.common.goal import GOAL_ERROR_DIM
from owm_envs.envs.common.policies import PolicyConfig
from owm_envs.envs.common.sensing import PRESETS
from owm_envs.envs.iss_numerical.config import NUM_LAYOUT, OBS_MODE_DIM, NumericalConfig
from owm_envs.envs.iss_numerical.dynamics import NumericalDynamics, relative_view
from owm_envs.envs.iss_numerical.observe import make_observe

NUMERICAL_SPEC = EnvSpec(
    name="iss-numerical",
    gym_id="ISS-Numerical-Docking-v1",
    config_cls=NumericalConfig,
    layout=NUM_LAYOUT,
    make_dynamics=NumericalDynamics,
    # ScanDriver never builds a vector env; the registry entry supplies the
    # real one in the task that registers this env.
    make_vector_env=lambda num_envs, cfg: None,
    view=relative_view,
    renderable=False,
    make_observe=make_observe,
)


def _cfg(mode, goal_error=False, sensor_noise=None) -> NumericalConfig:
    return NumericalConfig(
        dt=0.5,
        max_steps=6,
        max_range_m=None,
        dock={"enabled": False},
        physics={"collision_boxes_path": [], "linear_damping": 0.0},
        orbit={"start_radius_range_m": (80.0, 120.0)},
        observation={"mode": mode, "goal_error": goal_error},
        sensor_noise=sensor_noise or PRESETS["off"],
    )


def _rollout(mode, goal_error=False, env_spec=NUMERICAL_SPEC, policy_type="dock",
             sensor_noise=None, policy_observe="measurement"):
    driver = ScanDriver(
        cfg=_cfg(mode, goal_error, sensor_noise),
        policy_cfg=PolicyConfig(type=policy_type, observe=policy_observe),
        num_envs=2,
        env_spec=env_spec,
    )
    batch = driver.generate(RolloutSpec(num_episodes=2, max_steps=6, seed=0))
    batch.validate()
    return batch


@pytest.mark.parametrize("mode", sorted(OBS_MODE_DIM))
def test_recorded_observations_take_the_configured_mode_width(mode):
    batch = _rollout(mode)
    assert batch.observations.shape[2] == OBS_MODE_DIM[mode]
    # The state behind them is the full one regardless of what is reported.
    assert batch.true_state.shape[2] == NUM_LAYOUT.state_dim == 21


@pytest.mark.parametrize("policy_type", ["random", "dock", "orbit", "union"])
def test_the_goal_block_is_appended_to_the_mode_shaped_observation(policy_type):
    """Every policy type builds its own concatenation, and `union` builds its
    block inside a `lax.switch` whose branches must agree on dtype -- a zero
    literal beside two blocks the f64 view produced. Running all four through
    the scan is what proves the augment traces at all for an env that observes
    narrower than it integrates."""
    batch = _rollout("relative", goal_error=True, policy_type=policy_type)
    assert batch.observations.shape[2] == OBS_MODE_DIM["relative"] + GOAL_ERROR_DIM


def test_the_noise_range_is_the_standoff_not_the_orbit_radius():
    """`sigma_pos_frac_of_range` is 1% under the noncooperative preset: ~1 m
    of position error at a ~100 m standoff, or ~68 km measured against the
    chaser's absolute ECI radius instead. Only the driver supplying the range
    from the view keeps it the former.

    `observe="state"` flies the policy on the true state, so the noised and
    clean rollouts follow the same trajectory off the same seed and their
    difference is exactly the injected error.
    """
    clean = _rollout("relative", policy_observe="state")
    noisy = _rollout("relative", policy_observe="state",
                     sensor_noise=PRESETS["noncooperative"])
    length = int(clean.lengths[0])
    assert int(noisy.lengths[0]) == length
    error = np.linalg.norm(
        noisy.observations[0, :length, 2:5] - clean.observations[0, :length, 2:5], axis=1
    )
    assert error.max() < 10.0, error.max()   # metres, not kilometres
    assert error.mean() > 0.05               # and the noise really is switched on


def test_a_relative_observation_reports_the_standoff_not_the_orbit_radius():
    """The recorded rows really are in the world frame: ~100 m of separation
    where the state beside them holds a ~6.8e6 m ECI radius. Recomputing the
    view from the stored `true_state` would not check this -- that row is
    f32, and the differencing this mode does needs the f64 state (see
    `tests/envs/iss_numerical/test_observe.py`)."""
    batch = _rollout("relative")
    length = int(batch.lengths[0])
    standoff = np.linalg.norm(batch.observations[0, :length, 2:5], axis=1)
    assert np.all((standoff > 50.0) & (standoff < 200.0))
    assert np.all(np.linalg.norm(batch.true_state[0, :length, 8:11], axis=1) > 6e6)


def test_an_env_without_the_hook_records_its_state_unchanged():
    """`make_observe` defaults to None, which is the identity: the envs
    already in the suite see no new step between state and observation."""
    batch = _rollout("relative", env_spec=replace(NUMERICAL_SPEC, make_observe=None))
    assert batch.observations.shape[2] == NUM_LAYOUT.state_dim
    # The truth channel is stored at the dynamics' own float64 and the
    # observation at float32, so "unchanged" means unchanged up to that
    # narrowing -- exactly, once the narrowing is applied to both sides.
    np.testing.assert_array_equal(
        batch.observations, batch.true_state.astype(batch.observations.dtype)
    )
