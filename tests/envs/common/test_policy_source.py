"""TaskPolicySource's preference for `info["measured_state"]` over the raw
`observation` argument, for both `act()` and `augment_observation()`.

This preference exists for iss-numerical: its recorded observation can be
mode-shaped (narrower than, or reordered from, the raw state `view` and the
scripted policies expect -- see `envs/iss_numerical/observe.py`), so both
methods read the always-raw-layout `info["measured_state"]` when an env
publishes it, falling back to `observation` when it does not (iss and
iss-hcw, unchanged from before this key existed).

`augment_observation` never RECORDS `measured_state`, only reads it for the
goal-error block: the recorded row is always the driver's own `observation`,
verbatim (`make_augment`'s `observed=` override). Re-deriving the recorded
row from `measured_state` would difference iss-numerical's independently
float32-narrowed ~6.8e6 m ECI columns a second time, quantizing it to ~1 m
against the 0.1 m dock gate for no reason -- the env already computed that
exact row once, correctly.
"""

import jax
import jax.numpy as jnp
import numpy as np

from owm_envs.envs import ENV_REGISTRY
from owm_envs.envs.common.config import ObservationConfig, dock_target
from owm_envs.envs.common.goal import dock_goal_error
from owm_envs.envs.common.policies import PolicyConfig
from owm_envs.envs.common.policy_source import TaskPolicySource
from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss_numerical.config import NumericalConfig
from owm_envs.envs.iss_numerical.dynamics import relative_view


def _spy_source(cfg, policy_cfg, **kwargs):
    source = TaskPolicySource(cfg, policy_cfg, **kwargs)
    captured = {}

    def spy(view_state, key, extras):
        captured["state"] = np.asarray(view_state)
        return jnp.zeros(6, jnp.float32)

    source._policy_fn = spy
    return source, captured


def test_act_prefers_measured_state_over_observation_when_present():
    cfg = ISSConfig()
    policy_cfg = PolicyConfig(type="random")  # observe defaults to "measurement"
    source, captured = _spy_source(cfg, policy_cfg)
    episode_state = source.new_episode(0)

    observation = np.zeros(13, dtype=np.float32)
    measured = np.arange(13, dtype=np.float32)

    source.act(observation, episode_state, 0, {"measured_state": measured})
    np.testing.assert_array_equal(captured["state"], measured)


def test_act_falls_back_to_observation_when_measured_state_is_absent():
    # iss/iss-hcw never publish "measured_state" -- this is the exact
    # behaviour they had before the key existed.
    cfg = ISSConfig()
    policy_cfg = PolicyConfig(type="random")
    source, captured = _spy_source(cfg, policy_cfg)
    episode_state = source.new_episode(0)

    observation = np.arange(13, dtype=np.float32)
    source.act(observation, episode_state, 0, {})
    np.testing.assert_array_equal(captured["state"], observation)


def test_act_with_observe_state_still_prefers_info_state():
    cfg = ISSConfig()
    policy_cfg = PolicyConfig(type="random", observe="state")
    source, captured = _spy_source(cfg, policy_cfg)
    episode_state = source.new_episode(0)

    observation = np.zeros(13, dtype=np.float32)
    true_state = np.arange(13, dtype=np.float32) + 100.0
    measured = np.arange(13, dtype=np.float32)

    source.act(observation, episode_state, 0, {"state": true_state, "measured_state": measured})
    np.testing.assert_array_equal(captured["state"], true_state)


def test_augment_observation_records_the_env_observation_verbatim():
    # The goal block is computed from `measured_state` (here, exactly at the
    # dock target -- zero error), while the recorded row is the DRIVER'S
    # observation, unrelated to `measured_state` -- not re-derived from it.
    cfg = ISSConfig(observation=ObservationConfig(goal_error=True))
    policy_cfg = PolicyConfig(type="dock")
    source = TaskPolicySource(cfg, policy_cfg)
    episode_state = source.new_episode(0)

    target = dock_target(cfg)
    at_target = np.concatenate(
        [target[0:3], np.zeros(3, dtype=np.float32), target[3:7], np.zeros(3, dtype=np.float32)]
    ).astype(np.float32)
    different_observation = np.full(13, 999.0, dtype=np.float32)

    augmented = source.augment_observation(
        different_observation, episode_state, {"measured_state": at_target}
    )
    assert augmented.shape == (25,)
    np.testing.assert_array_equal(augmented[:13], different_observation)
    np.testing.assert_allclose(augmented[13:], 0.0, atol=1e-5)


def test_augment_observation_falls_back_to_observation_when_measured_state_is_absent():
    cfg = ISSConfig(observation=ObservationConfig(goal_error=True))
    policy_cfg = PolicyConfig(type="dock")
    source = TaskPolicySource(cfg, policy_cfg)
    episode_state = source.new_episode(0)

    target = dock_target(cfg)
    at_target = np.concatenate(
        [target[0:3], np.zeros(3, dtype=np.float32), target[3:7], np.zeros(3, dtype=np.float32)]
    ).astype(np.float32)

    augmented = source.augment_observation(at_target, episode_state, {})
    np.testing.assert_array_equal(augmented[:13], at_target)
    np.testing.assert_allclose(augmented[13:], 0.0, atol=1e-5)


def test_numerical_registry_entry_records_the_exact_mode_shaped_observation():
    """End-to-end check against the real iss-numerical registry entry: the
    row `TaskPolicySource.augment_observation` records is BYTE-IDENTICAL to
    the mode-shaped observation the vector env would have produced -- never
    re-derived from `info["measured_state"]`, which would difference its
    already-float32-narrowed ~6.8e6 m ECI columns a second time and quantize
    the recorded row to ~1 m against the 0.1 m dock gate for no reason (the
    bug this test guards against on the RECORDED side).

    The goal BLOCK is a separate story: it is computed from
    `info["measured_state"]`, which this suite's info dict always narrows to
    float32 (see `env.py`), so it inherits that channel's ECI-narrowing
    budget regardless of what `augment_observation` does with it -- checked
    here at a loose, budget-sized tolerance rather than float32 grain.
    """
    spec = ENV_REGISTRY["iss-numerical"]
    cfg = NumericalConfig(observation={"mode": "relative", "goal_error": True})
    policy_cfg = PolicyConfig(type="dock")
    source = TaskPolicySource(cfg, policy_cfg, view=spec.view)
    episode_state = source.new_episode(0)

    dynamics = spec.make_dynamics(cfg)
    raw_state = dynamics.reset(jax.random.PRNGKey(0))  # float64
    observe_fn = spec.make_observe(cfg)
    mode_shaped_obs = np.asarray(observe_fn(raw_state), dtype=np.float32)
    assert mode_shaped_obs.shape == (15,)
    measured_state = np.asarray(raw_state, dtype=np.float32)

    augmented = source.augment_observation(
        mode_shaped_obs, episode_state, {"measured_state": measured_state}
    )
    assert augmented.shape == (15 + 12,)
    # Byte-identical: never re-derived from measured_state.
    np.testing.assert_array_equal(augmented[:15], mode_shaped_obs)

    # The block, by contrast, is computed off `measured_state` (float32) --
    # a 1 m bound covers the ECI-narrowing budget on the position channel
    # (indices 0:3) with room to spare; the other channels are far tighter.
    expected = np.asarray(
        dock_goal_error(relative_view(raw_state), jnp.asarray(dock_target(cfg))), dtype=np.float64
    )
    np.testing.assert_allclose(augmented[15:18], expected[0:3], atol=1.0)
    np.testing.assert_allclose(augmented[18:], expected[3:], atol=1e-3)
