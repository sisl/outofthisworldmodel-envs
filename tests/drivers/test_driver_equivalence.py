"""The guard against the two drivers silently diverging.

Done-logic exists twice, once traced through JAX in `ScanDriver` and once in
numpy in `VectorEnvDriver`, and two copies of the same rule are free to drift
apart. The two implementations exist for real performance reasons, so they
must prove they agree.

Equivalence is tested with the `dock` policy, which is deterministic and ignores
its PRNG key, so the comparison isolates rollout mechanics -- stepping,
termination detection, episode segmentation, padding -- from PRNG plumbing.
Matching two independent PRNG pipelines bitwise is a tar pit and is not the
property that matters.

Bitwise observation/action/reward equality only holds, and is only asserted,
in the one-episode-per-lane regime (`num_episodes == num_envs`, and the
horizon exactly covers `max_steps`): `ScanDriver` seeds its lanes' *initial*
reset to match `VectorEnvDriver`'s, but the two drivers derive each lane's
*subsequent* per-episode reset key differently once autoreset kicks in mid-
horizon (`ScanDriver` threads an independent per-lane, per-step key through
the scan; the vector env splits one shared key each time any lane resets).
Beyond one episode per lane -- which is the regime real dataset generation
actually runs in, e.g. 512 episodes from 8 lanes -- the guarantee this test
suite provides is STRUCTURAL, not bitwise: identical episode segmentation,
lengths, and termination flags, and both drivers honoring the same episode
convention (terminal action pad, terminal observation present). That is
deliberate, not a gap: what can realistically drift between two independent
rollout implementations is the segmentation/termination logic (two copies of
"is this lane done, and where does the next episode start"), not the
per-step arithmetic, which both drivers call through the same backend
dynamics, `docking_reward`, and policy functions.
"""

import numpy as np
import pytest

from owm_envs.drivers.scan_driver import ScanDriver
from owm_envs.drivers.types import RolloutSpec
from owm_envs.drivers.vector_env_driver import VectorEnvDriver
from owm_envs.envs import ENV_REGISTRY
from owm_envs.envs.common.config import DockConfig, ObservationConfig, PhysicsConfig
from owm_envs.envs.common.policies import PolicyConfig
from owm_envs.envs.common.policy_source import TaskPolicySource
from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss.vector_env import ISSVectorEnv

# The contract below is env-independent -- segmentation and termination are
# the drivers' own logic, not the backend's -- so every registered env runs
# it. iss-hcw is the case that keeps it honest: a 15-wide float64 state
# behind a 13D task view, where a driver that assumed either width would
# still pass on iss alone.
ENV_NAMES = ("iss", "iss-hcw")

DETERMINISTIC = PolicyConfig(type="dock")


def start_shell(cfg):
    """The start-radius range this config's env actually disperses over.

    `iss` draws it from `physics`; `iss-hcw` draws it from its `orbit`
    dispersions and ignores the physics field entirely, so a test that needs
    the shell has to ask the config rather than assume the section.
    """
    orbit = getattr(cfg, "orbit", None)
    return cfg.physics.start_radius_range_m if orbit is None else orbit.start_radius_range_m


def free_flight_config(env_spec, **overrides):
    """No collision geometry and no dock gate: episodes end at a step limit only."""
    return env_spec.config_cls(
        physics=PhysicsConfig(collision_boxes_path=None),
        dock=DockConfig(enabled=False),
        **overrides,
    )


def collision_config(env_spec):
    """One collision box swallowing the whole start shell, so every episode
    terminates on its first step whatever radius it started from.

    The box is sized off the env's own dispersions rather than a fixed
    number: the two envs do not start from the same shell, and a box that
    covered one would leave starts outside the other.
    """
    size = 3.0 * start_shell(free_flight_config(env_spec))[1]
    return env_spec.config_cls(
        physics=PhysicsConfig(
            collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [size, size, size]}]
        ),
        dock=DockConfig(enabled=False),
    )


def drivers_for(cfg, env_spec, num_envs=2):
    vec = VectorEnvDriver(
        env_factory=lambda: env_spec.make_vector_env(num_envs, cfg),
        policy_source=TaskPolicySource(cfg, DETERMINISTIC, view=env_spec.view),
    )
    scan = ScanDriver(
        cfg=cfg, policy_cfg=DETERMINISTIC, num_envs=num_envs, env_spec=env_spec
    )
    return vec, scan


@pytest.mark.parametrize("env_name", ENV_NAMES)
@pytest.mark.parametrize(
    ("env_max_steps", "spec_max_steps", "expected_length"),
    [
        (10, 40, 11),  # the environment's own limit binds
        (40, 10, 11),  # the requested rollout horizon binds
        (20, 20, 21),  # both, together
    ],
)
def test_both_drivers_truncate_at_whichever_limit_comes_first(
    env_name, env_max_steps, spec_max_steps, expected_length
):
    # Two independent step limits exist: the config's own max_steps, which the
    # Gymnasium adapters truncate at, and RolloutSpec.max_steps, the horizon
    # this rollout asked for. An episode must end at the smaller of the two
    # regardless of driver -- otherwise --driver changes how long the
    # trajectories in a dataset are, for one unchanged config.
    env_spec = ENV_REGISTRY[env_name]
    cfg = free_flight_config(env_spec, max_steps=env_max_steps)
    vec, scan = drivers_for(cfg, env_spec)
    spec = RolloutSpec(num_episodes=2, max_steps=spec_max_steps, seed=0)

    a = vec.generate(spec)
    b = scan.generate(spec)

    assert a.lengths.tolist() == [expected_length] * 2
    np.testing.assert_array_equal(a.lengths, b.lengths)
    np.testing.assert_array_equal(a.truncated, b.truncated)


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_both_drivers_agree_on_free_flight_trajectories(env_name):
    env_spec = ENV_REGISTRY[env_name]
    cfg = free_flight_config(env_spec)
    vec, scan = drivers_for(cfg, env_spec)
    spec = RolloutSpec(num_episodes=2, max_steps=25, seed=0)

    a = vec.generate(spec)
    b = scan.generate(spec)

    np.testing.assert_array_equal(a.lengths, b.lengths)
    np.testing.assert_array_equal(a.terminated, b.terminated)
    np.testing.assert_array_equal(a.truncated, b.truncated)
    np.testing.assert_allclose(a.observations, b.observations, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(a.actions, b.actions, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(a.rewards, b.rewards, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(a.true_state, b.true_state, rtol=1e-4, atol=1e-4)
    _assert_truth_recorded(a, env_spec.layout.state_dim)
    _assert_truth_recorded(b, env_spec.layout.state_dim)


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_scan_and_vector_agree_on_goal_blocks_for_dock(env_name):
    # Goal-error augmentation lives in two places -- ScanDriver applies
    # make_augment in-scan, VectorEnvDriver applies it via
    # TaskPolicySource.augment_observation -- the same two-implementations-of-
    # one-rule drift risk the module docstring describes, now for the
    # appended goal block. One episode per lane keeps this in the bitwise
    # regime documented above. `observe` is pinned explicitly (not left to
    # the default) and identically on both sides, so a future default change
    # can't silently make this test compare two different policy inputs.
    env_spec = ENV_REGISTRY[env_name]
    state_dim = env_spec.layout.state_dim
    cfg = env_spec.config_cls(max_steps=10, observation={"goal_error": True})
    env_cfg = cfg.model_copy(update={"observation": cfg.observation.model_copy(update={"goal_error": False})})
    policy_cfg = PolicyConfig(type="dock", observe="state")
    spec = RolloutSpec(num_episodes=2, max_steps=10, seed=0)

    scan = ScanDriver(
        cfg=cfg, policy_cfg=policy_cfg, num_envs=2, env_spec=env_spec
    ).generate(spec)
    vector = VectorEnvDriver(
        env_factory=lambda: env_spec.make_vector_env(2, env_cfg),
        policy_source=TaskPolicySource(cfg, policy_cfg, view=env_spec.view),
    ).generate(spec)

    # The block is 12 wide and rides on top of the env's OWN state, whatever
    # that state's width is -- 25 for iss, 27 for iss-hcw. An augmentation
    # built against the 13D task view instead would give both envs 25.
    assert scan.observations.shape[-1] == state_dim + 12
    np.testing.assert_allclose(scan.observations, vector.observations, atol=1e-5)
    np.testing.assert_array_equal(scan.lengths, vector.lengths)
    # Truth is the un-augmented dynamics state on both sides: the goal block
    # widens `observations` only, so the two channels must still agree.
    np.testing.assert_allclose(scan.true_state, vector.true_state, atol=1e-5)
    _assert_truth_recorded(scan, state_dim)
    _assert_truth_recorded(vector, state_dim)


def test_scan_and_vector_agree_structurally_on_goal_blocks_for_union():
    # The union policy selects between orbit, dock, and random based on policy_id
    # derived from the PRNG, which the two drivers initialize differently. Beyond
    # one episode per lane, this test checks structural agreement: observation
    # shape, episode lengths, and termination flags match even if PRNG sequences
    # diverge. This mirrors test_both_drivers_agree_structurally_with_many_episodes_per_lane
    # but with goal-error augmentation enabled.
    cfg = ISSConfig(max_steps=10, observation={"goal_error": True})
    env_cfg = cfg.model_copy(update={"observation": cfg.observation.model_copy(update={"goal_error": False})})
    policy_cfg = PolicyConfig(type="union", observe="state")
    spec = RolloutSpec(num_episodes=4, max_steps=10, seed=0)

    scan = ScanDriver(cfg=cfg, policy_cfg=policy_cfg, num_envs=2).generate(spec)
    vector = VectorEnvDriver(
        env_factory=lambda: ISSVectorEnv(num_envs=2, cfg=env_cfg),
        policy_source=TaskPolicySource(cfg, policy_cfg),
    ).generate(spec)

    # Structural agreement: same shape and lengths, both honor the episode convention.
    assert scan.observations.shape == vector.observations.shape
    np.testing.assert_array_equal(scan.lengths, vector.lengths)
    np.testing.assert_array_equal(scan.terminated, vector.terminated)
    np.testing.assert_array_equal(scan.truncated, vector.truncated)
    scan.validate()
    vector.validate()
    _assert_terminal_convention(scan)
    _assert_terminal_convention(vector)


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_both_drivers_agree_when_episodes_terminate_on_collision(env_name):
    env_spec = ENV_REGISTRY[env_name]
    cfg = collision_config(env_spec)
    vec, scan = drivers_for(cfg, env_spec)
    spec = RolloutSpec(num_episodes=2, max_steps=50, seed=0)

    a = vec.generate(spec)
    b = scan.generate(spec)

    # Both drivers must actually collide, or the agreement below is agreement
    # about truncation and this test says nothing about termination at all.
    assert np.all(a.terminated) and np.all(b.terminated)
    np.testing.assert_array_equal(a.lengths, b.lengths)
    np.testing.assert_array_equal(a.terminated, b.terminated)


def _assert_terminal_convention(batch):
    """Every episode's final action slot is zero, and a terminated episode's
    final observation differs from its first (the terminal state was really
    reached, not just a zero-padded no-op)."""
    for i in range(batch.num_episodes):
        length = int(batch.lengths[i])
        final_action = batch.actions[i, length - 1]
        np.testing.assert_array_equal(final_action, np.zeros_like(final_action))
        if batch.terminated[i]:
            assert not np.array_equal(batch.observations[i, 0], batch.observations[i, length - 1])


def _assert_truth_recorded(batch, state_dim):
    """Both drivers must record the `state_dim`-wide true dynamics state on the
    same episode/time layout as the observations -- ScanDriver from the
    un-noised scan carry, VectorEnvDriver from the env's info["state"], the
    same two-implementations-of-one-rule drift risk as everything else here.

    `state_dim` is the env's own state width (13 for iss, 15 for iss-hcw), not
    the 13D task view: the truth channel records what the backend integrates.

    Every config in this module runs with sensor noise off, so truth and the
    stored observation are the same numbers; asserting that pins the
    alignment, which a shape check alone would not.
    """
    assert batch.true_state is not None
    assert batch.true_state.shape == batch.observations.shape[:2] + (state_dim,)
    np.testing.assert_allclose(
        batch.true_state, batch.observations[..., :state_dim], rtol=1e-5, atol=1e-5
    )


def _assert_every_episode_starts_from_a_reset(batch, cfg, layout):
    """Every episode's first observation must lie inside the start shell.

    A cheap, strong check that catches a driver silently starting a "new"
    episode from wherever the previous one happened to leave the physics,
    instead of from a real reset -- that state would essentially never sit
    inside the shell by chance.
    """
    low, high = start_shell(cfg)
    for i in range(batch.num_episodes):
        radius = np.linalg.norm(batch.observations[i, 0, layout.pos])
        assert low * (1 - 1e-4) <= radius <= high * (1 + 1e-4)


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_both_drivers_agree_structurally_with_many_episodes_per_lane(env_name):
    # 3 episodes per lane: past the one-episode-per-lane regime where the two
    # drivers' reset keys still coincide (see module docstring), so this
    # cannot assert bitwise equality. It asserts what must still hold:
    # segmentation, lengths, and termination flags agree, and both drivers
    # honor the episode convention.
    env_spec = ENV_REGISTRY[env_name]
    cfg = free_flight_config(env_spec)
    vec, scan = drivers_for(cfg, env_spec)
    spec = RolloutSpec(num_episodes=6, max_steps=10, seed=0)

    a = vec.generate(spec)
    b = scan.generate(spec)

    np.testing.assert_array_equal(a.lengths, b.lengths)
    np.testing.assert_array_equal(a.terminated, b.terminated)
    np.testing.assert_array_equal(a.truncated, b.truncated)
    a.validate()
    b.validate()
    _assert_terminal_convention(a)
    _assert_terminal_convention(b)
    _assert_every_episode_starts_from_a_reset(a, cfg, env_spec.layout)
    _assert_every_episode_starts_from_a_reset(b, cfg, env_spec.layout)
    _assert_truth_recorded(a, env_spec.layout.state_dim)
    _assert_truth_recorded(b, env_spec.layout.state_dim)


def test_both_drivers_start_every_episode_from_an_independent_reset():
    # A single lane past the one-episode-per-lane regime: with num_envs=1,
    # `finished`/`_segment` order is unambiguously that lane's episode
    # sequence, so "does episode N start where episode N-1 ended" is a direct
    # per-driver check rather than something inferred from interleaved,
    # multi-lane output order (which the two drivers do not even produce in
    # the same order -- see module docstring). spec.max_steps (5) is far
    # below either driver's own horizon, so this is the regime where a
    # driver that only reset between generate() calls would silently chain
    # episodes together instead of starting each one from a real reset.
    env_spec = ENV_REGISTRY["iss"]
    cfg = free_flight_config(env_spec)
    vec, scan = drivers_for(cfg, env_spec, num_envs=1)
    spec = RolloutSpec(num_episodes=3, max_steps=5, seed=0)

    for batch in (vec.generate(spec), scan.generate(spec)):
        batch.validate()
        assert np.all(batch.truncated)
        _assert_every_episode_starts_from_a_reset(batch, cfg, env_spec.layout)
        for i in range(1, batch.num_episodes):
            previous_terminal = batch.observations[i - 1, batch.lengths[i - 1] - 1]
            this_initial = batch.observations[i, 0]
            assert not np.allclose(this_initial, previous_terminal)


def test_both_drivers_agree_structurally_when_episodes_terminate_early():
    # The collision box covers the whole start sphere, so every episode
    # terminates almost immediately and lanes recycle several times over the
    # 6 requested episodes -- again past the one-episode-per-lane regime.
    env_spec = ENV_REGISTRY["iss"]
    cfg = collision_config(env_spec)
    vec, scan = drivers_for(cfg, env_spec)
    spec = RolloutSpec(num_episodes=6, max_steps=10, seed=0)

    a = vec.generate(spec)
    b = scan.generate(spec)

    np.testing.assert_array_equal(a.lengths, b.lengths)
    np.testing.assert_array_equal(a.terminated, b.terminated)
    np.testing.assert_array_equal(a.truncated, b.truncated)
    assert np.all(a.terminated)
    assert np.all(b.terminated)
    a.validate()
    b.validate()
    _assert_terminal_convention(a)
    _assert_terminal_convention(b)
    _assert_truth_recorded(a, env_spec.layout.state_dim)
    _assert_truth_recorded(b, env_spec.layout.state_dim)


def test_both_drivers_agree_on_episode_length_distribution_for_a_stochastic_policy():
    # Random actions cannot match trajectory-for-trajectory across two PRNG
    # pipelines, but a systematic difference in termination or segmentation
    # logic would still show up as a different length distribution.
    cfg = ISSConfig(physics=PhysicsConfig(collision_boxes_path=None), dock=DockConfig(enabled=False))
    stochastic = PolicyConfig(type="random")
    from owm_envs.envs.common.policy_source import TaskPolicySource

    vec = VectorEnvDriver(
        env_factory=lambda: ISSVectorEnv(num_envs=4, cfg=cfg),
        policy_source=TaskPolicySource(cfg, stochastic),
    )
    scan = ScanDriver(cfg=cfg, policy_cfg=stochastic, num_envs=4)
    spec = RolloutSpec(num_episodes=8, max_steps=20, seed=3)

    a = vec.generate(spec)
    b = scan.generate(spec)

    # Free flight with no termination: every episode must truncate at max_steps
    # in BOTH drivers. 20 steps -> 21 observations.
    assert np.all(a.lengths == 21)
    assert np.all(b.lengths == 21)
    assert np.all(a.truncated) and np.all(b.truncated)


def test_both_drivers_produce_batches_that_validate():
    env_spec = ENV_REGISTRY["iss"]
    cfg = free_flight_config(env_spec)
    vec, scan = drivers_for(cfg, env_spec)
    spec = RolloutSpec(num_episodes=2, max_steps=15, seed=1)
    vec.generate(spec).validate()
    scan.generate(spec).validate()


def _assert_batches_equal(a, b):
    np.testing.assert_array_equal(a.observations, b.observations)
    np.testing.assert_array_equal(a.actions, b.actions)
    np.testing.assert_array_equal(a.lengths, b.lengths)
    np.testing.assert_array_equal(a.true_state, b.true_state)


# Multi-split runs call generate() repeatedly (the CLI builds a driver per
# split, but reuse must also be safe): generate() must be a pure function of
# the RolloutSpec, or split contents would depend on generation order. These
# tests prove this property: calling generate() with one spec interleaved
# between calls with another must not perturb either spec's result, and a
# fresh driver instance must reproduce the same result too.


def test_scan_driver_generate_is_pure_per_spec():
    cfg = ISSConfig(max_steps=20)
    driver = ScanDriver(cfg=cfg, policy_cfg=PolicyConfig(), num_envs=2)
    spec_a = RolloutSpec(num_episodes=2, max_steps=20, seed=0)
    spec_b = RolloutSpec(num_episodes=2, max_steps=20, seed=1)

    first_b = driver.generate(spec_b)
    driver.generate(spec_a)  # interleave a different spec
    second_b = driver.generate(spec_b)  # must not be affected by it
    fresh_b = ScanDriver(cfg=cfg, policy_cfg=PolicyConfig(), num_envs=2).generate(spec_b)

    _assert_batches_equal(first_b, second_b)
    _assert_batches_equal(first_b, fresh_b)


def test_vector_env_driver_generate_is_pure_per_spec():
    cfg = ISSConfig(max_steps=20)

    def build_driver():
        return VectorEnvDriver(
            env_factory=lambda: ISSVectorEnv(num_envs=2, cfg=cfg),
            policy_source=TaskPolicySource(cfg, PolicyConfig()),
        )

    driver = build_driver()
    spec_a = RolloutSpec(num_episodes=2, max_steps=20, seed=0)
    spec_b = RolloutSpec(num_episodes=2, max_steps=20, seed=1)

    first_b = driver.generate(spec_b)
    driver.generate(spec_a)  # interleave a different spec
    second_b = driver.generate(spec_b)  # must not be affected by it
    fresh_b = build_driver().generate(spec_b)

    _assert_batches_equal(first_b, second_b)
    _assert_batches_equal(first_b, fresh_b)
