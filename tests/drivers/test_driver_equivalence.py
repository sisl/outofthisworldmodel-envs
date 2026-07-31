"""The guard against the two drivers silently diverging.

seamstress has exactly this bug: its done-logic is implemented twice, once in
JAX and once in numpy (environment_parallel.py:114-121 vs :221-226), free to
drift. This port has two rollout implementations for real performance reasons,
so it must prove they agree.

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
the scan; `ISSVectorEnv` splits one shared key each time any lane resets).
Beyond one episode per lane -- which is the regime real dataset generation
actually runs in, e.g. 512 episodes from 8 lanes -- the guarantee this test
suite provides is STRUCTURAL, not bitwise: identical episode segmentation,
lengths, and termination flags, and both drivers honoring the same episode
convention (terminal action pad, terminal observation present). That is
deliberate, not a gap: what can realistically drift between two independent
rollout implementations is the segmentation/termination logic (two copies of
"is this lane done, and where does the next episode start"), not the
per-step arithmetic, which both drivers call through the same `ISSDynamics`,
`iss_reward`, and policy functions.
"""

import numpy as np
import pytest

from owm_envs.drivers.scan_driver import ScanDriver
from owm_envs.drivers.types import RolloutSpec
from owm_envs.drivers.vector_env_driver import VectorEnvDriver
from owm_envs.envs.iss.config import DockConfig, ISSConfig, PhysicsConfig
from owm_envs.envs.iss.policies import PolicyConfig
from owm_envs.envs.iss.vector_env import ISSVectorEnv

START_RADIUS_M = ISSConfig().physics.start_radius_m

DETERMINISTIC = PolicyConfig(type="dock")


def drivers_for(cfg, num_envs=2):
    from owm_envs.envs.iss.policy_source import IssPolicySource

    vec = VectorEnvDriver(
        env_factory=lambda: ISSVectorEnv(num_envs=num_envs, cfg=cfg),
        policy_source=IssPolicySource(cfg, DETERMINISTIC),
    )
    scan = ScanDriver(cfg=cfg, policy_cfg=DETERMINISTIC, num_envs=num_envs)
    return vec, scan


def test_both_drivers_agree_on_free_flight_trajectories():
    cfg = ISSConfig(physics=PhysicsConfig(collision_boxes_path=None), dock=DockConfig(enabled=False))
    vec, scan = drivers_for(cfg)
    spec = RolloutSpec(num_episodes=2, max_steps=25, seed=0)

    a = vec.generate(spec)
    b = scan.generate(spec)

    np.testing.assert_array_equal(a.lengths, b.lengths)
    np.testing.assert_array_equal(a.terminated, b.terminated)
    np.testing.assert_array_equal(a.truncated, b.truncated)
    np.testing.assert_allclose(a.observations, b.observations, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(a.actions, b.actions, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(a.rewards, b.rewards, rtol=1e-3, atol=1e-2)


def test_both_drivers_agree_when_episodes_terminate_on_collision():
    cfg = ISSConfig(
        physics=PhysicsConfig(
            collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [400.0, 400.0, 400.0]}]
        ),
        dock=DockConfig(enabled=False),
    )
    vec, scan = drivers_for(cfg)
    spec = RolloutSpec(num_episodes=2, max_steps=50, seed=0)

    a = vec.generate(spec)
    b = scan.generate(spec)

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


def _assert_every_episode_starts_from_a_reset(batch):
    """Every episode's first observation must lie on the start sphere.

    A cheap, strong check that catches a driver silently starting a "new"
    episode from wherever the previous one happened to leave the physics,
    instead of from a real reset -- that state would essentially never sit
    exactly on the sphere by chance.
    """
    for i in range(batch.num_episodes):
        radius = np.linalg.norm(batch.observations[i, 0, 0:3])
        np.testing.assert_allclose(radius, START_RADIUS_M, rtol=1e-4)


def test_both_drivers_agree_structurally_with_many_episodes_per_lane():
    # 3 episodes per lane: past the one-episode-per-lane regime where the two
    # drivers' reset keys still coincide (see module docstring), so this
    # cannot assert bitwise equality. It asserts what must still hold:
    # segmentation, lengths, and termination flags agree, and both drivers
    # honor the episode convention.
    cfg = ISSConfig(physics=PhysicsConfig(collision_boxes_path=None), dock=DockConfig(enabled=False))
    vec, scan = drivers_for(cfg)
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
    _assert_every_episode_starts_from_a_reset(a)
    _assert_every_episode_starts_from_a_reset(b)


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
    cfg = ISSConfig(physics=PhysicsConfig(collision_boxes_path=None), dock=DockConfig(enabled=False))
    vec, scan = drivers_for(cfg, num_envs=1)
    spec = RolloutSpec(num_episodes=3, max_steps=5, seed=0)

    for batch in (vec.generate(spec), scan.generate(spec)):
        batch.validate()
        assert np.all(batch.truncated)
        _assert_every_episode_starts_from_a_reset(batch)
        for i in range(1, batch.num_episodes):
            previous_terminal = batch.observations[i - 1, batch.lengths[i - 1] - 1]
            this_initial = batch.observations[i, 0]
            assert not np.allclose(this_initial, previous_terminal)


def test_both_drivers_agree_structurally_when_episodes_terminate_early():
    # The collision box covers the whole start sphere, so every episode
    # terminates almost immediately and lanes recycle several times over the
    # 6 requested episodes -- again past the one-episode-per-lane regime.
    cfg = ISSConfig(
        physics=PhysicsConfig(
            collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [400.0, 400.0, 400.0]}]
        ),
        dock=DockConfig(enabled=False),
    )
    vec, scan = drivers_for(cfg)
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


def test_both_drivers_agree_on_episode_length_distribution_for_a_stochastic_policy():
    # Random actions cannot match trajectory-for-trajectory across two PRNG
    # pipelines, but a systematic difference in termination or segmentation
    # logic would still show up as a different length distribution.
    cfg = ISSConfig(physics=PhysicsConfig(collision_boxes_path=None), dock=DockConfig(enabled=False))
    stochastic = PolicyConfig(type="random")
    from owm_envs.envs.iss.policy_source import IssPolicySource

    vec = VectorEnvDriver(
        env_factory=lambda: ISSVectorEnv(num_envs=4, cfg=cfg),
        policy_source=IssPolicySource(cfg, stochastic),
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
    cfg = ISSConfig(physics=PhysicsConfig(collision_boxes_path=None), dock=DockConfig(enabled=False))
    vec, scan = drivers_for(cfg)
    spec = RolloutSpec(num_episodes=2, max_steps=15, seed=1)
    vec.generate(spec).validate()
    scan.generate(spec).validate()
