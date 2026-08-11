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

iss-numerical is the one env where even the per-step arithmetic is not
identical, and deliberately so. On the vector path `TaskPolicySource.act`
reads `info["measured_state"]`, which the Gymnasium adapters publish narrowed
to float32; `ScanDriver` flies the policy on the float64 state throughout.
That env's state carries ABSOLUTE ECI positions, so the narrowing is worth
~1 m on the policy's input and the two drivers fly measurably different
trajectories from the same seed. `NUMERICAL_DIVERGENCE` below derives what
that is allowed to come to and states what it actually measures; everything
structural still has to match exactly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from owm_envs.drivers.scan_driver import ScanDriver
from owm_envs.drivers.types import RolloutSpec
from owm_envs.drivers.vector_env_driver import VectorEnvDriver
from owm_envs.envs import ENV_REGISTRY
from owm_envs.envs.common.config import DockConfig, ObservationConfig, PhysicsConfig
from owm_envs.envs.common.goal import GOAL_ERROR_DIM
from owm_envs.envs.common.policies import PolicyConfig
from owm_envs.envs.common.policy_source import TaskPolicySource
from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss.vector_env import ISSVectorEnv
from owm_envs.envs.iss_numerical.config import NUM_LAYOUT

# The contract below is env-independent -- segmentation and termination are
# the drivers' own logic, not the backend's -- so every registered env runs
# it. iss-hcw is the case that keeps it honest: a 15-wide float64 state
# behind a 13D task view, where a driver that assumed either width would
# still pass on iss alone. iss-numerical adds the case where the recorded
# observation is not the state at all (see RECORDED_WIDTH).
ENV_NAMES = ("iss", "iss-hcw", "iss-numerical")

DETERMINISTIC = PolicyConfig(type="dock")

# Width of what each env RECORDS per step, before any goal-error block. For
# iss and iss-hcw that is the state itself -- their `EnvSpec.make_observe` is
# None, i.e. the identity. iss-numerical's is `OBS_MODE_DIM[mode]`, 15 under
# the shipped "relative" mode, which is NOT its 21-wide state: the observation
# is `[epoch | relative_view]`, element for element iss-hcw's own layout (see
# envs/iss_numerical/observe.py).
RECORDED_WIDTH = {"iss": 13, "iss-hcw": 15, "iss-numerical": 15}

# Where the chaser's world-frame relative position sits in that recorded
# observation. Deliberately not `env_spec.layout.pos`, which for
# iss-numerical names an absolute ECI slice of a state the observation is not
# -- reading the layout here would compare a 6.8e6 m radius against a 100 m
# start shell.
OBSERVED_POSITION = {
    "iss": slice(0, 3),
    "iss-hcw": slice(2, 5),
    "iss-numerical": slice(2, 5),
}

# How far the two drivers' iss-numerical trajectories may drift apart over
# TRAJECTORY_STEPS steps of the shipped 0.05 s dt. Derived, not fitted.
#
# `TaskPolicySource.act` on the vector path flies the policy on
# `info["measured_state"]`, which `NumericalEnv._observation_and_measured`
# narrows to float32; ScanDriver hands the policy the float64 state. This
# env's state holds ABSOLUTE ECI positions at ~6.8e6 m, where float32's ulp is
# 0.5 m, so the chief's and the chaser's positions each round by up to 0.25 m
# and `relative_view`'s difference of the two carries up to 0.5 m per axis --
# the ~1 m budget `TaskPolicySource.augment_observation` documents.
#
#   0.5 m per axis on the policy's position input
#     -> <= 0.5 * sqrt(3) = 0.87 m in NORM. The norm is what matters, not the
#        per-axis figure: the dock law forms its correction in world axes and
#        rotates it into the body frame, so a single commanded body component
#        can carry the whole vector error.
#   x DockParams.kp_position, 16 N/m         -> <= 13.9 N per component
#   + DockParams.kd_velocity, 880 N*s/m, on the velocity input's own float32
#     grain (4.9e-4 m/s per axis at 7.7e3 m/s, 8.5e-4 in norm)
#                                            -> +0.7 N, so <= 14.6 N in total
#   / PhysicsConfig.mass, 12000 kg           -> <= 1.22e-3 m/s^2
#   freely integrated over T = 25 * 0.05 = 1.25 s:
#       velocity <= 1.22e-3 * T         = 1.5e-3 m/s
#       position <= 0.5 * 1.22e-3 * T^2 = 9.5e-4 m
#
# Measured across 12 seeds: 7.6 N of action difference (the realized
# quantization runs about half the worst case), 5.0e-5 m of relative position,
# 1.0e-4 m/s of relative velocity, 9.9e-8 per quaternion component and 1.9e-9
# rad/s of body rate.
#
# Position and velocity are bounded at 3x their measured maxima, which leaves
# them 6.3x and 5.0x inside the budget above. The ACTION bound is set at 1.9x
# measured instead, because 3x would be 22.8 N against a 14.6 N budget -- and
# a bound past its own budget would admit a divergence the float32 input does
# not explain, which is exactly the thing this test exists to catch. Every
# bound here therefore sits inside what the narrowing allows, and anything
# past one is a bug to find rather than a tolerance to widen.
#
# The epoch prefix is exempt and asserted EXACT: it advances outside the
# integrator through `advance_epoch_state` and never sees an action.
NUMERICAL_DIVERGENCE = {
    "position": 1.5e-4,  # m,      measured 5.0e-5
    "velocity": 3.0e-4,  # m/s,    measured 1.0e-4
    "attitude": 3.0e-7,  # per quaternion component, measured 9.9e-8
    "rate": 6.0e-9,  # rad/s,  measured 1.9e-9
    # N. 1.9x measured 7.6 and inside the 14.6 N the budget above allows; a
    # 1600 N limit, so this is 0.9% of an action.
    "action": 14.5,
    # Relative, because `docking_reward`'s control-effort term is
    # -0.05 * sum(action**2) against ~1600 N actions, so the reward runs at
    # ~1e5-1e6 here and 16 N of action difference is worth ~2.6e3 of it.
    # Measured 6.8e-3.
    "reward": 2.0e-2,
}

# The horizon NUMERICAL_DIVERGENCE's arithmetic is written against. The bound
# grows quadratically in it, so a test that lengthens the rollout has to
# redo that sum rather than reuse these numbers.
TRAJECTORY_STEPS = 25


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
    size = 3.0 * free_flight_config(env_spec).start_shell()[1]
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
    spec = RolloutSpec(num_episodes=2, max_steps=TRAJECTORY_STEPS, seed=0)

    a = vec.generate(spec)
    b = scan.generate(spec)

    np.testing.assert_array_equal(a.lengths, b.lengths)
    np.testing.assert_array_equal(a.terminated, b.terminated)
    np.testing.assert_array_equal(a.truncated, b.truncated)
    _assert_trajectories_agree(a, b, env_name)
    _assert_truth_channels_agree(a, b, env_name, rtol=1e-4, atol=1e-4)
    _assert_truth_recorded(a, env_spec)
    _assert_truth_recorded(b, env_spec)


def test_both_drivers_reset_iss_numerical_lanes_to_the_same_state():
    # The premise every iss-numerical bound in this module is measured
    # against: the two drivers start from the same place. Their seeding is
    # mirrored (`ScanDriver._run_chunk` documents the mirror), so each lane
    # resets into the same float64 state and both narrow that same state to a
    # recorded row before any action has been taken. If that were not so,
    # "how far apart do the two drivers drift" would be unanswerable rather
    # than merely larger, and NUMERICAL_DIVERGENCE would be measuring the
    # wrong thing.
    #
    # Asserted on the recorded TRUTH -- the raw 21D state -- not only on the
    # observation. Matching relative views would not settle it: the view is a
    # difference of two ECI vectors, and two different chief/chaser pairs
    # displaced together along the orbit present the same one. The epoch,
    # chief ECI and chaser ECI columns come back BIT-identical.
    #
    # The attitude and rate columns do not, and that is compilation rather
    # than physics: `reset`'s quaternion composition is inlined into two
    # different XLA programs (a `jit(vmap(...))` call in the vector env, a
    # `lax.scan` body in the fused driver) which reassociate its float64
    # products differently, and the ~1e-16 that leaves is amplified to one
    # float32 ulp by the astrojax-pinned quaternion helpers underneath.
    env_spec = ENV_REGISTRY["iss-numerical"]
    cfg = free_flight_config(env_spec)
    vec, scan = drivers_for(cfg, env_spec)
    spec = RolloutSpec(num_episodes=2, max_steps=TRAJECTORY_STEPS, seed=0)

    a, b = vec.generate(spec), scan.generate(spec)
    state_a, state_b = a.true_state[:, 0], b.true_state[:, 0]

    for name, columns in (
        ("epoch", NUM_LAYOUT.epoch),
        ("chief ECI state", NUM_LAYOUT.chief),
        ("chaser ECI position", NUM_LAYOUT.pos),
        ("chaser ECI velocity", NUM_LAYOUT.vel),
    ):
        np.testing.assert_array_equal(
            state_a[..., columns], state_b[..., columns], err_msg=f"initial {name}"
        )
    # 2.4e-7 is four float32 ulps of a unit quaternion (2**-24 = 6.0e-8);
    # measured 6.0e-8, i.e. exactly one. The rate channel's measured 2.3e-10
    # is the same one-ulp story against the world frame's 1.1e-3 rad/s.
    _assert_within(
        state_a[..., NUM_LAYOUT.quat], state_b[..., NUM_LAYOUT.quat], 2.4e-7, "initial q_bi"
    )
    _assert_within(
        state_a[..., NUM_LAYOUT.omega], state_b[..., NUM_LAYOUT.omega], 1e-9, "initial rate"
    )

    # Non-vacuous: the columns just compared have to hold a real start, not
    # zeros. The chaser sits at an ISS radius, and the shipped dispersion puts
    # every episode 100 m off the chief.
    assert np.all(np.linalg.norm(state_a[..., NUM_LAYOUT.pos], axis=-1) > 6e6)
    assert np.all(np.linalg.norm(a.observations[:, 0, 2:5], axis=-1) > 1.0)

    # And the observation the truth is recorded beside agrees too, on the
    # channels the derivation carries through at float64.
    np.testing.assert_array_equal(
        a.observations[:, 0, 0:8], b.observations[:, 0, 0:8]
    )


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_scan_and_vector_agree_on_goal_blocks_for_dock(env_name):
    # Goal-error augmentation lives in two places -- ScanDriver applies
    # make_augment in-scan, VectorEnvDriver applies it via
    # TaskPolicySource.augment_observation -- the same two-implementations-of-
    # one-rule drift risk the module docstring describes, now for the
    # appended goal block. One episode per lane keeps this in the bitwise
    # regime documented above for iss and iss-hcw; iss-numerical is bounded
    # rather than bitwise even here, and by two separate budgets -- see
    # `_assert_trajectories_agree` for the observed part and
    # `_assert_goal_blocks_agree` for the block. `observe` is pinned
    # explicitly (not left to the default) and identically on both sides, so
    # a future default change can't silently make this test compare two
    # different policy inputs -- note it is pinned to "state", which on the
    # vector path is `info["state"]`, float32 like `info["measured_state"]`,
    # so it does not sidestep the narrowing.
    env_spec = ENV_REGISTRY[env_name]
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

    # The block is 12 wide and rides on top of what the env RECORDS -- 25 for
    # iss, 27 for iss-hcw (their states), 27 for iss-numerical (its 15-wide
    # "relative" observation, not its 21-wide state). An augmentation built
    # against the 13D task view instead would give all three 25.
    recorded = RECORDED_WIDTH[env_name]
    assert scan.observations.shape[-1] == recorded + GOAL_ERROR_DIM
    np.testing.assert_array_equal(scan.lengths, vector.lengths)
    # 1e-7/1e-5 on the observed columns, not the free-flight default: this
    # test has always compared them tightly, and the goal block riding on the
    # end does not make the part in front of it any less exact.
    _assert_trajectories_agree(
        scan, vector, env_name, observed_width=recorded, rtol=1e-7, atol=1e-5
    )
    _assert_goal_blocks_agree(scan, vector, env_name, observed_width=recorded)
    # Truth is the un-augmented dynamics state on both sides: the goal block
    # widens `observations` only, so the two channels must still agree.
    _assert_truth_channels_agree(scan, vector, env_name, rtol=1e-7, atol=1e-5)
    _assert_truth_recorded(scan, env_spec)
    _assert_truth_recorded(vector, env_spec)


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


def _assert_within(actual, expected, tolerance, what):
    """`|actual - expected| <= tolerance`, elementwise, reporting the worst
    element against the bound it broke.

    Used in preference to `np.testing.assert_allclose` wherever a bound is
    being asserted rather than approximate equality: allclose's default rtol
    of 1e-7 is the same size as the float32 world<->ECI rotation error these
    envs carry (`relative_view`'s docstring), so leaving it implicit would let
    the thing under test absorb itself. `tolerance` may be a scalar or a
    broadcastable array.
    """
    difference = np.abs(np.asarray(actual, np.float64) - np.asarray(expected, np.float64))
    bound = np.broadcast_to(np.asarray(tolerance, np.float64), difference.shape)
    if np.all(difference <= bound):
        return
    # The element that overshoots its OWN bound by the most, which for an
    # array-valued tolerance is not the element with the largest difference.
    worst = np.unravel_index(np.argmax(difference - bound), difference.shape)
    raise AssertionError(
        f"{what}: {difference[worst]:.4g} at index {tuple(int(i) for i in worst)} "
        f"exceeds its bound of {bound[worst]:.4g}"
    )


def _assert_trajectories_agree(a, b, env_name, observed_width=None, rtol=1e-4, atol=1e-4):
    """The two drivers' observations, actions and rewards, per env.

    iss and iss-hcw fly ONE trajectory between them -- both drivers hand the
    policy the same float64 state -- so the tolerances are the loose
    round-off ones they have always been. iss-numerical does not, and gets
    the derived, channel-by-channel bounds in `NUMERICAL_DIVERGENCE`: its
    recorded observation is `[epoch(2) | rel_pos(3) | rel_vel(3) | q_bw(4) |
    omega(3)]`, whose channels span 100 m and 1e-3 rad/s and so cannot share
    one number. `observed_width` trims a goal-error block off the end when
    the caller has one; the block is compared separately, on its own budget.

    `rtol`/`atol` govern the OBSERVATIONS for the non-numerical envs only --
    actions and rewards keep their own fixed tolerances, and iss-numerical
    reads `NUMERICAL_DIVERGENCE` regardless. They are a parameter because the
    two call sites are not equally demanding: the goal-block test passes
    1e-7/1e-5, the tight pair it has always used, where free flight has always
    run at the 1e-4 default. Collapsing the two onto the looser default would
    slacken the goal-block comparison ~800x at its largest-magnitude columns,
    and iss's measured 3.05e-5 there already sits at about half the tight
    bound -- so that bound is doing real work and must not be widened.
    """
    observations_a, observations_b = a.observations, b.observations
    if observed_width is not None:
        observations_a = observations_a[..., :observed_width]
        observations_b = observations_b[..., :observed_width]

    if env_name != "iss-numerical":
        np.testing.assert_allclose(observations_a, observations_b, rtol=rtol, atol=atol)
        np.testing.assert_allclose(a.actions, b.actions, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(a.rewards, b.rewards, rtol=1e-3, atol=1e-2)
        return

    # The epoch advances outside the integrator and never sees an action, so
    # no amount of trajectory divergence may move it.
    np.testing.assert_array_equal(observations_a[..., 0:2], observations_b[..., 0:2])
    for channel, columns in (
        ("position", slice(2, 5)),
        ("velocity", slice(5, 8)),
        ("attitude", slice(8, 12)),
        ("rate", slice(12, 15)),
    ):
        _assert_within(
            observations_a[..., columns],
            observations_b[..., columns],
            NUMERICAL_DIVERGENCE[channel],
            f"observed {channel}",
        )
    _assert_within(a.actions, b.actions, NUMERICAL_DIVERGENCE["action"], "actions")
    _assert_within(
        a.rewards,
        b.rewards,
        NUMERICAL_DIVERGENCE["reward"] * np.maximum(np.abs(a.rewards), 1.0),
        "rewards",
    )


def _assert_truth_channels_agree(a, b, env_name, rtol, atol):
    """The two drivers' recorded truth channels.

    iss and iss-hcw keep the plain `assert_allclose` they have always used --
    those two fly one trajectory between them. iss-numerical needs the bound
    below because its recorded truth is float32 of a 21D state spanning 6.8e6 m
    ECI positions and 1e-3 rad/s body rates.

    Two terms, and both are necessary. The float32 STORAGE contributes 2**-23
    of each element's own magnitude -- two roundings of one float64 value --
    which at the ECI columns is 0.8 m and is the only reason those columns can
    differ at all. The per-segment FLOOR is the float64 divergence itself,
    which cannot be written as a fraction of an element: at the reference
    epoch the chaser's ECI vx is exactly zero while its velocity is 7.7 km/s,
    so a purely relative bound would demand bit equality of a channel that has
    none. Measured worst cases are 4.9e-4 m and 4.9e-4 m/s on the ECI columns,
    1.2e-7 per quaternion component and 1.9e-9 rad/s, so the floors below sit
    at 20x, 6x, 2.5x and 3x of them -- the position floor is the loosest
    because it is the channel a real trajectory divergence would show up in
    first, and the sharp instrument for that is not this channel anyway but
    the OBSERVATION, whose relative view is differenced at float64 at a ~100 m
    scale.
    """
    if env_name != "iss-numerical":
        np.testing.assert_allclose(a.true_state, b.true_state, rtol=rtol, atol=atol)
        return

    floors = np.zeros(a.true_state.shape[-1])
    floors[NUM_LAYOUT.epoch] = 0.0  # advanced outside the integrator
    floors[NUM_LAYOUT.chief] = 1e-6  # no control acts on the chief
    floors[NUM_LAYOUT.pos] = 1e-2  # chaser ECI position [m]
    floors[NUM_LAYOUT.vel] = 3e-3  # chaser ECI velocity [m/s]
    floors[NUM_LAYOUT.quat] = 3e-7  # q_bi component
    floors[NUM_LAYOUT.omega] = 6e-9  # body rate [rad/s]
    _assert_within(
        a.true_state,
        b.true_state,
        floors + 2.0**-23 * np.abs(np.asarray(a.true_state, np.float64)),
        "true state",
    )


def _assert_goal_blocks_agree(scan, vector, env_name, observed_width):
    """The appended `[pos_err(3), vel_err(3), att_err(3), rate_err(3)]` block.

    iss and iss-hcw compute it from the same numbers on both paths. For
    iss-numerical the two paths genuinely differ: ScanDriver's `make_augment`
    reads the float64 state, while `TaskPolicySource.augment_observation`
    reads `info["measured_state"]`, float32 -- so the block's position error
    carries the full ~1 m of differencing two float32 ECI positions.
    `policy_source.py` documents that as a deliberate, recorded residual, and
    documents equally that the RECORDED row is never re-derived from
    `measured` -- which is why the observed part beside it pays only the
    trajectory divergence bounded in `_assert_trajectories_agree`, three
    orders smaller, and not this.

    The bounds are that budget, not a fitted number: 0.5 m per axis on each of
    two positions is <= 0.87 m in norm, and the velocity's 4.9e-4 m/s ulp at
    7.7e3 m/s gives <= 1.7e-3 m/s. Measured: 0.39 m and 7.0e-4 m/s, with
    4.8e-7 rad and 9.3e-10 rad/s on the two attitude channels, which pass
    through no ECI difference at all.
    """
    block_scan = scan.observations[..., observed_width:]
    block_vector = vector.observations[..., observed_width:]
    if env_name != "iss-numerical":
        np.testing.assert_allclose(block_scan, block_vector, atol=1e-5)
        return
    for name, columns, bound in (
        ("position error", slice(0, 3), 1.0),
        ("velocity error", slice(3, 6), 3.0e-3),
        ("attitude error", slice(6, 9), 2.0e-6),
        ("rate error", slice(9, 12), 5.0e-9),
    ):
        _assert_within(
            block_scan[..., columns], block_vector[..., columns], bound, f"goal {name}"
        )


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


def _assert_truth_recorded(batch, env_spec):
    """Both drivers must record the env's own true dynamics state on the same
    episode/time layout as the observations -- ScanDriver from the un-noised
    scan carry, VectorEnvDriver from the env's info["state"], the same
    two-implementations-of-one-rule drift risk as everything else here.

    The truth channel records what the backend integrates -- 13 elements for
    iss, 15 for iss-hcw, 21 for iss-numerical -- never the 13D task view and
    never the recorded observation, which for iss-numerical is neither.

    What pins the ALIGNMENT (a shape check alone would not) depends on
    whether the env records its state verbatim. iss and iss-hcw do, and every
    config in this module runs with sensor noise off, so truth and the stored
    observation are the same numbers. iss-numerical's observation is
    `observe(state)`, so instead: the epoch prefix, which every mode carries
    through untouched and which strictly increases along an episode, has to
    match element for element; and the relative position recovered from the
    recorded truth has to be the one the observation reports, to the ~1 m that
    recovering it from a float32 21D state costs (two 6.8e6 m ECI positions,
    0.5 m ulp each) -- three orders below the ~100 m that separates two lanes'
    episodes, so a row taken from the wrong lane or the wrong step still fails.
    """
    state_dim = env_spec.layout.state_dim
    assert batch.true_state is not None
    assert batch.true_state.shape == batch.observations.shape[:2] + (state_dim,)

    if env_spec.make_observe is None:
        np.testing.assert_allclose(
            batch.true_state, batch.observations[..., :state_dim], rtol=1e-5, atol=1e-5
        )
        return

    epoch = env_spec.layout.epoch
    np.testing.assert_array_equal(
        batch.true_state[..., epoch], batch.observations[..., epoch]
    )
    recovered = np.asarray(
        jax.vmap(env_spec.view)(jnp.asarray(batch.true_state.reshape(-1, state_dim))),
        np.float64,
    )[:, 0:3]
    observed = batch.observations[..., OBSERVED_POSITION[env_spec.name]].reshape(-1, 3)
    _assert_within(recovered, observed, 1.0, "recorded position against its truth")


def _assert_every_episode_starts_from_a_reset(batch, cfg, env_name):
    """Every episode's first observation must lie inside the start shell.

    A cheap, strong check that catches a driver silently starting a "new"
    episode from wherever the previous one happened to leave the physics,
    instead of from a real reset -- that state would essentially never sit
    inside the shell by chance.

    Read through `OBSERVED_POSITION` rather than the env's `layout.pos`: for
    iss-numerical the layout names an absolute ECI slice of a 21D state, and
    the recorded observation is a 15-wide relative one.
    """
    low, high = cfg.start_shell()
    for i in range(batch.num_episodes):
        radius = np.linalg.norm(batch.observations[i, 0, OBSERVED_POSITION[env_name]])
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
    _assert_every_episode_starts_from_a_reset(a, cfg, env_name)
    _assert_every_episode_starts_from_a_reset(b, cfg, env_name)
    _assert_truth_recorded(a, env_spec)
    _assert_truth_recorded(b, env_spec)


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
        _assert_every_episode_starts_from_a_reset(batch, cfg, env_spec.name)
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
    _assert_truth_recorded(a, env_spec)
    _assert_truth_recorded(b, env_spec)


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
