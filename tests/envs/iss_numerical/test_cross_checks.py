"""iss-numerical against iss-hcw: two independent physics implementations of
the same relative-motion problem, held against each other.

`test_dynamics.py` checks this env against closed forms and against itself
(`test_two_body_conserves_energy_and_momentum_over_one_orbit` is the two-body
energy/momentum conservation over a full orbit in f64, and is not repeated
here). What those cannot catch is a mistake shared between a derivation and
the reference it is checked against. iss-hcw solves the same task from a
completely different starting point -- a linearized relative ODE about an
analytic Keplerian chief, rather than two ECI state vectors through a full
force model -- so where the linearization is valid the two have to agree, and
where they do not agree the disagreement has to be HCW's own modelling error
and nothing else. That is what the first two tests establish, quantitatively.

The third documents a deliberate divergence rather than an agreement: the two
envs carry the body->world quaternion under different sign conventions, and
the suite documents that rather than unifying it: unifying would move iss-hcw
off the numerics its golden fixtures were recorded against, for a difference
no consumer reading rotations can observe.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from owm_envs.core.quaternion import quat_to_rotmat
from owm_envs.envs.common.epoch_state import advance_epoch_state
from owm_envs.envs.iss_hcw.config import HCW_LAYOUT, HCWConfig
from owm_envs.envs.iss_hcw.dynamics import HCWDynamics
from owm_envs.envs.iss_numerical.config import NumericalConfig
from owm_envs.envs.iss_numerical.dynamics import (
    NumericalDynamics,
    chaser_state_from_view,
    relative_view,
)

ZERO_ACTION = jnp.zeros(6, jnp.float64)

# 200 s of coasting at dt = 0.5 rather than the shipped 0.05: 400 steps
# instead of 4000, for a comparison the step size provably does not enter.
# Measured, the numerical-to-HCW gap below is 1.273740e-2 m at dt = 1,
# 1.273741e-2 at dt = 0.5 and 1.273743e-2 at dt = 0.25 -- invariant to six
# figures across a 4x change in step size, so RK4 truncation is nowhere near
# the quantity these tests measure.
DT = 0.5
DURATION_S = 200.0
STEPS = int(round(DURATION_S / DT))

# The shared initial view: 100 m straight up from the chief on the radial
# axis (world +z, see `envs/common/orbit.RTN_FROM_WORLD`), at rest in the
# ROTATING LVLH frame, body aligned with the world frame and not rotating
# relative to it.
STANDOFF_M = 100.0
RADIAL_STANDOFF_VIEW = jnp.zeros(13, jnp.float64).at[2].set(STANDOFF_M).at[6].set(1.0)


def _configs(dt=DT, **overrides):
    """An (hcw_cfg, numerical_cfg) pair differing only in what their own
    dynamics require: same dt, same chief orbit, no perturbations on the
    numerical side so both are pure two-body, and no gates or geometry so
    nothing terminates the coast."""
    common = dict(
        dt=dt,
        max_range_m=None,
        dock={"enabled": False},
        physics={"collision_boxes_path": []},
        **overrides,
    )
    return (
        HCWConfig(**common),
        NumericalConfig(perturbations={"zonal_max_degree": 0}, **common),
    )


def _paired_start(num: NumericalDynamics, view: jnp.ndarray):
    """(numerical_state, hcw_state) presenting the same relative view.

    The numerical state is built through `chaser_state_from_view`, the same
    reset inversion `NumericalDynamics.reset` uses, against the chief that
    env's own reset places -- so the pair differs in representation only. The
    HCW state carries the view directly, which is what its 15D layout is.
    """
    start = num.reset(jax.random.PRNGKey(0))
    prefix, chief = start[0:2], start[2:8]
    numerical = jnp.concatenate([prefix, chief, chaser_state_from_view(chief, view)])
    return numerical, jnp.concatenate([prefix, view])


def _coast(num: NumericalDynamics, hcw: HCWDynamics, view: jnp.ndarray, steps: int):
    """Fly both envs from the same view under zero control; return the
    per-step (numerical_position, hcw_position) relative-position histories,
    each (steps, 3) in world axes."""
    numerical, hcw_state = _paired_start(num, view)
    step_num, step_hcw = jax.jit(num.step), jax.jit(hcw.step)

    numerical_track, hcw_track = [], []
    for _ in range(steps):
        numerical, _ = step_num(numerical, ZERO_ACTION)
        hcw_state, _ = step_hcw(hcw_state, ZERO_ACTION)
        numerical_track.append(np.asarray(relative_view(numerical)[0:3], np.float64))
        hcw_track.append(np.asarray(hcw_state[HCW_LAYOUT.pos], np.float64))
    return np.stack(numerical_track), np.stack(hcw_track)


def test_numerical_matches_hcw_within_linearization():
    """The whole-model cross-check: two independent integrations of the same
    200 s coast agree to 0.012% of the separation they are flown at.

    A 100 m radial standoff at rest in LVLH is the canonical CW case -- it
    does not stay put, it drifts along-track and outward, which is exactly
    what makes the comparison non-trivial: over 200 s the pair separates from
    100.00 m to 107.60 m and picks up 1.14 m of along-track offset, and both
    envs have to produce that same motion from equations that share no code.

    Measured: 1.27e-2 m of disagreement at 200 s -- 0.0127% of the 100 m the
    pair is closest at, 0.0118% of the 107.6 m it ends at.
    `test_the_gap_is_hcws_circular_chief_assumption` identifies what that
    residual is.

    Two bounds, because they say different things. The 1%-of-separation and
    1 m absolute pair is the REQUIREMENT -- what "agrees within HCW's own
    linearization error" has to mean for this env to be usable as a check on
    that one. The 0.03% is what the code actually does, 2.4x the measured
    figure, and it is the one that would catch a regression: the quantity is
    fully deterministic (fixed key, fixed config, no dispersion), so there is
    no seed spread for a margin to absorb, and leaving only the requirement
    asserted would admit an 80x degradation silently.
    """
    hcw_cfg, num_cfg = _configs()
    numerical, hcw = _coast(
        NumericalDynamics(num_cfg), HCWDynamics(hcw_cfg), RADIAL_STANDOFF_VIEW, STEPS
    )

    gap = np.linalg.norm(numerical - hcw, axis=1)
    separation = np.linalg.norm(hcw, axis=1)
    assert gap.max() < 0.01 * separation.min()
    assert gap.max() < 1.0
    assert gap.max() < 3e-4 * separation.min()

    # Not vacuous in either of the two ways it could be. The trajectory has to
    # actually move -- a comparison of two vehicles parked at their start
    # would pass the bound above trivially -- and the gap has to actually
    # resolve, growing with the elapsed time the linearization error
    # accumulates over rather than sitting at f64 round-off.
    assert separation[-1] - separation[0] > 7.0
    assert abs(hcw[-1, 1]) > 1.0
    early = int(round(5.0 / DT)) - 1
    assert gap[-1] > 100.0 * gap[early] > 0.0


def _gap_at_eccentricity(ecc):
    """Peak numerical-to-HCW position gap over the standard coast, at a chief
    eccentricity of `ecc`. Returns (gap, minimum separation), both metres."""
    hcw_cfg, num_cfg = _configs(orbit={"ecc": ecc})
    numerical, hcw = _coast(
        NumericalDynamics(num_cfg), HCWDynamics(hcw_cfg), RADIAL_STANDOFF_VIEW, STEPS
    )
    return (
        float(np.linalg.norm(numerical - hcw, axis=1).max()),
        float(np.linalg.norm(hcw, axis=1).min()),
    )


def test_the_gap_is_hcws_circular_chief_assumption():
    """What separates the two envs is the chief's eccentricity, and nothing
    else worth naming -- which is what makes the bound in the test above a
    statement about HCW's model rather than an arbitrary number.

    Clohessy-Wiltshire is derived about a CIRCULAR chief; the shipped orbit
    has e = 5e-4, which the numerical env propagates and the linearization
    cannot express. Two things are asserted, and the second is the one that
    identifies the term:

    * At e = 0 the gap collapses ~100x, to 1.24e-4 m (1.2e-6 of the
      separation) -- the second-order (rho / r_chief) residual that survives
      even a circular chief, and all that is left once the eccentricity is
      gone.
    * Between e = 1e-4 and e = 5e-4 the gap grows by very nearly the SAME
      factor of 5 the eccentricity does (measured 2.45e-3 m and 1.27e-2 m, a
      ratio of 5.19 against the 5.00 of an exactly first-order term, the 4%
      excess being the circular floor above and O(e^2) riding along). That is
      what "first order in e" means, and it is what distinguishes this from a
      force-model discrepancy: an error in either gravity model would not care
      about the chief's eccentricity at all, and a second-order term would
      have grown by 25.
    """
    circular_gap, separation = _gap_at_eccentricity(0.0)
    assert circular_gap < 1e-5 * separation
    # Still resolved above f64 round-off, so this is the second-order term
    # measured rather than the two integrations agreeing to the last bit --
    # which they must not, being different equations.
    assert circular_gap > 1e-8 * separation

    small_gap, _ = _gap_at_eccentricity(1e-4)
    shipped_gap, _ = _gap_at_eccentricity(5e-4)
    assert circular_gap < small_gap < shipped_gap

    # +-20% around the first-order 5.0, which comfortably clears the measured
    # 5.19 while still rejecting both the 25 a second-order term would give
    # and the 1 of a residual that did not depend on eccentricity at all.
    ratio = shipped_gap / small_gap
    assert 4.0 < ratio < 6.0, f"gap ratio {ratio:.4g} is not first order in eccentricity"


# The epoch test's own start and step size, chosen so that a narrowed prefix
# could not survive them. `OrbitConfig`'s default epoch lands on 43200.0 s of
# day, which is exactly representable in float32 -- an env that had silently
# narrowed its prefix would pass a comparison anchored there. Offsetting the
# start by 137.37 s moves it to 43337.37, where float32's ulp is 0.00390625 s
# and the value itself is not representable, and stepping at the shipped
# dt = 0.05 makes each add 12.8 of those ulps wide: the biased-rounding regime
# `envs/common/epoch_state.py` measures at 290 s of loss per orbit.
EPOCH_TEST_DT = 0.05
EPOCH_TEST_STEPS = 400
EPOCH_TEST_OFFSET_S = 137.37


def test_epoch_prefix_advances_identically_in_both_envs():
    """Both envs advance the [jd_day, sec_of_day] prefix through the same
    `advance_epoch_state`, outside their integrators. Same dt and same start,
    so the two prefixes must stay BIT-identical for the whole coast -- and
    equal to the helper called directly on its own. If either drifts, the
    shared helper is being bypassed or re-implemented somewhere.

    Run from a start float32 cannot represent, at a dt that is 12.8 float32
    ulps wide there (see the constants above): a prefix that had narrowed
    anywhere along either env's path would separate from the float64
    reference within a few hundred steps rather than compare equal to it.
    """
    hcw_cfg, num_cfg = _configs(
        dt=EPOCH_TEST_DT,
        orbit={"epoch_offset_range_s": (EPOCH_TEST_OFFSET_S, EPOCH_TEST_OFFSET_S)},
    )
    num, hcw = NumericalDynamics(num_cfg), HCWDynamics(hcw_cfg)
    numerical, hcw_state = _paired_start(num, RADIAL_STANDOFF_VIEW)

    assert numerical.dtype == jnp.float64 and hcw_state.dtype == jnp.float64
    started_at = float(numerical[1])
    # float() on both sides: comparing a np.float32 against a Python float
    # directly would run the comparison IN float32 under NEP 50's weak
    # promotion and report every value as representable.
    assert float(np.float32(started_at)) != started_at, "start epoch is float32-exact"

    expected = numerical[0:2]
    step_num, step_hcw = jax.jit(num.step), jax.jit(hcw.step)
    for _ in range(EPOCH_TEST_STEPS):
        numerical, _ = step_num(numerical, ZERO_ACTION)
        hcw_state, _ = step_hcw(hcw_state, ZERO_ACTION)
        expected = advance_epoch_state(expected, EPOCH_TEST_DT)
        np.testing.assert_array_equal(np.asarray(numerical[0:2]), np.asarray(hcw_state[0:2]))
        np.testing.assert_array_equal(np.asarray(numerical[0:2]), np.asarray(expected))

    # Non-vacuous: the prefix has to have actually advanced by the elapsed
    # duration, not sat still and compared equal to itself.
    #
    # 1e-8 s, not exact. float64 has the same biased-rounding shape float32
    # does, eight orders smaller: one ulp at 4.3e4 s is 7.3e-12 s, so 400 adds
    # of a dt that is not a whole number of them accumulate ~1.2e-9 s (that is
    # the measured value). The float32 counterfactual over these same 400
    # steps is 0.31 s, which this bound rejects by seven orders.
    elapsed = float(numerical[1]) - started_at
    assert elapsed == pytest.approx(EPOCH_TEST_STEPS * EPOCH_TEST_DT, abs=1e-8)


def test_the_two_envs_disagree_on_quaternion_sign_but_not_on_the_rotation():
    """The one convention the two envs do NOT share, recorded rather than
    reconciled.

    `relative_view` canonicalizes q_bw onto the w >= 0 hemisphere, which is
    the only thing a pure state -> view function can do: it has no previous
    view to stay continuous against. iss-hcw carries q_bw IN state and so
    resolves the q/-q double cover the other way, flipping each step to stay
    on the same hemisphere as the last one -- which means it holds whatever
    hemisphere its episode started on, indefinitely.

    So the same physical attitude reaches a consumer as q from one env and -q
    from the other. The ROTATION is identical, and every gate in the task
    layer reads it as one (`EventChecker.docked` takes |w| of the error
    quaternion), but a consumer reading raw quaternion components across both
    envs must expect differing sign distributions -- iss-numerical's q_bw_w is
    non-negative by construction while iss-hcw's is not. That is the
    documented contract, not a defect: unifying it would move iss-hcw off the
    numerics its golden fixtures pin.
    """
    hcw_cfg, num_cfg = _configs()
    num, hcw = NumericalDynamics(num_cfg), HCWDynamics(hcw_cfg)

    # A 200 deg rotation about the world x axis: w = cos(100 deg) < 0, so the
    # w >= 0 canonicalization has something to do.
    half = np.deg2rad(100.0)
    far_side = jnp.asarray([np.cos(half), np.sin(half), 0.0, 0.0], jnp.float64)
    assert far_side[0] < 0.0
    view = RADIAL_STANDOFF_VIEW.at[6:10].set(far_side)

    numerical, hcw_state = _paired_start(num, view)
    viewed = relative_view(numerical)[6:10]
    # HCW carries q_bw in state at HCW_LAYOUT.quat (8:12), two past the view's
    # own 6:10 -- the epoch prefix sits in front of it there.
    in_state = hcw_state[HCW_LAYOUT.quat]

    # The divergence itself: the view came back negated, not merely close.
    assert float(viewed[0]) > 0.0
    np.testing.assert_allclose(np.asarray(viewed), -np.asarray(far_side), rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(in_state), np.asarray(far_side))

    # And the agreement underneath it, checked on the rotations themselves
    # rather than on the components -- the level at which the two envs are
    # required to be the same env.
    np.testing.assert_allclose(
        np.asarray(quat_to_rotmat(viewed), np.float64),
        np.asarray(quat_to_rotmat(in_state), np.float64),
        rtol=0.0,
        atol=1e-12,
    )

    # The conventions persist under stepping rather than being a property of
    # the initial state: iss-hcw stays on the hemisphere it started on and
    # iss-numerical's view stays on the canonical one, for as long as the
    # episode runs.
    step_num, step_hcw = jax.jit(num.step), jax.jit(hcw.step)
    for _ in range(20):
        numerical, _ = step_num(numerical, ZERO_ACTION)
        hcw_state, _ = step_hcw(hcw_state, ZERO_ACTION)
        assert float(relative_view(numerical)[6]) >= 0.0
        assert float(hcw_state[HCW_LAYOUT.quat][0]) < 0.0
