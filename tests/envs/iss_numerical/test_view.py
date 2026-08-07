"""`relative_view` and the frame rate it is built on.

The view is the only thing the task layer sees of this env, and it is a
derived quantity -- `NUM_LAYOUT`'s slices hold absolute ECI numbers. So these
tests pin the three conversions the derivation makes (offset, frame-relative
velocity, world-relative attitude and rate), the exactness of its inverse
`chaser_state_from_view`, and the axis of the frame rate both of them use.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from owm_envs.core.quaternion import quat_from_body_z_to, quat_to_rotmat
from owm_envs.envs.common.orbit import RTN_FROM_WORLD, world_from_eci
from owm_envs.envs.iss_numerical.config import NumericalConfig
from owm_envs.envs.iss_numerical.dynamics import (
    NumericalDynamics,
    chaser_state_from_view,
    frame_rate_eci,
    relative_view,
)

ZERO_ACTION = jnp.zeros(6, jnp.float64)
IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

# Wide enough that no channel is exercised only near zero: a standoff spanning
# two decades, a start speed and body rate an order above the defaults, and an
# attitude error large enough to reach every hemisphere of q_bw.
WIDE_DISPERSION = {
    "start_radius_range_m": (5.0, 500.0),
    "start_speed_max_m_s": 2.0,
    "start_rate_max_rad_s": 0.2,
    "start_attitude_error_max_deg": 175.0,
    "epoch_offset_range_s": (0.0, 5600.0),
}


def _cfg(**overrides) -> NumericalConfig:
    physics = {"collision_boxes_path": [], "linear_damping": 0.0, "angular_damping": 0.0}
    physics.update(overrides.pop("physics", {}))
    return NumericalConfig(
        max_range_m=None, dock={"enabled": False}, physics=physics, **overrides
    )


def _chord(a, b) -> float:
    """Angle between two rotations, 2 asin(|a - b| / 2), resolving the q/-q
    double cover. Never arccos of a dot product: that loses all precision as
    the angle goes to zero, which is exactly where these tests measure."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    gap = min(float(np.linalg.norm(a - b)), float(np.linalg.norm(a + b)))
    return 2.0 * float(np.arcsin(min(gap / 2.0, 1.0)))


def _world_from_eci_f64(chief) -> np.ndarray:
    """`orbit.world_from_eci` rebuilt in f64 from the chief's state.

    The shipped one narrows to astrojax's f32 (see `envs/common/orbit.py`),
    which is harmless where it is used but would swamp a finite difference of
    the rotation over a fraction of a second.
    """
    r, v = np.asarray(chief[0:3], np.float64), np.asarray(chief[3:6], np.float64)
    radial = r / np.linalg.norm(r)
    normal = np.cross(r, v)
    normal = normal / np.linalg.norm(normal)
    transverse = np.cross(normal, radial)
    return RTN_FROM_WORLD.T @ np.stack([radial, transverse, normal])


def _omega_frame_fd(dyn: NumericalDynamics, state: jnp.ndarray) -> np.ndarray:
    """Angular velocity of the world triad in ECI axes, by finite difference.

    Each row of `world_from_eci` is a world axis expressed in ECI, so it obeys
    e_dot = omega x e; the triad being orthonormal, omega is recovered as
    0.5 * sum_i e_i x e_dot_i, which uses all three rows and needs no inverse.
    """
    nxt, _ = dyn.step(state, ZERO_ACTION)
    rot = _world_from_eci_f64(state[2:8])
    rot_dot = (_world_from_eci_f64(nxt[2:8]) - rot) / dyn.cfg.dt
    return 0.5 * sum(np.cross(rot[i], rot_dot[i]) for i in range(3))


def test_colocated_comoving_chaser_with_world_attitude_is_the_identity_view():
    """The fixed point of the whole derivation: a chaser at the chief's own
    position and velocity, oriented with the world frame and rotating with it,
    views as zeros and the identity quaternion. Every one of the four
    conversions has to cancel exactly for this to hold."""
    dyn = NumericalDynamics(_cfg())
    start = dyn.reset(jax.random.PRNGKey(0))
    chief = start[2:8]

    identity_view = jnp.zeros(13, jnp.float64).at[6].set(1.0)
    view = relative_view(
        jnp.concatenate([start[0:2], chief, chaser_state_from_view(chief, identity_view)])
    )

    np.testing.assert_allclose(np.asarray(view[0:6]), 0.0, rtol=0.0, atol=1e-12)
    assert _chord(view[6:10], IDENTITY_QUAT) < 1e-12
    np.testing.assert_allclose(np.asarray(view[10:13]), 0.0, rtol=0.0, atol=1e-12)


def test_radial_standoff_at_rest_in_lvlh_has_no_relative_velocity():
    """A chaser 100 m straight up from the chief and at rest in the ROTATING
    world frame. Its ECI velocity differs from the chief's -- it has to, being
    on a longer lever arm of the same rotation -- and the view has to report
    that difference as zero. This is the frame-rotation term alone: omega x r
    at 100 m and 1.1e-3 rad/s is 0.11 m/s, 22% of the 0.5 m/s dock velocity
    gate, so dropping it would not be a rounding matter."""
    dyn = NumericalDynamics(_cfg())
    start = dyn.reset(jax.random.PRNGKey(0))
    chief = start[2:8]

    at_rest = jnp.zeros(13, jnp.float64).at[2].set(100.0).at[6].set(1.0)
    state = jnp.concatenate([start[0:2], chief, chaser_state_from_view(chief, at_rest)])

    delta_v_eci = np.asarray(state[11:14] - chief[3:6], np.float64)
    assert float(np.linalg.norm(delta_v_eci)) == pytest.approx(0.112828, abs=1e-5)

    view = relative_view(state)
    # 1e-5 m on the position, not f64 grain: the f32 world<->ECI rotation is
    # orthonormal only to ~1e-7 and the offset goes out through R.T and back
    # through R. The velocity does NOT pay that -- both directions difference
    # the frame's motion at the same ECI offset, so it cancels identically,
    # and at rest in the world frame the channel is exactly zero.
    np.testing.assert_allclose(np.asarray(view[0:3]), [0.0, 0.0, 100.0], rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(view[3:6]), 0.0, rtol=0.0, atol=1e-12)


def test_view_and_chaser_state_from_view_invert_each_other():
    """Round trip 100 wide-dispersion views through the state and back.

    Two of the four channels invert EXACTLY, at f64, and two do not, and the
    split is not arbitrary. The attitude and the rate both pass through the
    world frame as a quaternion (`q_wi`) or as a single additive term built
    from it, and a unit quaternion's conjugate is its exact inverse -- so
    whatever error the f32 world<->ECI rotation put into `q_wi` is applied and
    then removed identically. Position and velocity pass through the rotation
    as a MATRIX, and R.T is only the approximate inverse of R: this one is
    f32-derived, so R @ R.T differs from the identity by ~1e-7 and each
    channel round-trips to that fraction of its own magnitude. The bounds
    below are written against that scale rather than as absolutes, which is
    also why the tolerances are stated explicitly -- `assert_allclose`'s
    default rtol of 1e-7 is exactly the error being measured here and would
    otherwise absorb it.
    """
    dyn = NumericalDynamics(_cfg())
    start = dyn.reset(jax.random.PRNGKey(0))
    prefix, chief = start[0:2], start[2:8]

    for seed in range(100):
        k_pos, k_vel, k_quat, k_rate = jax.random.split(jax.random.PRNGKey(seed), 4)
        raw_quat = jax.random.normal(k_quat, (4,), dtype=jnp.float64)
        quat = raw_quat / jnp.linalg.norm(raw_quat)
        view = jnp.concatenate(
            [
                jax.random.uniform(k_pos, (3,), jnp.float64, minval=-500.0, maxval=500.0),
                jax.random.uniform(k_vel, (3,), jnp.float64, minval=-2.0, maxval=2.0),
                jnp.where(quat[0] < 0.0, -quat, quat),
                jax.random.uniform(k_rate, (3,), jnp.float64, minval=-0.2, maxval=0.2),
            ]
        )
        state = jnp.concatenate([prefix, chief, chaser_state_from_view(chief, view)])
        back = np.asarray(relative_view(state), np.float64)
        view = np.asarray(view, np.float64)

        # 1e-6 leaves one decade over the ~1e-7 the rotation is orthonormal
        # to; 1e-12 leaves four over the f64 grain the other two channels
        # actually hit (3e-16 rad, 1e-17 rad/s) while still rejecting the
        # 1.4e-7 an f32 quaternion product would put here.
        assert np.linalg.norm(back[0:3] - view[0:3]) <= 1e-6 * np.linalg.norm(view[0:3])
        assert np.linalg.norm(back[3:6] - view[3:6]) <= 1e-6 * np.linalg.norm(view[3:6])
        assert _chord(back[6:10], view[6:10]) < 1e-12
        np.testing.assert_allclose(back[10:13], view[10:13], rtol=0.0, atol=1e-12)


def test_state_round_trips_as_a_rotation_but_not_componentwise():
    """The other direction, state -> view -> state, where the w >= 0 flip is
    visible: a chaser whose q_bi puts q_bw on the far hemisphere comes back
    NEGATED. Same rotation, same dynamics, different four numbers -- and the
    reason `relative_view` documents its inverse as an inverse of the rotation
    rather than of the components. Both branches are exercised, and the
    negated ones are checked to be exactly that and not merely close."""
    dyn = NumericalDynamics(_cfg())
    start = dyn.reset(jax.random.PRNGKey(0))
    prefix, chief = start[0:2], start[2:8]

    flipped = 0
    for seed in range(100):
        k_off, k_vel, k_quat, k_rate = jax.random.split(jax.random.PRNGKey(1000 + seed), 4)
        raw_quat = jax.random.normal(k_quat, (4,), dtype=jnp.float64)
        chaser = jnp.concatenate(
            [
                chief[0:3] + jax.random.uniform(k_off, (3,), jnp.float64, minval=-300.0, maxval=300.0),
                chief[3:6] + jax.random.uniform(k_vel, (3,), jnp.float64, minval=-1.0, maxval=1.0),
                raw_quat / jnp.linalg.norm(raw_quat),
                jax.random.uniform(k_rate, (3,), jnp.float64, minval=-0.2, maxval=0.2),
            ]
        )
        state = jnp.concatenate([prefix, chief, chaser])
        back = np.asarray(chaser_state_from_view(chief, relative_view(state)), np.float64)
        chaser = np.asarray(chaser, np.float64)

        # Both ECI channels are bounded by the RELATIVE quantity they were
        # derived from, never by the 6.8e6 m absolute they are stored as: the
        # chief cancels before the rotation is applied. The velocity's scale
        # includes the frame's own motion at that offset, which the view
        # removes and this puts back.
        offset = chaser[0:3] - np.asarray(chief[0:3], np.float64)
        delta_v = chaser[3:6] - np.asarray(chief[3:6], np.float64)
        speed_scale = np.linalg.norm(delta_v) + 1.2e-3 * np.linalg.norm(offset)
        assert np.linalg.norm(back[0:3] - chaser[0:3]) <= 1e-6 * np.linalg.norm(offset)
        assert np.linalg.norm(back[3:6] - chaser[3:6]) <= 1e-6 * speed_scale
        assert _chord(back[6:10], chaser[6:10]) < 1e-12
        np.testing.assert_allclose(back[10:13], chaser[10:13], rtol=0.0, atol=1e-12)

        sign = -1.0 if np.dot(back[6:10], chaser[6:10]) < 0.0 else 1.0
        flipped += sign < 0.0
        np.testing.assert_allclose(back[6:10], sign * chaser[6:10], rtol=0.0, atol=1e-12)

    # Uniform quaternions split evenly, so both branches have to be well
    # represented -- otherwise the assertions above never see the flip.
    assert 20 < flipped < 80


def test_reset_views_back_as_the_dispersions_it_sampled():
    """`reset` samples a relative view, stores it as absolute ECI state, and
    `relative_view` has to hand the same view back. Over 100 keys of wide
    dispersion, every sampled bound reappears: the standoff radius inside its
    range, the world-frame speed and body rate inside their balls, and the
    nose pointed at the ISS to within the attitude dispersion."""
    cfg = _cfg(orbit=WIDE_DISPERSION)
    dyn = NumericalDynamics(cfg)
    views = jax.vmap(lambda k: relative_view(dyn.reset(k)))(
        jax.random.split(jax.random.PRNGKey(7), 100)
    )

    # The slack on each bound is relative, not absolute: what separates the
    # view from the dispersion that produced it is the f32 world<->ECI
    # rotation's ~1e-7 non-orthonormality, which scales with the channel (see
    # test_view_and_chaser_state_from_view_invert_each_other). The rate
    # channel inverts exactly and only needs slack for f32 sampling.
    radius = np.linalg.norm(np.asarray(views[:, 0:3], np.float64), axis=1)
    lo, hi = cfg.orbit.start_radius_range_m
    assert radius.min() >= lo * (1.0 - 1e-6) and radius.max() <= hi * (1.0 + 1e-6)
    # Non-vacuous: the draws have to actually span the range they are bounded by.
    assert radius.max() - radius.min() > 0.8 * (hi - lo)

    speed = np.linalg.norm(np.asarray(views[:, 3:6], np.float64), axis=1)
    assert speed.max() <= cfg.orbit.start_speed_max_m_s * (1.0 + 1e-6)
    assert speed.max() > 0.9 * cfg.orbit.start_speed_max_m_s

    rate = np.linalg.norm(np.asarray(views[:, 10:13], np.float64), axis=1)
    assert rate.max() <= cfg.orbit.start_rate_max_rad_s * (1.0 + 1e-6)
    assert rate.max() > 0.9 * cfg.orbit.start_rate_max_rad_s

    # The sampled attitude is nose (body +z) on the ISS, missed by at most the
    # configured dispersion. Recovering the miss needs only the view itself:
    # the ISS sits at the world origin, so the intended nose direction is
    # -rel_pos, and the achieved one is q_bw applied to body +z.
    rotations = np.asarray(jax.vmap(quat_to_rotmat)(views[:, 6:10]), np.float64)
    nose = rotations[:, :, 2]
    intended = -np.asarray(views[:, 0:3], np.float64) / radius[:, None]
    miss = np.arccos(np.clip(np.einsum("ki,ki->k", nose, intended), -1.0, 1.0))
    assert miss.max() <= np.deg2rad(cfg.orbit.start_attitude_error_max_deg) + 1e-6
    assert miss.max() > 0.5 * np.deg2rad(cfg.orbit.start_attitude_error_max_deg)


def test_reset_zero_width_attitude_points_the_nose_exactly_at_the_iss():
    """The zero-width default of the same dispersion, where the bound above
    collapses to an equality: nose exactly on the ISS, to the f32 grain of
    `quat_from_body_z_to`, which is what `reset` samples with."""
    dyn = NumericalDynamics(_cfg())
    view = relative_view(dyn.reset(jax.random.PRNGKey(3)))
    direction = np.asarray(view[0:3], np.float64)
    direction = direction / np.linalg.norm(direction)
    assert _chord(view[6:10], quat_from_body_z_to(jnp.asarray(-direction, jnp.float64))) < 1e-6


def test_frame_rate_axis_matches_a_finite_difference_of_the_rotation():
    """`frame_rate_eci` returns a closed form; nothing pinned its direction.

    Point-mass gravity, so the out-of-plane term the closed form omits is
    identically zero and the only discrepancy left is the forward difference's
    own O(dt) truncation. Both rotations are rebuilt in f64 -- the shipped
    astrojax f32 one is orthonormal only to ~1e-7, three orders above the
    ~1e-5 rad the triad turns through in one dt.
    """
    dyn = NumericalDynamics(_cfg(dt=0.5, perturbations={"zonal_max_degree": 0}))
    state = dyn.reset(jax.random.PRNGKey(0))

    code = np.asarray(frame_rate_eci(state[2:5], state[5:8]), np.float64)
    fd = _omega_frame_fd(dyn, state)

    cosine = float(np.dot(fd, code) / (np.linalg.norm(fd) * np.linalg.norm(code)))
    assert cosine > 1.0 - 1e-6
    assert float(np.linalg.norm(fd - code)) < 1e-9
    # h / r^2 is the TRUE-anomaly rate, close to the mean motion the orbit's
    # sma implies but deliberately not equal to it: at e = 5e-4 it swings by
    # 2e = 1e-3 either side of n over a revolution.
    assert float(np.linalg.norm(code)) == pytest.approx(dyn.ref.mean_motion, rel=3e-3)


def test_omitted_frame_rate_term_is_radial_and_within_its_documented_bound():
    """What the in-plane-only frame rate misses, at zonal degree 6.

    `frame_rate_eci` documents the full rate as (h/r^2) N + (r a_N / h) R and
    returns only the first term, so the residual against a finite difference
    has to lie along the RADIAL axis, not somewhere unmodelled -- and stay
    under the 1.1e-6 rad/s peak `relative_view` quotes its velocity error
    from. Both are checked, the direction being much the sharper of the two.
    """
    dyn = NumericalDynamics(_cfg(dt=0.5, perturbations={"zonal_max_degree": 6}))
    state = dyn.reset(jax.random.PRNGKey(0))

    residual = _omega_frame_fd(dyn, state) - np.asarray(
        frame_rate_eci(state[2:5], state[5:8]), np.float64
    )
    radial = np.asarray(state[2:5], np.float64)
    radial = radial / np.linalg.norm(radial)

    assert float(np.linalg.norm(residual)) < 1.1e-6
    # Non-vacuous only if the residual is resolved at all above the point-mass
    # truncation floor measured in the test above (6e-11 rad/s).
    assert float(np.linalg.norm(residual)) > 1e-9
    assert abs(float(np.dot(residual, radial))) / float(np.linalg.norm(residual)) > 0.95


def test_view_is_f64_throughout():
    """The whole point of the derivation: no channel narrows to the astrojax
    f32 the world rotation and `core.quaternion` arrive at."""
    dyn = NumericalDynamics(_cfg())
    view = relative_view(dyn.reset(jax.random.PRNGKey(0)))
    assert view.shape == (13,)
    assert view.dtype == jnp.float64
    assert chaser_state_from_view(dyn.reset(jax.random.PRNGKey(0))[2:8], view).dtype == jnp.float64


@pytest.mark.parametrize("seed", range(20))
def test_view_rotation_agrees_with_the_world_frame_it_names(seed):
    """`q_bw` composes with the chief's world<->ECI rotation the way its name
    says: R(q_bw) is the stored body -> ECI rotation carried into world axes,
    checked against the matrices directly rather than through the quaternion
    algebra the view uses to get there.

    Being a check on ROTATIONS, it is blind to one thing by construction:
    R(q) == R(-q), so no matrix comparison can catch a sign flip, and the
    hemisphere is pinned separately above. What the sweep buys instead is
    which path through `quat_from_rotmat` gets exercised in context.
    Shepperd's method divides by whichever of the four squared components is
    largest, and that is decided by the chief's own rotation -- so one key
    reaches exactly one of the four. Over these twenty (measured) all four
    are reached; a single key would leave three untested here and covered
    only against hand-built matrices in `tests/core`.
    """
    dyn = NumericalDynamics(_cfg(orbit=WIDE_DISPERSION))
    state = dyn.reset(jax.random.PRNGKey(seed))
    view = relative_view(state)

    expected = np.asarray(world_from_eci(state[2:8]), np.float64) @ np.asarray(
        quat_to_rotmat(state[14:18]), np.float64
    )
    # Absolute only (rtol=0.0): the entries of a rotation matrix run down to
    # zero, and a relative tolerance on those would demand a precision no
    # channel here has. The bound is the astrojax f32 rotation's own ~1e-7
    # non-orthonormality, measured at 3.0e-7 worst over these keys.
    np.testing.assert_allclose(
        np.asarray(quat_to_rotmat(view[6:10]), np.float64), expected, atol=1e-6, rtol=0.0
    )
