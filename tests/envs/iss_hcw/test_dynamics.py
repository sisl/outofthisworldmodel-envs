import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astrojax.constants import GM_EARTH
from astrojax.relative_motion.hcw_dynamics import hcw_stm

from owm_envs.core.quaternion import rotate_body_to_world
from owm_envs.envs.common.config import dock_target
from owm_envs.envs.common.epoch_state import epoch_prefix, seconds_between
from owm_envs.envs.common.orbit import RTN_FROM_WORLD
from owm_envs.envs.iss_hcw.config import HCWConfig
from owm_envs.envs.iss_hcw.dynamics import STATE_LABELS, HCWDynamics

ZERO_ACTION = jnp.zeros(6, jnp.float64)
BODY_Z = jnp.array([0.0, 0.0, 1.0], jnp.float64)


def _wide_cfg() -> HCWConfig:
    return HCWConfig(
        orbit={
            "epoch_offset_range_s": (0.0, 5400.0),
            "start_radius_range_m": (80.0, 120.0),
            "start_speed_max_m_s": 0.1,
            "start_attitude_error_max_deg": 10.0,
            "start_rate_max_rad_s": 0.01,
        }
    )


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _nose_errors_deg(states: jnp.ndarray) -> np.ndarray:
    """Angle between each state's body +z in world and the direction to the ISS.

    Measured as a chord, 2 asin(|a - b| / 2), not as arccos of the dot
    product: the quaternion comes back from the f32-pinned helpers with |q|
    off unity by ~4e-9, and arccos near 1 turns that norm deficiency into a
    spurious 1e-4 rad "misalignment" that swamps the real error. The chord is
    well conditioned at zero and exact out to 10 deg.
    """
    body_z = jnp.broadcast_to(BODY_Z, (states.shape[0], 3))
    nose = _unit(np.asarray(jax.vmap(rotate_body_to_world)(states[:, 8:12], body_z)))
    to_iss = _unit(np.asarray(-states[:, 2:5]))
    half_chord = np.clip(np.linalg.norm(nose - to_iss, axis=1) / 2.0, 0.0, 1.0)
    return np.rad2deg(2.0 * np.arcsin(half_chord))


def _free_flight_cfg() -> HCWConfig:
    # No dock termination, no boxes, no domain bound: pure dynamics.
    return HCWConfig(
        max_range_m=None,
        dock={"enabled": False},
        physics={"collision_boxes_path": [], "linear_damping": 0.0, "angular_damping": 0.0},
    )


def _rollout(dyn: HCWDynamics, state: jnp.ndarray, steps: int) -> jnp.ndarray:
    body = jax.jit(lambda _, s: dyn.step(s, ZERO_ACTION)[0])
    return jax.lax.fori_loop(0, steps, body, state)


def test_translation_matches_hcw_stm_quarter_orbit():
    cfg = _free_flight_cfg()
    dyn = HCWDynamics(cfg)
    n = dyn.ref.mean_motion
    state0 = dyn.reset(jax.random.PRNGKey(0))
    rel0_world = state0[2:8]
    steps = int((0.25 * 2 * np.pi / n) / cfg.dt)
    end = _rollout(dyn, state0, steps)

    R = jnp.asarray(RTN_FROM_WORLD, jnp.float64)
    x0 = jnp.concatenate([R @ rel0_world[0:3], R @ rel0_world[3:6]])
    x_ref = hcw_stm(steps * cfg.dt, n) @ x0
    rel_end_rtn = jnp.concatenate([R @ end[2:5], R @ end[5:8]])
    # RK4 at dt=0.05 over ~28k steps: the tolerance covers integration error
    # plus astrojax's f32-pinned STM/derivative evaluation, both far below it
    # (worst channel 4.2e-5 m). atol has to stay tight enough to constrain the
    # cross-track channel, which passes through zero here and so is governed
    # by atol alone rather than by rtol; at 1e-3 it keeps ~3 orders of margin.
    np.testing.assert_allclose(np.asarray(rel_end_rtn), np.asarray(x_ref), rtol=2e-3, atol=1e-3)


def test_cw_sign_probes():
    """Five qualitative CW facts, one step each from rest at 100 m offsets.

    Frame: R=+z_w, T=-y_w, N=+x_w.
    1. +R offset -> outward radial acceleration (3 n^2 x): rel_z accelerates +.
    2. -R offset -> inward: rel_z accelerates -.
    3. +N offset (x_w) -> restoring: rel_x accelerates -.
    4. Radial velocity +R -> Coriolis along -T: y_ddot = -2 n x_dot in RTN,
       T=-y_w so rel_y accelerates + when moving radially outward.
    5. Along-track offset alone -> no acceleration (y enters HCW only via rates).
    """
    cfg = _free_flight_cfg()
    dyn = HCWDynamics(cfg)
    step = jax.jit(lambda s: dyn.step(s, ZERO_ACTION)[0])

    def accel(rel_pos, rel_vel):
        s = dyn.reset(jax.random.PRNGKey(0))
        s = s.at[2:5].set(jnp.asarray(rel_pos, jnp.float64))
        s = s.at[5:8].set(jnp.asarray(rel_vel, jnp.float64))
        return (np.asarray(step(s)[5:8]) - np.asarray(s[5:8])) / cfg.dt

    assert accel([0, 0, 100], [0, 0, 0])[2] > 0  # 1
    assert accel([0, 0, -100], [0, 0, 0])[2] < 0  # 2
    assert accel([100, 0, 0], [0, 0, 0])[0] < 0  # 3
    assert accel([0, 0, 0], [0, 0, 0.1])[1] > 0  # 4
    np.testing.assert_allclose(accel([0, -100, 0], [0, 0, 0]), 0.0, atol=1e-6)  # 5


def test_gravity_gradient_torque_closed_form():
    """Principal-axis body tilted by theta about world x, radial = body z
    rotated: closed form T_x = (3 mu / 2 r^3) (I_zz - I_yy) sin(2 theta).
    Compare the one-step angular-rate change at rtol 1e-4."""
    cfg = _free_flight_cfg()
    dyn = HCWDynamics(cfg)
    ixx, iyy, izz = cfg.physics.inertia_diag
    theta = 0.3
    q = jnp.asarray([np.cos(theta / 2), np.sin(theta / 2), 0.0, 0.0], jnp.float64)
    s = dyn.reset(jax.random.PRNGKey(0))
    s = s.at[2:5].set(jnp.zeros(3))  # at the chief: r = r_chief
    s = s.at[8:12].set(q)
    nxt = dyn.step(s, ZERO_ACTION)[0]
    domega = (np.asarray(nxt[12:15]) - np.asarray(s[12:15])) / cfg.dt

    r = np.linalg.norm(np.asarray(dyn.ref.chief_state_eci(0.0))[:3])
    t_x = 1.5 * GM_EARTH / r**3 * (izz - iyy) * np.sin(2 * theta)
    np.testing.assert_allclose(domega[0], t_x / ixx, rtol=1e-4, atol=1e-12)
    np.testing.assert_allclose(domega[1:], 0.0, atol=1e-9)


def test_free_precession_matches_the_axisymmetric_rate():
    """The gravity-gradient probe spins at zero rate, so it never exercises
    the gyroscopic term. A tumbling body does: the default inertia is
    axisymmetric (Ixx == Iyy), for which Euler's equations give a transverse
    rate of constant magnitude precessing at lambda = (I_t - I_z) / I_t * w_z
    while w_z itself is conserved. Over 60 s the gravity-gradient torque
    perturbs these by ~5e-4 relative, which sets the tolerances.
    """
    cfg = _free_flight_cfg()
    dyn = HCWDynamics(cfg)
    i_t, i_t2, i_z = cfg.physics.inertia_diag
    assert i_t == i_t2, "this test assumes an axisymmetric default inertia"

    w_t, w_z = 0.05, 0.1
    s = dyn.reset(jax.random.PRNGKey(0))
    s = s.at[12:15].set(jnp.asarray([w_t, 0.0, w_z], jnp.float64))
    duration = 60.0
    end = _rollout(dyn, s, int(duration / cfg.dt))

    np.testing.assert_allclose(np.linalg.norm(np.asarray(end[12:14])), w_t, rtol=1e-3)
    np.testing.assert_allclose(float(end[14]), w_z, rtol=1e-3)

    # (wx, wy) rotates at -lambda: wx_dot = lambda*wy, wy_dot = -lambda*wx.
    # 60 s of it is 2.25 rad, inside atan2's branch, so no unwrapping.
    lam = (i_t - i_z) / i_t * w_z
    np.testing.assert_allclose(
        np.arctan2(float(end[13]), float(end[12])), -lam * duration, atol=5e-3
    )
    # Renormalized every step, so RK4's norm drift never accumulates. The
    # floor is f32, not f64: `quat_normalize` goes through astrojax's
    # f32-pinned Quaternion, so |q| lands within ~1e-8 of 1 and stays there.
    # rtol=0 because numpy's 1e-7 default would otherwise be doing the work.
    np.testing.assert_allclose(float(jnp.linalg.norm(end[8:12])), 1.0, rtol=0, atol=1e-7)


def test_epoch_advances_and_rolls_over():
    cfg = _free_flight_cfg()
    dyn = HCWDynamics(cfg)
    s = dyn.reset(jax.random.PRNGKey(0))

    nxt = dyn.step(s, ZERO_ACTION)[0]
    assert float(seconds_between(nxt[0:2], s[0:2])) == pytest.approx(cfg.dt, abs=1e-9)

    # Day rollover happens outside the integrator, so check it explicitly:
    # half a step short of midnight, the seconds-of-day must wrap and the
    # Julian day number must tick.
    late = s.at[1].set(86400.0 - 0.5 * cfg.dt)
    rolled = dyn.step(late, ZERO_ACTION)[0]
    assert float(rolled[0]) == float(late[0]) + 1.0
    assert float(rolled[1]) == pytest.approx(0.5 * cfg.dt, abs=1e-9)


def test_epoch_prefix_stays_exact_over_a_long_rollout():
    """The reason the state is f64: an f32 seconds-of-day biases every add in
    the same direction and loses ~290 s per orbit at dt=0.05 (see
    envs/common/epoch_state.py). f64 keeps the drift at the nanosecond level.
    """
    cfg = _free_flight_cfg()
    dyn = HCWDynamics(cfg)
    s = dyn.reset(jax.random.PRNGKey(0))
    steps = 20_000
    end = _rollout(dyn, s, steps)
    elapsed = float(seconds_between(end[0:2], s[0:2]))
    assert elapsed == pytest.approx(steps * cfg.dt, abs=1e-6)


def test_reset_zero_width_config_is_the_undispersed_start():
    """Every sampling range in `OrbitConfig` defaults to zero width, and that
    case has to collapse exactly: epoch0, the configured radius, at rest,
    nose at the ISS."""
    cfg = HCWConfig()
    dyn = HCWDynamics(cfg)
    s = dyn.reset(jax.random.PRNGKey(3))

    np.testing.assert_array_equal(np.asarray(s[0:2]), np.asarray(epoch_prefix(dyn.ref.epoch0)))
    assert float(jnp.linalg.norm(s[2:5])) == pytest.approx(100.0, abs=1e-9)
    np.testing.assert_array_equal(np.asarray(s[5:8]), 0.0)
    np.testing.assert_array_equal(np.asarray(s[12:15]), 0.0)
    # The nose direction runs through the f32-pinned quaternion helpers, so
    # 1e-6 is the grain here, not f64.
    assert float(jnp.linalg.norm(s[8:12])) == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(_nose_errors_deg(s[None, :]), 0.0, atol=1e-4)


def test_reset_disperses_across_the_configured_ranges():
    """Wide ranges, 500 vmapped keys: every channel inside its bound and
    actually spread, not pinned to one value."""
    dyn = HCWDynamics(_wide_cfg())
    states = jax.vmap(dyn.reset)(jax.random.split(jax.random.PRNGKey(0), 500))
    assert states.dtype == jnp.float64

    radii = np.asarray(jnp.linalg.norm(states[:, 2:5], axis=1))
    assert radii.min() >= 80.0 - 1e-6 and radii.max() <= 120.0 + 1e-6
    assert radii.std() > 5.0
    # The radius is uniform in RADIUS, not uniform in volume -- the standoff
    # range is a mission constraint, not a ball to fill, so the shell weighting
    # `sample_vector_in_ball` applies to velocity and rates would be wrong
    # here. The spread above cannot tell the two apart: volume weighting over
    # [80, 120] has std 11.31 against uniform's 11.55. Only the mean separates
    # them, 102.63 against 100.0, which is 5.1 standard errors at n = 500.
    assert abs(radii.mean() - 100.0) < 1.5

    # The ball samplers draw at f32, so a vector right at the boundary can
    # land a few f32 ulps outside it -- hence a relative slack on the caps.
    speeds = np.asarray(jnp.linalg.norm(states[:, 5:8], axis=1))
    assert speeds.max() <= 0.1 * (1 + 1e-6)
    assert speeds.max() > 0.05

    rates = np.asarray(jnp.linalg.norm(states[:, 12:15], axis=1))
    assert rates.max() <= 0.01 * (1 + 1e-6)
    assert rates.max() > 0.005

    offsets = np.asarray(seconds_between(states[:, 0:2], epoch_prefix(dyn.ref.epoch0)))
    assert offsets.min() >= 0.0 and offsets.max() <= 5400.0
    assert offsets.std() > 1000.0

    nose_deg = _nose_errors_deg(states)
    # The error rotation is applied about a uniformly random axis, so the
    # nose swings by at most the sampled angle and by less when the axis
    # leans toward body +z -- hence a bound at 10 deg but a mean well below.
    assert nose_deg.max() <= 10.0 + 1e-4
    assert nose_deg.std() > 1.0


def test_interface_dimensions():
    dyn = HCWDynamics(_free_flight_cfg())
    assert dyn.state_dim == 15
    assert dyn.action_dim == 6
    assert dyn.reset(jax.random.PRNGKey(0)).shape == (15,)
    assert len(STATE_LABELS) == 15


def test_state_stays_float64():
    cfg = _free_flight_cfg()
    dyn = HCWDynamics(cfg)
    s = dyn.reset(jax.random.PRNGKey(0))
    assert s.dtype == jnp.float64
    # An f32 state handed in must be widened, not carried at f32.
    nxt = dyn.step(s.astype(jnp.float32), jnp.zeros(6, jnp.float32))[0]
    assert nxt.dtype == jnp.float64


def test_events_fire_like_iss():
    """Dock/collision/escape run through the shared EventChecker on the view:
    place the chaser at the dock pose at rest -> docked fires."""
    cfg = HCWConfig()
    dyn = HCWDynamics(cfg)
    s = dyn.reset(jax.random.PRNGKey(0))
    target = jnp.asarray(dock_target(cfg), jnp.float64)
    s = s.at[2:5].set(target[0:3])
    s = s.at[5:8].set(jnp.zeros(3))
    s = s.at[8:12].set(target[3:7])
    s = s.at[12:15].set(jnp.zeros(3))
    _, events = dyn.step(s, ZERO_ACTION)
    assert bool(events.docked)
    assert not bool(events.escaped)
