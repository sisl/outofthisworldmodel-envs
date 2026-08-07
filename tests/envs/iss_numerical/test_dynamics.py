import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astrojax.constants import GM_EARTH, GM_MOON, GM_SUN, J2_EARTH, OMEGA_EARTH, R_EARTH
from astrojax.epoch import Epoch
from astrojax.orbit_dynamics.density import density_harris_priester
from astrojax.orbit_dynamics.ephemerides import moon_position, sun_position

from owm_envs.core.quaternion import quat_to_rotmat
from owm_envs.envs.common.config import dock_target
from owm_envs.envs.common.epoch_state import epoch_from_prefix, epoch_prefix, seconds_between
from owm_envs.envs.common.orbit import world_from_eci
from owm_envs.envs.iss_numerical.config import NumericalConfig
from owm_envs.envs.iss_numerical.dynamics import (
    STATE_LABELS,
    NumericalDynamics,
    accel_perturbed,
    chaser_state_from_view,
    frame_rate_eci,
    relative_view,
)

ZERO_ACTION = jnp.zeros(6, jnp.float64)

CHIEF = slice(2, 8)
CHASER = slice(8, 14)


def _free_flight_cfg(**overrides) -> NumericalConfig:
    """No dock termination, no boxes, no domain bound: pure dynamics."""
    physics = {"collision_boxes_path": [], "linear_damping": 0.0, "angular_damping": 0.0}
    physics.update(overrides.pop("physics", {}))
    return NumericalConfig(
        max_range_m=None, dock={"enabled": False}, physics=physics, **overrides
    )


def _rollout(dyn: NumericalDynamics, state: jnp.ndarray, steps: int) -> jnp.ndarray:
    body = jax.jit(lambda _, s: dyn.step(s, ZERO_ACTION)[0])
    return jax.lax.fori_loop(0, steps, body, state)


def _orbit_steps(dyn: NumericalDynamics) -> int:
    return int(round(2.0 * np.pi / dyn.ref.mean_motion / dyn.cfg.dt))


def _specific_energy(rv: np.ndarray) -> float:
    return 0.5 * float(np.dot(rv[3:6], rv[3:6])) - GM_EARTH / float(np.linalg.norm(rv[0:3]))


def _angular_momentum(rv: np.ndarray) -> np.ndarray:
    return np.cross(rv[0:3], rv[3:6])


def _raan(rv: np.ndarray) -> float:
    """Right ascension of the ascending node from an ECI state.

    The node vector is z_hat x h = (-h_y, h_x, 0), so RAAN = atan2(h_x, -h_y).
    """
    h = _angular_momentum(rv)
    return float(np.arctan2(h[0], -h[1]))


def test_interface_dimensions():
    dyn = NumericalDynamics(_free_flight_cfg())
    assert dyn.state_dim == 21
    assert dyn.action_dim == 6
    assert dyn.reset(jax.random.PRNGKey(0)).shape == (21,)
    assert len(STATE_LABELS) == 21


def test_linear_damping_is_rejected():
    with pytest.raises(ValueError, match="linear_damping"):
        NumericalDynamics(_free_flight_cfg(physics={"linear_damping": 0.1}))


def test_two_body_conserves_energy_and_momentum_over_one_orbit():
    """Perturbations off, no control: specific orbital energy and |h| of both
    vehicles drift < 1e-9 relative over a full period at dt = 0.5 in f64.

    RK4 is not symplectic, so this is a bound on its secular energy error, not
    a conservation law the integrator enforces. At n*dt = 5.6e-4 the per-step
    error goes as (n*dt)^5 ~ 5e-17 relative, which over ~11k steps leaves
    three orders of margin under the bound. In f32 the state's own 6e-8 grain
    would blow through it on the first step.
    """
    cfg = _free_flight_cfg(dt=0.5)
    dyn = NumericalDynamics(cfg)
    start = dyn.reset(jax.random.PRNGKey(0))
    end = _rollout(dyn, start, _orbit_steps(dyn))

    for name, vehicle in (("chief", CHIEF), ("chaser", CHASER)):
        rv0, rv1 = np.asarray(start[vehicle]), np.asarray(end[vehicle])
        energy0, energy1 = _specific_energy(rv0), _specific_energy(rv1)
        h0 = float(np.linalg.norm(_angular_momentum(rv0)))
        h1 = float(np.linalg.norm(_angular_momentum(rv1)))
        assert abs(energy1 / energy0 - 1.0) < 1e-9, f"{name} energy"
        assert abs(h1 / h0 - 1.0) < 1e-9, f"{name} |h|"


def test_zonal_gate_changes_chief_trajectory():
    """One orbit with zonal_max_degree=2 against 0: the chief's RAAN drifts by
    the J2 secular rate, -3/2 n J2 (Re/p)^2 cos i ~= -1.0e-6 rad/s at these
    elements (the ISS's familiar ~5 deg/day nodal regression), while the
    point-mass run holds its node fixed.

    Measured over exactly one period so the short-period J2 oscillation in
    RAAN -- ~4e-4 rad, roughly 8% of the secular term accumulated here --
    returns to its starting phase instead of contaminating the difference.
    """
    cfg = _free_flight_cfg(dt=0.5)
    point_mass = NumericalDynamics(cfg)
    oblate = NumericalDynamics(_free_flight_cfg(dt=0.5, perturbations={"zonal_max_degree": 2}))

    start = point_mass.reset(jax.random.PRNGKey(0))
    steps = _orbit_steps(point_mass)
    elapsed = steps * cfg.dt

    raan0 = _raan(np.asarray(start[CHIEF]))
    drift_point_mass = _raan(np.asarray(_rollout(point_mass, start, steps)[CHIEF])) - raan0
    drift_oblate = _raan(np.asarray(_rollout(oblate, start, steps)[CHIEF])) - raan0

    semi_latus = cfg.orbit.sma_m * (1.0 - cfg.orbit.ecc**2)
    expected = (
        -1.5
        * point_mass.ref.mean_motion
        * J2_EARTH
        * (R_EARTH / semi_latus) ** 2
        * np.cos(np.deg2rad(cfg.orbit.inc_deg))
        * elapsed
    )
    assert abs(drift_oblate / expected - 1.0) < 0.2
    assert abs(drift_point_mass) < 0.02 * abs(expected)


def test_third_body_and_drag_gates_are_inert_when_off():
    """Each perturbation gate is a static Python branch, so switching one off
    removes the term rather than adding a zero: 100 steps against a
    default-constructed config must agree bitwise. The converse is what makes
    that mean anything -- switching each on has to move the state.
    """
    default = NumericalDynamics(_free_flight_cfg())
    explicit_off = NumericalDynamics(
        _free_flight_cfg(
            perturbations={"third_body_sun": False, "third_body_moon": False, "drag": False}
        )
    )
    start = default.reset(jax.random.PRNGKey(0))
    baseline = _rollout(default, start, 100)

    np.testing.assert_array_equal(
        np.asarray(baseline), np.asarray(_rollout(explicit_off, start, 100))
    )

    for flag in ("third_body_sun", "third_body_moon", "drag"):
        on = NumericalDynamics(_free_flight_cfg(perturbations={flag: True}))
        moved = _rollout(on, start, 100)
        assert not np.array_equal(np.asarray(moved), np.asarray(baseline)), flag


# ~390 km altitude, inside Harris-Priester's 100-1000 km validity band, and
# off-axis in all three components so a transposed or swapped frame shows.
PROBE_R = jnp.array([3.4e6, 4.9e6, 3.2e6], jnp.float64)
PROBE_V = jnp.array([-6.0e3, 2.2e3, 3.1e3], jnp.float64)


def _probe_epoch(cfg):
    """The epoch through the same prefix round-trip `step` uses."""
    return epoch_from_prefix(epoch_prefix(Epoch(cfg.orbit.epoch)))


def _term(cfg, flag, value, mass, ballistic):
    """One gate's contribution: the force model with it on, minus without."""
    epoch = _probe_epoch(cfg)
    off = accel_perturbed(PROBE_R, PROBE_V, epoch, mass, ballistic, cfg.perturbations)
    on = accel_perturbed(
        PROBE_R,
        PROBE_V,
        epoch,
        mass,
        ballistic,
        cfg.perturbations.model_copy(update={flag: value}),
    )
    return np.asarray(on - off, np.float64), np.asarray(off, np.float64)


def test_point_mass_gravity_is_the_f64_closed_form():
    """With every gate off the force model is exactly -GM r / |r|^3, to f64
    and not to astrojax's f32 -- the reason gravity comes from
    envs/common/zonal_gravity rather than astrojax's point-mass helper."""
    cfg = _free_flight_cfg()
    accel = accel_perturbed(
        PROBE_R,
        PROBE_V,
        _probe_epoch(cfg),
        cfg.physics.mass,
        cfg.perturbations.chaser_ballistic,
        cfg.perturbations,
    )
    r = np.asarray(PROBE_R, np.float64)
    np.testing.assert_allclose(
        np.asarray(accel, np.float64), -GM_EARTH * r / np.linalg.norm(r) ** 3, rtol=1e-15
    )


@pytest.mark.parametrize(
    "flag, gm, ephemeris, atol",
    [
        # The tolerance is ABSOLUTE, because astrojax's f32 error here is: a
        # third-body acceleration is the difference of two nearly equal
        # ~5.9e-3 m/s^2 vectors, and at 1.5e11 m that cancels four digits, so
        # the error is set by the cancellation, not by the answer's size --
        # f32's grain on the two large terms arrives almost intact on their
        # 3e-7 m/s^2 difference -- 8e-10 m/s^2, wherever the components
        # happen to fall. A per-component rtol would be the wrong instrument:
        # the smallest component here is 25x below the vector's norm, so it
        # carries the same 8e-10 as a 5% relative error. The moon, 400x
        # closer, cancels far less and lands at 2e-12.
        ("third_body_sun", GM_SUN, sun_position, 3e-9),
        ("third_body_moon", GM_MOON, moon_position, 1e-11),
    ],
)
def test_third_body_terms_match_the_differential_closed_form(flag, gm, ephemeris, atol):
    """GM ((s - r)/|s - r|^3 - s/|s|^3), written out here rather than taken
    from astrojax: the direct pull on the satellite minus the pull on the
    Earth the ECI frame is attached to. This is what a bitwise inertness check
    cannot see -- a dropped indirect term, a swapped body, or a sign."""
    cfg = _free_flight_cfg()
    term, _ = _term(cfg, flag, True, 12_000.0, cfg.perturbations.chaser_ballistic)

    body = np.asarray(ephemeris(_probe_epoch(cfg)), np.float64)
    offset = body - np.asarray(PROBE_R, np.float64)
    expected = gm * (
        offset / np.linalg.norm(offset) ** 3 - body / np.linalg.norm(body) ** 3
    )
    np.testing.assert_allclose(term, expected, rtol=1e-4, atol=atol)
    # Without the indirect term the magnitude would be ~4 orders larger.
    assert np.linalg.norm(expected) < 1e-5


def _drag_closed_form(r, v, density, mass, ballistic):
    """-1/2 Cd (A/m) rho |v_rel| v_rel against an atmosphere co-rotating at
    OMEGA_EARTH, written out independently of astrojax."""
    v_rel = np.asarray(v, np.float64) - np.cross(
        [0.0, 0.0, OMEGA_EARTH], np.asarray(r, np.float64)
    )
    scale = -0.5 * ballistic.cd * (ballistic.area_m2 / mass) * density
    return scale * np.linalg.norm(v_rel) * v_rel, v_rel


def _ballistics(cfg, vehicle):
    if vehicle == "chief":
        return cfg.perturbations.chief_mass_kg, cfg.perturbations.chief_ballistic
    return cfg.physics.mass, cfg.perturbations.chaser_ballistic


@pytest.mark.parametrize("vehicle", ["chief", "chaser"])
def test_drag_opposes_the_co_rotating_relative_wind(vehicle):
    """The drag term against its closed form, and pointing exactly into the
    relative wind. This constrains the formula only -- which vehicle's
    ballistics reach it is `_eom`'s job and is tested separately below."""
    cfg = _free_flight_cfg()
    mass, ballistic = _ballistics(cfg, vehicle)
    term, _ = _term(cfg, "drag", True, mass, ballistic)

    density = float(density_harris_priester(PROBE_R, sun_position(_probe_epoch(cfg))))
    expected, v_rel = _drag_closed_form(PROBE_R, PROBE_V, density, mass, ballistic)
    np.testing.assert_allclose(term, expected, rtol=1e-5)
    # Drag removes energy: it must point exactly against the relative wind.
    cosine = np.dot(term, -v_rel) / (np.linalg.norm(term) * np.linalg.norm(v_rel))
    assert cosine == pytest.approx(1.0, abs=1e-9)


def test_drag_routes_each_vehicle_through_its_own_ballistics():
    """Both vehicles' drag out of one real step, each against the closed form
    with its OWN mass and ballistic config.

    The test above supplies those by hand and so cannot see a swap; `_eom` is
    where the routing actually happens. The signal is smaller than the
    configured numbers suggest -- the ISS has 30x the area but 35x the mass,
    so the two ballistic coefficients differ by only 14% (3.57e-3 vs 4.17e-3
    m^2/kg) -- but that is still seven times this test's tolerance.
    """
    cfg = _free_flight_cfg()
    coasting = NumericalDynamics(cfg)
    dragging = NumericalDynamics(_free_flight_cfg(perturbations={"drag": True}))
    start = coasting.reset(jax.random.PRNGKey(0))
    epoch = epoch_from_prefix(start[0:2])

    without = coasting.step(start, ZERO_ACTION)[0]
    with_drag = dragging.step(start, ZERO_ACTION)[0]

    coefficients = []
    for vehicle, position, velocity in (
        ("chief", slice(2, 5), slice(5, 8)),
        ("chaser", slice(8, 11), slice(11, 14)),
    ):
        mass, ballistic = _ballistics(cfg, vehicle)
        coefficients.append(ballistic.area_m2 / mass)
        measured = np.asarray(with_drag[velocity] - without[velocity], np.float64) / cfg.dt
        density = float(density_harris_priester(start[position], sun_position(epoch)))
        expected, _ = _drag_closed_form(
            start[position], start[velocity], density, mass, ballistic
        )
        # One step's velocity change is the RK4-averaged acceleration, not the
        # acceleration at the step start; over 385 m of along-track motion the
        # density moves that by well under a percent. atol covers the same
        # thing on the cross-track component, which is exactly zero at this
        # start (r along ECI x, so omega x r has no x term) and picks up 1e-11
        # from the position rotating within the step.
        np.testing.assert_allclose(
            measured, expected, rtol=2e-2, atol=1e-9, err_msg=vehicle
        )

    # The assertions above only catch a swap if the two coefficients are far
    # enough apart to clear the tolerance; they are, by 14% against 2%.
    assert abs(coefficients[0] / coefficients[1] - 1.0) > 5 * 2e-2


def test_relative_geometry_resolves_below_f32_eci_grain():
    """Why the ECI state is f64. A 1e-4 m displacement of the chaser -- 2e-4
    of one f32 ulp at 6.8e6 m -- has to survive a step and arrive intact in
    the relative view, where f32 would have quantized it away entirely."""
    dyn = NumericalDynamics(_free_flight_cfg())
    start = dyn.reset(jax.random.PRNGKey(0))
    nudge = 1e-4
    assert float(jnp.spacing(jnp.float32(6.8e6))) > 1000 * nudge

    before = relative_view(dyn.step(start, ZERO_ACTION)[0])
    after = relative_view(dyn.step(start.at[8].add(nudge), ZERO_ACTION)[0])
    moved = float(jnp.linalg.norm(after[0:3] - before[0:3]))
    assert moved == pytest.approx(nudge, rel=1e-4)


@pytest.mark.parametrize(
    "quaternion, thrust_axis_eci",
    [
        # Identity: body axes are ECI axes, so body +z pushes along ECI +z.
        ([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        # 90 deg about x maps body +z onto ECI -y. This is the case that
        # separates q_bi from its transpose: the identity case cannot.
        ([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0], [0.0, -1.0, 0.0]),
        # 90 deg about y maps body +z onto ECI +x.
        ([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0], [1.0, 0.0, 0.0]),
    ],
)
def test_control_force_acts_in_body_frame(quaternion, thrust_axis_eci):
    """A 1 N body +z force for one step: the chaser's ECI velocity gains
    dt/m along whichever ECI axis q_bi maps body +z onto, and nothing else.
    The chief, which the control never reaches, is bitwise unchanged."""
    cfg = _free_flight_cfg()
    dyn = NumericalDynamics(cfg)
    start = dyn.reset(jax.random.PRNGKey(0))
    start = start.at[14:18].set(jnp.asarray(quaternion, jnp.float64))
    start = start.at[18:21].set(jnp.zeros(3, jnp.float64))

    coasting = dyn.step(start, ZERO_ACTION)[0]
    pushed = dyn.step(start, jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], jnp.float64))[0]

    delta_v = np.asarray(pushed[11:14] - coasting[11:14])
    expected = np.asarray(thrust_axis_eci) * (cfg.dt / cfg.physics.mass)
    # rtol covers the gravity difference over the ~1e-7 m the thrust displaces
    # the chaser within the step (~1e-9 relative); atol covers the f32-pinned
    # quaternion rotation leaking ~1e-7 of the thrust onto the other two axes.
    np.testing.assert_allclose(delta_v, expected, rtol=1e-6, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(pushed[CHIEF]), np.asarray(coasting[CHIEF]))


def test_gravity_gradient_torque_closed_form():
    """Principal-axis body tilted by theta about x with the chaser on the ECI
    +z axis: closed form T_x = (3 mu / 2 r^3) (I_zz - I_yy) sin(2 theta).

    This is also what pins the quaternion convention handed to astrojax's
    `torque_gravity_gradient`: passing q_bi straight through flips the sign,
    and the conjugate is what agrees with the closed form.
    """
    cfg = _free_flight_cfg()
    dyn = NumericalDynamics(cfg)
    ixx, iyy, izz = cfg.physics.inertia_diag
    theta = 0.3
    radius = cfg.orbit.sma_m

    start = dyn.reset(jax.random.PRNGKey(0))
    start = start.at[8:11].set(jnp.array([0.0, 0.0, radius], jnp.float64))
    # At rest, so the radial direction barely moves within the step: an
    # orbital velocity here would tilt r_hat by 5.7e-5 rad over one dt and
    # perturb the answer by ~1e-4, right at the tolerance below.
    start = start.at[11:14].set(jnp.zeros(3, jnp.float64))
    start = start.at[14:18].set(
        jnp.array([np.cos(theta / 2), np.sin(theta / 2), 0.0, 0.0], jnp.float64)
    )
    start = start.at[18:21].set(jnp.zeros(3, jnp.float64))

    rate_change = np.asarray(dyn.step(start, ZERO_ACTION)[0][18:21]) / cfg.dt
    torque_x = 1.5 * GM_EARTH / radius**3 * (izz - iyy) * np.sin(2.0 * theta)
    np.testing.assert_allclose(rate_change[0], torque_x / ixx, rtol=1e-4)
    np.testing.assert_allclose(rate_change[1:], 0.0, atol=1e-9)


def test_state_is_float64_through_step():
    """f32 would put 0.5 m of grain on a 6.8e6 m ECI position -- five times the
    dock gate -- so the state is f64 and an f32 input is widened, not carried."""
    dyn = NumericalDynamics(_free_flight_cfg())
    start = dyn.reset(jax.random.PRNGKey(0))
    assert start.dtype == jnp.float64

    widened = dyn.step(start.astype(jnp.float32), jnp.zeros(6, jnp.float32))[0]
    assert widened.dtype == jnp.float64
    assert relative_view(widened).dtype == jnp.float64


def test_dock_gate_resolves_below_f32_eci_grain():
    """The 10 cm dock gate, applied to vehicles 6.8e6 m from the Earth's
    centre: 9 cm off the port docks and 11 cm does not.

    KNOWN FAILING. `relative_view` does not yet derive the chaser's
    world-frame attitude -- it reports the identity quaternion -- so the dock
    gate on attitude error sees a 90 deg miss against the port's pose and
    never opens, at either distance. The position resolution this test is
    really about is already there; the attitude channel is not.
    """
    cfg = NumericalConfig(physics={"collision_boxes_path": []})
    dyn = NumericalDynamics(cfg)
    target = jnp.asarray(dock_target(cfg), jnp.float64)
    start = dyn.reset(jax.random.PRNGKey(0))
    chief = start[CHIEF]

    def state_offset_by(distance: float) -> jnp.ndarray:
        view = jnp.concatenate(
            [
                target[0:3] + jnp.array([0.0, 0.0, distance], jnp.float64),
                jnp.zeros(3, jnp.float64),
                target[3:7],
                jnp.zeros(3, jnp.float64),
            ]
        )
        return jnp.concatenate([start[0:2], chief, chaser_state_from_view(chief, view)])

    assert bool(dyn.step(state_offset_by(0.09), ZERO_ACTION)[1].docked)
    assert not bool(dyn.step(state_offset_by(0.11), ZERO_ACTION)[1].docked)


def test_reset_zero_width_config_is_the_undispersed_start():
    """Every sampling range defaults to zero width, and that case has to
    collapse exactly: epoch0, the chief on its reference orbit, the chaser at
    the configured radius and at rest IN THE WORLD FRAME -- which in ECI means
    carrying the chief's velocity plus the frame's motion at that offset."""
    dyn = NumericalDynamics(_free_flight_cfg())
    start = dyn.reset(jax.random.PRNGKey(3))

    np.testing.assert_array_equal(
        np.asarray(start[0:2]), np.asarray(epoch_prefix(dyn.ref.epoch0))
    )
    np.testing.assert_array_equal(
        np.asarray(start[CHIEF]), np.asarray(dyn.ref.chief_state_eci(0.0).astype(jnp.float64))
    )

    view = relative_view(start)
    # 1e-4 m, not f64 grain: the world <-> ECI rotation is astrojax f32 and so
    # orthonormal only to ~1e-7, which is ~1e-5 m of round-trip error on a
    # 100 m standoff (see the dynamics module docstring).
    assert float(jnp.linalg.norm(view[0:3])) == pytest.approx(100.0, abs=1e-4)
    np.testing.assert_allclose(np.asarray(view[3:6]), 0.0, atol=1e-9)

    # At rest in the world frame is not at rest in ECI: the stored velocity
    # carries the chief's plus the world frame's own motion at the offset.
    chief_r, chief_v = np.asarray(start[2:5]), np.asarray(start[5:8])
    omega_frame = np.cross(chief_r, chief_v) / np.dot(chief_r, chief_r)
    offset_eci = np.asarray(start[8:11]) - chief_r
    np.testing.assert_allclose(
        np.asarray(start[11:14]) - chief_v, np.cross(omega_frame, offset_eci), atol=1e-9
    )


def test_reset_composes_inertial_attitude_and_rate_from_world_relative_ones():
    """`reset` samples a world-relative attitude and body rate but stores
    inertial ones, and `relative_view`'s attitude channels are a placeholder,
    so neither composition can be checked by a view round trip. Both are
    reconstructed here independently, from q_bi and the chief's state.
    """
    dyn = NumericalDynamics(_free_flight_cfg())
    start = dyn.reset(jax.random.PRNGKey(3))
    chief = start[CHIEF]
    world_from_chief = np.asarray(world_from_eci(chief), np.float64)
    omega_frame = np.asarray(frame_rate_eci(chief[0:3], chief[3:6]), np.float64)

    # q_bi is body -> ECI, so composing with world <- ECI gives body -> world.
    body_to_world = world_from_chief @ np.asarray(quat_to_rotmat(start[14:18]), np.float64)

    # The default attitude dispersion is zero width: the nose (body +z) points
    # exactly at the ISS. Measured as a chord, 2 asin(|a - b| / 2), not as
    # arccos of a dot product -- the quaternion arrives with |q| off unity by
    # ~1e-8 from the f32-pinned helpers, which arccos near 1 would turn into a
    # spurious 1e-4 rad of "misalignment".
    nose_world = body_to_world @ np.array([0.0, 0.0, 1.0])
    to_iss = -np.asarray(relative_view(start)[0:3], np.float64)
    to_iss /= np.linalg.norm(to_iss)
    half_chord = np.clip(np.linalg.norm(nose_world - to_iss) / 2.0, 0.0, 1.0)
    assert np.rad2deg(2.0 * np.arcsin(half_chord)) < 1e-3

    # The rate dispersion is zero width too, so the stored inertial body rate
    # is exactly the world frame's own rotation expressed in body axes.
    np.testing.assert_allclose(
        np.asarray(start[18:21]),
        body_to_world.T @ (world_from_chief @ omega_frame),
        rtol=1e-5,
        atol=1e-9,
    )
    # Close to the mean motion but deliberately not equal to it: h/r^2 is the
    # TRUE-anomaly rate, which at e = 5e-4 swings by 2e = 1e-3 either side of
    # n over a revolution.
    assert float(jnp.linalg.norm(start[18:21])) == pytest.approx(
        dyn.ref.mean_motion, rel=3e-3
    )


def test_reset_disperses_the_relative_geometry():
    """Wide ranges, 200 vmapped keys: the sampled dispersions come back out of
    the relative view, so the world -> ECI placement inverts the view rather
    than merely producing plausible numbers."""
    dyn = NumericalDynamics(
        _free_flight_cfg(
            orbit={
                "epoch_offset_range_s": (0.0, 5400.0),
                "start_radius_range_m": (80.0, 120.0),
                "start_speed_max_m_s": 0.1,
            }
        )
    )
    states = jax.vmap(dyn.reset)(jax.random.split(jax.random.PRNGKey(0), 200))
    assert states.dtype == jnp.float64
    views = jax.vmap(relative_view)(states)

    radii = np.asarray(jnp.linalg.norm(views[:, 0:3], axis=1))
    assert radii.min() >= 80.0 - 1e-6 and radii.max() <= 120.0 + 1e-6
    assert radii.std() > 5.0

    # The ball sampler draws at f32, so a vector right at the cap can land a
    # few f32 ulps outside it.
    speeds = np.asarray(jnp.linalg.norm(views[:, 3:6], axis=1))
    assert speeds.max() <= 0.1 * (1 + 1e-5)
    assert speeds.max() > 0.05

    offsets = np.asarray(seconds_between(states[:, 0:2], epoch_prefix(dyn.ref.epoch0)))
    assert offsets.min() >= 0.0 and offsets.max() <= 5400.0
    assert offsets.std() > 1000.0
