"""Numerically propagated chief and chaser in ECI, plus chaser attitude.

State (21D)

  0..1   epoch prefix [jd_day, sec_of_day] (`envs/common/epoch_state.py`)
  2..7   chief ECI position [m] and velocity [m/s]
  8..13  chaser ECI position [m] and velocity [m/s]
  14..17 quaternion q_bi = [w,x,y,z] (chaser body -> ECI)
  18..20 chaser angular velocity, body frame, INERTIAL rates [rad/s]

Control (6D)
  0..2   body-frame force [N]
  3..5   body-frame torque [N*m]

Where this differs from `iss_hcw`: there the chief is an analytic Keplerian
ephemeris and only the chaser's motion RELATIVE to it is integrated, through
the linearized Clohessy-Wiltshire equations. Here both vehicles are ordinary
ECI state vectors integrated through the same force model, and the relative
geometry the task layer consumes is derived from the difference. That is what
buys the perturbations: J2..J6 bend the chief's orbit as much as the chaser's,
and only the residual between the two shows up in the relative view, which is
a distinction the CW linearization cannot express at all.

Both vehicles carry the full force model, gated by `PerturbationsConfig`.
Every gate is a static Python branch on the config -- `step` runs under
jit/vmap but the config is not a traced array -- so a term that is off costs
nothing and contributes no rounding at all, not merely a zero.

The dominant gravity term always comes from `envs/common/zonal_gravity.py`,
which is f64 and includes the point mass. astrojax's own point-mass helper is
never used for it: that one casts to astrojax's module-wide dtype (f32 by
default, independent of this package's x64 flag), which would cap the term
that carries essentially all of the acceleration at ~5e-8 relative.

The perturbations do go through astrojax at f32, and that is a different
budget entirely, because they are all ~1e-6 m/s^2 against gravity's ~8.6.
Measured against f64 references at ISS radii, the f32 cost is 5e-14 m/s^2 for
drag, 2e-12 for the moon, and 8e-10 for the sun. The sun is three orders
worse than f32's 6e-8 relative grain would suggest, and not because the
ephemeris is coarse: a third-body acceleration is the DIFFERENCE of two
almost equal vectors (the pull on the satellite and the pull on the Earth),
and at 1.5e11 m that difference cancels four decimal digits, so f32's grain
on the two 5.9e-3 m/s^2 terms lands almost intact on their 3e-7 m/s^2
difference. It is still only 8e-10 m/s^2 -- ~1 cm of absolute position per
orbit, and largely common-mode between two vehicles 100 m apart, so far less
than that in the relative view the task scores.

Model error dwarfs all of it regardless: astrojax's sun and moon are
Montenbruck & Gill low-precision ephemerides (~0.1 deg, i.e. ~2e-3 relative
on the direction) and Harris-Priester is a static-tables density model good
to maybe 10-20%.

Per-step structure, following `iss_hcw`:

* The epoch prefix advances OUTSIDE the integrator, since a day rollover is
  not smooth. The 19D dynamic part goes through the shared RK4.
* Epoch-dependent perturbations -- sun, moon, and the drag model's diurnal
  bulge -- are evaluated once at the step-start epoch and held constant
  across the four RK4 stages. At these step sizes that is not even an
  approximation. astrojax's ephemerides evaluate at f32, where one dt of
  0.05 s (or 0.5 s) does not advance their Julian-century argument by a
  representable amount: the sun and moon positions before and after a step
  are bit-identical, so re-deriving them per stage would return the same
  numbers four times.

  That exactness has a bound, and dt is a config field, so the bound is
  worth stating rather than leaving to be rediscovered: measured against the
  shipped ephemerides, the positions either side of one step stay
  bit-identical up to dt = 30 s and have separated by dt = 45 s. Past that
  the hold stops being free -- the stages ask for the epoch-dependent terms
  at three intermediate times and get the step-start value instead, so those
  terms integrate to first order while everything else stays 4th, and
  nothing anywhere reports it. Nothing enforces the bound either; the
  shipped configs run at 0.05 s and 0.5 s, well inside it.
* Drag reads the ECI position where Harris-Priester asks for a true-of-date
  one, and rotates ECI->ECEF with the identity. Neither is the sidereal
  rotation error it looks like. The Earth's rotation ABOUT the polar axis is
  already carried by the model's own omega x r co-rotation term rather than
  by that matrix, so what the identity drops is only the bias/precession/
  nutation TILT between the two frames -- 6.5e-3 rad (0.37 deg) by 2026,
  essentially all of it precession since J2000. The diurnal bulge does not
  feel that at all: the model reads the sun's direction against the
  satellite's, and both are passed in the SAME frame, so the angle between
  them is exact. The tilt reaches the answer only through geodetic altitude,
  which over a lat/lon sweep at ISS altitude moves the density by at most
  0.449% -- the tilt taken in its worst direction, a bound rather than a
  typical value, since a tilt about the polar axis alone changes only
  longitude and moves the density by 0.002%. Against Harris-Priester's own
  10-20%, either way.

Attitude is INERTIAL here, not chief-relative: `q_bi` maps body to ECI and
the stored rates are rates with respect to ECI, so the kinematics are the
plain quaternion ones with no transport term. `iss_hcw` has to treat its
LVLH/world frame as inertial and eat the orbit's own 1.1e-3 rad/s as an
unmodelled bias; this env has no such approximation to make, and the frame
rate appears only where the relative view and its inverse convert between the
two conventions. The gravity-gradient torque likewise measures the chaser
from its own true ECI radius rather than from an offset against the chief's.

Dtype: the state is f64 for the reason `iss_hcw`'s is -- an f32
seconds-of-day biases every add in the same direction and loses ~290 s per
orbit -- and here also because the chaser's position is now an absolute ECI
one at ~6.8e6 m, where f32's 0.5 m grain would swamp the 0.1 m dock gate
outright. Three places still narrow to f32 and are stated rather than
corrected:

* `ReferenceOrbit.chief_state_eci` at reset (astrojax f32, ~4 m of position
  and ~5e-4 m/s of velocity, per `envs/common/orbit.py`). Both vehicles are
  built from the SAME chief state -- the chaser as chief plus a relative
  offset -- so this displaces the pair along the orbit together and leaves
  the relative geometry the task scores untouched.
* The world<->ECI rotation, also astrojax f32. Orthonormal to ~1e-7, which is
  ~1e-5 m of round-trip error on a 100 m standoff and ~1e-8 m at the dock.
* The quaternion inside the integrator, through the f32-pinned
  `core.quaternion` helpers, exactly as in `iss_hcw`: bounded per-step
  quantization at the 1e-8 level rather than a directional bias, because it is
  re-derived each step and not summed into. `relative_view` and
  `chaser_state_from_view` are the exception and compose their quaternions at
  the state's own width (`_quat_compose`), because those two have to be exact
  inverses and an f32 product would floor the round trip at ~1.4e-7 rad.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from astrojax.attitude_dynamics.gravity_gradient import torque_gravity_gradient
from astrojax.epoch import Epoch
from astrojax.orbit_dynamics.density import density_harris_priester
from astrojax.orbit_dynamics.drag import accel_drag
from astrojax.orbit_dynamics.ephemerides import sun_position
from astrojax.orbit_dynamics.third_body import accel_third_body_moon, accel_third_body_sun

from ...core.integrator import Integrator
from ...core.quaternion import (
    quat_conjugate,
    quat_derivative_from_omega_body,
    quat_from_body_z_to,
    quat_from_rotmat,
    quat_multiply,
    quat_normalize,
    quat_to_rotmat,
    rotate_body_to_world,
)
from ..common.config import dock_target
from ..common.epoch_state import advance_epoch_state, epoch_from_prefix, epoch_prefix
from ..common.events import EventChecker, Events
from ..common.orbit import ReferenceOrbit, world_from_eci
from ..common.sampling import sample_small_rotation, sample_vector_in_ball
from ..common.zonal_gravity import accel_zonal_gravity
from .config import NUM_LAYOUT, BallisticConfig, NumericalConfig, PerturbationsConfig

STATE_LABELS: tuple[str, ...] = NUM_LAYOUT.labels

# Fills `accel_drag`'s frame-rotation slot, which that function documents as
# ECI -> ECEF but uses as ECI -> true-of-date: it applies the Earth's rotation
# separately, through its own omega x r term. The identity is therefore not
# the sidereal-rotation blunder its position in the signature suggests -- it
# drops only the precession/nutation tilt. See the module docstring.
_DRAG_FRAME_ROTATION = jnp.eye(3, dtype=jnp.float64)


def _quat_conjugate_wide(q: jnp.ndarray) -> jnp.ndarray:
    """Conjugate of [w,x,y,z] at the input's own dtype. See `_quat_compose`."""
    return jnp.concatenate([q[0:1], -q[1:4]], axis=0)


def _quat_compose(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Unit-normalized Hamilton product q1 (x) q2, at the inputs' own dtype.

    `core.quaternion`'s `quat_multiply`/`quat_normalize` route through
    astrojax's `Quaternion`, which casts to astrojax's module-wide dtype --
    f32, independent of this package's x64 flag. That is the right trade
    inside the integrator, where the quaternion is re-derived from the
    kinematics every step and never summed into, but it is the wrong one for
    `relative_view` and `chaser_state_from_view`: those two are required to be
    exact inverses, and an f32 product caps a round trip at ~1.4e-7 rad
    against the ~3e-16 the f64 state itself carries. Composing here keeps the
    view at the state's own width.

    Deliberately private and local rather than a second public helper in
    `core.quaternion`: widening the ones already there would move `iss_hcw`'s
    integrator off the numerics its golden fixtures were recorded against,
    and adding parallel wide variants beside them would leave the shared
    module offering two products with no way for a caller to tell which one
    its dtype needs. Consolidating is a decision about all three envs, not
    about this view.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    q = jnp.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )
    return q / jnp.linalg.norm(q)


def accel_perturbed(
    r_eci: jnp.ndarray,
    v_eci: jnp.ndarray,
    epoch: Epoch,
    mass: float,
    ballistic: BallisticConfig,
    pert: PerturbationsConfig,
) -> jnp.ndarray:
    """Total inertial acceleration on one vehicle [m/s^2], (3,).

    Shared by chief and chaser -- they differ only in mass and ballistic
    properties, never in which terms apply. `mass`, `ballistic` and `pert` are
    static Python values; `r_eci`, `v_eci` and `epoch` may be traced.
    """
    accel = accel_zonal_gravity(r_eci, pert.zonal_max_degree)
    if pert.third_body_sun:
        accel = accel + accel_third_body_sun(epoch, r_eci)
    if pert.third_body_moon:
        accel = accel + accel_third_body_moon(epoch, r_eci)
    if pert.drag:
        density = density_harris_priester(r_eci, sun_position(epoch))
        accel = accel + accel_drag(
            jnp.concatenate([r_eci, v_eci]),
            density,
            mass,
            ballistic.area_m2,
            ballistic.cd,
            _DRAG_FRAME_ROTATION,
        )
    return accel


def frame_rate_eci(r_chief: jnp.ndarray, v_chief: jnp.ndarray) -> jnp.ndarray:
    """Angular velocity of the world (LVLH) frame with respect to ECI, in ECI
    axes [rad/s] -- the in-plane term only.

    The world axes are a fixed relabelling of RTN (`RTN_FROM_WORLD`), built
    from the chief's current r and v. Decomposing that triad's true angular
    velocity leaves exactly two non-zero components,

        omega = (h / r^2) N_hat  +  (r a_N / h) R_hat

    with a_N the chief's out-of-plane acceleration: the radial axis sweeping
    along the orbit, and the orbit normal itself tilting. This returns the
    first term, `r x v / |r|^2`.

    The second vanishes identically whenever the chief's acceleration is
    central, which covers two-body motion and, notably, the Keplerian chief
    `reset` builds against -- so the reset conversion is exact, not
    approximate. It is non-zero only while a non-central perturbation is
    switched on, and then only in `relative_view`. Measured against a finite
    difference of the actual rotation over an eighth of an orbit at zonal
    degree 6, the omitted term peaks at 1.1e-6 rad/s: 1.1e-4 m/s of relative
    velocity at a 100 m standoff, 2.8e-5 m/s at the dock, 0.006% of the
    0.5 m/s dock velocity gate. Restoring it needs the chief's acceleration,
    which this function cannot obtain without being handed the force model.
    """
    return jnp.cross(r_chief, v_chief) / jnp.dot(r_chief, r_chief)


def relative_view(state: jnp.ndarray) -> jnp.ndarray:
    """The canonical 13D world-frame relative view `[pos, vel, q_bw, omega]`
    from a (21,) numerical state -- what the task layer (events, reward,
    policies) consumes in place of `NUM_LAYOUT.slice_view`, whose slices hold
    the chaser's ABSOLUTE ECI state.

    All four channels are measured against the ROTATING world frame, so the
    view matches what `iss_hcw` carries in-state directly: position and
    velocity are the chaser from the chief in world axes with the frame's own
    motion at that offset removed, `q_bw` is body -> world rather than the
    stored body -> ECI, and the rates are the body rates the world frame's own
    rotation has been subtracted from. `chaser_state_from_view` inverts it to
    f64 -- exactly for the attitude and rate, and to the f32 world<->ECI
    rotation's ~1e-7 non-orthonormality for position and velocity, which pass
    through that rotation as a matrix whose transpose is only an approximate
    inverse where a unit quaternion's conjugate is an exact one.

    `q_bw` is returned on the w >= 0 hemisphere, matching `quat_from_rotmat`.
    That resolves the q/-q double cover the way a pure state -> view function
    has to -- there is no previous view to stay continuous against, unlike
    `step`, which flips against the last one. It is also the one place the
    inverse is an inverse of the ROTATION rather than of the components: a
    state whose q_bi puts q_bw on the w < 0 hemisphere comes back through
    `chaser_state_from_view` negated, which is the same attitude and the same
    dynamics but not the same four numbers. The dock gate is indifferent
    (`EventChecker.docked` takes |w| of the error quaternion), but a consumer
    reading the raw components will see a sign flip where `iss_hcw`, carrying
    q_bw in-state, would not.

    The rates inherit `frame_rate_eci`'s in-plane-only frame rate, which is
    exact whenever the chief's acceleration is central -- two-body motion, and
    every state `reset` builds. With a non-central perturbation on, the
    omitted out-of-plane term peaks at 1.1e-6 rad/s at zonal degree 6, which
    reaches the velocity channel as 1.1e-4 m/s at a 100 m standoff and 2.8e-5
    m/s at the dock, 0.006% of the 0.5 m/s dock velocity gate. Restoring it
    needs the chief's acceleration and so the force model; the closed form is
    in `frame_rate_eci`'s docstring, and keeping it out of here is what makes
    this function a pure state -> view map.
    """
    r_chief, v_chief = state[2:5], state[5:8]
    r_chaser, v_chaser = state[8:11], state[11:14]

    rotation = jnp.asarray(world_from_eci(state[2:8]), jnp.float64)
    omega_frame = frame_rate_eci(r_chief, v_chief)
    offset_eci = r_chaser - r_chief

    rel_pos = rotation @ offset_eci
    rel_vel = rotation @ (v_chaser - v_chief - jnp.cross(omega_frame, offset_eci))

    q_wi = quat_from_rotmat(rotation.T)
    q_bw = _quat_compose(_quat_conjugate_wide(q_wi), state[14:18])
    q_bw = jnp.where(q_bw[0] < 0.0, -q_bw, q_bw)
    omega_rel = state[18:21] - quat_to_rotmat(q_bw).T @ (rotation @ omega_frame)
    return jnp.concatenate([rel_pos, rel_vel, q_bw, omega_rel], axis=0)


def chaser_state_from_view(chief_eci: jnp.ndarray, view: jnp.ndarray) -> jnp.ndarray:
    """Inverse of `relative_view`'s mapping: the chaser's 13D ABSOLUTE state
    `[r_eci, v_eci, q_bi, omega_b]` that presents as `view` against `chief_eci`.

    Applied to a view that came out of `relative_view`, it returns the same
    ROTATION, which on the w < 0 hemisphere means the negated quaternion --
    see that function on why the view canonicalizes and this does not.

    The three conversions each undo one of the view's: the world-frame offset
    rotates back into ECI and adds to the chief's position; the world-frame
    velocity adds back the frame's own motion at the chaser's offset; and the
    world-relative attitude and rates compose with the world frame's own
    orientation and rotation to give inertial ones.
    """
    r_chief, v_chief = chief_eci[0:3], chief_eci[3:6]
    rotation = jnp.asarray(world_from_eci(chief_eci), jnp.float64)
    omega_frame = frame_rate_eci(r_chief, v_chief)

    offset_eci = rotation.T @ view[0:3]
    r_chaser = r_chief + offset_eci
    v_chaser = v_chief + rotation.T @ view[3:6] + jnp.cross(omega_frame, offset_eci)

    q_bw = view[6:10]
    q_wi = quat_from_rotmat(rotation.T)
    q_bi = _quat_compose(q_wi, q_bw)
    omega_b = view[10:13] + quat_to_rotmat(q_bw).T @ (rotation @ omega_frame)
    return jnp.concatenate([r_chaser, v_chaser, q_bi, omega_b], axis=0)


class NumericalDynamics:
    def __init__(self, cfg: NumericalConfig):
        if cfg.physics.linear_damping != 0.0:
            raise ValueError(
                "iss-numerical requires physics.linear_damping == 0, got "
                f"{cfg.physics.linear_damping}. Linear damping was the kinematic "
                "env's artificial stabilizer; here the chaser flies a real ECI "
                "orbit, where there is no medium to damp its inertial velocity "
                "against. Applying it would decay the orbital velocity itself "
                "(~7.7 km/s) and drop the vehicle out of orbit within seconds. "
                "Atmospheric drag is the physical version of this and is "
                "modelled separately: set perturbations.drag."
            )

        self.cfg = cfg
        self.state_dim = 21
        self.action_dim = 6

        self.ref = ReferenceOrbit(cfg.orbit)
        self._events = EventChecker(cfg)
        self._integrator = Integrator(cfg.dt)
        self._epoch0 = epoch_prefix(self.ref.epoch0)

        self._pert = cfg.perturbations
        self._chief_mass = float(cfg.perturbations.chief_mass_kg)
        self._chief_ballistic = cfg.perturbations.chief_ballistic
        self._chaser_ballistic = cfg.perturbations.chaser_ballistic

        inertia = jnp.asarray(cfg.physics.inertia_diag, dtype=jnp.float64)
        self._mass = float(cfg.physics.mass)
        self._inertia_diag = jnp.maximum(inertia, 1e-6)
        self._inertia_mat = jnp.diag(self._inertia_diag)
        self._inv_inertia_diag = 1.0 / self._inertia_diag
        self._angular_damping = jnp.asarray(cfg.physics.angular_damping, jnp.float64)

        # Default target when `step` is not given one, as in `HCWDynamics`.
        self._dock_target = jnp.asarray(dock_target(cfg), jnp.float64)

        # Reset dispersion bounds, each in the dtype its draw uses (see
        # `reset`). Ranges are validated at config load, so lo <= hi holds.
        self._start_radius_low, self._start_radius_high = (
            jnp.asarray(v, jnp.float64) for v in cfg.orbit.start_radius_range_m
        )
        self._epoch_offset_low, self._epoch_offset_high = (
            jnp.asarray(v, jnp.float64) for v in cfg.orbit.epoch_offset_range_s
        )
        self._start_speed_max = jnp.asarray(cfg.orbit.start_speed_max_m_s, jnp.float32)
        self._start_rate_max = jnp.asarray(cfg.orbit.start_rate_max_rad_s, jnp.float32)
        self._start_attitude_error_max_rad = jnp.asarray(
            jnp.deg2rad(cfg.orbit.start_attitude_error_max_deg), jnp.float32
        )

    def _eom(self, x: jnp.ndarray, args: tuple) -> jnp.ndarray:
        """19D dynamic derivative (the state without its epoch prefix).
        `args = (u, epoch)` is held across the RK4 stages -- `Integrator.rk4`
        forwards its third argument to `f` untouched, so a tuple rides through
        where an action array would."""
        u, epoch = args
        r_chief, v_chief = x[0:3], x[3:6]
        r_chaser, v_chaser = x[6:9], x[9:12]
        q_bi = quat_normalize(x[12:16])
        omega_b = x[16:19]

        accel_chief = accel_perturbed(
            r_chief, v_chief, epoch, self._chief_mass, self._chief_ballistic, self._pert
        )
        accel_chaser = accel_perturbed(
            r_chaser, v_chaser, epoch, self._mass, self._chaser_ballistic, self._pert
        )
        accel_chaser = accel_chaser + rotate_body_to_world(q_bi, u[0:3]) / self._mass

        # astrojax documents `q` as body -> inertial, but builds its matrix
        # with `quaternion_to_rotation_matrix`, which returns reference ->
        # body -- the transpose of what this package calls body -> world (see
        # `core.quaternion.quat_to_rotmat`, which transposes it back). Passing
        # q_bi straight through flips the torque's sign against the closed
        # form in test_gravity_gradient_torque_closed_form; the conjugate is
        # what makes the two agree, exactly as in `iss_hcw`.
        torque_gg = torque_gravity_gradient(quat_conjugate(q_bi), r_chaser, self._inertia_mat)
        coriolis = jnp.cross(omega_b, self._inertia_diag * omega_b)
        omega_dot = self._inv_inertia_diag * (
            u[3:6] + torque_gg - coriolis - self._angular_damping * omega_b
        )

        q_dot = quat_derivative_from_omega_body(q_bi, omega_b)
        return jnp.concatenate(
            [v_chief, accel_chief, v_chaser, accel_chaser, q_dot, omega_dot], axis=0
        )

    def step(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        dock_pose: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, Events]:
        """Advance one step. `dock_pose` is a (7,) [position, quaternion] row;
        omitted, the pose in `DockConfig` is used."""
        s = state.astype(jnp.float64)
        u = action.astype(jnp.float64)
        target = self._dock_target if dock_pose is None else jnp.asarray(dock_pose, jnp.float64)

        prefix, dyn = s[0:2], s[2:21]
        q_prev = quat_normalize(dyn[12:16])
        dyn = dyn.at[12:16].set(q_prev)

        epoch = epoch_from_prefix(prefix)
        # One epoch rides through all four stages: the sun, moon and diurnal
        # bulge are held at the step-start value. Exact only while one dt does
        # not move astrojax's f32 ephemerides at all -- bit-identical up to
        # dt = 30 s, separated by 45 s, past which those terms silently drop
        # to first order. See the module docstring.
        dyn_next = self._integrator.rk4(self._eom, dyn, (u, epoch))

        q_next = quat_normalize(dyn_next[12:16])
        # Keep the quaternion on the same hemisphere as the previous step so
        # the trajectory doesn't show a spurious sign flip (q and -q are the
        # same rotation), as the other envs do.
        q_next = jnp.where(jnp.dot(q_next, q_prev) < 0.0, -q_next, q_next)
        dyn_next = dyn_next.at[12:16].set(q_next)

        prefix_next = advance_epoch_state(prefix, self.cfg.dt)
        state_next = jnp.concatenate([prefix_next, dyn_next], axis=0)

        # Events read the derived relative view, never the raw slices: those
        # hold absolute ECI quantities, and a dock check against a 24 m
        # world-frame port would be meaningless applied to a 6.8e6 m radius.
        events = self._events.events(
            relative_view(jnp.concatenate([prefix, dyn], axis=0)),
            relative_view(state_next),
            target,
        )
        return state_next, events

    def reset(self, key: jax.Array) -> jnp.ndarray:
        """Sample an episode start state. Scalar `key`; vmap for batches.

        The dispersions are the same ones `HCWDynamics.reset` draws, in the
        same order off the same split, so the two envs start from statistically
        identical RELATIVE geometry given the same key: standoff distance from
        `start_radius_range_m` (uniform direction, uniform radius), start time
        along the chief's orbit from `epoch_offset_range_s`, initial motion
        from `start_speed_max_m_s` and `start_rate_max_rad_s`, and nose-to-ISS
        pointing missed by up to `start_attitude_error_max_deg`. Every default
        is zero-width, and that case collapses exactly: the configured radius,
        at rest in the world frame, nose on the ISS, at epoch0.

        What differs is that those dispersions describe a relative view, while
        the state stores absolute ECI vectors -- so the sampled view is run
        back through `chaser_state_from_view` to place the chaser. "At rest"
        therefore means at rest in the rotating world frame: the stored ECI
        velocity is the chief's plus the frame's motion at the offset, and the
        stored body rate is the world frame's own 1.1e-3 rad/s rotation.

        Dtypes follow the module docstring, and the reasoning is `iss_hcw`'s:
        the radius and epoch offset draw f64 so a zero-width range lands
        exactly on the configured value and so the prefix (whose carriage is
        why this state is f64 at all) starts clean, while the dispersion draws
        stay f32 because a start velocity or body rate is re-derived against
        the dynamics every step and gains nothing from the extra width.
        """
        key_pos, key_epoch, key_vel, key_att, key_rate = jax.random.split(key, 5)
        key_direction, key_radius = jax.random.split(key_pos)

        raw = jax.random.normal(key_direction, (3,), dtype=jnp.float64)
        direction = raw / jnp.maximum(jnp.linalg.norm(raw), 1e-8)
        radius = jax.random.uniform(
            key_radius,
            (),
            dtype=jnp.float64,
            minval=self._start_radius_low,
            maxval=self._start_radius_high,
        )

        offset = jax.random.uniform(
            key_epoch,
            (),
            dtype=jnp.float64,
            minval=self._epoch_offset_low,
            maxval=self._epoch_offset_high,
        )
        prefix = advance_epoch_state(self._epoch0, offset)
        chief = self.ref.chief_state_eci(offset).astype(jnp.float64)

        q_bw = quat_normalize(
            quat_multiply(
                quat_from_body_z_to(-direction),
                sample_small_rotation(key_att, self._start_attitude_error_max_rad),
            )
        )
        view = jnp.concatenate(
            [
                direction * radius,
                sample_vector_in_ball(key_vel, self._start_speed_max).astype(jnp.float64),
                q_bw.astype(jnp.float64),
                sample_vector_in_ball(key_rate, self._start_rate_max).astype(jnp.float64),
            ],
            axis=0,
        )
        return jnp.concatenate([prefix, chief, chaser_state_from_view(chief, view)], axis=0)
