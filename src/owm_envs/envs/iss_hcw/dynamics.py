"""HCW relative dynamics + gravity-gradient torque about a Keplerian chief.

State (15D)

  0..1   epoch prefix [jd_day, sec_of_day] (`envs/common/epoch_state.py`)
  2..4   relative position to the chief, world frame [m]
  5..7   relative velocity, world frame [m/s]
  8..11  quaternion q_bw = [w,x,y,z] (body -> world)
  12..14 angular velocity, body frame [rad/s]

Control (6D)
  0..2   body-frame force [N]
  3..5   body-frame torque [N*m]

The state is FLOAT64 in sim, unlike `iss`'s f32 one. The epoch prefix is what
forces it: carrying seconds-of-day in f32 biases every add in the same
direction and loses ~290 s per orbit at dt = 0.05 (measured; see
`envs/common/epoch_state.py`). The prefix cannot be widened on its own without
splitting the state pytree, so the whole vector is f64 and narrows to f32 only
where observations and dataset records leave the env.

That buys exact accumulation, not exact evaluation: astrojax pins its own
float dtype to f32 (`astrojax.config`), so `hcw_derivative`,
`torque_gravity_gradient`, and the `core.quaternion` helpers all evaluate at
f32 and hand back f32 values which the f64 accumulators then absorb. That is
tolerable precisely because those quantities are re-derived from the state
each step rather than summed into it, so the ~1e-7 relative perturbation
stays a perturbation instead of biasing a running total the way the epoch's
did. Measured against the same equations of motion written entirely in f64,
over a full 7200-step episode with a tumbling, drifting chaser: 3.4e-7 m of
position, 2.8e-6 rad of attitude, 2.0e-9 rad/s of body rate -- five orders
below the tightest dock gate (0.1 m, 5 deg, 0.0087 rad/s).

Per-step structure: the epoch prefix advances OUTSIDE the integrator, since a
day rollover is not smooth; translation and attitude integrate with the shared
RK4 over the 13D view. The chief's ECI state is evaluated once per step at the
step-start epoch and held constant through the RK4 stages -- over one dt the
chief moves 1e-5 of a radian, and re-deriving it per stage would cost four
Kepler solves for nothing.

Modelling approximations, both stated rather than corrected:

* Attitude treats the LVLH/world frame as inertial, so the body rates carry no
  orbital-rate bias term. The neglected term is the orbit's own 1.1e-3 rad/s.
* The gravity-gradient torque measures the chaser from the Earth's centre as
  `r_world = rel_pos + [0, 0, |r_chief|]` -- exact in direction only to the
  extent that world +z stays the chief's radial direction, which it does by
  construction (`RTN_FROM_WORLD`), while the along-track and cross-track
  offsets tilt it by up to 1e-4 rad over the domain.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from astrojax.attitude_dynamics.gravity_gradient import torque_gravity_gradient
from astrojax.relative_motion.hcw_dynamics import hcw_derivative

from ...core.integrator import Integrator
from ...core.quaternion import (
    quat_conjugate,
    quat_derivative_from_omega_body,
    quat_from_body_z_to,
    quat_normalize,
    rotate_body_to_world,
)
from ..common.config import dock_target
from ..common.epoch_state import advance_epoch_state, epoch_prefix, seconds_between
from ..common.events import EventChecker, Events
from ..common.orbit import RTN_FROM_WORLD, ReferenceOrbit
from .config import HCW_LAYOUT, HCWConfig

STATE_LABELS: tuple[str, ...] = HCW_LAYOUT.labels

_WORLD_RADIAL = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64)


class HCWDynamics:
    def __init__(self, cfg: HCWConfig):
        self.cfg = cfg
        self.state_dim = 15
        self.action_dim = 6

        self.ref = ReferenceOrbit(cfg.orbit)
        self._events = EventChecker(cfg)
        self._integrator = Integrator(cfg.dt)
        self._epoch0 = epoch_prefix(self.ref.epoch0)

        inertia = jnp.asarray(cfg.physics.inertia_diag, dtype=jnp.float64)
        self._mass = jnp.asarray(cfg.physics.mass, jnp.float64)
        self._inertia_diag = jnp.maximum(inertia, 1e-6)
        self._inertia_mat = jnp.diag(self._inertia_diag)
        self._inv_inertia_diag = 1.0 / self._inertia_diag
        self._linear_damping = jnp.asarray(cfg.physics.linear_damping, jnp.float64)
        self._angular_damping = jnp.asarray(cfg.physics.angular_damping, jnp.float64)
        self._n = jnp.asarray(self.ref.mean_motion, jnp.float64)
        self._rtn_from_world = jnp.asarray(RTN_FROM_WORLD, jnp.float64)

        # Default target when `step` is not given one, as in `ISSDynamics`.
        self._dock_target = jnp.asarray(dock_target(cfg), jnp.float64)

    def _eom(self, x: jnp.ndarray, args: tuple) -> jnp.ndarray:
        """13D view derivative. `args = (u, r_chief_mag)` is held across the
        RK4 stages -- `Integrator.rk4` forwards its third argument to `f`
        untouched, so a tuple rides through where an action array would."""
        u, r_chief_mag = args
        pos_w, vel_w = x[0:3], x[3:6]
        q_bw = quat_normalize(x[6:10])
        omega_b = x[10:13]

        rel_rtn = jnp.concatenate([self._rtn_from_world @ pos_w, self._rtn_from_world @ vel_w])
        acc_rtn = hcw_derivative(rel_rtn, self._n)[3:6]
        acc_world = self._rtn_from_world.T @ acc_rtn

        force_w = rotate_body_to_world(q_bw, u[0:3])
        vel_dot = acc_world + force_w / self._mass - self._linear_damping * vel_w

        r_world = pos_w + _WORLD_RADIAL * r_chief_mag
        # astrojax documents `q` as body -> inertial, but builds its matrix
        # with `quaternion_to_rotation_matrix`, which returns reference ->
        # body -- the transpose of what this package calls body -> world (see
        # `core.quaternion.quat_to_rotmat`, which transposes it back). Passing
        # q_bw straight through flips the torque's sign against the closed
        # form in test_gravity_gradient_torque_closed_form; the conjugate is
        # what makes the two agree.
        torque_gg = torque_gravity_gradient(quat_conjugate(q_bw), r_world, self._inertia_mat)
        coriolis = jnp.cross(omega_b, self._inertia_diag * omega_b)
        omega_dot = self._inv_inertia_diag * (
            u[3:6] + torque_gg - coriolis - self._angular_damping * omega_b
        )

        q_dot = quat_derivative_from_omega_body(q_bw, omega_b)
        return jnp.concatenate([vel_w, vel_dot, q_dot, omega_dot], axis=0)

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

        prefix, view = s[0:2], s[2:15]
        q_prev = quat_normalize(view[6:10])
        view = view.at[6:10].set(q_prev)

        # `chief_state_eci` returns astrojax's f32, so `r_chief_mag` carries
        # the ~4 m reconstruction error documented in `envs/common/orbit.py`.
        # It enters only the gravity-gradient magnitude (3 mu / r^3), where
        # 6e-7 relative is far below the model error of treating LVLH as
        # inertial.
        t = seconds_between(prefix, self._epoch0)
        chief = self.ref.chief_state_eci(t)
        r_chief_mag = jnp.linalg.norm(chief[0:3]).astype(jnp.float64)

        view_next = self._integrator.rk4(self._eom, view, (u, r_chief_mag))

        q_next = quat_normalize(view_next[6:10])
        # Keep the quaternion on the same hemisphere as the previous step so
        # the trajectory doesn't show a spurious sign flip (q and -q are the
        # same rotation), as `ISSDynamics` does.
        q_next = jnp.where(jnp.dot(q_next, q_prev) < 0.0, -q_next, q_next)
        view_next = view_next.at[6:10].set(q_next)

        events = self._events.events(view, view_next, target)
        prefix_next = advance_epoch_state(prefix, self.cfg.dt)
        return jnp.concatenate([prefix_next, view_next], axis=0), events

    def reset(self, key: jax.Array) -> jnp.ndarray:
        """Uniform direction at the low end of the start radius range, at
        rest, nose at the ISS, at the configured epoch. Task 3 replaces this
        with the per-episode dispersion sampling; this stub is the zero-width
        default those ranges collapse to."""
        raw = jax.random.normal(key, (3,), dtype=jnp.float64)
        direction = raw / jnp.maximum(jnp.linalg.norm(raw), 1e-8)
        pos = direction * jnp.asarray(self.cfg.orbit.start_radius_range_m[0], jnp.float64)
        q_bw = quat_from_body_z_to(-direction).astype(jnp.float64)
        return jnp.concatenate(
            [self._epoch0, pos, jnp.zeros(3, jnp.float64), q_bw, jnp.zeros(3, jnp.float64)],
            axis=0,
        )
