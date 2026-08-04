"""ISS-centered free-flyer dynamics for a Dragon chaser.

State (13D). Per-step outcomes (collision, dock success, leaving the domain)
are deliberately NOT folded into the state as an absorbing terminal flag;
`step` returns them as `Events` instead, so that the three stay
distinguishable and the Gymnasium adapters can map them onto
terminated/truncated/info.

  0..2   relative position to ISS, world frame [m]
  3..5   relative velocity, world frame [m/s]
  6..9   quaternion q_bw = [w,x,y,z] (body -> world)
  10..12 angular velocity, body frame [rad/s]

Control (6D)
  0..2   body-frame force [N]
  3..5   body-frame torque [N*m]
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ...core.integrator import Integrator
from ...core.quaternion import (
    quat_conjugate,
    quat_derivative_from_omega_body,
    quat_from_body_z_to,
    quat_multiply,
    quat_normalize,
    rotate_body_to_world,
)
from .config import ISSConfig, dock_target, load_collision_boxes

STATE_LABELS: tuple[str, ...] = (
    "rel_x_m", "rel_y_m", "rel_z_m",
    "rel_vx_m_s", "rel_vy_m_s", "rel_vz_m_s",
    "q_w", "q_x", "q_y", "q_z",
    "omega_x_rad_s", "omega_y_rad_s", "omega_z_rad_s",
)


class Events(NamedTuple):
    """Per-step outcomes. All three are boolean scalars (or boolean arrays
    under vmap), and they are independent: a step can raise more than one.

    `escaped` -- the chaser left the spherical domain of radius
    `cfg.max_range_m` -- is an absorbing outcome like `collision`, not a time
    limit: consumers map it onto terminated, never truncated. It carries no
    reward term of its own. The position penalty already scores being far
    from the dock, and a separate escape bonus or penalty would be a second,
    unweighted opinion on the same thing.
    """

    collision: jnp.ndarray
    docked: jnp.ndarray
    escaped: jnp.ndarray


class ISSDynamics:
    def __init__(self, cfg: ISSConfig):
        self.cfg = cfg
        self.state_dim = 13
        self.action_dim = 6

        self._integrator = Integrator(cfg.dt)

        inertia = jnp.asarray(cfg.physics.inertia_diag, dtype=jnp.float32)
        self._mass = jnp.asarray(cfg.physics.mass, dtype=jnp.float32)
        self._inertia_diag = jnp.maximum(inertia, 1e-6)
        self._inv_inertia_diag = 1.0 / self._inertia_diag
        self._linear_damping = jnp.asarray(cfg.physics.linear_damping, dtype=jnp.float32)
        self._angular_damping = jnp.asarray(cfg.physics.angular_damping, dtype=jnp.float32)
        self._chaser_radius = jnp.asarray(cfg.physics.dragon_collision_radius_m, dtype=jnp.float32)
        self._start_radius = jnp.asarray(cfg.physics.start_radius_m, dtype=jnp.float32)
        if cfg.max_range_m is not None:
            self._max_range = jnp.asarray(cfg.max_range_m, dtype=jnp.float32)

        centers, half_extents = load_collision_boxes(cfg.physics.collision_boxes_path)
        self._box_centers = jnp.asarray(centers, dtype=jnp.float32)
        self._box_half_extents = jnp.asarray(half_extents, dtype=jnp.float32)

        # Default target when `step` is not given one: the pose named by
        # DockConfig. A multi-port rollout passes the episode's own target in.
        self._dock_target = jnp.asarray(dock_target(cfg), dtype=jnp.float32)
        self._dock_max_distance = jnp.asarray(cfg.dock.max_distance_m, dtype=jnp.float32)
        self._dock_max_velocity = jnp.asarray(cfg.dock.max_velocity_m_s, dtype=jnp.float32)
        if cfg.dock.max_attitude_error_deg is not None:
            self._dock_max_attitude_error_rad = jnp.asarray(
                jnp.deg2rad(cfg.dock.max_attitude_error_deg), dtype=jnp.float32
            )
        if cfg.dock.max_body_rate_rad_s is not None:
            self._dock_max_body_rate = jnp.asarray(cfg.dock.max_body_rate_rad_s, dtype=jnp.float32)

    def _eom(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        vel_w = x[3:6]
        q_bw = quat_normalize(x[6:10])
        omega_b = x[10:13]

        force_w = rotate_body_to_world(q_bw, u[0:3])
        torque_b = u[3:6]

        pos_dot = vel_w
        vel_dot = force_w / self._mass - self._linear_damping * vel_w
        q_dot = quat_derivative_from_omega_body(q_bw, omega_b)
        coriolis = jnp.cross(omega_b, self._inertia_diag * omega_b)
        omega_dot = self._inv_inertia_diag * (
            torque_b - coriolis - self._angular_damping * omega_b
        )
        return jnp.concatenate([pos_dot, vel_dot, q_dot, omega_dot], axis=0)

    def _collision(self, pos_prev: jnp.ndarray, pos_next: jnp.ndarray) -> jnp.ndarray:
        # Swept-segment test: does the straight-line path from pos_prev to
        # pos_next (the whole integration step, not just its endpoint) pass
        # within the chaser's radius of any box? A fast chaser can tunnel
        # through a thin box between two samples and land clear on the far
        # side, so the endpoint alone is not enough.
        #
        # Each box is expanded per-axis by the chaser radius and the segment
        # is tested against that expanded AABB with the standard slab method
        # (ray/segment vs box). The per-axis expansion is a rectangular
        # superset of the true rounded-corner Minkowski sum of the box and a
        # sphere of that radius, so this never misses a genuine crossing; it
        # can only be conservative right at the corners, which is an
        # accepted approximation here. On the degenerate axes -- where the
        # segment doesn't move along that axis at all -- the interval is
        # collapsed to a plain membership test instead of dividing by zero,
        # so a stationary chaser (pos_prev == pos_next) reduces exactly to a
        # point-in-box test with no NaN.
        if self._box_centers.shape[0] == 0:
            return jnp.array(False)

        expanded_half = self._box_half_extents + self._chaser_radius
        box_min = self._box_centers - expanded_half
        box_max = self._box_centers + expanded_half

        d = pos_next - pos_prev
        eps = jnp.float32(1e-12)
        is_parallel = jnp.abs(d)[None, :] < eps
        safe_d = jnp.where(jnp.abs(d) < eps, eps, d)

        t1 = (box_min - pos_prev[None, :]) / safe_d[None, :]
        t2 = (box_max - pos_prev[None, :]) / safe_d[None, :]
        tmin_axis = jnp.minimum(t1, t2)
        tmax_axis = jnp.maximum(t1, t2)

        inside_slab = jnp.logical_and(
            pos_prev[None, :] >= box_min, pos_prev[None, :] <= box_max
        )
        tmin_axis = jnp.where(
            is_parallel, jnp.where(inside_slab, -jnp.inf, jnp.inf), tmin_axis
        )
        tmax_axis = jnp.where(
            is_parallel, jnp.where(inside_slab, jnp.inf, -jnp.inf), tmax_axis
        )

        t_enter = jnp.maximum(jnp.max(tmin_axis, axis=1), 0.0)
        t_exit = jnp.minimum(jnp.min(tmax_axis, axis=1), 1.0)
        return jnp.any(t_enter <= t_exit)

    def _docked(
        self,
        pos_w: jnp.ndarray,
        vel_w: jnp.ndarray,
        q_bw: jnp.ndarray,
        omega_b: jnp.ndarray,
        target: jnp.ndarray,
    ) -> jnp.ndarray:
        if not self.cfg.dock.enabled:
            return jnp.array(False)
        near = jnp.linalg.norm(pos_w - target[0:3]) <= self._dock_max_distance
        slow = jnp.linalg.norm(vel_w) <= self._dock_max_velocity
        docked = jnp.logical_and(near, slow)

        # Both gates are optional and, being static Python values, are branched
        # on at trace time rather than with jnp.where -- `_docked` runs inside
        # jit/vmap, but `self.cfg.dock.*` is not a traced array.
        if self.cfg.dock.max_attitude_error_deg is not None:
            q_err = quat_multiply(quat_conjugate(q_bw), target[3:7])
            # abs() handles the q/-q double cover: q and -q are the same
            # rotation, but without it their w components differ in sign and
            # give angles 2*pi apart.
            w_err = jnp.clip(jnp.abs(q_err[0]), -1.0, 1.0)
            angle = 2.0 * jnp.arccos(w_err)
            aligned = angle <= self._dock_max_attitude_error_rad
            docked = jnp.logical_and(docked, aligned)

        if self.cfg.dock.max_body_rate_rad_s is not None:
            still = jnp.linalg.norm(omega_b) <= self._dock_max_body_rate
            docked = jnp.logical_and(docked, still)

        return docked

    def _escaped(self, pos_w: jnp.ndarray) -> jnp.ndarray:
        # The endpoint alone, not the swept segment `_collision` tests: the
        # domain is a sphere the chaser is inside, so it cannot cross the
        # boundary and return within one step without being outside at some
        # sampled position long before -- the tunnelling a thin box invites
        # has no analogue here.
        #
        # `cfg.max_range_m is None` is a static Python value branched on at
        # trace time, like the optional dock gates: `step` runs under jit/vmap
        # but the config is not a traced array.
        if self.cfg.max_range_m is None:
            return jnp.array(False)
        return jnp.linalg.norm(pos_w) > self._max_range

    def step(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        dock_pose: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, Events]:
        """Advance one step. `dock_pose` is a (7,) [position, quaternion] row;
        omitted, the pose in `DockConfig` is used, which is what the Gymnasium
        adapters do."""
        s = state.astype(jnp.float32)
        a = action.astype(jnp.float32)
        target = self._dock_target if dock_pose is None else jnp.asarray(dock_pose, jnp.float32)

        q_prev = quat_normalize(s[6:10])
        s = s.at[6:10].set(q_prev)

        s_next = self._integrator.rk4(self._eom, s, a)

        q_next = quat_normalize(s_next[6:10])
        # Keep the quaternion on the same hemisphere as the previous step so the
        # trajectory doesn't show a spurious sign flip (q and -q are the same rotation).
        q_next = jnp.where(jnp.dot(q_next, q_prev) < 0.0, -q_next, q_next)
        s_next = s_next.at[6:10].set(q_next)

        events = Events(
            collision=self._collision(s[0:3], s_next[0:3]),
            docked=self._docked(s_next[0:3], s_next[3:6], s_next[6:10], s_next[10:13], target),
            escaped=self._escaped(s_next[0:3]),
        )
        return s_next, events

    def reset(self, key: jax.Array) -> jnp.ndarray:
        """Uniform random point on the start sphere, at rest, nose pointed at the ISS."""
        raw = jax.random.normal(key, (3,), dtype=jnp.float32)
        direction = raw / jnp.maximum(jnp.linalg.norm(raw), 1e-8)
        pos = direction * self._start_radius
        nose_to_iss = -direction
        q_bw = quat_from_body_z_to(nose_to_iss)
        return jnp.concatenate(
            [pos, jnp.zeros((3,), jnp.float32), q_bw, jnp.zeros((3,), jnp.float32)], axis=0
        )
