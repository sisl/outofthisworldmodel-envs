"""ISS-centered free-flyer dynamics for a Dragon chaser.

Ported from seamstress branch iss2,
src/seamstress/dynamics/international_space_station.py.

State (13D) -- note this is 13, not seamstress's 14. The absorbing terminal flag
that lived at index 13 is NOT part of the state here; `step` returns it as
`Events` instead, so that collision and dock-success stay distinguishable and
the Gymnasium adapters can map them onto terminated/truncated/info.

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
    quat_derivative_from_omega_body,
    quat_normalize,
    rotate_body_to_world,
)
from .config import ISSConfig, load_collision_boxes

STATE_LABELS: tuple[str, ...] = (
    "rel_x_m", "rel_y_m", "rel_z_m",
    "rel_vx_m_s", "rel_vy_m_s", "rel_vz_m_s",
    "q_w", "q_x", "q_y", "q_z",
    "omega_x_rad_s", "omega_y_rad_s", "omega_z_rad_s",
)

BODY_Z = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)


class Events(NamedTuple):
    """Per-step outcomes. Both are boolean scalars (or boolean arrays under vmap)."""

    collision: jnp.ndarray
    docked: jnp.ndarray


class ISSDynamics:
    def __init__(self, cfg: ISSConfig):
        self.cfg = cfg
        self.state_dim = 13
        self.action_dim = 6

        self._integrator = Integrator(cfg.dt)

        inertia = jnp.asarray(cfg.inertia_diag, dtype=jnp.float32)
        self._mass = jnp.asarray(cfg.mass, dtype=jnp.float32)
        self._inertia_diag = jnp.maximum(inertia, 1e-6)
        self._inv_inertia_diag = 1.0 / self._inertia_diag
        self._linear_damping = jnp.asarray(cfg.linear_damping, dtype=jnp.float32)
        self._angular_damping = jnp.asarray(cfg.angular_damping, dtype=jnp.float32)
        self._chaser_radius = jnp.asarray(cfg.dragon_collision_radius_m, dtype=jnp.float32)
        self._start_radius = jnp.asarray(cfg.start_radius_m, dtype=jnp.float32)

        centers, half_extents = load_collision_boxes(cfg.collision_boxes_path)
        self._box_centers = jnp.asarray(centers, dtype=jnp.float32)
        self._box_half_extents = jnp.asarray(half_extents, dtype=jnp.float32)

        self._dock_position = jnp.asarray(cfg.dock_position, dtype=jnp.float32)
        self._dock_max_distance = jnp.asarray(cfg.dock_max_distance_m, dtype=jnp.float32)
        self._dock_max_velocity = jnp.asarray(cfg.dock_max_velocity_m_s, dtype=jnp.float32)

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

    def _collision(self, pos_w: jnp.ndarray) -> jnp.ndarray:
        # Distance from the point to each AABB surface (0 when inside), then
        # contact if any is within the chaser's bounding radius.
        if self._box_centers.shape[0] == 0:
            return jnp.array(False)
        delta = jnp.abs(pos_w[None, :] - self._box_centers) - self._box_half_extents
        outside = jnp.maximum(delta, 0.0)
        dist = jnp.linalg.norm(outside, axis=1)
        return jnp.any(dist <= self._chaser_radius)

    def _docked(self, pos_w: jnp.ndarray, vel_w: jnp.ndarray) -> jnp.ndarray:
        if not self.cfg.dock_enabled:
            return jnp.array(False)
        near = jnp.linalg.norm(pos_w - self._dock_position) <= self._dock_max_distance
        slow = jnp.linalg.norm(vel_w) <= self._dock_max_velocity
        return jnp.logical_and(near, slow)

    def step(self, state: jnp.ndarray, action: jnp.ndarray) -> tuple[jnp.ndarray, Events]:
        s = state.astype(jnp.float32)
        a = action.astype(jnp.float32)

        q_prev = quat_normalize(s[6:10])
        s = s.at[6:10].set(q_prev)

        s_next = self._integrator.rk4(self._eom, s, a)

        q_next = quat_normalize(s_next[6:10])
        # Keep the quaternion on the same hemisphere as the previous step so the
        # trajectory doesn't show a spurious sign flip (q and -q are the same rotation).
        q_next = jnp.where(jnp.dot(q_next, q_prev) < 0.0, -q_next, q_next)
        s_next = s_next.at[6:10].set(q_next)

        events = Events(
            collision=self._collision(s_next[0:3]),
            docked=self._docked(s_next[0:3], s_next[3:6]),
        )
        return s_next, events

    def reset(self, key: jax.Array) -> jnp.ndarray:
        """Uniform random point on the start sphere, at rest, nose pointed at the ISS."""
        raw = jax.random.normal(key, (3,), dtype=jnp.float32)
        direction = raw / jnp.maximum(jnp.linalg.norm(raw), 1e-8)
        pos = direction * self._start_radius
        nose_to_iss = -direction
        q_bw = _quat_from_body_z_to(nose_to_iss)
        return jnp.concatenate(
            [pos, jnp.zeros((3,), jnp.float32), q_bw, jnp.zeros((3,), jnp.float32)], axis=0
        )


def _quat_from_body_z_to(target_dir: jnp.ndarray) -> jnp.ndarray:
    """Shortest-arc quaternion rotating body +z onto `target_dir` (assumed unit)."""
    dot = jnp.clip(jnp.dot(BODY_Z, target_dir), -1.0, 1.0)
    cross = jnp.cross(BODY_Z, target_dir)
    # The w = 1 + dot form degenerates when target_dir is exactly -z; fall back to
    # a 180-degree rotation about x, which is a valid shortest arc in that case.
    q = jnp.concatenate([jnp.array([1.0 + dot], dtype=jnp.float32), cross], axis=0)
    antiparallel = dot < -1.0 + 1e-6
    q = jnp.where(antiparallel, jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float32), q)
    return quat_normalize(q)
