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

import jax
import jax.numpy as jnp

from ...core.integrator import Integrator
from ...core.quaternion import (
    quat_derivative_from_omega_body,
    quat_from_body_z_to,
    quat_normalize,
    rotate_body_to_world,
)
from ..common.config import dock_target
from ..common.events import EventChecker, Events
from ..common.layout import ISS_LAYOUT
from .config import ISSConfig

STATE_LABELS: tuple[str, ...] = ISS_LAYOUT.labels


class ISSDynamics:
    def __init__(self, cfg: ISSConfig):
        self.state_dim = 13
        self.action_dim = 6

        self._integrator = Integrator(cfg.dt)

        inertia = jnp.asarray(cfg.physics.inertia_diag, dtype=jnp.float32)
        self._mass = jnp.asarray(cfg.physics.mass, dtype=jnp.float32)
        self._inertia_diag = jnp.maximum(inertia, 1e-6)
        self._inv_inertia_diag = 1.0 / self._inertia_diag
        self._linear_damping = jnp.asarray(cfg.physics.linear_damping, dtype=jnp.float32)
        self._angular_damping = jnp.asarray(cfg.physics.angular_damping, dtype=jnp.float32)
        start_low, start_high = cfg.physics.start_radius_range_m
        self._start_radius_low = jnp.asarray(start_low, dtype=jnp.float32)
        self._start_radius_high = jnp.asarray(start_high, dtype=jnp.float32)

        self._events = EventChecker(cfg)

        # Default target when `step` is not given one: the pose named by
        # DockConfig. A multi-port rollout passes the episode's own target in.
        self._dock_target = jnp.asarray(dock_target(cfg), dtype=jnp.float32)

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

        events = self._events.events(s, s_next, target)
        return s_next, events

    def reset(self, key: jax.Array) -> jnp.ndarray:
        """Uniform direction, uniform radius in the start range, at rest, nose at the ISS."""
        key_direction, key_radius = jax.random.split(key)
        raw = jax.random.normal(key_direction, (3,), dtype=jnp.float32)
        direction = raw / jnp.maximum(jnp.linalg.norm(raw), 1e-8)
        radius = jax.random.uniform(
            key_radius,
            (),
            dtype=jnp.float32,
            minval=self._start_radius_low,
            maxval=self._start_radius_high,
        )
        pos = direction * radius
        nose_to_iss = -direction
        q_bw = quat_from_body_z_to(nose_to_iss)
        return jnp.concatenate(
            [pos, jnp.zeros((3,), jnp.float32), q_bw, jnp.zeros((3,), jnp.float32)], axis=0
        )
