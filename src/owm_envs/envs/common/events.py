"""Per-step outcome checks shared by every environment posing the docking task.

Collision, dock success and domain escape are properties of the task geometry,
not of the dynamics that produced the motion, so they live here and read only
the canonical 13D world-frame relative view.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from ...core.quaternion import quat_angle_between
from .config import BaseTaskConfig, load_collision_boxes


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


class EventChecker:
    """Collision / dock-success / domain-escape checks, shared by every env.

    Operates on world-frame relative quantities (the canonical view), so the
    same instance serves any dynamics that can produce them. Built from the
    shared config sections; all cfg reads are static Python values branched
    at trace time, exactly as before the extraction.
    """

    def __init__(self, cfg: BaseTaskConfig):
        self.cfg = cfg

        self._chaser_radius = jnp.asarray(cfg.physics.dragon_collision_radius_m, dtype=jnp.float32)
        if cfg.max_range_m is not None:
            self._max_range = jnp.asarray(cfg.max_range_m, dtype=jnp.float32)

        centers, half_extents = load_collision_boxes(cfg.physics.collision_boxes_path)
        self._box_centers = jnp.asarray(centers, dtype=jnp.float32)
        self._box_half_extents = jnp.asarray(half_extents, dtype=jnp.float32)

        self._dock_max_distance = jnp.asarray(cfg.dock.max_distance_m, dtype=jnp.float32)
        self._dock_max_velocity = jnp.asarray(cfg.dock.max_velocity_m_s, dtype=jnp.float32)
        if cfg.dock.max_attitude_error_deg is not None:
            self._dock_max_attitude_error_rad = jnp.asarray(
                jnp.deg2rad(cfg.dock.max_attitude_error_deg), dtype=jnp.float32
            )
        if cfg.dock.max_body_rate_rad_s is not None:
            self._dock_max_body_rate = jnp.asarray(cfg.dock.max_body_rate_rad_s, dtype=jnp.float32)

    def collision(self, pos_prev: jnp.ndarray, pos_next: jnp.ndarray) -> jnp.ndarray:
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

    def docked(
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
        # on at trace time rather than with jnp.where -- `docked` runs inside
        # jit/vmap, but `self.cfg.dock.*` is not a traced array.
        if self.cfg.dock.max_attitude_error_deg is not None:
            angle = quat_angle_between(q_bw, target[3:7])
            aligned = angle <= self._dock_max_attitude_error_rad
            docked = jnp.logical_and(docked, aligned)

        if self.cfg.dock.max_body_rate_rad_s is not None:
            still = jnp.linalg.norm(omega_b) <= self._dock_max_body_rate
            docked = jnp.logical_and(docked, still)

        return docked

    def escaped(self, pos_w: jnp.ndarray) -> jnp.ndarray:
        # The endpoint alone, not the swept segment `collision` tests: the
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

    def events(
        self, prev_view: jnp.ndarray, next_view: jnp.ndarray, target: jnp.ndarray
    ) -> Events:
        """All three checks for one step, from the canonical view before and
        after it. `target` is a (7,) [position, quaternion] row."""
        return Events(
            collision=self.collision(prev_view[0:3], next_view[0:3]),
            docked=self.docked(
                next_view[0:3], next_view[3:6], next_view[6:10], next_view[10:13], target
            ),
            escaped=self.escaped(next_view[0:3]),
        )
