"""Goal-error observation block: [pos_err, vel_err, att_err_axis_angle, rate_err].

One pure implementation used by the Gymnasium envs (dock-pose goal) and by
both rollout drivers (per-policy goal via make_augment), so every consumer
appends byte-identical blocks for the same inputs. Position, velocity and rate
errors are measured minus target; the attitude block is the body-frame rotation
FROM the measured attitude TO the target (the controllers' error convention),
i.e. goal-minus-current in the rotational sense. All are computed from the SAME
(possibly noisy) observation the dataset records -- a navigation system's output,
not privileged truth.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from ...core.quaternion import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_from_body_z_to,
    quat_multiply,
    quat_normalize,
)
from .config import ISSConfig
from .policies import PolicyConfig

GOAL_ERROR_DIM = 12

GOAL_ERROR_LABELS: tuple[str, ...] = (
    "goal_pos_err_x_m", "goal_pos_err_y_m", "goal_pos_err_z_m",
    "goal_vel_err_x_m_s", "goal_vel_err_y_m_s", "goal_vel_err_z_m_s",
    "goal_att_err_axis_angle_x", "goal_att_err_axis_angle_y", "goal_att_err_axis_angle_z",
    "goal_rate_err_x_rad_s", "goal_rate_err_y_rad_s", "goal_rate_err_z_rad_s",
)


def goal_error(measured, target_pos, target_vel, target_quat, target_rate):
    q_err = quat_multiply(quat_conjugate(quat_normalize(measured[6:10])),
                          quat_normalize(target_quat))
    return jnp.concatenate([
        measured[0:3] - target_pos,
        measured[3:6] - target_vel,
        axis_angle_from_quat(q_err),
        measured[10:13] - target_rate,
    ])


def dock_goal_error(measured, cfg: ISSConfig):
    zeros = jnp.zeros((3,), jnp.float32)
    return goal_error(measured, jnp.asarray(cfg.dock.position, jnp.float32), zeros,
                      jnp.asarray(cfg.dock.quaternion, jnp.float32), zeros)


def _orbit_goal_error(measured, orbit_extras):
    axis = orbit_extras[0:3]
    radius, omega = orbit_extras[3], orbit_extras[4]
    pos, zeros = measured[0:3], jnp.zeros((3,), jnp.float32)
    pos_planar = pos - jnp.dot(pos, axis) * axis
    r_hat = pos_planar / jnp.maximum(jnp.linalg.norm(pos_planar), 1e-8)
    p_des = radius * r_hat
    v_des = omega * radius * jnp.cross(axis, r_hat)
    return goal_error(measured, p_des, v_des, quat_from_body_z_to(-r_hat), zeros)


def make_augment(cfg: ISSConfig, policy_cfg: PolicyConfig) -> Callable | None:
    """Observation-augment fn for a policy type, or None when disabled."""
    if not cfg.observation.goal_error:
        return None
    zeros12 = jnp.zeros((GOAL_ERROR_DIM,), jnp.float32)
    if policy_cfg.type == "random":
        return lambda measured, extras: jnp.concatenate([measured, zeros12])
    if policy_cfg.type == "dock":
        return lambda measured, extras: jnp.concatenate([measured, dock_goal_error(measured, cfg)])
    if policy_cfg.type == "orbit":
        return lambda measured, extras: jnp.concatenate([measured, _orbit_goal_error(measured, extras)])
    if policy_cfg.type == "union":
        def augment(measured, extras):
            block = jax.lax.switch(
                extras[0].astype(jnp.int32),
                [lambda: zeros12,
                 lambda: _orbit_goal_error(measured, extras[1:6]),
                 lambda: dock_goal_error(measured, cfg)],
            )
            return jnp.concatenate([measured, block])
        return augment
    raise ValueError(f"unknown policy type '{policy_cfg.type}'")
