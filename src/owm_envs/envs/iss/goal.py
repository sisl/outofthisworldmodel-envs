"""Goal-error observation block: [pos_err, vel_err, att_err_axis_angle, rate_err].

One pure implementation used by the Gymnasium envs (dock-pose goal) and by
both rollout drivers (per-policy goal via make_augment), so every consumer
appends byte-identical blocks for the same inputs. Position, velocity and rate
errors are measured minus target; the attitude block is the body-frame rotation
FROM the measured attitude TO the target (the controllers' error convention),
i.e. goal-minus-current in the rotational sense. All are computed from the SAME
(possibly noisy) observation the dataset records -- a navigation system's output,
not privileged truth.

The orbit goal targets the reference state ONE TIMESTEP AHEAD along the
commanded circle -- the measured position's planar projection advanced by
one control step -- which is also what the orbit control law regulates to
(see `policies.orbit_reference`, the single function both consume). The
recorded goal error is therefore exactly the position/velocity/attitude/rate
residual the controller is driving toward zero, not a synthetic look-ahead
target it never chases. Dock targets whichever port the episode was assigned at reset (see
`policies.dock_target_selector`), which is a static final-state target within
an episode; random remains zeros.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from ...core.quaternion import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_multiply,
    quat_normalize,
)
from .config import ISSConfig, dock_target
from .policies import PolicyConfig, dock_target_selector, orbit_reference

GOAL_ERROR_DIM = 12

GOAL_ERROR_LABELS: tuple[str, ...] = (
    "goal_pos_err_x_m", "goal_pos_err_y_m", "goal_pos_err_z_m",
    "goal_vel_err_x_m_s", "goal_vel_err_y_m_s", "goal_vel_err_z_m_s",
    "goal_att_err_axis_angle_x", "goal_att_err_axis_angle_y", "goal_att_err_axis_angle_z",
    "goal_rate_err_x_rad_s", "goal_rate_err_y_rad_s", "goal_rate_err_z_rad_s",
)


# Names, in block order, of the four magnitudes `goal_error_norms` returns.
GOAL_ERROR_NORM_LABELS: tuple[str, ...] = ("pos_m", "vel_mps", "att_rad", "rate_radps")


def goal_error_norms(block):
    """The four per-quantity magnitudes of a goal-error block, in block order.

    How far the state is from the goal in each of the four quantities the
    block measures, as one (4,) array. `att_rad` is the rotation angle to the
    target attitude and nothing else: the axis-angle block is axis * angle
    with a hemisphere-corrected angle in [0, pi] (see
    `quaternion.axis_angle_from_quat`), so its norm is that angle.
    """
    return jnp.stack([
        jnp.linalg.norm(block[0:3]),
        jnp.linalg.norm(block[3:6]),
        jnp.linalg.norm(block[6:9]),
        jnp.linalg.norm(block[9:12]),
    ])


def goal_error(measured, target_pos, target_vel, target_quat, target_rate):
    q_err = quat_multiply(quat_conjugate(quat_normalize(measured[6:10])),
                          quat_normalize(target_quat))
    return jnp.concatenate([
        measured[0:3] - target_pos,
        measured[3:6] - target_vel,
        axis_angle_from_quat(q_err),
        measured[10:13] - target_rate,
    ])


def dock_goal_error(measured, target):
    """`target` is a (7,) [position, quaternion] row, from `config.dock_target`
    for the environment's own dock pose or from `policies.dock_target_selector`
    for the port an episode was assigned."""
    zeros = jnp.zeros((3,), jnp.float32)
    return goal_error(measured, target[0:3], zeros, target[3:7], zeros)


def _orbit_goal_error(measured, orbit_extras, dt):
    # Thin wrapper: the reference state is shared with the orbit control law
    # via `policies.orbit_reference` (see module docstring).
    p_des, v_des, q_des = orbit_reference(measured[0:3], orbit_extras, dt)
    zeros = jnp.zeros((3,), jnp.float32)
    return goal_error(measured, p_des, v_des, q_des, zeros)


def make_augment(cfg: ISSConfig, policy_cfg: PolicyConfig) -> Callable | None:
    """Observation-augment fn for a policy type, or None when disabled."""
    if not cfg.observation.goal_error:
        return None
    zeros12 = jnp.zeros((GOAL_ERROR_DIM,), jnp.float32)
    if policy_cfg.type == "random":
        return lambda measured, extras: jnp.concatenate([measured, zeros12])
    select_target = dock_target_selector(cfg, policy_cfg)
    if policy_cfg.type == "dock":
        return lambda measured, extras: jnp.concatenate(
            [measured, dock_goal_error(measured, select_target(extras))]
        )
    if policy_cfg.type == "orbit":
        return lambda measured, extras: jnp.concatenate([measured, _orbit_goal_error(measured, extras, cfg.dt)])
    if policy_cfg.type == "union":
        def augment(measured, extras):
            block = jax.lax.switch(
                extras[0].astype(jnp.int32),
                [lambda: zeros12,
                 lambda: _orbit_goal_error(measured, extras[1:6], cfg.dt),
                 lambda: dock_goal_error(measured, select_target(extras))],
            )
            return jnp.concatenate([measured, block])
        return augment
    raise ValueError(f"unknown policy type '{policy_cfg.type}'")
