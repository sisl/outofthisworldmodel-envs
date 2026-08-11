"""Docking reward.

A per-step reward, not a trajectory-level cost: each term is computed
pointwise from one state and the terms are summed into one scalar. Gymnasium
expects a per-step scalar, and the discounted sum over an episode takes the
place of the horizon average an MPPI-style cost function would compute.

Every error enters as a NORM passed through a normalised Huber,

    f(e; delta, scale) = (sqrt(e**2 + delta**2) - delta) / scale

which is ~e/scale for e >> delta and ~e**2/(2*delta*scale) for e << delta. The
far field therefore has a constant gradient, so a chaser at the outer edge of
the start shell feels a steady pull toward the port; the near field is smooth
with zero slope at the origin, so closing the last metre keeps paying. A plain
squared error gives the opposite of both -- it explodes far out and goes flat
exactly where precision matters.

That normalisation is also what makes the collision penalty mean anything.
The weights sum to 1 and `position_scale_m` is set to 225 m, the outer edge
of the start shell the iss-numerical variants disperse over, so a worst-case
step there costs ~0.73 and a 7200-step rollout ~-5,225 against a -1e6
collision -- a 191x margin. This module is shared by all three envs, though,
and `PhysicsConfig.start_radius_range_m` goes out to 500 m with
`max_range_m` at 750 m; the same worst-case step at those radii costs ~1.33
and ~1.89, ~-9,607 and ~-13,604 over a full rollout, still a 104x and 73x
margin. Under the previous summed-square terms a 225 m error cost -50,625
per step and a rollout integrated to ~-3.6e8, leaving the collision penalty
at 0.3% of the trajectory -- an agent could fly into the station and barely
feel it.

The rotational pair is gated on range to the port. Far out the task is to
close the distance and pointing hardly matters; at the gate, attitude and body
rate matter as much as position and velocity. See `RewardShapingConfig`.

The position term targets the dock goal, ~24.6 m from the ISS origin, rather
than the origin itself. That is deliberate -- this is a docking task, and
pulling the chaser toward the ISS origin would reward it for approaching the
very structure the collision term penalises it for hitting; the origin sits
inside the station's collision hull while the dock position does not.
`cfg.reward_goal_position` can override the target for callers who deliberately
want that trade-off.

Which dock goal that is depends on the episode. A multi-port rollout assigns
each episode a port at reset, and the caller passes that port's POSE in as
`dock_pose` so both the position and the attitude are shaped toward the point
the episode is actually flying to -- the same target the dynamics score
`docked` against and the goal-error block records. Callers with no per-episode
target (the Gymnasium adapters, and any single-pose config) pass nothing and
get `cfg.dock`.
"""

from __future__ import annotations

import jax.numpy as jnp

from ...core.quaternion import quat_angle_between
from .config import BaseTaskConfig
from .events import Events


def _huber(error: jnp.ndarray, delta: float, scale: float) -> jnp.ndarray:
    """Normalised Huber of a non-negative error. Zero at zero error."""
    return (jnp.sqrt(error * error + delta * delta) - delta) / scale


def docking_reward(
    state: jnp.ndarray,
    action: jnp.ndarray,
    events: Events,
    cfg: BaseTaskConfig,
    dock_pose: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Per-step reward. Returns a float32 scalar (or a batch under vmap).

    `state` is the canonical 13D world-frame relative view
    `[pos, vel, q_bw, omega]` -- which for the iss env is the state itself,
    and for an env with a wider state is what that env's `view` derives.

    `action` is accepted but not read. No term depends on it since the control
    effort penalty was dropped; it stays in the signature because a fuel or
    actuator-wear term is a plausible thing to want back, and adding one should
    be a change to this function rather than to seven call sites.

    `dock_pose` is this episode's `(7,)` `[position, quaternion]` goal -- the
    target row `policies.dock_target_selector` resolved for it. Omitted, the
    pose in `DockConfig` is used. An explicit `cfg.reward_goal_position`
    outranks both for the POSITION only: it is an instruction to shape the
    reward toward some other point, which says nothing about what attitude to
    hold on arrival, so the quaternion still comes from `dock_pose` or
    `DockConfig`.
    """
    w = cfg.reward_weights
    shaping = cfg.reward_shaping

    if dock_pose is not None:
        pose = jnp.asarray(dock_pose, dtype=jnp.float32)
        goal_position, goal_quaternion = pose[0:3], pose[3:7]
    else:
        goal_position = jnp.asarray(cfg.dock.position, dtype=jnp.float32)
        goal_quaternion = jnp.asarray(cfg.dock.quaternion, dtype=jnp.float32)
    if cfg.reward_goal_position is not None:
        goal_position = jnp.asarray(cfg.reward_goal_position, dtype=jnp.float32)

    distance = jnp.linalg.norm(state[0:3] - goal_position)
    speed = jnp.linalg.norm(state[3:6])
    attitude_error = quat_angle_between(state[6:10], goal_quaternion)
    body_rate = jnp.linalg.norm(state[10:13])

    rotation_gate = shaping.rotation_gate_far + (1.0 - shaping.rotation_gate_far) / (
        1.0 + (distance / shaping.rotation_gate_range_m) ** 2
    )

    return (
        w.position * _huber(distance, shaping.position_delta_m, shaping.position_scale_m)
        + w.velocity * _huber(speed, shaping.velocity_delta_m_s, shaping.velocity_scale_m_s)
        + rotation_gate
        * (
            w.attitude
            * _huber(attitude_error, shaping.attitude_delta_rad, shaping.attitude_scale_rad)
            + w.body_rate
            * _huber(body_rate, shaping.rate_delta_rad_s, shaping.rate_scale_rad_s)
        )
        + w.collision * events.collision.astype(jnp.float32)
        + w.dock_success * events.docked.astype(jnp.float32)
    ).astype(jnp.float32)
