"""Docking reward.

This is a per-step reward, not a trajectory-level cost: each term is computed
pointwise from a single (state, action) pair and the five terms are summed
into one scalar. An MPPI-style cost function instead takes (N, H, Ds) states
and (N, H, Da) actions and averages each term over a horizon H, but Gymnasium
expects a per-step scalar; the discounted sum of per-step rewards over an
episode takes the place of that horizon average.

The distance terms compute sum(diff**2) -- a summed squared difference, not a
Euclidean norm. This is deliberate: it keeps every term a simple quadratic,
differentiable everywhere including at zero error, and on a comparable scale
to the other quadratic penalty terms below.

The position term targets the dock goal, ~24.6 m from the ISS origin, rather
than the origin itself. That is deliberate, not an oversight -- this is a
docking task, and pulling the chaser toward the ISS origin would reward it
for approaching the very structure the collision term penalises it for
hitting; indeed, the origin sits inside the station's collision hull while
the dock position does not. `cfg.reward_goal_position` can override the
target to the origin (or elsewhere) for callers who deliberately want that
trade-off.

Which dock goal that is depends on the episode. A multi-port rollout assigns
each episode a port at reset, and the caller passes that port's position in
as `dock_position` so the reward is shaped toward the point the episode is
actually flying to -- the same target the dynamics score `docked` against and
the goal-error block records. Callers with no per-episode target (the
Gymnasium adapters, and any single-pose config) pass nothing and get
`cfg.dock.position`.
"""

from __future__ import annotations

import jax.numpy as jnp

from .config import BaseTaskConfig
from .events import Events


def docking_reward(
    state: jnp.ndarray,
    action: jnp.ndarray,
    events: Events,
    cfg: BaseTaskConfig,
    dock_position: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Per-step reward. Returns a float32 scalar (or a batch under vmap).

    `state` is the canonical 13D world-frame relative view
    `[pos, vel, q_bw, omega]` -- which for the iss env is the state itself,
    and for an env with a wider state is what `StateLayout.slice_view`
    extracts.

    `dock_position` is this episode's dock goal position -- the (3,) prefix of
    the target row `policies.dock_target_selector` resolved for it. Omitted,
    the position in `DockConfig` is used. An explicit
    `cfg.reward_goal_position` outranks both: it is an instruction to shape
    the reward toward some other point entirely, which a per-episode port does
    not revoke.
    """
    w = cfg.reward_weights

    if cfg.reward_goal_position is not None:
        goal = cfg.reward_goal_position
    elif dock_position is not None:
        goal = dock_position
    else:
        goal = cfg.dock.position
    goal_position = jnp.asarray(goal, dtype=jnp.float32)

    position_error = jnp.sum((state[0:3] - goal_position) ** 2)
    velocity_error = jnp.sum(state[3:6] ** 2)
    angular_velocity_error = jnp.sum(state[10:13] ** 2)
    control_effort = jnp.sum(action**2)
    collision = events.collision.astype(jnp.float32)

    return (
        w.position * position_error
        + w.velocity * velocity_error
        + w.angular_velocity * angular_velocity_error
        + w.control_effort * control_effort
        + w.collision * collision
    ).astype(jnp.float32)
