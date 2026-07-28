"""ISS reward.

Ported from seamstress branch iss2, src/seamstress/rewards/international_space_station.py.

The seamstress original is a CompositeReward over five trajectory-level terms:
each takes (N, H, Ds) states and (N, H, Da) actions and returns (N,), averaging
over the horizon -- an MPPI cost function. Gymnasium needs a per-step scalar, so
this port drops the mean-over-horizon and computes each term pointwise. The
discounted sum of per-step rewards takes the place of the horizon mean.

Note that seamstress's distance_type="euclidean" computes sum(diff**2) -- a
summed squared difference, not a Euclidean norm, despite the name. That
behaviour is preserved here.

One term formula does diverge: seamstress's reward.goal is the all-zero state
(conf/control/control_international_space_station.yaml), so its position term
targets the ISS origin. This port instead targets `cfg.dock_position`, ~24.6 m
away. That divergence is deliberate, not an oversight -- this is a docking
task, and pulling the chaser toward the ISS origin would reward it for
approaching the very structure the collision term penalises it for hitting.
"""

from __future__ import annotations

import jax.numpy as jnp

from .config import ISSConfig
from .dynamics import Events


def iss_reward(
    state: jnp.ndarray,
    action: jnp.ndarray,
    events: Events,
    cfg: ISSConfig,
) -> jnp.ndarray:
    """Per-step reward. Returns a float32 scalar (or a batch under vmap)."""
    w = cfg.reward_weights

    dock_position = jnp.asarray(cfg.dock_position, dtype=jnp.float32)

    position_error = jnp.sum((state[0:3] - dock_position) ** 2)
    velocity_error = jnp.sum(state[3:6] ** 2)
    angular_velocity_error = jnp.sum(state[10:13] ** 2)
    control_effort = jnp.sum(action**2)
    collision = events.collision.astype(jnp.float32)

    return -(
        w.position * position_error
        + w.velocity * velocity_error
        + w.angular_velocity * angular_velocity_error
        + w.control_effort * control_effort
        + w.collision * collision
    ).astype(jnp.float32)
