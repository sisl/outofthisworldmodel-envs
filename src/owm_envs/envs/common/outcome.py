"""Why an episode ended, read off a finished `TrajectoryBatch`.

`TrajectoryBatch` records `terminated` and `truncated`, but collision, a
successful dock and leaving the domain all set `terminated` -- so those two
alone cannot say which happened. That distinction is what a curated rollout
needs: "keep going until three episodes actually docked" is not answerable from
a boolean that a crash also sets.

The authority is `batch.terminal_events`, the events the driver recorded at the
moment the final step happened, at the precision the rollout ran at.

FALLBACK. A batch that carries no recorded events has its final step re-run
through the same `EventChecker` the rollout used, from `batch.true_state`. That
keeps one definition of each event, but it is sound only where float32 storage
is fine-grained enough to answer the question being asked. A stored position
lands on a grid of one float32 ulp of its own magnitude, and a relative view
differenced from two such columns inherits that whole ulp on each axis. For an
env that stores the canonical relative view directly, at ~100 m, the ulp is
7.6e-6 m and the gate is never in doubt. For one storing absolute ECI columns,
at an orbital radius of ~6.8e6 m, it is 0.5 m per axis -- and the gate tests
the NORM of the three axes, which reaches 0.5 * sqrt(3) = 0.87 m, over half a
metre of it observed in practice. That is five to nearly nine times a 0.1 m
gate, so whether the chaser was inside it is not a question the stored numbers
can answer. `classify_batch` refuses such a batch rather than answering it
wrongly.

Truth, not measurement. The terminal state used by the fallback and by the
error metrics comes from `batch.true_state`, never from `observations` -- under
sensor noise an observation is what the chaser believed, and whether it docked
is a fact about where it actually was. It goes through the env's own `view`
because an env's stored state need not be the canonical relative one
(iss-numerical stores absolute ECI columns).

The error metrics are diagnostics, not gates, and are read from the stored
state on every env: up to ~0.87 m of quantization on a reported terminal
position error is not material where a boolean gate an order finer is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ...core.quaternion import quat_angle_between
from ...drivers.types import TrajectoryBatch
from .. import EnvSpec
from .config import BaseTaskConfig
from .events import EventChecker
from .layout import StateLayout


@dataclass(frozen=True)
class EpisodeOutcome:
    """How one episode ended, and how close it got.

    The three event flags are not exclusive -- a step can raise more than one,
    exactly as `Events` documents -- so a consumer wanting "docked and nothing
    else" must check the others too.
    """

    docked: bool
    collided: bool
    escaped: bool
    truncated: bool
    steps: int
    position_error_m: float
    velocity_error_m_s: float
    attitude_error_rad: float
    body_rate_rad_s: float


def classify_episode(
    batch: TrajectoryBatch,
    index: int,
    checker: EventChecker,
    view,
) -> EpisodeOutcome:
    """Classify episode `index`. `view` is the env's canonical-view derivation."""
    if batch.true_state is None:
        raise ValueError(
            "cannot classify a batch with no true_state: the outcome is a fact "
            "about where the chaser actually was, and observations may be noisy"
        )
    if batch.dock_targets is None:
        raise ValueError(
            "cannot classify a batch with no dock_targets: each episode has to "
            "be scored against the pose it was actually flying to"
        )

    length = int(batch.lengths[index])
    target = jnp.asarray(batch.dock_targets[index], dtype=jnp.float32)
    final = view(jnp.asarray(batch.true_state[index, length - 1], dtype=jnp.float32))

    if batch.terminal_events is not None:
        collided, docked, escaped = (bool(flag) for flag in batch.terminal_events[index])
    elif length < 2:
        # One observation, no step taken: there is no swept segment to test a
        # collision against, and no motion to have left the domain during.
        docked = collided = escaped = False
    else:
        previous = view(jnp.asarray(batch.true_state[index, length - 2], dtype=jnp.float32))
        events = checker.events(previous, final, target)
        docked = bool(events.docked)
        collided = bool(events.collision)
        escaped = bool(events.escaped)

    return EpisodeOutcome(
        docked=docked,
        collided=collided,
        escaped=escaped,
        truncated=bool(batch.truncated[index]),
        steps=length - 1,
        position_error_m=float(jnp.linalg.norm(final[0:3] - target[0:3])),
        velocity_error_m_s=float(jnp.linalg.norm(final[3:6])),
        attitude_error_rad=float(quat_angle_between(final[6:10], target[3:7])),
        body_rate_rad_s=float(jnp.linalg.norm(final[10:13])),
    )


def _refuse_unresolvable_fallback(
    batch: TrajectoryBatch, cfg: BaseTaskConfig, layout: StateLayout
) -> None:
    """Raise if re-deriving events from `batch.true_state` cannot resolve the
    dock gate for this env.

    `true_state` is stored float32, so each stored column sits on a grid of one
    float32 ulp of its own magnitude, and the canonical view -- a difference of
    the chaser and chief blocks for an env storing absolute positions --
    inherits the ulp of the LARGER of the two. Both blocks are therefore
    measured, not just the chaser's: they are the operands the view
    differences, and either one being large is enough to coarsen the result.

    The grain is per axis, but the gate tests the NORM of three of them, so the
    error the gate actually has to survive reaches `grain * sqrt(3)` -- the
    same factor the module docstring derives. The comparison against
    `dock.max_distance_m` is strict, because a gate exactly one grid step wide
    is one a single rounding can carry the chaser across.
    """
    blocks = [batch.true_state[..., layout.pos]]
    if layout.chief is not None:
        blocks.append(batch.true_state[..., layout.chief])
    largest = max(float(np.abs(block).max()) for block in blocks)
    grain = float(np.spacing(np.float32(largest)))
    reachable = grain * math.sqrt(3.0)
    if reachable < cfg.dock.max_distance_m:
        return
    raise ValueError(
        f"cannot classify this batch: it carries no terminal_events, and "
        f"re-deriving them is not possible for a state whose view is "
        f"differenced from coordinates as large as {largest:.3g} m, where "
        f"float32 storage lands on a {grain:.3g} m per-axis grid -- "
        f"{reachable:.3g} m across the three-axis norm the gate tests -- "
        f"against a {cfg.dock.max_distance_m} m dock gate. Generate it with a "
        f"driver that records terminal_events"
    )


def classify_batch(
    batch: TrajectoryBatch, cfg: BaseTaskConfig, env_spec: EnvSpec
) -> list[EpisodeOutcome]:
    """Classify every episode. One `EventChecker` is built and shared: it loads
    the station's 313-box collision geometry, which is not worth doing per
    episode."""
    if batch.terminal_events is None and batch.true_state is not None:
        _refuse_unresolvable_fallback(batch, cfg, env_spec.layout)
    checker = EventChecker(cfg)
    return [
        classify_episode(batch, index, checker, env_spec.view)
        for index in range(batch.num_episodes)
    ]
