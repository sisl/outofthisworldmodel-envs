"""Why an episode ended, recovered from a finished `TrajectoryBatch`.

`TrajectoryBatch` records `terminated` and `truncated`, but collision, a
successful dock and leaving the domain all set `terminated` -- so the batch
alone cannot say which happened. That distinction is what a curated rollout
needs: "keep going until three episodes actually docked" is not answerable from
a boolean that a crash also sets.

Rather than widen the batch, the outcome is re-derived by running the episode's
last step back through the same `EventChecker` the rollout used. That keeps one
definition of each event: an episode counted as docked here is docked by
exactly the gate the environment scored it against.

Truth, not measurement. The terminal state comes from `batch.true_state`, never
from `observations` -- under sensor noise an observation is what the chaser
believed, and whether it docked is a fact about where it actually was. It goes
through the env's own `view` because an env's stored state need not be the
canonical relative one (iss-numerical stores absolute ECI columns).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from ...core.quaternion import quat_angle_between
from ...drivers.types import TrajectoryBatch
from .. import EnvSpec
from .config import BaseTaskConfig
from .events import EventChecker


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

    if length < 2:
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


def classify_batch(
    batch: TrajectoryBatch, cfg: BaseTaskConfig, env_spec: EnvSpec
) -> list[EpisodeOutcome]:
    """Classify every episode. One `EventChecker` is built and shared: it loads
    the station's 313-box collision geometry, which is not worth doing per
    episode."""
    checker = EventChecker(cfg)
    return [
        classify_episode(batch, index, checker, env_spec.view)
        for index in range(batch.num_episodes)
    ]
