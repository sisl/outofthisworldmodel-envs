"""iss-hcw environment configuration: HCW relative dynamics + gravity-gradient
torque about an analytic Keplerian chief.

Unlike `iss`, whose chaser moves in fixed world-frame kinematics with no
notion of orbital position, iss-hcw derives the chaser's relative motion from
Clohessy-Wiltshire (Hill's) equations linearized about a chief following a
two-body Keplerian orbit (`OrbitConfig`, `ReferenceOrbit`), plus a
gravity-gradient torque on the chaser's attitude. Absolute time advances
alongside the relative state, carried in-state as a `[jd, sec]` epoch prefix
(`EPOCH_LABELS`) so the chief's position and the sun/moon/eclipse geometry it
drives can be recovered at any point along a trajectory without external
bookkeeping.
"""

from __future__ import annotations

from pydantic import Field

from ..common.config import BaseTaskConfig
from ..common.epoch_state import EPOCH_LABELS
from ..common.layout import StateLayout, VIEW_LABELS
from ..common.orbit import OrbitConfig


class HCWConfig(BaseTaskConfig):
    """`BaseTaskConfig` plus the chief's reference orbit."""

    orbit: OrbitConfig = Field(default_factory=OrbitConfig)


HCW_LAYOUT = StateLayout(
    state_dim=15,
    pos=slice(2, 5),
    vel=slice(5, 8),
    quat=slice(8, 12),
    omega=slice(12, 15),
    epoch=slice(0, 2),
    labels=EPOCH_LABELS + VIEW_LABELS,
)
