"""iss-hcw's render adapter: chief geometry and lighting from the state's epoch.

Unlike iss, whose view row already is a posable frame, iss-hcw's state carries
an epoch prefix ahead of the 13D view (`HCW_LAYOUT`), and the chief that
drives sun/moon/eclipse geometry is not carried in the state at all -- it is
reconstructed analytically from the reference orbit (`ReferenceOrbit`) at the
state's own epoch, exactly as `HCWDynamics.step` does for the gravity-gradient
torque.

`make_render_adapter` is module-level, as `iss`'s is, so render worker
processes can pickle it across a process boundary when rebuilding an adapter
from (env_name, cfg) via ENV_REGISTRY. Unlike `iss`, this adapter needs
per-config state -- the `ReferenceOrbit` built from `cfg.orbit` -- built once
and reused across calls rather than once per frame. A closure cannot do that
and still pickle (pickle cannot resolve a nested function by reference), so
the built adapter is a callable class instance instead, itself module-level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...render.inputs import Lighting, RenderInputs
from ..common.epoch_state import epoch_from_prefix, epoch_prefix, seconds_between
from ..common.orbit import ReferenceOrbit, illumination, moon_vector_world, sun_direction_world
from .config import HCW_LAYOUT

if TYPE_CHECKING:
    from .config import HCWConfig


class _HCWRenderAdapter:
    """Built once per config by `make_render_adapter`; callable (state,
    action) -> RenderInputs. Holds the `ReferenceOrbit` built from that
    config so it is not rebuilt on every frame."""

    def __init__(self, cfg: HCWConfig) -> None:
        self._ref = ReferenceOrbit(cfg.orbit)
        self._epoch0_prefix = epoch_prefix(self._ref.epoch0)

    def __call__(self, state: np.ndarray, action: np.ndarray | None = None) -> RenderInputs:
        prefix = state[0:2]
        epoch = epoch_from_prefix(prefix)
        t = seconds_between(prefix, self._epoch0_prefix)
        chief = self._ref.chief_state_eci(t)
        chief_pos = chief[0:3]

        lighting = Lighting(
            sun_direction_world=np.asarray(sun_direction_world(chief, epoch)),
            illumination=float(illumination(chief_pos, epoch)),
            chief_distance_m=float(np.linalg.norm(np.asarray(chief_pos))),
            moon_vector_world=np.asarray(moon_vector_world(chief, epoch)),
        )
        return RenderInputs.from_view(HCW_LAYOUT.slice_view(state), action, lighting=lighting)


def make_render_adapter(cfg: HCWConfig) -> _HCWRenderAdapter:
    return _HCWRenderAdapter(cfg)
