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

All of the ephemeris runs inside one jitted kernel. Called op by op it is
uncompiled JAX dispatch on a handful of scalars, which measured 72 ms per
frame -- four times what drawing the frame costs, and ten hours of a
500k-frame split spent in the adapter. Compiled it is 0.11 ms. The numbers
are the same ones to well inside `envs/common/orbit.py`'s budgets: the sun
direction moves by 1.1e-5 rad against a ~2e-3 rad model budget, illumination
not at all, the chief distance by 0.5 m against ~4 m, and the moon by 61 km,
which is 0.009 deg of its position against the same ~0.1 deg budget.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from ...render.inputs import Lighting, RenderInputs
from ..common.epoch_state import epoch_from_prefix, epoch_prefix, seconds_between
from ..common.orbit import (
    ReferenceOrbit,
    earth_rotation_world,
    illumination,
    moon_vector_world,
    sun_direction_world,
)
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
        self._kernel = jax.jit(self._compute)

    def _compute(self, prefix: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
        """Everything the epoch prefix determines, as one traced computation."""
        epoch = epoch_from_prefix(prefix)
        t = seconds_between(prefix, self._epoch0_prefix)
        chief = self._ref.chief_state_eci(t)
        chief_pos = chief[0:3]
        return (
            sun_direction_world(epoch, chief),
            illumination(epoch, chief_pos),
            jnp.linalg.norm(chief_pos),
            moon_vector_world(epoch, chief),
            earth_rotation_world(epoch, chief),
        )

    def __getstate__(self) -> dict[str, Any]:
        # The jitted callable does not cross a process boundary: it holds a
        # compilation cache and a reference back to this instance's bound
        # method. Only what the kernel is built FROM is pickled, and each
        # worker jits its own -- one trace per worker, amortised over the
        # episodes it renders.
        return {"_ref": self._ref, "_epoch0_prefix": self._epoch0_prefix}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._kernel = jax.jit(self._compute)

    def __call__(self, state: np.ndarray, action: np.ndarray | None = None) -> RenderInputs:
        sun, illum, distance, moon, earth = self._kernel(jnp.asarray(state[0:2]))
        lighting = Lighting(
            sun_direction_world=np.asarray(sun),
            illumination=float(illum),
            chief_distance_m=float(distance),
            moon_vector_world=np.asarray(moon),
            earth_rotation_world=np.asarray(earth),
        )
        return RenderInputs.from_view(HCW_LAYOUT.slice_view(state), action, lighting=lighting)


def make_render_adapter(cfg: HCWConfig) -> _HCWRenderAdapter:
    return _HCWRenderAdapter(cfg)
