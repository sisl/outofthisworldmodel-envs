"""iss-numerical's render adapter: chief geometry and lighting read straight
off the state, and the chaser's pose from the derived relative view.

Unlike `iss_hcw`, whose chief is not carried in the state at all and has to be
reconstructed analytically from a `ReferenceOrbit` at the state's own epoch,
`iss-numerical`'s chief is a propagated ECI state sitting right in the state
vector (`NUM_LAYOUT.chief`, columns 2:8) -- this adapter reads it directly, no
`ReferenceOrbit` involved. The chaser's world-frame pose is not carried
directly either (the raw slices hold its absolute ECI state), so the kernel
calls `relative_view` the same way the task layer does.

`make_render_adapter` is module-level for the same reason `iss` and `iss_hcw`'s
are: render worker processes rebuild an adapter from (env_name, cfg) via
ENV_REGISTRY and have to pickle it across a process boundary. This adapter
carries no per-config state at all (there is no `ReferenceOrbit` to build), so
`__getstate__`/`__setstate__` exist only to drop the unpicklable jitted kernel,
exactly as `iss_hcw`'s do for its `ReferenceOrbit`.

The kernel is jitted for the same reason `iss_hcw`'s is: called op by op, the
sun/moon/eclipse ephemeris is uncompiled JAX dispatch over a handful of
scalars, measured there at 72 ms a frame against 0.11 ms compiled -- the same
astrojax calls, so the same cost either way here.

Both the live `render()` path and the recorded-dataset video path hand this
adapter an already float32 state (`env.py`'s `_true_state`/
`_observation_and_measured` narrow before recording, and `TrajectoryBatch
.true_state` is stored float32),
so the chief and chaser ECI columns the kernel differences inside
`relative_view` are already at the ~0.5 m grain that narrowing an ~6.8e6 m
position to float32 costs -- accepted for rendering, the same trade
`dynamics.py`'s module docstring documents for the state's own f32 narrowing
points, and far below what a viewer can see.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from ...render.inputs import Lighting, RenderInputs
from ..common.epoch_state import epoch_from_prefix
from ..common.orbit import illumination, moon_vector_world, sun_direction_world
from .dynamics import relative_view

if TYPE_CHECKING:
    from .config import NumericalConfig


class _NumericalRenderAdapter:
    """Built once per config by `make_render_adapter`; callable (state,
    action) -> RenderInputs. `cfg` is accepted only to match the registry's
    per-env factory signature -- unlike `iss_hcw`, nothing here is built from
    it, since the chief comes straight out of the state."""

    def __init__(self, cfg: NumericalConfig) -> None:
        del cfg
        self._kernel = jax.jit(self._compute)

    def _compute(self, state: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
        """Everything the state determines, as one traced computation."""
        epoch = epoch_from_prefix(state[0:2])
        chief = state[2:8]
        chief_pos = chief[0:3]
        return (
            sun_direction_world(chief, epoch),
            illumination(chief_pos, epoch),
            jnp.linalg.norm(chief_pos),
            moon_vector_world(chief, epoch),
            relative_view(state),
        )

    def __getstate__(self) -> dict[str, Any]:
        # The jitted callable does not cross a process boundary: it holds a
        # compilation cache and a reference back to this instance's bound
        # method. There is nothing else to carry -- each worker jits its own
        # from an otherwise-empty instance.
        return {}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._kernel = jax.jit(self._compute)

    def __call__(self, state: np.ndarray, action: np.ndarray | None = None) -> RenderInputs:
        sun, illum, distance, moon, view = self._kernel(jnp.asarray(state))
        lighting = Lighting(
            sun_direction_world=np.asarray(sun),
            illumination=float(illum),
            chief_distance_m=float(distance),
            moon_vector_world=np.asarray(moon),
        )
        return RenderInputs.from_view(np.asarray(view), action, lighting=lighting)


def make_render_adapter(cfg: NumericalConfig) -> _NumericalRenderAdapter:
    return _NumericalRenderAdapter(cfg)
