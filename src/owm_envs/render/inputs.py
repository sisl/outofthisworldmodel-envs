"""RenderInputs: the numpy-only seam between an env's state and a renderer.

Every env's render adapter (see `EnvSpec.make_render_adapter`) turns a raw
state or view row into one of these, so the renderer itself never has to know
which env produced the frame it's posing. Nothing here may import pygfx or
jax -- render worker processes rebuild adapters from (env_name, cfg) and pose
frames through this module alone, without paying for a GPU stack or an
autodiff one just to do it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Lighting:
    sun_direction_world: np.ndarray  # unit (3,)
    # [0, 1] conical-eclipse factor, only as sharp as the ephemeris behind it:
    # `envs/common/orbit.py` documents ~0.1 deg on the sun direction, which at
    # LEO orbital rate puts an eclipse edge in rendered output a second or two
    # early or late. Model-limited, not a rounding artefact of this seam.
    illumination: float
    chief_distance_m: float  # Earth center sits at -z * this
    moon_vector_world: np.ndarray  # geocentric, meters, (3,)
    # (3, 3) mapping ECEF axes onto world axes: the globe's attitude, which
    # decides which terrain lies under the station. A spin angle would not do
    # -- the world frame is RTN at the chief, so most of what this rotation
    # does between frames is the chief's own motion around the planet.
    earth_rotation_world: np.ndarray


@dataclass(frozen=True)
class RenderInputs:
    position_world: np.ndarray  # chaser (3,)
    quaternion_bw: np.ndarray  # [w, x, y, z] (4,)
    action: np.ndarray | None = None  # (6,) debug overlays
    lighting: Lighting | None = None  # None = static config lighting

    @classmethod
    def from_view(
        cls,
        view: np.ndarray,
        action: np.ndarray | None = None,
        lighting: Lighting | None = None,
    ) -> RenderInputs:
        """From a canonical 13D view row: pos=view[0:3], q=view[6:10]."""
        view = np.asarray(view)
        return cls(
            position_world=view[0:3],
            quaternion_bw=view[6:10],
            action=action,
            lighting=lighting,
        )
