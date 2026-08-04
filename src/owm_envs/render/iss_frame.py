"""The ISS station frame: where the render asset sits in world coordinates.

Free of rendering imports on purpose: asset tooling
(`scripts/check_iss_asset.py`) reads these constants without the render
extra installed, so nothing here may import pygfx, pylinalg or imageio.
"""

from __future__ import annotations

import numpy as np

# Where the station asset's own origin has to move so the station sits at the
# world origin, expressed in the frame the upright rotation leaves it in.
#
# Pinned rather than recomputed. It was originally the mean of every visible
# ISS vertex, which is a tessellation-weighted quantity: unlinking the PMM
# moves it 0.085 m, and any future remesh or added module would move it again.
# The dock success gate is 0.1 m wide and the collision hull and dock pose in
# `envs/iss` are both authored against this exact frame, so the offset is a
# constant of the environment, not something the renderer gets to re-derive.
ISS_RECENTRE_OFFSET: tuple[float, float, float] = (-0.2330057, -0.1656835, 2.6015187)

# Rotation the loaded assets need so their +Y faces this environment's +Z.
UPRIGHT_EULER_XYZ: tuple[float, float, float] = (np.pi / 2.0, 0.0, 0.0)
