"""Renders one episode's stored states into an egocentric video clip.

Kept separate from `lerobot_writer.py` so that module stays the sole lerobot
call site: rendering needs `owm_envs.render`, an optional extra of its own,
so the import here is lazy -- nothing in this package may pull in pygfx at
module level outside the render package itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..drivers.types import TrajectoryBatch


def render_episode_frames(
    batch: TrajectoryBatch,
    episode_index: int,
    cfg: Any,
    view: str = "DRAGON_FPV",
) -> np.ndarray:
    """Render one episode's observations to an `(L, H, W, 3)` uint8 clip.

    `L` is `batch.lengths[episode_index]` -- only real, non-padded frames are
    rendered. `cfg` is an `owm_envs.render.iss_scene.RenderConfig`.
    """
    from ..render.renderer import ISSRenderer

    length = int(batch.lengths[episode_index])
    renderer = ISSRenderer(cfg)
    try:
        frames = np.empty((length, cfg.image_height, cfg.image_width, 3), dtype=np.uint8)
        for t in range(length):
            state = batch.observations[episode_index, t]
            action = batch.actions[episode_index, t]
            frames[t] = renderer.render(state, action=action, view=view)
    finally:
        renderer.close()
    return frames
