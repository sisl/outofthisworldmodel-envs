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
    renderer: Any | None = None,
    epoch_offset_s: float = 0.0,
    dt: float = 0.0,
) -> np.ndarray:
    """Render one episode's observations to an `(L, H, W, 3)` uint8 clip.

    `L` is `batch.lengths[episode_index]` -- only real, non-padded frames are
    rendered. `cfg` is an `owm_envs.render.iss_scene.RenderConfig`.

    `renderer`, when given, is a live `ISSRenderer` to render into -- reused
    across episodes so a multi-episode batch pays the cost of loading the
    scene's GLBs, cubemap and Earth textures once, not once per episode. The
    caller owns that renderer's lifetime and must close it itself. When
    omitted, a renderer is built and closed just for this one episode.

    `epoch_offset_s` and `dt` give frame `t`'s simulation time,
    `epoch_offset_s + t * dt`, past the orbit epoch -- read by the renderer
    only when `cfg.sun_from_epoch` is on.
    """
    length = int(batch.lengths[episode_index])
    owns_renderer = renderer is None
    if owns_renderer:
        from ..render.renderer import ISSRenderer

        renderer = ISSRenderer(cfg)
    try:
        frames = np.empty((length, cfg.image_height, cfg.image_width, 3), dtype=np.uint8)
        for t in range(length):
            state = batch.observations[episode_index, t]
            action = batch.actions[episode_index, t]
            frames[t] = renderer.render(
                state, action=action, view=view, t_offset_s=epoch_offset_s + t * dt
            )
    finally:
        if owns_renderer:
            renderer.close()
    return frames


def render_batch_frames(
    batch: TrajectoryBatch,
    cfg: Any,
    view: str = "DRAGON_FPV",
    dt: float = 0.0,
) -> list[np.ndarray]:
    """Render every episode in `batch` to an `(L, H, W, 3)` uint8 clip.

    Builds one `ISSRenderer` for the whole batch and reuses it across
    episodes, then closes it exactly once -- see `render_episode_frames`'s
    `renderer` argument for why that matters.

    Each episode's `epoch_offset_s` comes from `batch.epoch_offsets` when
    present (i.e. `cfg.orbit.enabled` at generation time), else 0.0; see
    `render_episode_frames` for how it combines with `dt`.
    """
    from ..render.renderer import ISSRenderer

    renderer = ISSRenderer(cfg)
    try:
        return [
            render_episode_frames(
                batch,
                i,
                cfg,
                view=view,
                renderer=renderer,
                epoch_offset_s=(
                    float(batch.epoch_offsets[i]) if batch.epoch_offsets is not None else 0.0
                ),
                dt=dt,
            )
            for i in range(batch.num_episodes)
        ]
    finally:
        renderer.close()
