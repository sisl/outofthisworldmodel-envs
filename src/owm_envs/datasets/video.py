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
) -> np.ndarray:
    """Render one episode's true states to an `(L, H, W, 3)` uint8 clip.

    `L` is `batch.lengths[episode_index]` -- only real, non-padded frames are
    rendered. `cfg` is an `owm_envs.render.iss_scene.RenderConfig`.

    A batch whose source cannot supply truth -- a foreign env that emits no
    `info["state"]`, or a batch built before the truth channel existed -- is
    rendered from its first 13 observation dims instead. That is the MEASURED
    state, so such a clip does shake with navigation error; it is the best
    available for that batch, not an equivalent substitute.

    `renderer`, when given, is a live `ISSRenderer` to render into -- reused
    across episodes so a multi-episode batch pays the cost of loading the
    scene's GLBs, cubemap and Earth textures once, not once per episode. The
    caller owns that renderer's lifetime and must close it itself. When
    omitted, a renderer is built and closed just for this one episode.
    """
    length = int(batch.lengths[episode_index])
    owns_renderer = renderer is None
    if owns_renderer:
        from ..render.renderer import ISSRenderer

        renderer = ISSRenderer(cfg)
    try:
        frames = np.empty((length, cfg.image_height, cfg.image_width, 3), dtype=np.uint8)
        for t in range(length):
            # Pose the TRUE state whenever the batch carries one: the camera
            # must not shake with navigation error, and an observation
            # carrying the goal-error block is not renderable geometry (25
            # dims into a renderer that poses 13). The fallback is measured,
            # not true -- see this function's docstring.
            if batch.true_state is not None:
                state = batch.true_state[episode_index, t]
            else:
                state = batch.observations[episode_index, t][:13]
            action = batch.actions[episode_index, t]
            frames[t] = renderer.render(state, action=action, view=view)
    finally:
        if owns_renderer:
            renderer.close()
    return frames


def render_batch_frames(
    batch: TrajectoryBatch,
    cfg: Any,
    view: str = "DRAGON_FPV",
) -> list[np.ndarray]:
    """Render every episode in `batch` to an `(L, H, W, 3)` uint8 clip.

    Builds one `ISSRenderer` for the whole batch and reuses it across
    episodes, then closes it exactly once -- see `render_episode_frames`'s
    `renderer` argument for why that matters.
    """
    from ..render.renderer import ISSRenderer

    renderer = ISSRenderer(cfg)
    try:
        return [
            render_episode_frames(batch, i, cfg, view=view, renderer=renderer)
            for i in range(batch.num_episodes)
        ]
    finally:
        renderer.close()
