"""Renders one episode's stored states into an egocentric video clip.

Kept separate from `lerobot_writer.py` so that module stays the sole lerobot
call site: rendering needs `owm_envs.render`, an optional extra of its own,
so the import here is lazy -- nothing in this package may pull in pygfx at
module level outside the render package itself.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os
from collections import deque
from typing import Any, Iterator

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


_WORKER_RENDERER: Any = None
_WORKER_CFG: Any = None
_WORKER_VIEW: str = "DRAGON_FPV"


def _worker_init(cfg_json: str, view: str, gpu_index: int | None) -> None:
    """Build this worker's own renderer, once, at pool start-up.

    The GPU is chosen first and exactly once: pygfx pins one shared wgpu
    device per process the moment a scene is built, and selecting an adapter
    after that raises. JAX is pushed to the CPU because a worker only replays
    states that are already stored -- it steps no dynamics, so a JAX GPU
    context here would take VRAM from the renderer for nothing.

    `cfg` crosses as JSON rather than as an object because the pool is a
    spawn one: the child re-imports this module from scratch, and a
    RenderConfig round-tripped through its own validator is reconstructed by
    the same code that would have built it in the parent.
    """
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from ..render.device import select_gpu

    select_gpu(gpu_index)
    from ..render.iss_scene import RenderConfig
    from ..render.renderer import ISSRenderer

    global _WORKER_RENDERER, _WORKER_CFG, _WORKER_VIEW
    _WORKER_CFG = RenderConfig.model_validate_json(cfg_json)
    _WORKER_VIEW = view
    _WORKER_RENDERER = ISSRenderer(_WORKER_CFG)


def _worker_render(payload: tuple[np.ndarray, np.ndarray, int]) -> np.ndarray:
    """Render one episode from its own states and actions.

    Takes the trimmed arrays rather than a batch and an index: only this
    episode's real frames cross the pickle boundary, not the whole padded
    batch once per episode.
    """
    states, actions, length = payload
    frames = np.empty(
        (length, _WORKER_CFG.image_height, _WORKER_CFG.image_width, 3), dtype=np.uint8
    )
    for t in range(length):
        frames[t] = _WORKER_RENDERER.render(states[t], action=actions[t], view=_WORKER_VIEW)
    return frames


def iter_batch_frames(
    batch: TrajectoryBatch,
    cfg: Any,
    view: str = "DRAGON_FPV",
    workers: int = 1,
    gpu_index: int | None = None,
) -> Iterator[np.ndarray]:
    """Yield each episode's `(L, H, W, 3)` uint8 clip, in episode order.

    Streams: a consumer that writes each clip and drops it holds one episode
    at a time, not the whole split. At 500k frames of 256x256x3 the whole
    split is ~98 GB, so materialising it is not an option.

    `workers` > 1 renders episodes in a spawn-context process pool, each
    worker owning its own renderer (~1.9 GiB of VRAM each). The context must
    be spawn: a forked child inherits the parent's wgpu device and JAX state,
    neither of which survives a fork. Clips still come back in episode order,
    and at most one episode per worker is in flight, so peak memory stays
    bounded by the pool rather than by the split.

    The truth-vs-observation source choice matches `render_episode_frames` --
    see its docstring for why a batch without a truth channel is rendered
    from measured state.
    """
    source = (
        batch.true_observations
        if batch.true_observations is not None
        else batch.observations[..., :13]
    )

    def payload(episode: int) -> tuple[np.ndarray, np.ndarray, int]:
        length = int(batch.lengths[episode])
        return (
            source[episode, :length].copy(),
            batch.actions[episode, :length].copy(),
            length,
        )

    if workers <= 1:
        from ..render.device import select_gpu
        from ..render.renderer import ISSRenderer

        select_gpu(gpu_index)
        renderer = ISSRenderer(cfg)
        try:
            for episode in range(batch.num_episodes):
                yield render_episode_frames(batch, episode, cfg, view=view, renderer=renderer)
        finally:
            renderer.close()
        return

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        workers, initializer=_worker_init, initargs=(cfg.model_dump_json(), view, gpu_index)
    ) as pool:
        episodes = iter(range(batch.num_episodes))
        # Submitted through a bounded window rather than `imap`: imap's
        # result handler buffers every clip that has completed but not yet
        # been consumed, so a writer slower than the renderers would rebuild
        # the whole split in RAM -- the very thing this streams to avoid. One
        # spare episode beyond the worker count keeps the pool busy across a
        # hand-over without buffering more than that.
        pending = deque(
            pool.apply_async(_worker_render, (payload(i),))
            for i in itertools.islice(episodes, workers + 1)
        )
        while pending:
            clip = pending.popleft().get()
            for i in itertools.islice(episodes, 1):
                pending.append(pool.apply_async(_worker_render, (payload(i),)))
            yield clip
