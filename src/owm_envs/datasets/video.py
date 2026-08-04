"""Renders one episode's stored states into video clips.

Kept separate from `lerobot_writer.py` so that module stays the sole lerobot
call site: rendering needs `owm_envs.render`, an optional extra of its own,
so the import here is lazy -- nothing in this package may pull in pygfx at
module level outside the render package itself.

This module also owns which dataset feature a clip is written as: the renderer
knows nothing about datasets, and the writer takes whatever keys it is handed.

Two features are possible. `observation.images.fpv` is the egocentric training
view and is always produced -- downstream training configs name that key
directly, and quickdraw reads exactly one camera per run. The optional
`observation.images.composite` is a single frame tiling all six named views,
for looking at rather than training on: one extra feature and roughly one
extra view's worth of storage, rather than the six a feature per view would
cost.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Iterator

import numpy as np

from ..drivers.types import TrajectoryBatch

FPV_VIEW = "DRAGON_FPV"
FPV_KEY = "observation.images.fpv"
COMPOSITE_KEY = "observation.images.composite"

# Row-major tile order for the composite: the capsule's three views above the
# station's three, each row reading first-person, isometric, top-down.
COMPOSITE_VIEWS: tuple[str, ...] = (
    "DRAGON_FPV",
    "DRAGON_ISO",
    "DRAGON_TOP",
    "ISS_FPV",
    "ISS_ISO",
    "ISS_TOP",
)
COMPOSITE_COLUMNS = 3
COMPOSITE_ROWS = 2

OUTPUT_KEYS: tuple[str, ...] = (FPV_KEY, COMPOSITE_KEY)


def _episode_state(batch: TrajectoryBatch, episode_index: int, t: int) -> np.ndarray:
    # Pose the TRUE state whenever the batch carries one: the camera must not
    # shake with navigation error, and an observation carrying the goal-error
    # block is not renderable geometry (25 dims into a renderer that poses
    # 13). The fallback is measured, not true -- see `render_episode_frames`.
    if batch.true_state is not None:
        return batch.true_state[episode_index, t]
    return batch.observations[episode_index, t][:13]


def tile_views(rendered: dict[str, np.ndarray], height: int, width: int) -> np.ndarray:
    """Tile the six named views into one `(height, width, 3)` uint8 frame.

    The mosaic is deliberately the same size as a single view rather than six
    times it: this is a debug artifact, and a feature six times the area would
    cost more to store and encode than the training view it sits beside. Each
    view is downscaled into its cell, which for a square render config is not
    the view's own aspect -- the picture is squeezed horizontally. That is the
    trade for keeping the whole field of view of all six; cropping to fit would
    lose the edges instead.

    Cell edges are proportional rather than a fixed tile size, so neighbours
    may differ by a pixel and the six of them cover the frame exactly. Flooring
    to a common size instead leaves a black strip whenever the width does not
    divide by three -- 512 px, the default, is one such width.
    """
    from PIL import Image

    rows = [round(i * height / COMPOSITE_ROWS) for i in range(COMPOSITE_ROWS + 1)]
    columns = [round(i * width / COMPOSITE_COLUMNS) for i in range(COMPOSITE_COLUMNS + 1)]
    frame = np.empty((height, width, 3), dtype=np.uint8)
    for index, view in enumerate(COMPOSITE_VIEWS):
        row, column = divmod(index, COMPOSITE_COLUMNS)
        top, bottom = rows[row], rows[row + 1]
        left, right = columns[column], columns[column + 1]
        tile = Image.fromarray(rendered[view]).resize(
            (right - left, bottom - top), Image.BILINEAR
        )
        frame[top:bottom, left:right] = np.asarray(tile)
    return frame


def render_episode_frames(
    batch: TrajectoryBatch,
    episode_index: int,
    cfg: Any,
    composite: bool = False,
    renderer: Any | None = None,
) -> dict[str, np.ndarray]:
    """Render one episode to `(L, H, W, 3)` uint8 clips, keyed by feature name.

    Always produces `observation.images.fpv`; with `composite`, also produces
    `observation.images.composite`, which needs all six views of every frame
    and so costs six renders per frame rather than one.

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
    views = COMPOSITE_VIEWS if composite else (FPV_VIEW,)
    try:
        clips = {
            key: np.empty((length, cfg.image_height, cfg.image_width, 3), dtype=np.uint8)
            for key in (OUTPUT_KEYS if composite else (FPV_KEY,))
        }
        for t in range(length):
            state = _episode_state(batch, episode_index, t)
            action = batch.actions[episode_index, t]
            # One pose serves every view of this frame.
            rendered = renderer.render_views(state, action=action, views=views)
            clips[FPV_KEY][t] = rendered[FPV_VIEW]
            if composite:
                clips[COMPOSITE_KEY][t] = tile_views(
                    rendered, cfg.image_height, cfg.image_width
                )
    finally:
        if owns_renderer:
            renderer.close()
    return clips


def render_batch_frames(
    batch: TrajectoryBatch,
    cfg: Any,
    composite: bool = False,
) -> list[dict[str, np.ndarray]]:
    """Render every episode in `batch` to its clips, keyed by feature name.

    Builds one `ISSRenderer` for the whole batch and reuses it across
    episodes, then closes it exactly once -- see `render_episode_frames`'s
    `renderer` argument for why that matters.
    """
    from ..render.renderer import ISSRenderer

    renderer = ISSRenderer(cfg)
    try:
        return [
            render_episode_frames(batch, i, cfg, composite=composite, renderer=renderer)
            for i in range(batch.num_episodes)
        ]
    finally:
        renderer.close()


_WORKER_RENDERER: Any = None
_WORKER_CFG: Any = None
_WORKER_COMPOSITE: bool = False


def _worker_init(cfg_json: str, composite: bool, gpu_index: int | None) -> None:
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

    global _WORKER_RENDERER, _WORKER_CFG, _WORKER_COMPOSITE
    _WORKER_CFG = RenderConfig.model_validate_json(cfg_json)
    _WORKER_COMPOSITE = bool(composite)
    # Downloads off: the parent resolved all three Earth textures before the
    # pool started. A worker allowed to fetch its own would put them back the
    # moment the parent's fetch failed -- every worker retrying the same
    # multi-gigabyte download at once, and a worker whose retry succeeded
    # rendering at a different Earth resolution from one whose retry did not.
    _WORKER_RENDERER = ISSRenderer(_WORKER_CFG, download_textures=False)


def _worker_render(payload: tuple[np.ndarray, np.ndarray, int]) -> dict[str, np.ndarray]:
    """Render one episode from its own states and actions.

    Takes the trimmed arrays rather than a batch and an index: only this
    episode's real frames cross the pickle boundary, not the whole padded
    batch once per episode.
    """
    states, actions, length = payload
    views = COMPOSITE_VIEWS if _WORKER_COMPOSITE else (FPV_VIEW,)
    keys = OUTPUT_KEYS if _WORKER_COMPOSITE else (FPV_KEY,)
    height, width = _WORKER_CFG.image_height, _WORKER_CFG.image_width
    clips = {key: np.empty((length, height, width, 3), dtype=np.uint8) for key in keys}
    for t in range(length):
        rendered = _WORKER_RENDERER.render_views(states[t], action=actions[t], views=views)
        clips[FPV_KEY][t] = rendered[FPV_VIEW]
        if _WORKER_COMPOSITE:
            clips[COMPOSITE_KEY][t] = tile_views(rendered, height, width)
    return clips


def iter_batch_frames(
    batch: TrajectoryBatch,
    cfg: Any,
    composite: bool = False,
    workers: int = 1,
    gpu_index: int | None = None,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield each episode's clips, keyed by feature name, in episode order.

    Streams: a consumer that writes each episode's clips and drops them holds
    one episode at a time, not the whole split. At 500k frames of 256x256x3 a
    single view over the whole split is ~98 GB, so materialising it is not an
    option; every extra view multiplies that, and the per-episode bound with
    it.

    `workers` > 1 renders episodes in a spawn-context process pool, each
    worker owning its own renderer (~1.9 GiB of VRAM each). The context must
    be spawn: a forked child inherits the parent's wgpu device and JAX state,
    neither of which survives a fork. Clips still come back in episode order,
    and at most one episode per worker is in flight, so peak memory stays
    bounded by the pool rather than by the split. A worker that cannot start
    or dies mid-episode raises `BrokenProcessPool` here, with its own
    traceback on stderr naming the real cause.

    The truth-vs-observation source choice matches `render_episode_frames` --
    see its docstring for why a batch without a truth channel is rendered
    from measured state.
    """
    source = (
        batch.true_state
        if batch.true_state is not None
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
                yield render_episode_frames(
                    batch, episode, cfg, composite=composite, renderer=renderer
                )
        finally:
            renderer.close()
        return

    # ProcessPoolExecutor rather than multiprocessing.Pool: a Pool silently
    # respawns a worker whose initializer raised, for ever, while the parent
    # blocks on a result that never arrives -- a mistyped gpu index would
    # hang a ten-hour render rather than fail it. The executor marks itself
    # broken instead, and every pending clip raises. The same applies to a
    # renderer that dies mid-episode.
    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=_worker_init,
        initargs=(cfg.model_dump_json(), composite, gpu_index),
    )
    try:
        episodes = iter(range(batch.num_episodes))
        # Submitted through a bounded window, not all at once: results are
        # whole clips, and every clip that has been rendered but not yet
        # consumed sits in the parent's memory. Submitting the batch would
        # rebuild the whole split in RAM whenever the writer is slower than
        # the renderers -- the very thing this streams to avoid. One spare
        # episode beyond the worker count keeps every worker fed across a
        # hand-over without buffering more than that.
        pending = deque(
            executor.submit(_worker_render, payload(i))
            for i in itertools.islice(episodes, workers + 1)
        )
        while pending:
            clip = pending.popleft().result()
            for i in itertools.islice(episodes, 1):
                pending.append(executor.submit(_worker_render, payload(i)))
            yield clip
    finally:
        # Abandoning the iterator costs the window's episodes and no more:
        # the ones past it were never submitted. It is not less than that --
        # the executor moves the whole window into its call queue at once,
        # which marks even the spare that no worker has picked up as
        # running, so `cancel_futures` usually has nothing left to cancel.
        executor.shutdown(cancel_futures=True)


def tee_episode_clips(
    frames: Iterator[dict[str, np.ndarray]],
    media_dir: Any,
    fps: int,
    key: str = FPV_KEY,
) -> Iterator[dict[str, np.ndarray]]:
    """Write each episode's `key` clip to its own mp4, and pass it along.

    The dataset stores video the way lerobot does, concatenated into chunk
    files that need its index to cut back apart. One file per episode is what
    anyone reviewing a run actually reaches for, and what the training side's
    own tooling writes, so a copy goes to `media_dir/ep_%04d.mp4`. These are
    auxiliary files, deliberately not dataset features: nothing reads them
    back, and a consumer that wants frames should use the feature.

    A tee rather than a second render: the clips are already in hand on their
    way to the writer, and re-rendering an episode to look at it would double
    the cost of the whole run.

    Standing between the render pool and the writer makes this responsible for
    both of the properties that chain already had. It must not hold an episode
    while the next one is produced -- that is the one-episode video bound, and
    with a composite alongside the training view there is twice as much of it
    to hold -- and closing it must close the pool behind it, since the writer
    now closes this rather than the iterator that owns the workers.
    """
    from pathlib import Path

    import imageio.v3 as iio

    media_dir = Path(media_dir)
    media_dir.mkdir(parents=True, exist_ok=True)
    source = iter(frames)
    episode = 0
    try:
        while True:
            try:
                clips = next(source)
            except StopIteration:
                return
            iio.imwrite(
                media_dir / f"ep_{episode:04d}.mp4",
                clips[key],
                fps=fps,
                codec="libx264",
                # yuv420p and even dimensions: the default yuv444p is rejected
                # by most players, and libx264 cannot subsample an odd frame.
                pixelformat="yuv420p",
                macro_block_size=2,
            )
            yield clips
            # Before pulling the next episode, not after: `for clips in source`
            # would keep this one bound across that call and hold two at once.
            del clips
            episode += 1
    finally:
        close = getattr(source, "close", None)
        if close is not None:
            close()
