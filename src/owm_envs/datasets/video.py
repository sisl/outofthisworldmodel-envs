"""Renders one episode's stored states into video clips.

Kept separate from `lerobot_writer.py` so that module stays the sole lerobot
call site: rendering needs `owm_envs.render`, an optional extra of its own,
so the import here is lazy -- nothing in this package may pull in pygfx at
module level outside the render package itself. `render.inputs` is the
exception that proves it: it is numpy and nothing else, which is what lets a
render worker pose a frame without a GPU stack.

This module also owns which dataset feature a clip is written as: the renderer
knows nothing about datasets, and the writer takes whatever keys it is handed.

Each named camera has a feature of its own, and `observation.images.composite`
is a seventh: one frame tiling all six, for scrubbing a run without opening
six streams side by side. `observation.images.fpv` is the egocentric training
view -- downstream training configs name that key directly, and quickdraw reads
exactly one camera per run -- so it keeps its name and meaning whatever else is
asked for, and the extra keys are additive.

Which of them a run produces is the caller's choice, because they are not
priced alike: the six cameras come from one pose apiece, so asking for more
views costs draws rather than poses, while every key asked for is a video
stream of its own to encode and store.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import warnings
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from ..drivers.types import TrajectoryBatch
from ..render.inputs import RenderInputs

# What an env's render adapter is, from this module's side: the thing that
# turns one of that env's stored rows into a posable frame. `render/inputs.py`
# is numpy-only, so naming the type here costs no pygfx and no jax.
RenderAdapter = Callable[[np.ndarray, np.ndarray | None], RenderInputs]

FPV_VIEW = "DRAGON_FPV"
FPV_KEY = "observation.images.fpv"
COMPOSITE_KEY = "observation.images.composite"

# Row-major tile order for the composite: the capsule's three views above the
# station's three, each row reading first-person, isometric, top-down. Also the
# order every per-view key is reported in, so a run's features read the same
# way its mosaic does.
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

# `DRAGON_FPV` is `observation.images.fpv` rather than `dragon_fpv`: that key
# is the training contract, named directly by downstream configs, and renaming
# it to match its siblings would break every one of them for tidiness.
VIEW_KEYS: dict[str, str] = {
    "DRAGON_FPV": FPV_KEY,
    "DRAGON_ISO": "observation.images.dragon_iso",
    "DRAGON_TOP": "observation.images.dragon_top",
    "ISS_FPV": "observation.images.iss_fpv",
    "ISS_ISO": "observation.images.iss_iso",
    "ISS_TOP": "observation.images.iss_top",
}
KEY_VIEWS: dict[str, str] = {key: view for view, key in VIEW_KEYS.items()}

OUTPUT_KEYS: tuple[str, ...] = tuple(VIEW_KEYS[view] for view in COMPOSITE_VIEWS) + (
    COMPOSITE_KEY,
)


# The short tail of each key -- `fpv`, `iss_top`, `composite` -- which is how
# a run selects views on the command line and in a generation config. The keys
# themselves are the dataset's interface; these are the names people type.
VIEW_NAMES: tuple[str, ...] = tuple(key.rsplit(".", 1)[-1] for key in OUTPUT_KEYS)
_KEYS_BY_NAME: dict[str, str] = dict(zip(VIEW_NAMES, OUTPUT_KEYS))


def parse_view_names(spec: str | Sequence[str]) -> tuple[str, ...]:
    """A view selection -> the short names it means, canonically ordered.

    Takes either a comma-joined string or a sequence, since the same selection
    arrives from a command line as one and from a config file as the other.
    `all` stands for every name, and an empty selection means the same rather
    than nothing: a run rendering no view would pay the render cost and write
    no video for it.

    Ordered by `OUTPUT_KEYS` rather than by how it was written, so two runs
    asking for the same set record and declare it identically.
    """
    parts = spec.replace("+", ",").split(",") if isinstance(spec, str) else list(spec)
    names = [name for part in parts if (name := str(part).strip().lower())]
    # Checked even when `all` is among them: `all,typo` is a typo the caller
    # wants to hear about, and reading `all` first would swallow it.
    unknown = [name for name in names if name != "all" and name not in _KEYS_BY_NAME]
    if unknown:
        raise ValueError(
            f"unknown view {unknown[0]!r}; expected 'all' or a comma-joined list of "
            f"{', '.join(VIEW_NAMES)}"
        )
    if not names or "all" in names:
        return VIEW_NAMES
    wanted = set(names)
    return tuple(name for name in VIEW_NAMES if name in wanted)


def keys_for_names(names: str | Sequence[str]) -> tuple[str, ...]:
    """A view selection -> the dataset feature keys it writes."""
    return tuple(_KEYS_BY_NAME[name] for name in parse_view_names(names))


def views_for(keys: Sequence[str]) -> tuple[str, ...]:
    """Which cameras have to be drawn to produce `keys`.

    The composite is not a camera: it needs all six drawn whether or not their
    own keys were asked for. Reported in `COMPOSITE_VIEWS` order so a frame's
    draws do not reshuffle with the order the keys arrived in.
    """
    unknown = [key for key in keys if key not in KEY_VIEWS and key != COMPOSITE_KEY]
    if unknown:
        raise ValueError(f"unknown video feature {unknown[0]!r}; expected one of {OUTPUT_KEYS}")
    if COMPOSITE_KEY in keys:
        return COMPOSITE_VIEWS
    wanted = {KEY_VIEWS[key] for key in keys}
    return tuple(view for view in COMPOSITE_VIEWS if view in wanted)


def _state_source(batch: TrajectoryBatch, state_dim: int) -> np.ndarray:
    """The rows to pose from, `(episodes, steps, state_dim)`.

    Truth whenever the batch carries it: the camera must not shake with
    navigation error. Failing that, the first `state_dim` observation dims,
    which is the measured state -- an observation may carry a goal-error block
    past the state itself, and those extra dims are not geometry any adapter
    reads. The width is the producing env's, not a constant: an iss row is 13
    wide and an iss-hcw row 15, and an env whose adapter reads past 13 would
    be handed a truncated row.

    That fallback assumes the observation's own first `state_dim` columns ARE
    the state, in the state's own layout -- true for every env before
    iss-numerical, whose `observe` hook can narrow, reorder, or otherwise
    reshape what gets recorded (`envs/iss_numerical/observe.py`), so its
    observation can come out NARROWER than `state_dim` even though nothing
    was truncated. Slicing such a batch here would silently hand the adapter
    a truncated, wrongly-laid-out row instead of the state it expects, so a
    batch that reaches this fallback too narrow to hold `state_dim` columns
    fails loudly instead: an env whose observation can do this must be
    rendered from its truth channel, not this one.

    That guard is a WIDTH check and promises no more: a reshaped row as wide
    as the state -- iss-numerical's three absolute modes are all 21 -- passes
    it and would be posed from columns in the wrong order. Nothing in the
    tree can reach that, because every env whose observation can be reshaped
    also publishes its true state to the drivers, so its batches always
    arrive with a truth channel and never reach this fallback at all. Width
    is what this function can see of a row it is otherwise handed
    uninterpreted, and layout is what it cannot.

    One rule for both the in-process and the pooled path, so a split rendered
    across workers is the same video as one rendered here.
    """
    if batch.true_state is not None:
        return batch.true_state
    width = batch.observations.shape[-1]
    if width < state_dim:
        raise ValueError(
            f"no truth channel to render from, and the observation is only "
            f"{width} columns wide -- narrower than this env's {state_dim}-wide "
            f"state, so its first {state_dim} columns cannot be the state in "
            f"the state's own layout. Rerun with a truth channel recorded "
            f"(the video path needs one for any env whose observation can be "
            f"narrower than its state, e.g. iss-numerical's observation modes)."
        )
    return batch.observations[..., :state_dim]


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


def _fill_frame(
    clips: dict[str, np.ndarray],
    rendered: dict[str, np.ndarray],
    t: int,
    keys: Sequence[str],
    height: int,
    width: int,
) -> None:
    """Write frame `t` of every requested key from one frame's draws."""
    for key in keys:
        if key == COMPOSITE_KEY:
            clips[key][t] = tile_views(rendered, height, width)
        else:
            clips[key][t] = rendered[KEY_VIEWS[key]]


def render_episode_frames(
    batch: TrajectoryBatch,
    episode_index: int,
    cfg: Any,
    keys: Sequence[str] = (FPV_KEY,),
    renderer: Any | None = None,
    *,
    adapter: RenderAdapter,
    state_dim: int,
) -> dict[str, np.ndarray]:
    """Render one episode to `(L, H, W, 3)` uint8 clips, keyed by feature name.

    `adapter` is the source environment's render adapter -- see
    `EnvSpec.make_render_adapter` -- which is what turns each stored row into
    something the renderer can pose, and `state_dim` is that same
    environment's state width (`EnvSpec.layout.state_dim`), which is how wide
    a row the adapter is owed. Neither has a default because getting either
    wrong is silently wrong video rather than an error: every caller says
    which environment produced the batch it is handing over.

    `keys` names the features to produce, out of `OUTPUT_KEYS`. Only the
    cameras they need are drawn, so the per-frame cost tracks the number of
    distinct views asked for rather than the number of keys -- except that
    `observation.images.composite` needs all six whatever else is requested.

    `L` is `batch.lengths[episode_index]` -- only real, non-padded frames are
    rendered. `cfg` is an `owm_envs.render.iss_scene.RenderConfig`.

    A batch whose source cannot supply truth -- a foreign env that emits no
    `info["state"]`, or a batch built before the truth channel existed -- is
    rendered from its first `state_dim` observation dims instead. That is the
    MEASURED state, so such a clip does shake with navigation error; it is the
    best available for that batch, not an equivalent substitute.

    `renderer`, when given, is a live `ISSRenderer` to render into -- reused
    across episodes so a multi-episode batch pays the cost of loading the
    scene's GLBs, cubemap and Earth textures once, not once per episode. The
    caller owns that renderer's lifetime and must close it itself. When
    omitted, a renderer is built and closed just for this one episode.
    """
    length = int(batch.lengths[episode_index])
    views = views_for(keys)
    owns_renderer = renderer is None
    if owns_renderer:
        from ..render.renderer import ISSRenderer

        renderer = ISSRenderer(cfg)
    try:
        clips = {
            key: np.empty((length, cfg.image_height, cfg.image_width, 3), dtype=np.uint8)
            for key in keys
        }
        source = _state_source(batch, state_dim)
        for t in range(length):
            state = source[episode_index, t]
            action = batch.actions[episode_index, t]
            # One pose serves every view of this frame.
            rendered = renderer.render_views(adapter(state, action), views=views)
            _fill_frame(clips, rendered, t, keys, cfg.image_height, cfg.image_width)
    finally:
        if owns_renderer:
            renderer.close()
    return clips


def render_batch_frames(
    batch: TrajectoryBatch,
    cfg: Any,
    keys: Sequence[str] = (FPV_KEY,),
    *,
    adapter: RenderAdapter,
    state_dim: int,
) -> list[dict[str, np.ndarray]]:
    """Render every episode in `batch` to its clips, keyed by feature name.

    Builds one `ISSRenderer` for the whole batch and reuses it across
    episodes, then closes it exactly once -- see `render_episode_frames`'s
    `renderer` argument for why that matters, and its `adapter` and
    `state_dim` for what those are.
    """
    from ..render.renderer import ISSRenderer

    renderer = ISSRenderer(cfg)
    try:
        return [
            render_episode_frames(
                batch, i, cfg, keys=keys, renderer=renderer, adapter=adapter, state_dim=state_dim
            )
            for i in range(batch.num_episodes)
        ]
    finally:
        renderer.close()


def render_adapter_for(env_name: str, env_cfg: Any) -> RenderAdapter:
    """The named environment's render adapter, built for `env_cfg`.

    The one way this module obtains an adapter, in the parent and in a worker
    alike: an adapter is a property of the environment that produced a batch,
    and the registry is where that lives.
    """
    from ..envs import ENV_REGISTRY

    spec = ENV_REGISTRY[env_name]
    if spec.make_render_adapter is None:
        raise ValueError(
            f"environment {env_name!r} has no render adapter, so its rows cannot "
            "be posed; see EnvSpec.make_render_adapter"
        )
    return spec.make_render_adapter(env_cfg)


_WORKER_RENDERER: Any = None
_WORKER_CFG: Any = None
_WORKER_KEYS: tuple[str, ...] = (FPV_KEY,)
_WORKER_ADAPTER: RenderAdapter | None = None


def _worker_init(
    cfg_json: str,
    keys: Sequence[str],
    gpu_index: int | None,
    env_name: str,
    env_cfg_json: str,
) -> None:
    """Build this worker's own renderer and render adapter, once, at pool
    start-up.

    The GPU is chosen first and exactly once: pygfx pins one shared wgpu
    device per process the moment a scene is built, and selecting an adapter
    after that raises. JAX is pushed to the CPU because a worker only replays
    states that are already stored -- it steps no dynamics, so a JAX GPU
    context here would take VRAM from the renderer for nothing. The render
    adapter is built after that, since reaching the registry imports the
    environments and so imports JAX.

    Both configs cross as JSON rather than as objects because the pool is a
    spawn one: the child re-imports this module from scratch, and a config
    round-tripped through its own validator is reconstructed by the same code
    that would have built it in the parent. The adapter is rebuilt here rather
    than sent, because what the parent holds is a callable bound to a config
    the child does not have.
    """
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from ..render.device import select_gpu

    select_gpu(gpu_index)
    from ..envs import ENV_REGISTRY
    from ..render.iss_scene import RenderConfig
    from ..render.renderer import ISSRenderer

    global _WORKER_RENDERER, _WORKER_CFG, _WORKER_KEYS, _WORKER_ADAPTER
    _WORKER_CFG = RenderConfig.model_validate_json(cfg_json)
    _WORKER_KEYS = tuple(keys)
    _WORKER_ADAPTER = render_adapter_for(
        env_name, ENV_REGISTRY[env_name].config_cls.model_validate_json(env_cfg_json)
    )
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
    keys = _WORKER_KEYS
    views = views_for(keys)
    height, width = _WORKER_CFG.image_height, _WORKER_CFG.image_width
    clips = {key: np.empty((length, height, width, 3), dtype=np.uint8) for key in keys}
    for t in range(length):
        rendered = _WORKER_RENDERER.render_views(
            _WORKER_ADAPTER(states[t], actions[t]), views=views
        )
        _fill_frame(clips, rendered, t, keys, height, width)
    return clips


def iter_batch_frames(
    batch: TrajectoryBatch,
    cfg: Any,
    keys: Sequence[str] = (FPV_KEY,),
    workers: int = 1,
    gpu_index: int | None = None,
    *,
    env_name: str,
    env_cfg: Any,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield each episode's clips, keyed by feature name, in episode order.

    `(env_name, env_cfg)` rather than a built adapter, and neither with a
    default: an adapter is a callable bound to a config, and a spawned worker
    can be handed neither. The pair is what a worker can rebuild its own
    adapter from, so it is what this takes on both paths.

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

    Both paths pose from `_state_source` at the named env's own state width --
    see its docstring for why a batch without a truth channel is rendered from
    measured state, and why that width cannot be a constant. Sharing the one
    rule is what makes a split rendered across workers the same video as one
    rendered in this process.
    """
    from ..envs import ENV_REGISTRY

    state_dim = ENV_REGISTRY[env_name].layout.state_dim
    source = _state_source(batch, state_dim)

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
        adapter = render_adapter_for(env_name, env_cfg)
        renderer = ISSRenderer(cfg)
        try:
            for episode in range(batch.num_episodes):
                yield render_episode_frames(
                    batch,
                    episode,
                    cfg,
                    keys=keys,
                    renderer=renderer,
                    adapter=adapter,
                    state_dim=state_dim,
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
        initargs=(
            cfg.model_dump_json(),
            tuple(keys),
            gpu_index,
            env_name,
            env_cfg.model_dump_json(),
        ),
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
    media_root: Any,
    split: str,
    fps: int,
) -> Iterator[dict[str, np.ndarray]]:
    """Write every view's per-episode mp4 under `media_root`, and pass it along.

    The dataset stores video the way lerobot does, concatenated into chunk
    files that need its index to cut back apart. One file per episode is what
    anyone reviewing a run actually reaches for, and what the training side's
    own tooling writes, so a copy of each view goes to
    `media_root/<view>/<split>/ep_%04d.mp4` -- the view named by the tail of
    its feature key. Whatever the run rendered gets a copy; there is no second
    selection here, since a view worth a dataset feature is worth reviewing.

    These are auxiliary files, deliberately not dataset features: nothing reads
    them back, and a consumer that wants frames should use the feature. They
    are not free -- the same frames encoded a second time, and `push` ships
    them with the run -- but they are much cheaper than a second copy: cut per
    episode and encoded at imageio's libx264 defaults, they came to a fifth of
    the dataset's own video on a measured seven-view run.

    A tee rather than a second render: the clips are already in hand on their
    way to the writer, and re-rendering an episode to look at it would double
    the cost of the whole run.

    Standing between the render pool and the writer makes this responsible for
    both of the properties that chain already had. It must not hold an episode
    while the next one is produced -- that is the one-episode video bound, and
    with every view rendered there is a multiple of it to hold -- and closing
    it must close the pool behind it, since the writer now closes this rather
    than the iterator that owns the workers.
    """
    from pathlib import Path

    import imageio.v3 as iio

    media_root = Path(media_root)
    source = iter(frames)
    episode = 0
    try:
        while True:
            try:
                clips = next(source)
            except StopIteration:
                return
            # Indexed rather than unpacked: a `for key, clip in ...` binding
            # outlives the loop and would hold one view of this episode across
            # the next `next(source)`, on top of the episode itself.
            for key in clips:
                path = media_root / key.rsplit(".", 1)[-1] / split / f"ep_{episode:04d}.mp4"
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    iio.imwrite(
                        path,
                        clips[key],
                        fps=fps,
                        codec="libx264",
                        # yuv420p and even dimensions: the default yuv444p is
                        # rejected by most players, and libx264 cannot
                        # subsample an odd frame.
                        pixelformat="yuv420p",
                        macro_block_size=2,
                    )
                except Exception as error:
                    # These clips are auxiliary, so failing to write one must
                    # never take down the dataset write running downstream of
                    # this -- that would strand the episodes already on disk.
                    # Per clip rather than latching off: whatever failed may
                    # not recur, and it says nothing about the other views of
                    # the same episode, which are separate files.
                    warnings.warn(
                        f"could not write debug clip {path}: {error!r}",
                        stacklevel=2,
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


def tee_episode_stills(
    frames: Iterator[dict[str, np.ndarray]],
    stills_root: Any,
    stride: int,
) -> Iterator[dict[str, np.ndarray]]:
    """Write every `stride`-th frame of every view as a PNG, and pass it along.

    A tee like `tee_episode_clips`, for the same reason: the frames are already
    in hand, and re-rendering an episode to pull stills out of it would double
    the cost of the run. Written under
    `stills_root/ep_%04d/<view>_%06d.png`, indexed by the frame's own
    position in the episode so a still can be located in the clip beside it.

    `stride` of 0 writes nothing and is the default everywhere: stills are for
    figures, not for review, and a run that did not ask for them should not pay
    for them.

    In a rollout this wraps `tee_episode_clips` rather than the render pool
    directly, so it carries the same two obligations that chain already had:
    it must not hold an episode while the next one is produced, and closing
    it must close the iterator behind it, since the consumer now closes this
    rather than the tee underneath it.
    """
    if stride < 0:
        raise ValueError(f"stride must be >= 0, got {stride}")

    from pathlib import Path

    import imageio.v3 as iio

    stills_root = Path(stills_root)
    source = iter(frames)
    episode = 0
    try:
        while True:
            try:
                clips = next(source)
            except StopIteration:
                return
            if stride > 0:
                episode_dir = stills_root / f"ep_{episode:04d}"
                episode_dir.mkdir(parents=True, exist_ok=True)
                # Indexed rather than unpacked, as in `tee_episode_clips`: a
                # `for key, clip in ...` binding would outlive this loop and
                # survive the `del clips` below, holding one view across the
                # next `next(source)` on top of the episode itself.
                for key in clips:
                    view = key.rsplit(".", 1)[-1]
                    for frame_index in range(0, int(clips[key].shape[0]), stride):
                        iio.imwrite(
                            episode_dir / f"{view}_{frame_index:06d}.png",
                            clips[key][frame_index],
                        )
            yield clips
            del clips
            episode += 1
    finally:
        close = getattr(source, "close", None)
        if close is not None:
            close()
