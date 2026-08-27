"""Camera clips from a stored trajectory, one mp4 per requested view.

Every row is posed through the env's own render adapter, which reads the
epoch and the chief orbit out of the state and so lights the scene as the
simulation would have -- the sun, the eclipse and the terrain under the
station all follow from the file, not from a render setting.
"""

from __future__ import annotations

import os
from contextlib import ExitStack
from pathlib import Path
from typing import Sequence

import numpy as np

from ..envs import ENV_REGISTRY
from .trajectory import SHORT_VIEW_NAMES, Trajectory
from .video import (
    COMPOSITE_KEY,
    KEY_VIEWS,
    keys_for_names,
    render_adapter_for,
    tile_views,
    views_for,
)

ENCODE = dict(codec="libx264", pixelformat="yuv420p", macro_block_size=2)


def frame_indices(steps: int, dt: float, fps: float | None, stride: int) -> np.ndarray:
    """Row indices (into the T+1 states) drawn for a clip at `fps`, thinned by `stride`.

    Nearest-row selection, never interpolation: a row is a state the
    simulation actually reached, and a frame between two of them would be a
    pose it never held. Asking for more frames per second than the file has
    rows is refused for the same reason.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if fps is not None and (not np.isfinite(fps) or fps <= 0.0):
        raise ValueError(f"fps must be finite and > 0, got {fps}")
    source_rate = 1.0 / dt
    if fps is None:
        rows = np.arange(steps + 1)
    else:
        if fps > source_rate + 1e-9:
            raise ValueError(
                f"fps {fps} is faster than the file's {source_rate:g} Hz rows; a clip "
                "cannot show states the simulation never reached"
            )
        times = np.arange(0.0, steps * dt + 1e-9, 1.0 / fps)
        rows = np.unique(np.rint(times / dt).astype(int))
        rows = rows[rows <= steps]
    return rows[::stride]


def render_trajectory_clips(
    traj: Trajectory,
    directory: str | Path,
    views: str | Sequence[str],
    fps: float | None = None,
    stride: int = 1,
) -> dict[str, Path]:
    """Write `<method>_<short>.mp4` under `directory` for each view; return the paths.

    Publication is atomic per file: each clip is renamed into place only once
    every writer closed, and a rename that fails removes the `.part` files
    still waiting rather than leaving them beside the clips already published.
    """
    from ..render.iss_scene import RenderConfig
    from ..render.renderer import ISSRenderer
    import imageio.v2 as iio

    traj.validate()
    spec = ENV_REGISTRY[traj.meta["env"]]
    if not spec.renderable:
        raise ValueError(f"env '{spec.name}' has no render adapter, so it cannot be drawn")
    cfg = spec.config_cls.model_validate(traj.meta["env_config"])
    adapter = render_adapter_for(spec.name, cfg)
    keys = keys_for_names(views)
    draw = views_for(keys)

    rows = frame_indices(traj.steps, traj.dt, fps, stride)
    # No rounding anywhere: the clip is written at exactly the rate its rows
    # were selected at, so `--fps 7.5` plays at 7.5 and a fps below 1 is a
    # rate rather than a zero. `frame_indices` has already refused a fps that
    # is not positive and finite, and `validate()` a dt that is not, so this
    # is positive by construction.
    clip_fps = (1.0 / traj.dt if fps is None else float(fps)) / stride

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    method = traj.meta["method"]
    written: dict[str, Path] = {}
    writers: dict[str, object] = {}
    # Every clip is written under a `.part.mp4` name and renamed only once all
    # of them closed cleanly, so a run that dies mid-episode -- a lost GPU, a
    # full disk -- leaves no half-written clip where a whole one used to be,
    # and no partial file for a later reader to mistake for output.
    partials: list[tuple[Path, Path]] = []
    try:
        with ExitStack() as stack:
            for key in keys:
                short = SHORT_VIEW_NAMES[key.rsplit(".", 1)[-1]]
                final = directory / f"{method}_{short}.mp4"
                partial = final.with_suffix(".part.mp4")
                partials.append((partial, final))
                writers[key] = stack.enter_context(iio.get_writer(partial, fps=clip_fps, **ENCODE))
                written[short] = final

            renderer = ISSRenderer(RenderConfig(**(cfg.render or {})))
            stack.callback(renderer.close)
            for row in rows:
                action = traj.action_phys[row - 1] if row > 0 else None
                rendered = renderer.render_views(adapter(traj.state[row], action), views=draw)
                for key in keys:
                    if key == COMPOSITE_KEY:
                        frame = tile_views(
                            rendered, renderer.cfg.image_height, renderer.cfg.image_width
                        )
                    else:
                        frame = rendered[KEY_VIEWS[key]]
                    writers[key].append_data(frame)
    except BaseException:
        for partial, _ in partials:
            partial.unlink(missing_ok=True)
        raise

    try:
        for partial, final in partials:
            os.replace(partial, final)
    except BaseException:
        for partial, _ in partials:
            partial.unlink(missing_ok=True)
        raise
    return written
