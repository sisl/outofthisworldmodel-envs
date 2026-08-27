"""The station-frame path of one trajectory, as a still and as a video.

The still shows the whole path coloured by speed against the station's
collision hull; the video grows the same path in step with the episode clock
so it can play beside the rendered camera views.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

from ..envs import ENV_REGISTRY  # noqa: E402
from ..envs.common.config import load_collision_boxes  # noqa: E402
from .trajectory import Trajectory  # noqa: E402

FIGSIZE = (10.0, 8.0)
DPI = 100
HULL_COLOR = "#7f7f7f"
HULL_ALPHA = 0.18
START_COLOR = "black"
PORT_COLOR = "red"
CMAP = "viridis"

# Vertex pairs of a unit box's twelve edges, as signs on the half extents.
_CORNERS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8)
          if np.sum(_CORNERS[a] != _CORNERS[b]) == 1]


def box_edges(centers: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    corners = centers[:, None, :] + _CORNERS[None, :, :] * half_extents[:, None, :]
    segments = [[corners[i, a], corners[i, b]] for i in range(len(centers)) for a, b in _EDGES]
    return np.asarray(segments, dtype=float).reshape(-1, 2, 3)


def _hull(traj: Trajectory) -> np.ndarray:
    spec = ENV_REGISTRY[traj.meta["env"]]
    cfg = spec.config_cls.model_validate(traj.meta["env_config"])
    centers, half = load_collision_boxes(cfg.physics.collision_boxes_path)
    return box_edges(centers, half)


def _speeds(traj: Trajectory) -> np.ndarray:
    return np.linalg.norm(traj.rel_view[:, 3:6], axis=1)


def _path_segments(pos: np.ndarray) -> np.ndarray:
    return np.stack([pos[:-1], pos[1:]], axis=1)


def _axes(traj: Trajectory):
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_subplot(projection="3d")
    hull = _hull(traj)
    ax.add_collection3d(Line3DCollection(hull, colors=HULL_COLOR, alpha=HULL_ALPHA, linewidths=0.5))
    pos = traj.rel_view[:, 0:3]
    goal = traj.dock_target[0:3]
    points = np.concatenate([pos, goal[None], hull.reshape(-1, 3)])
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = (hi - lo).max() * 0.5 + 5.0
    mid = 0.5 * (lo + hi)
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.scatter(*pos[0], color=START_COLOR, s=40, label="start", depthshade=False)
    ax.scatter(*goal, color=PORT_COLOR, marker="*", s=200, label="port", depthshade=False)
    meta = traj.meta
    ax.set_title(f"{meta['method']} · {meta['port']} · seed {meta['seed']} · {meta['outcome']}")
    ax.legend(loc="upper right")
    return fig, ax


def _path_collection(traj: Trajectory, upto: int):
    speeds = _speeds(traj)
    segments = _path_segments(traj.rel_view[:upto + 1, 0:3])
    collection = Line3DCollection(segments, cmap=CMAP, linewidths=2.5)
    collection.set_array(speeds[:upto])
    collection.set_clim(0.0, float(speeds.max()) if speeds.max() > 0 else 1.0)
    return collection


def plot_trajectory_png(traj: Trajectory, path: str | Path) -> Path:
    traj.validate()
    fig, ax = _axes(traj)
    collection = _path_collection(traj, traj.steps)
    ax.add_collection3d(collection)
    fig.colorbar(collection, ax=ax, shrink=0.6, pad=0.1, label="speed (m/s)")
    path = Path(path)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_trajectory_video(traj: Trajectory, path: str | Path, fps: int = 10) -> Path:
    import imageio.v2 as iio

    traj.validate()
    duration = traj.steps * traj.dt
    ticks = np.arange(0.0, duration + 1e-9, 1.0 / fps)
    rows = np.minimum(np.rint(ticks / traj.dt).astype(int), traj.steps)
    fig, ax = _axes(traj)
    full = _path_collection(traj, traj.steps)
    fig.colorbar(full, ax=ax, shrink=0.6, pad=0.1, label="speed (m/s)")
    path = Path(path)
    collection = None
    with iio.get_writer(path, fps=fps, codec="libx264",
                         pixelformat="yuv420p", macro_block_size=2) as writer:
        for row in rows:
            if collection is not None:
                collection.remove()
            collection = _path_collection(traj, max(int(row), 1))
            ax.add_collection3d(collection)
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(rgba[..., :3])
    plt.close(fig)
    return path
