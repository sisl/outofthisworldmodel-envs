#!/usr/bin/env python
"""Plot a run's episode trajectories in 3D, one HTML page per split.

Every episode's chaser path in the ISS-fixed frame, plus the dock poses those
episodes were flying to and the station origin, on equal-aspect axes so the
approach geometry is undistorted. This is the eyeball check on a generated run:
every path should start on the start-radius sphere, and a dock or union policy's
episodes should be seen converging on the port they were assigned.

Paths are drawn from `state_vector`, the true simulator state, falling back to
`observation_vector` for a run with no truth column; the page title says which.

Episodes are coloured along one blue ramp by episode index, which separates
neighbouring paths without implying eight kinds of episode; identity comes from
the legend (click an entry to isolate one path) and the hover readout.

Usage:
    uv run --extra datasets python scripts/plot_trajectories_3d.py RUN_DIR
    uv run --extra datasets python scripts/plot_trajectories_3d.py RUN_DIR --out logs/traj
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Chart surface and ink for the dark page this renders on, and the two ends of
# the ordinal blue ramp episode colour interpolates along -- the light end is
# the ramp's step 100, the dark end its step 600, which is the darkest step
# that still clears 2:1 against this surface.
# Points drawn per episode. 64 paths at this length is a page of a few
# megabytes that a browser rotates smoothly; the full 7200-step episodes the
# datasets hold are not.
MAX_POINTS = 1000

SURFACE = "#1a1a19"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#c3c2b7"
RAMP_LIGHT = (0xCD, 0xE2, 0xFB)
RAMP_DARK = (0x18, 0x4F, 0x95)
DOCK_COLOR = "#eda100"
ORIGIN_COLOR = "#e34948"


def episode_color(position: float) -> str:
    """Ramp colour at `position` in [0, 1], light end first."""
    r, g, b = (round(lo + (hi - lo) * position) for lo, hi in zip(RAMP_LIGHT, RAMP_DARK))
    return f"#{r:02x}{g:02x}{b:02x}"


def load_split(run_dir: Path, split: str, max_episodes: int) -> tuple[pd.DataFrame, int]:
    """One split's frames in episode and frame order, and its episode count.

    Globs `**/*.parquet` under the split's data directory so a change in
    lerobot's chunk/file layout does not silently read nothing. Only the first
    `max_episodes` episodes are kept: a published split holds thousands, and
    every point of every one of them goes into the page as text.

    `episode_index` is read on its own first so that the files holding none of
    those episodes are never read at all. A 500k-frame split is hundreds of
    megabytes of per-frame vectors, each of which pandas holds as its own numpy
    object; reading all of it to draw the first 64 episodes is what this avoids.
    """
    files = sorted((run_dir / split / "data").rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {run_dir / split / 'data'}")

    per_file = [pd.read_parquet(f, columns=["episode_index"])["episode_index"] for f in files]
    episodes = np.unique(np.concatenate([ids.to_numpy() for ids in per_file]))
    kept = set(episodes[:max_episodes].tolist())

    parts = []
    for path, ids in zip(files, per_file):
        if kept.intersection(ids.to_numpy().tolist()):
            part = pd.read_parquet(path)
            parts.append(part[part["episode_index"].isin(kept)])
    frame = pd.concat(parts, ignore_index=True)
    return frame.sort_values(["episode_index", "frame_index"], ignore_index=True), len(episodes)


def dock_positions(frame: pd.DataFrame) -> np.ndarray:
    """The distinct dock-target positions the split's episodes flew to.

    A batch whose driver could not supply a target writes all-NaN rows; those
    are dropped rather than drawn at the origin.

    `dock_target` is declared to lerobot at shape (1, 7) (see
    datasets/lerobot_writer), so each cell reads back nested rather than flat.
    """
    if "dock_target" not in frame.columns:
        return np.empty((0, 3))
    targets = np.stack([
        np.ravel(np.stack(np.asarray(row))).astype(float)[:3]
        for row in frame.drop_duplicates("episode_index")["dock_target"]
    ])
    return np.unique(targets[np.isfinite(targets).all(axis=1)], axis=0)


def figure(frame: pd.DataFrame, title: str, source: str, max_points: int = MAX_POINTS):
    import plotly.graph_objects as go

    groups = list(frame.groupby("episode_index", sort=True))
    starts = []
    fig = go.Figure()
    for i, (episode, rows) in enumerate(groups):
        # Episodes run to 7200 steps, and every point of every path is written
        # into the page as text. Thinning by a stride keeps the shape of a
        # trajectory that a 20 Hz sim oversamples anyway, and the hover keeps
        # reporting the true step index because frame_index is thinned with it.
        step = max(1, -(-len(rows) // max_points))
        keep = np.arange(0, len(rows), step)
        if keep[-1] != len(rows) - 1:
            # The last frame is where the episode ended up, which is the
            # question being asked of a docking approach; a stride that steps
            # over it is the one point that must not be thinned away.
            keep = np.append(keep, len(rows) - 1)
        rows = rows.iloc[keep]
        path = np.stack(rows[source].to_numpy())[:, 0:3]
        starts.append(path[0])
        fig.add_trace(go.Scatter3d(
            x=path[:, 0], y=path[:, 1], z=path[:, 2],
            mode="lines", name=f"ep {episode}",
            line={"color": episode_color(i / max(len(groups) - 1, 1)), "width": 2},
            customdata=rows["frame_index"].to_numpy(),
            hovertemplate=(f"ep {episode} step %{{customdata}}<br>"
                           "%{x:.1f}, %{y:.1f}, %{z:.1f} m<extra></extra>"),
        ))

    # One trace for every episode's first frame rather than a marker per path:
    # it reads as an annotation on the paths, keeps one legend entry, and puts
    # the start-radius sphere on screen -- a path that did not begin on it is
    # a reset that leaked state from the previous episode.
    origins = np.stack(starts)
    fig.add_trace(go.Scatter3d(
        x=origins[:, 0], y=origins[:, 1], z=origins[:, 2],
        mode="markers", name="episode start",
        marker={"size": 4, "color": TEXT_SECONDARY},
        hovertemplate="episode start<br>%{x:.1f}, %{y:.1f}, %{z:.1f} m<extra></extra>",
    ))

    docks = dock_positions(frame)
    if len(docks):
        fig.add_trace(go.Scatter3d(
            x=docks[:, 0], y=docks[:, 1], z=docks[:, 2],
            mode="markers", name="dock target",
            marker={"size": 8, "symbol": "diamond", "color": DOCK_COLOR},
            hovertemplate="dock target<br>%{x:.2f}, %{y:.2f}, %{z:.2f} m<extra></extra>",
        ))
    fig.add_trace(go.Scatter3d(
        x=[0.0], y=[0.0], z=[0.0], mode="markers", name="ISS origin",
        marker={"size": 8, "symbol": "cross", "color": ORIGIN_COLOR},
        hovertemplate="ISS origin<extra></extra>",
    ))

    axis = {"backgroundcolor": SURFACE, "gridcolor": "#3a3a37",
            "zerolinecolor": "#52514e", "color": TEXT_SECONDARY}
    fig.update_layout(
        title={"text": title, "font": {"color": TEXT_PRIMARY}},
        paper_bgcolor=SURFACE, font={"color": TEXT_SECONDARY},
        legend={"font": {"color": TEXT_SECONDARY}},
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        scene={"aspectmode": "data",
               "xaxis": {"title": "x (m)", **axis},
               "yaxis": {"title": "y (m)", **axis},
               "zaxis": {"title": "z (m)", **axis}},
    )
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="run directory written by owm-envs generate")
    parser.add_argument(
        "--out", type=Path, default=Path("trajectories"),
        help="directory to write trajectories_<split>.html into (default ./trajectories); "
             "kept out of the run directory so it is never published with the dataset",
    )
    parser.add_argument(
        "--split", action="append", dest="splits",
        help="split to plot (repeatable); default is every split in the run",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=64,
        help="episodes to draw per split, lowest index first (default 64). A "
             "published split holds thousands, which no browser will open.",
    )
    parser.add_argument(
        "--max-points", type=int, default=MAX_POINTS,
        help=f"points drawn per episode (default {MAX_POINTS}); longer paths are "
             "thinned by a stride, and the hover still reports the true step index",
    )
    args = parser.parse_args()

    try:
        import plotly  # noqa: F401
    except ImportError:
        raise SystemExit(
            "plotly is required by this script: install the datasets extra "
            "(uv sync --extra datasets) or run it with 'uv run --extra datasets'."
        )

    if not args.run_dir.is_dir():
        raise SystemExit(f"{args.run_dir} is not a directory")
    splits = args.splits or sorted(
        p.name for p in args.run_dir.iterdir() if (p / "data").is_dir()
    )
    if not splits:
        raise SystemExit(f"no splits found under {args.run_dir}")
    args.out.mkdir(parents=True, exist_ok=True)

    for split in splits:
        frame, total = load_split(args.run_dir, split, args.max_episodes)
        source = "state_vector" if "state_vector" in frame.columns else "observation_vector"
        drawn = frame["episode_index"].nunique()
        shown = f"{drawn} episodes" if drawn == total else f"{drawn} of {total} episodes"
        title = (f"{args.run_dir.name} / {split} -- {shown}, "
                 f"{len(frame)} frames ({source.replace('_', ' ')})")
        path = args.out / f"trajectories_{split}.html"
        figure(frame, title, source, args.max_points).write_html(path, include_plotlyjs="inline")
        print(f"wrote {path} ({shown})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
