#!/usr/bin/env python
"""Plot a run's episode trajectories in 3D, split by policy type, one plotly
HTML page and one matplotlib PNG per (split, policy).

Every episode's chaser path in the ISS-fixed frame, plus the dock poses those
episodes were flying to and the station origin, on equal-aspect axes so the
approach geometry is undistorted. Episode endpoints are marked by how the
episode ended -- docked, collision, escaped the domain, or timeout -- each
with its own colour AND marker shape, so the verdict on a run is readable at
a glance: dock episodes should end in green diamonds on their port, orbit
episodes should trace closed loops to a grey timeout ring, and random
episodes should wander (some away from the station entirely).

Outcomes are inferred from the as-run configs written next to the data:
`terminated`/`truncated` flags plus the final state against the run's dock
gates (position + velocity) and domain radius. The attitude gates are not
re-checked -- an episode that terminated inside the position/velocity gates
terminated by docking in practice.

Paths are drawn from `state_vector`, the true simulator state, falling back to
`observation_vector` for a run with no truth column; the page title says which.

Episodes are coloured along one blue ramp by episode index, which separates
neighbouring paths without implying eight kinds of episode; identity comes from
the legend (click an entry to isolate one path) and the hover readout.

Usage:
    uv run python scripts/plot_trajectories_3d.py RUN_DIR
    uv run python scripts/plot_trajectories_3d.py RUN_DIR --out logs/traj
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Chart surface and ink for the dark page this renders on, and the two ends of
# the ordinal blue ramp episode colour interpolates along -- the light end is
# the ramp's step 100, the dark end its step 600, which is the darkest step
# that still clears 2:1 against this surface.
SURFACE = "#1a1a19"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#c3c2b7"
GRID = "#3a3a37"
ZEROLINE = "#52514e"
RAMP_LIGHT = (0xCD, 0xE2, 0xFB)
RAMP_DARK = (0x18, 0x4F, 0x95)
DOCK_COLOR = "#eda100"
ORIGIN_COLOR = "#e34948"

# Endpoint status colours (validated against this dark surface) paired with
# distinct marker shapes, so an outcome is never encoded by colour alone.
# Keys are the labels used in legends and filenames.
OUTCOMES = {
    "docked": {"color": "#0ca30c", "plotly": "diamond", "mpl": "D"},
    "collision": {"color": "#d03b3b", "plotly": "x", "mpl": "x"},
    "escaped": {"color": "#fab219", "plotly": "square-open", "mpl": "s"},
    "timeout": {"color": TEXT_SECONDARY, "plotly": "circle-open", "mpl": "o"},
}

# Positional order of _build_union's branches; policy_id indexes this list in
# a union split. Non-union splits are a single group named by the split's own
# policy type.
UNION_POLICIES = ("random", "orbit", "dock")

# Points drawn per episode. 64 paths at this length is a page of a few
# megabytes that a browser rotates smoothly; the full-length episodes the
# datasets hold are not.
MAX_POINTS = 1000


def positive_int(text: str) -> int:
    """An argparse type for counts that are meaningless below one."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


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


def _cell(value) -> np.ndarray:
    """A lerobot cell -- possibly nested arrays -- as one flat float vector."""
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = np.stack([np.ravel(np.asarray(v)) for v in arr])
    return np.ravel(arr).astype(float)


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
        _cell(row)[:3] for row in frame.drop_duplicates("episode_index")["dock_target"]
    ])
    return np.unique(targets[np.isfinite(targets).all(axis=1)], axis=0)


def load_run_meta(run_dir: Path) -> dict:
    """The as-run settings outcome classification and policy grouping need.

    `env_config.yaml` and `generation_config.yaml` are written next to the
    data by `owm-envs generate`; a run directory without them (or without a
    split's entry) still plots, with outcomes degraded to terminated/timeout
    and every episode grouped under one unlabelled policy.
    """
    meta = {"max_range_m": None, "dock_max_distance_m": None,
            "dock_max_velocity_m_s": None, "split_policy": {}}
    env_path = run_dir / "env_config.yaml"
    if env_path.is_file():
        env = yaml.safe_load(env_path.read_text()) or {}
        meta["max_range_m"] = env.get("max_range_m")
        dock = env.get("dock") or {}
        meta["dock_max_distance_m"] = dock.get("max_distance_m")
        meta["dock_max_velocity_m_s"] = dock.get("max_velocity_m_s")

    run_policy = None
    policy_path = run_dir / "policy_config.yaml"
    if policy_path.is_file():
        run_policy = (yaml.safe_load(policy_path.read_text()) or {}).get("type")
    gen_path = run_dir / "generation_config.yaml"
    if gen_path.is_file():
        gen = yaml.safe_load(gen_path.read_text()) or {}
        for name, spec in (gen.get("splits") or {}).items():
            split_policy = (spec.get("policy") or {}).get("type") or run_policy
            meta["split_policy"][name] = split_policy
    return meta


def classify_outcomes(frame: pd.DataFrame, meta: dict, source: str) -> pd.Series:
    """episode_index -> outcome label, from each episode's final frame.

    `terminated` separates the absorbing outcomes from `timeout`; which
    absorbing outcome is recovered from the final state: inside the run's
    dock gates is `docked`, outside the domain radius is `escaped`, anything
    else terminated by hitting the station.
    """
    last = frame.drop_duplicates("episode_index", keep="last").set_index("episode_index")
    outcomes = {}
    for episode, row in last.iterrows():
        if not bool(row.get("terminated", False)):
            outcomes[episode] = "timeout"
            continue
        state = _cell(row[source])
        pos, vel = state[0:3], state[3:6]
        label = "collision"
        max_range = meta["max_range_m"]
        if max_range is not None and np.linalg.norm(pos) >= 0.999 * max_range:
            label = "escaped"
        max_dist = meta["dock_max_distance_m"]
        if max_dist is not None and "dock_target" in frame.columns:
            target = _cell(row["dock_target"])[:3]
            max_vel = meta["dock_max_velocity_m_s"] or np.inf
            if (np.isfinite(target).all()
                    and np.linalg.norm(pos - target) <= max_dist * 1.001
                    and np.linalg.norm(vel) <= max_vel * 1.001):
                label = "docked"
        outcomes[episode] = label
    return pd.Series(outcomes, name="outcome")


def policy_groups(frame: pd.DataFrame, split_policy: str | None) -> dict[str, list]:
    """policy label -> ordered episode ids, splitting a union split by the
    recorded per-episode `policy_id` (which indexes UNION_POLICIES)."""
    episodes = frame.drop_duplicates("episode_index")
    if split_policy != "union" or "policy_id" not in frame.columns:
        label = split_policy or "policy"
        return {label: episodes["episode_index"].tolist()}
    groups: dict[str, list] = {}
    for _, row in episodes.iterrows():
        pid = int(_cell(row["policy_id"])[0])
        label = UNION_POLICIES[pid] if 0 <= pid < len(UNION_POLICIES) else f"policy{pid}"
        groups.setdefault(label, []).append(row["episode_index"])
    return groups


def _thin(rows: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """Every path point of a long episode, thinned by a stride that always
    keeps the final frame -- the one point that answers where it ended up."""
    step = max(1, -(-len(rows) // max_points))
    keep = np.arange(0, len(rows), step)
    if keep[-1] != len(rows) - 1:
        keep = np.append(keep, len(rows) - 1)
    return rows.iloc[keep]


def _paths(frame: pd.DataFrame, episodes: list, source: str, max_points: int):
    """[(episode, (K,3) path, frame indices)] for the group, in episode order."""
    out = []
    for episode in episodes:
        rows = _thin(frame[frame["episode_index"] == episode], max_points)
        path = np.stack([_cell(v)[:3] for v in rows[source]])
        out.append((episode, path, rows["frame_index"].to_numpy()))
    return out


def plotly_figure(paths, outcomes: pd.Series, docks: np.ndarray, title: str):
    import plotly.graph_objects as go

    fig = go.Figure()
    for i, (episode, path, frames) in enumerate(paths):
        fig.add_trace(go.Scatter3d(
            x=path[:, 0], y=path[:, 1], z=path[:, 2],
            mode="lines", name=f"ep {episode}",
            line={"color": episode_color(i / max(len(paths) - 1, 1)), "width": 2},
            customdata=frames,
            hovertemplate=(f"ep {episode} step %{{customdata}}<br>"
                           "%{x:.1f}, %{y:.1f}, %{z:.1f} m<extra></extra>"),
        ))

    starts = np.stack([path[0] for _, path, _ in paths])
    fig.add_trace(go.Scatter3d(
        x=starts[:, 0], y=starts[:, 1], z=starts[:, 2],
        mode="markers", name="episode start",
        marker={"size": 4, "color": TEXT_SECONDARY},
        hovertemplate="episode start<br>%{x:.1f}, %{y:.1f}, %{z:.1f} m<extra></extra>",
    ))

    # One trace per outcome present, so the legend doubles as the tally and a
    # click isolates e.g. every collision endpoint at once.
    for label, style in OUTCOMES.items():
        eps = [(ep, path) for ep, path, _ in paths if outcomes.get(ep) == label]
        if not eps:
            continue
        ends = np.stack([path[-1] for _, path in eps])
        fig.add_trace(go.Scatter3d(
            x=ends[:, 0], y=ends[:, 1], z=ends[:, 2],
            mode="markers", name=f"{label} ({len(eps)})",
            marker={"size": 6, "symbol": style["plotly"], "color": style["color"]},
            customdata=[ep for ep, _ in eps],
            hovertemplate=(f"{label} -- ep %{{customdata}}<br>"
                           "%{x:.1f}, %{y:.1f}, %{z:.1f} m<extra></extra>"),
        ))

    if len(docks):
        fig.add_trace(go.Scatter3d(
            x=docks[:, 0], y=docks[:, 1], z=docks[:, 2],
            mode="markers", name="dock target",
            marker={"size": 8, "symbol": "diamond-open", "color": DOCK_COLOR},
            hovertemplate="dock target<br>%{x:.2f}, %{y:.2f}, %{z:.2f} m<extra></extra>",
        ))
    fig.add_trace(go.Scatter3d(
        x=[0.0], y=[0.0], z=[0.0], mode="markers", name="ISS origin",
        marker={"size": 8, "symbol": "cross", "color": ORIGIN_COLOR},
        hovertemplate="ISS origin<extra></extra>",
    ))

    axis = {"backgroundcolor": SURFACE, "gridcolor": GRID,
            "zerolinecolor": ZEROLINE, "color": TEXT_SECONDARY}
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


def matplotlib_figure(paths, outcomes: pd.Series, docks: np.ndarray, title: str):
    """A static counterpart: the 3D view plus the three orthographic slices,
    which is where a circle actually reads as a circle."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 10), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 3, height_ratios=(2.2, 1.0), hspace=0.16, wspace=0.28)
    ax3d = fig.add_subplot(grid[0, :], projection="3d")
    planes = [  # (axes indices into xyz, x label, y label)
        ((0, 1), "x (m)", "y (m)"),
        ((0, 2), "x (m)", "z (m)"),
        ((1, 2), "y (m)", "z (m)"),
    ]
    ax2d = [fig.add_subplot(grid[1, k]) for k in range(3)]

    for i, (episode, path, _) in enumerate(paths):
        color = episode_color(i / max(len(paths) - 1, 1))
        ax3d.plot(path[:, 0], path[:, 1], path[:, 2], color=color, lw=0.9)
        for ax, ((a, b), _, _) in zip(ax2d, planes):
            ax.plot(path[:, a], path[:, b], color=color, lw=0.8)

    starts = np.stack([path[0] for _, path, _ in paths])
    ax3d.scatter(*starts.T, s=12, color=TEXT_SECONDARY, marker="o",
                 facecolors="none", label="episode start")
    for label, style in OUTCOMES.items():
        eps = [path[-1] for ep, path, _ in paths if outcomes.get(ep) == label]
        if not eps:
            continue
        ends = np.stack(eps)
        open_marker = style["mpl"] in ("s", "o")
        kw = {"facecolors": "none", "edgecolors": style["color"]} if open_marker \
            else {"color": style["color"]}
        ax3d.scatter(*ends.T, s=36, marker=style["mpl"],
                     label=f"{label} ({len(eps)})", **kw)
        for ax, ((a, b), _, _) in zip(ax2d, planes):
            ax.scatter(ends[:, a], ends[:, b], s=24, marker=style["mpl"], **kw)
    if len(docks):
        ax3d.scatter(*docks.T, s=48, marker="D", facecolors="none",
                     edgecolors=DOCK_COLOR, label="dock target")
        for ax, ((a, b), _, _) in zip(ax2d, planes):
            ax.scatter(docks[:, a], docks[:, b], s=32, marker="D",
                       facecolors="none", edgecolors=DOCK_COLOR)
    ax3d.scatter([0.0], [0.0], [0.0], s=48, marker="+", color=ORIGIN_COLOR,
                 label="ISS origin")
    for ax in ax2d:
        ax.scatter([0.0], [0.0], s=32, marker="+", color=ORIGIN_COLOR)

    every = np.concatenate([path for _, path, _ in paths] + ([docks] if len(docks) else []))
    span = np.abs(every).max() * 1.05
    ax3d.set_xlim(-span, span); ax3d.set_ylim(-span, span); ax3d.set_zlim(-span, span)
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.set_facecolor(SURFACE)
    ax3d.set_xlabel("x (m)"); ax3d.set_ylabel("y (m)"); ax3d.set_zlabel("z (m)")
    for item in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        item.label.set_color(TEXT_SECONDARY)
        item.set_pane_color((0, 0, 0, 0))
        item._axinfo["grid"]["color"] = GRID
    ax3d.tick_params(colors=TEXT_SECONDARY)
    ax3d.legend(loc="upper left", facecolor=SURFACE, edgecolor=GRID,
                labelcolor=TEXT_SECONDARY, fontsize=8)

    for ax, ((_, _), xlabel, ylabel) in zip(ax2d, planes):
        ax.set_xlim(-span, span); ax.set_ylim(-span, span)
        ax.set_aspect("equal")
        ax.set_facecolor(SURFACE)
        ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=8)
        ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=8)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(color=GRID, lw=0.4)
    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=11)
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="run directory written by owm-envs generate")
    parser.add_argument(
        "--out", type=Path, default=Path("trajectories"),
        help="directory to write trajectories_<split>_<policy>.{html,png} into "
             "(default ./trajectories); kept out of the run directory so it is "
             "never published with the dataset",
    )
    parser.add_argument(
        "--split", action="append", dest="splits",
        help="split to plot (repeatable); default is every split in the run",
    )
    parser.add_argument(
        "--max-episodes", type=positive_int, default=64,
        help="episodes to draw per split, lowest index first (default 64). A "
             "published split holds thousands, which no browser will open.",
    )
    parser.add_argument(
        "--max-points", type=positive_int, default=MAX_POINTS,
        help=f"points drawn per episode (default {MAX_POINTS}); longer paths are "
             "thinned by a stride, and the hover still reports the true step index",
    )
    args = parser.parse_args()

    try:
        import matplotlib
        import plotly  # noqa: F401
    except ImportError as missing:
        raise SystemExit(
            f"{missing.name} is required by this script and ships in the base "
            "install: rebuild the project environment with 'uv sync'."
        )
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not args.run_dir.is_dir():
        raise SystemExit(f"{args.run_dir} is not a directory")
    splits = args.splits or sorted(
        p.name for p in args.run_dir.iterdir() if (p / "data").is_dir()
    )
    if not splits:
        raise SystemExit(f"no splits found under {args.run_dir}")
    args.out.mkdir(parents=True, exist_ok=True)

    meta = load_run_meta(args.run_dir)
    for split in splits:
        frame, total = load_split(args.run_dir, split, args.max_episodes)
        source = "state_vector" if "state_vector" in frame.columns else "observation_vector"
        outcomes = classify_outcomes(frame, meta, source)
        groups = policy_groups(frame, meta["split_policy"].get(split))
        for policy, episodes in sorted(groups.items()):
            rows = frame[frame["episode_index"].isin(episodes)]
            paths = _paths(rows, episodes, source, args.max_points)
            docks = dock_positions(rows)
            tally = ", ".join(
                f"{label} {sum(outcomes.get(ep) == label for ep in episodes)}"
                for label in OUTCOMES
                if any(outcomes.get(ep) == label for ep in episodes)
            )
            shown = (f"{len(episodes)} episodes" if len(groups) == 1 and len(episodes) == total
                     else f"{len(episodes)} of {total} episodes")
            title = (f"{args.run_dir.name} / {split} / {policy} -- {shown} "
                     f"({source.replace('_', ' ')}) -- {tally}")

            html = args.out / f"trajectories_{split}_{policy}.html"
            plotly_figure(paths, outcomes, docks, title).write_html(
                html, include_plotlyjs="inline")
            png = args.out / f"trajectories_{split}_{policy}.png"
            fig = matplotlib_figure(paths, outcomes, docks, title)
            fig.savefig(png, dpi=150, facecolor=SURFACE, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {html} and {png} ({shown}; {tally})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
