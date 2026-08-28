import numpy as np
import pytest

from owm_envs.datasets.trajectory_plot import (
    box_edges,
    plot_trajectory_png,
    plot_trajectory_video,
)


def test_box_edges_are_twelve_per_box():
    centers = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    half = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    edges = box_edges(centers, half)
    assert edges.shape == (24, 2, 3)
    # Every edge of the first box is axis-aligned and spans a full side.
    lengths = np.linalg.norm(edges[:12, 1] - edges[:12, 0], axis=1)
    assert sorted(set(np.round(lengths, 6))) == [2.0, 4.0, 6.0]


def test_png_is_written(short_trajectory, tmp_path):
    path = plot_trajectory_png(short_trajectory, tmp_path / "test_traj.png")
    assert path.exists() and path.stat().st_size > 0


def test_video_has_one_frame_per_output_tick(short_trajectory, tmp_path):
    import imageio.v3 as iio

    path = plot_trajectory_video(short_trajectory, tmp_path / "test_traj.mp4", fps=10)
    frames = iio.imread(path)
    # 6 steps at dt=0.05 is 0.3 s: at 10 fps that is 4 ticks (0, 0.1, 0.2, 0.3).
    assert frames.shape[0] == 4
    assert frames.shape[1] % 2 == 0 and frames.shape[2] % 2 == 0
    assert list(tmp_path.glob("*.part.*")) == []


def test_video_opens_on_the_start_state_with_no_path_drawn(short_trajectory, tmp_path):
    """The first tick is the reset state: a point flown from, not a path."""
    import imageio.v3 as iio

    path = plot_trajectory_video(short_trajectory, tmp_path / "test_traj.mp4", fps=10)
    frames = iio.imread(path)
    assert not np.array_equal(frames[0], frames[1])


def test_a_failed_video_leaves_no_partial_file(short_trajectory, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    def explode(*args, **kwargs):
        raise RuntimeError("canvas is gone")

    monkeypatch.setattr(plt.Figure, "draw", explode)
    with pytest.raises(RuntimeError, match="canvas is gone"):
        plot_trajectory_video(short_trajectory, tmp_path / "test_traj.mp4", fps=10)
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_video_refuses_a_non_positive_or_non_finite_fps(short_trajectory, tmp_path):
    for value in (0, -5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="fps must be finite"):
            plot_trajectory_video(short_trajectory, tmp_path / "test_traj.mp4", fps=value)
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_a_failed_png_rename_leaves_no_partial_file(short_trajectory, tmp_path, monkeypatch):
    import os

    def refuse(src, dst):
        raise OSError("rename refused")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(OSError, match="rename refused"):
        plot_trajectory_png(short_trajectory, tmp_path / "test_traj.png")
    assert sorted(p.name for p in tmp_path.iterdir()) == []
