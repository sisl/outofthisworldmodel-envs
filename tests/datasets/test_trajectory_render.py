import os
from pathlib import Path

import numpy as np
import pytest

from owm_envs.datasets.trajectory_render import frame_indices, render_trajectory_clips


def test_frame_indices_at_source_rate_are_every_row():
    assert frame_indices(steps=5, dt=0.05, fps=None, stride=1).tolist() == [0, 1, 2, 3, 4, 5]


def test_frame_indices_resample_to_nearest_row():
    # 20 Hz source, 10 fps output: every other row.
    assert frame_indices(steps=6, dt=0.05, fps=10, stride=1).tolist() == [0, 2, 4, 6]


def test_frame_indices_stride_thins_the_output():
    assert frame_indices(steps=6, dt=0.05, fps=None, stride=3).tolist() == [0, 3, 6]


def test_frame_indices_refuse_upsampling():
    with pytest.raises(ValueError, match="faster"):
        frame_indices(steps=5, dt=1.0, fps=30, stride=1)


def test_render_writes_one_clip_per_view(short_trajectory, tmp_path):
    pygfx = pytest.importorskip("pygfx")
    wgpu = pytest.importorskip("wgpu")
    if wgpu.gpu.request_adapter_sync(power_preference="high-performance") is None:
        pytest.skip("no wgpu adapter")
    import imageio.v3 as iio

    from owm_envs.datasets.trajectory_render import render_trajectory_clips

    written = render_trajectory_clips(short_trajectory, tmp_path, "fpv,dragon_iso", stride=3)
    assert set(written) == {"fpv", "iso"}
    for short, path in written.items():
        assert path.name == f"test_{short}.mp4"
        frames = iio.imread(path)
        assert frames.shape[0] == len(frame_indices(short_trajectory.steps, 0.05, None, 3))
        assert frames.shape[-1] == 3
    assert list(tmp_path.glob("*.part.mp4")) == []


def test_frame_indices_refuse_a_non_positive_fps():
    with pytest.raises(ValueError, match="fps must be finite"):
        frame_indices(steps=5, dt=0.05, fps=0.0, stride=1)
    with pytest.raises(ValueError, match="fps must be finite"):
        frame_indices(steps=5, dt=0.05, fps=-2.0, stride=1)


def test_frame_indices_refuse_a_non_finite_fps():
    with pytest.raises(ValueError, match="fps must be finite"):
        frame_indices(steps=5, dt=0.05, fps=float("nan"), stride=1)
    with pytest.raises(ValueError, match="fps must be finite"):
        frame_indices(steps=5, dt=0.05, fps=float("inf"), stride=1)


def test_a_failed_render_leaves_no_partial_clips(short_trajectory, tmp_path, monkeypatch):
    """A renderer that dies mid-episode must not leave half-written mp4s behind."""

    def explode(*args, **kwargs):
        raise RuntimeError("no GPU today")

    monkeypatch.setattr("owm_envs.render.renderer.ISSRenderer", explode)
    with pytest.raises(RuntimeError, match="no GPU today"):
        render_trajectory_clips(short_trajectory, tmp_path, "fpv,dragon_iso")
    assert sorted(p.name for p in tmp_path.iterdir()) == []


class _FakeWriter:
    """Stands in for an imageio writer: creates its file on open, like the real one."""

    def __init__(self, path, fps):
        self.path = Path(path)
        self.fps = fps
        self.frames = 0
        self.path.write_bytes(b"partial")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def append_data(self, frame):
        self.frames += 1


class _FakeRenderer:
    class cfg:
        image_height = 64
        image_width = 64

    def __init__(self, config):
        pass

    def render_views(self, inputs, views):
        return {view: np.zeros((64, 64, 3), dtype=np.uint8) for view in views}

    def close(self):
        pass


@pytest.fixture
def fake_render(monkeypatch):
    """Drive render_trajectory_clips without a GPU or an ffmpeg subprocess."""
    writers = {}

    def get_writer(path, fps, **kwargs):
        writer = _FakeWriter(path, fps)
        writers[Path(path).name] = writer
        return writer

    import imageio.v2 as iio

    monkeypatch.setattr(iio, "get_writer", get_writer)
    monkeypatch.setattr("owm_envs.render.renderer.ISSRenderer", _FakeRenderer)
    monkeypatch.setattr(
        "owm_envs.datasets.trajectory_render.render_adapter_for",
        lambda name, cfg: (lambda state, action=None: None),
    )
    return writers


def test_a_fps_below_one_selects_every_fiftieth_row():
    # 0.4 fps on a 20 Hz file is one frame every 2.5 s, i.e. every 50th row.
    assert frame_indices(steps=200, dt=0.05, fps=0.4, stride=1).tolist() == [0, 50, 100, 150, 200]


def test_the_writer_is_given_the_requested_fractional_fps(short_trajectory, tmp_path, fake_render):
    render_trajectory_clips(short_trajectory, tmp_path, "fpv", fps=0.4)
    assert [w.fps for w in fake_render.values()] == [0.4]


def test_a_fractional_fps_is_not_rounded(short_trajectory, tmp_path, fake_render):
    render_trajectory_clips(short_trajectory, tmp_path, "fpv", fps=7.5)
    assert [w.fps for w in fake_render.values()] == [7.5]


def test_a_stride_divides_the_clip_rate(short_trajectory, tmp_path, fake_render):
    # 20 Hz rows kept every third: the clip plays at 20/3, not at 7.
    render_trajectory_clips(short_trajectory, tmp_path, "fpv", stride=3)
    assert [w.fps for w in fake_render.values()] == [20.0 / 3.0]


def test_a_failed_rename_leaves_no_partial_clips(short_trajectory, tmp_path, fake_render):
    """A rename that fails must not leave `.part` files beside the published clips."""
    real_replace = os.replace
    calls = []

    def replace(src, dst):
        calls.append(src)
        if len(calls) == 2:
            raise OSError("rename refused")
        return real_replace(src, dst)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "replace", replace)
        with pytest.raises(OSError, match="rename refused"):
            render_trajectory_clips(short_trajectory, tmp_path, "fpv,dragon_iso")

    assert list(tmp_path.glob("*.part.mp4")) == []
    # The clip published before the failure stays where it was put.
    assert (tmp_path / "test_fpv.mp4").exists()
