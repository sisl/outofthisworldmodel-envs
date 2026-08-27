import numpy as np
import pytest

from owm_envs.datasets.trajectory_render import frame_indices, output_fps


def test_frame_indices_at_source_rate_are_every_row():
    assert frame_indices(steps=5, dt=0.05, fps=None, stride=1).tolist() == [0, 1, 2, 3, 4, 5]


def test_frame_indices_resample_to_nearest_row():
    # 20 Hz source, 10 fps output: every other row.
    assert frame_indices(steps=6, dt=0.05, fps=10, stride=1).tolist() == [0, 2, 4, 6]


def test_frame_indices_stride_thins_the_output():
    assert frame_indices(steps=6, dt=0.05, fps=None, stride=3).tolist() == [0, 3, 6]


def test_output_fps_is_source_rate_by_default():
    assert output_fps(dt=0.05, fps=None) == 20
    assert output_fps(dt=1.0, fps=None) == 1
    assert output_fps(dt=0.05, fps=12.0) == 12


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
