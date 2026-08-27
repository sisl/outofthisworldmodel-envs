from pathlib import Path

from typer.testing import CliRunner

from owm_envs.cli import app
from owm_envs.datasets.trajectory import save_trajectory

runner = CliRunner()


def test_plot_trajectory_writes_png_and_mp4(short_trajectory, tmp_path):
    save_trajectory(short_trajectory, tmp_path)
    result = runner.invoke(app, ["plot-trajectory", str(tmp_path), "--fps", "10"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "test_traj.png").exists()
    assert (tmp_path / "test_traj.mp4").exists()
    assert "test_traj.mp4" in result.output


def test_plot_trajectory_refuses_a_directory_without_a_file(tmp_path):
    result = runner.invoke(app, ["plot-trajectory", str(tmp_path)])
    assert result.exit_code != 0
    assert "trajectory.npz" in result.output


def test_render_trajectory_rejects_bad_view(short_trajectory, tmp_path):
    save_trajectory(short_trajectory, tmp_path)
    result = runner.invoke(app, ["render-trajectory", str(tmp_path), "--views", "sideways"])
    assert result.exit_code != 0
    assert "sideways" in result.output
    assert "--views" in result.output
    assert "--render-views" not in result.output


def test_render_trajectory_rejects_bad_stride(short_trajectory, tmp_path):
    save_trajectory(short_trajectory, tmp_path)
    result = runner.invoke(app, ["render-trajectory", str(tmp_path), "--stride", "0"])
    assert result.exit_code != 0
    assert "--stride" in result.output


def test_render_trajectory_rejects_fps_faster_than_rows(short_trajectory, tmp_path, monkeypatch):
    save_trajectory(short_trajectory, tmp_path)

    def _unreached(*args, **kwargs):
        raise AssertionError("probe reached")

    monkeypatch.setattr("owm_envs.render.device.select_gpu", _unreached)
    result = runner.invoke(app, ["render-trajectory", str(tmp_path), "--fps", "60"])
    assert result.exit_code != 0
    assert "--fps" in result.output


def test_render_trajectory_rejects_a_non_positive_fps(short_trajectory, tmp_path, monkeypatch):
    save_trajectory(short_trajectory, tmp_path)

    def _unreached(*args, **kwargs):
        raise AssertionError("probe reached")

    monkeypatch.setattr("owm_envs.render.device.select_gpu", _unreached)
    for value in ("0", "-5", "nan", "inf"):
        result = runner.invoke(app, ["render-trajectory", str(tmp_path), "--fps", value])
        assert result.exit_code != 0, value
        assert "--fps" in result.output
        assert "finite and > 0" in result.output, value
