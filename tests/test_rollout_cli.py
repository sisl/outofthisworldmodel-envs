import json

import pytest
from typer.testing import CliRunner

from owm_envs.cli import app

runner = CliRunner()


def run(*args):
    """Invoke `rollout` against the iss env with rendering off.

    The iss env rather than iss-numerical: same task, same reward, same
    outcome classification, and a rollout finishes in a fraction of the time.
    Rendering is what makes a rollout slow and needs a GPU, and none of these
    tests are about the video.
    """
    return runner.invoke(app, ["rollout", "--env", "iss", "--no-render", *args])


def test_rollout_writes_a_manifest(tmp_path):
    result = run("--out", str(tmp_path), "--policy", "dock",
                 "--port", "harmony_fwd_pma2", "--episodes", "2", "--steps", "600")
    assert result.exit_code == 0, result.output
    manifest = json.loads((tmp_path / "rollout.json").read_text())
    assert len(manifest["episodes"]) == 2
    assert manifest["policy"] == "dock"
    assert manifest["port"] == "harmony_fwd_pma2"
    for episode in manifest["episodes"]:
        assert set(episode) >= {"seed", "docked", "steps", "position_error_m"}


def test_rollout_writes_the_as_run_env_config(tmp_path):
    result = run("--out", str(tmp_path), "--policy", "dock",
                 "--port", "harmony_fwd_pma2", "--episodes", "1", "--steps", "600")
    assert result.exit_code == 0, result.output
    assert (tmp_path / "env_config.toml").exists()


def test_require_dock_keeps_only_successful_episodes(tmp_path):
    # The full horizon, because this is the one test that needs episodes to
    # actually reach the gate. Do not shorten it to speed the file up -- a
    # --require-dock that silently keeps a non-docked episode is the failure
    # this whole command exists to prevent.
    result = run("--out", str(tmp_path), "--policy", "dock",
                 "--port", "harmony_fwd_pma2", "--episodes", "2",
                 "--require-dock", "--steps", "7200")
    assert result.exit_code == 0, result.output
    manifest = json.loads((tmp_path / "rollout.json").read_text())
    assert len(manifest["episodes"]) == 2
    assert all(e["docked"] for e in manifest["episodes"])


def test_without_require_dock_every_episode_is_kept(tmp_path):
    result = run("--out", str(tmp_path), "--policy", "random",
                 "--episodes", "3", "--steps", "200")
    assert result.exit_code == 0, result.output
    manifest = json.loads((tmp_path / "rollout.json").read_text())
    assert len(manifest["episodes"]) == 3


def test_exhausting_attempts_fails_with_the_counts(tmp_path):
    # A random policy essentially never docks, so the retry loop must give up
    # and say how hard it tried rather than spin.
    result = run("--out", str(tmp_path), "--policy", "random",
                 "--episodes", "1", "--require-dock", "--max-attempts", "4",
                 "--steps", "200")
    assert result.exit_code != 0
    assert "4" in result.output
    assert "docked" in result.output.lower()


def test_a_port_is_rejected_for_a_policy_that_ignores_it(tmp_path):
    result = run("--out", str(tmp_path), "--policy", "orbit",
                 "--port", "harmony_fwd_pma2", "--episodes", "1", "--steps", "200")
    assert result.exit_code != 0
    assert "orbit" in result.output


def test_union_is_not_an_offered_policy(tmp_path):
    result = run("--out", str(tmp_path), "--policy", "union",
                 "--episodes", "1", "--steps", "200")
    assert result.exit_code != 0


def test_frame_stride_is_rejected_without_rendering(tmp_path):
    result = run("--out", str(tmp_path), "--policy", "random",
                 "--episodes", "1", "--steps", "200", "--frame-stride", "10")
    assert result.exit_code != 0


def test_an_unknown_port_is_rejected(tmp_path):
    result = run("--out", str(tmp_path), "--policy", "dock",
                 "--port", "not_a_port", "--episodes", "1", "--steps", "200")
    assert result.exit_code != 0
