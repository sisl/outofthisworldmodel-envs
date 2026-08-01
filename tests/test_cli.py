import builtins
import json

import pytest
import typer
from typer.testing import CliRunner

from owm_envs.cli import app, _parse_split_flags
from owm_envs.datasets.stats import SplitSpec
from owm_envs.envs.iss.policies import PolicyConfig

runner = CliRunner()


def test_list_shows_the_iss_environment():
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "iss" in result.stdout.lower()


def test_generate_writes_a_run_directory(tmp_path):
    out = tmp_path / "run"
    result = runner.invoke(
        app,
        ["generate", "--out", str(out), "--episodes", "2", "--steps", "8",
         "--policy", "dock", "--num-envs", "2", "--driver", "vector", "--no-lerobot"],
    )
    assert result.exit_code == 0, result.stdout
    for name in ("normalization_stats.json", "dataset_card.json", "summary.json",
                 "env_config.yaml", "policy_config.yaml"):
        assert (out / name).exists(), f"missing {name}"


def test_generated_run_records_the_policy_actually_used(tmp_path):
    out = tmp_path / "run"
    runner.invoke(
        app,
        ["generate", "--out", str(out), "--episodes", "2", "--steps", "8",
         "--policy", "orbit", "--num-envs", "2", "--driver", "vector", "--no-lerobot"],
    )
    card = json.loads((out / "dataset_card.json").read_text())
    assert card["policy_type"] == "orbit"

    from owm_envs.envs.iss.policies import PolicyConfig

    assert PolicyConfig.from_yaml(out / "policy_config.yaml").type == "orbit"


def test_both_drivers_are_selectable(tmp_path):
    for driver in ("vector", "scan"):
        out = tmp_path / driver
        result = runner.invoke(
            app,
            ["generate", "--out", str(out), "--episodes", "2", "--steps", "8",
             "--policy", "dock", "--num-envs", "2", "--driver", driver, "--no-lerobot"],
        )
        assert result.exit_code == 0, result.stdout
        assert (out / "summary.json").exists()


def test_auto_driver_picks_the_fused_path_for_iss(tmp_path):
    out = tmp_path / "auto"
    result = runner.invoke(
        app,
        ["generate", "--out", str(out), "--episodes", "2", "--steps", "8",
         "--policy", "dock", "--num-envs", "2", "--driver", "auto", "--no-lerobot"],
    )
    assert result.exit_code == 0, result.stdout
    # ISS is JAX-traceable, so auto must resolve to the scan driver and say so.
    assert "scan" in result.stdout.lower()


def test_config_file_is_loaded_and_recorded(tmp_path):
    from owm_envs.envs.iss.config import ISSConfig, PhysicsConfig

    cfg_path = tmp_path / "env.yaml"
    ISSConfig(physics=PhysicsConfig(start_radius_m=175.0)).to_yaml(cfg_path)

    out = tmp_path / "run"
    result = runner.invoke(
        app,
        ["generate", "--out", str(out), "--config", str(cfg_path), "--episodes", "2",
         "--steps", "8", "--policy", "dock", "--num-envs", "2", "--driver", "vector",
         "--no-lerobot"],
    )
    assert result.exit_code == 0, result.stdout
    assert ISSConfig.from_yaml(out / "env_config.yaml").physics.start_radius_m == 175.0


def test_summary_reports_the_episode_count_requested(tmp_path):
    out = tmp_path / "run"
    runner.invoke(
        app,
        ["generate", "--out", str(out), "--episodes", "3", "--steps", "8",
         "--policy", "dock", "--num-envs", "2", "--driver", "vector", "--no-lerobot"],
    )
    summary = json.loads((out / "summary.json").read_text())
    assert summary["counts"]["train"]["episodes"] == 3


def test_fps_defaults_to_the_simulation_rate(tmp_path):
    # One frame is recorded per step, so a dataset stamped with any rate other
    # than 1/dt reports an inter-frame interval the physics never used.
    out = tmp_path / "run"
    result = runner.invoke(
        app,
        ["generate", "--out", str(out), "--episodes", "2", "--steps", "8",
         "--policy", "dock", "--num-envs", "2", "--driver", "vector", "--no-lerobot"],
    )
    assert result.exit_code == 0, result.stdout
    card = json.loads((out / "dataset_card.json").read_text())
    assert card["fps"] == round(1.0 / card["dt"])


def test_fps_default_tracks_a_non_default_dt(tmp_path):
    from owm_envs.envs.iss.config import ISSConfig

    cfg_path = tmp_path / "env.yaml"
    ISSConfig(dt=0.01).to_yaml(cfg_path)

    out = tmp_path / "run"
    result = runner.invoke(
        app,
        ["generate", "--out", str(out), "--config", str(cfg_path), "--episodes", "2",
         "--steps", "8", "--policy", "dock", "--num-envs", "2", "--driver", "vector",
         "--no-lerobot"],
    )
    assert result.exit_code == 0, result.stdout
    assert json.loads((out / "dataset_card.json").read_text())["fps"] == 100


def test_explicit_fps_is_honoured_but_warns_when_it_contradicts_dt(tmp_path):
    out = tmp_path / "run"
    result = runner.invoke(
        app,
        ["generate", "--out", str(out), "--episodes", "2", "--steps", "8",
         "--policy", "dock", "--num-envs", "2", "--driver", "vector", "--no-lerobot",
         "--fps", "24"],
    )
    assert result.exit_code == 0, result.stdout
    assert json.loads((out / "dataset_card.json").read_text())["fps"] == 24
    assert "does not match the simulation rate" in result.output


def test_explicit_fps_matching_the_simulation_rate_is_silent(tmp_path):
    # Pins that the warnings above discriminate: an implementation that warned
    # on every explicit --fps would satisfy them without being right.
    out = tmp_path / "run"
    result = runner.invoke(
        app,
        ["generate", "--out", str(out), "--episodes", "2", "--steps", "8",
         "--policy", "dock", "--num-envs", "2", "--driver", "vector", "--no-lerobot",
         "--fps", "20"],
    )
    assert result.exit_code == 0, result.stdout
    assert "[warn]" not in result.output


def test_explicit_fps_warns_when_the_rate_is_not_whole(tmp_path):
    # dt = 0.03 is 33.33... frames per second. 33 is the nearest integer but
    # still wrong, and it is the value most likely to be passed by hand, so
    # accepting it silently would stamp inaccurate metadata with no signal.
    from owm_envs.envs.iss.config import ISSConfig

    cfg_path = tmp_path / "env.yaml"
    ISSConfig(dt=0.03).to_yaml(cfg_path)

    out = tmp_path / "run"
    result = runner.invoke(
        app,
        ["generate", "--out", str(out), "--config", str(cfg_path), "--episodes", "2",
         "--steps", "8", "--policy", "dock", "--num-envs", "2", "--driver", "vector",
         "--no-lerobot", "--fps", "33"],
    )
    assert result.exit_code == 0, result.stdout
    assert json.loads((out / "dataset_card.json").read_text())["fps"] == 33
    assert "cannot match the simulation rate" in result.output


def test_fps_default_is_refused_when_the_rate_is_not_whole(tmp_path):
    # dt = 0.03 gives 33.33... frames per second, which no integer fps
    # represents; guessing 33 would silently misstate the interval.
    from owm_envs.envs.iss.config import ISSConfig

    cfg_path = tmp_path / "env.yaml"
    ISSConfig(dt=0.03).to_yaml(cfg_path)

    out = tmp_path / "run"
    result = runner.invoke(
        app,
        ["generate", "--out", str(out), "--config", str(cfg_path), "--episodes", "1",
         "--steps", "4", "--policy", "dock", "--num-envs", "1", "--driver", "vector",
         "--no-lerobot"],
    )
    assert result.exit_code != 0
    assert "--fps" in result.output
    assert not out.exists()


def test_failed_dataset_write_leaves_no_completed_run_marker(tmp_path, monkeypatch):
    # The run metadata is what makes a directory look like a finished run.
    # If the LeRobot write fails, downstream automation must be able to tell
    # -- so none of those files may exist.
    import owm_envs.datasets.lerobot_writer as writer

    def boom(*args, **kwargs):
        raise RuntimeError("encoder exploded")

    monkeypatch.setattr(writer, "write_lerobot_split", boom)

    out = tmp_path / "run"
    result = runner.invoke(
        app,
        ["generate", "--out", str(out), "--episodes", "2", "--steps", "8",
         "--policy", "dock", "--num-envs", "2", "--driver", "vector"],
    )
    # The run must have got as far as the dataset write and failed there,
    # otherwise this passes trivially by never reaching the metadata step.
    assert isinstance(result.exception, RuntimeError)
    assert "encoder exploded" in str(result.exception)
    for name in ("normalization_stats.json", "dataset_card.json", "summary.json",
                 "env_config.yaml", "policy_config.yaml"):
        assert not (out / name).exists(), f"{name} survived a failed run"


def test_non_positive_num_envs_is_rejected(tmp_path):
    out = tmp_path / "run"
    result = runner.invoke(
        app,
        ["generate", "--out", str(out), "--episodes", "1", "--steps", "4",
         "--policy", "dock", "--num-envs", "0", "--driver", "scan", "--no-lerobot"],
    )
    assert result.exit_code != 0
    assert "num-envs" in result.output.lower()
    # Failed before doing any rollout work or writing partial output.
    assert not out.exists()


def test_unknown_policy_is_rejected(tmp_path):
    result = runner.invoke(
        app,
        ["generate", "--out", str(tmp_path / "run"), "--episodes", "1", "--steps", "4",
         "--policy", "teleport", "--no-lerobot"],
    )
    assert result.exit_code != 0


def test_generate_gives_a_legible_error_when_lerobot_is_missing(tmp_path, monkeypatch):
    # --lerobot defaults to True, but lerobot is declared only in the
    # optional 'datasets' extra. A base install must be told why it isn't
    # getting a dataset, not fail with a bare ModuleNotFoundError deep in
    # write_lerobot_split, and not silently fall back to metadata only.
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "lerobot":
            raise ModuleNotFoundError("No module named 'lerobot'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    out = tmp_path / "run"
    result = runner.invoke(
        app,
        ["generate", "--out", str(out), "--episodes", "1", "--steps", "4",
         "--policy", "dock", "--num-envs", "1", "--driver", "vector"],
    )
    assert result.exit_code != 0
    assert "datasets" in result.output.lower()
    # Failed before doing any rollout work or writing partial output.
    assert not out.exists()


def test_failed_render_leaves_no_completed_run_marker(tmp_path, monkeypatch):
    # Rendering is the longest and most failure-prone step (GPU init, encoder,
    # missing extras) and it runs after the metadata is built. Building must
    # not have written anything, or a run that dies here looks finished.
    import owm_envs.datasets.video as video

    def boom(*args, **kwargs):
        raise RuntimeError("gpu unavailable")

    monkeypatch.setattr(video, "render_batch_frames", boom)

    out = tmp_path / "run"
    result = runner.invoke(
        app,
        ["generate", "--out", str(out), "--episodes", "1", "--steps", "4",
         "--policy", "dock", "--num-envs", "1", "--driver", "vector", "--render"],
    )
    assert isinstance(result.exception, RuntimeError)
    assert "gpu unavailable" in str(result.exception)
    for name in ("normalization_stats.json", "dataset_card.json", "summary.json",
                 "env_config.yaml", "policy_config.yaml"):
        assert not (out / name).exists(), f"{name} survived a failed render"


def test_render_without_lerobot_is_rejected(tmp_path):
    # --render with --no-lerobot would render every frame and then throw them
    # all away -- nothing consumes them. This must fail fast, before paying
    # any render cost, rather than silently doing (or skipping) the work.
    result = runner.invoke(
        app,
        ["generate", "--out", str(tmp_path / "run"), "--episodes", "1", "--steps", "4",
         "--policy", "dock", "--num-envs", "1", "--driver", "vector",
         "--render", "--no-lerobot"],
    )
    assert result.exit_code != 0
    assert "no effect with --no-lerobot" in result.output
    assert not (tmp_path / "run").exists()


def test_parse_split_flags_builds_specs():
    splits = _parse_split_flags(["train:4:0", "val:2:1"], steps=150)
    assert splits == {
        "train": SplitSpec(num_episodes=4, max_steps=150, seed=0),
        "val": SplitSpec(num_episodes=2, max_steps=150, seed=1),
    }


def test_parse_split_flags_accepts_a_per_split_policy():
    splits = _parse_split_flags(["train:4:0:union", "val:2:1:dock"], steps=150)
    assert splits["train"].policy == PolicyConfig(type="union")
    assert splits["val"].policy == PolicyConfig(type="dock")


@pytest.mark.parametrize("bad", ["train", "train:4", "train:4:0:bogus",
                                 "train:4:0:dock:extra", ":4:0",
                                 "train:x:0", "train:4:y", "train:0:0"])
def test_parse_split_flags_rejects_malformed_entries(bad):
    with pytest.raises(typer.BadParameter):
        _parse_split_flags([bad], steps=150)


def test_parse_split_flags_rejects_duplicate_names():
    with pytest.raises(typer.BadParameter, match="duplicate"):
        _parse_split_flags(["train:4:0", "train:2:1"], steps=150)
