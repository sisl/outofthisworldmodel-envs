import builtins
import json

from typer.testing import CliRunner

from owm_envs.cli import app

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
