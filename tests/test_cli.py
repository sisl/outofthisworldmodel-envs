import builtins
import json
import re

import pytest
import typer
from typer.testing import CliRunner

from owm_envs.cli import app, _parse_split_flags
from owm_envs.datasets.stats import GenerationConfig, SplitSpec
from owm_envs.envs.common.docking_ports import PORT_NAMES
from owm_envs.envs.common.policies import DockParams, PolicyConfig
from owm_envs.envs.common.sensing import PRESETS
from owm_envs.envs.iss.config import ISSConfig

runner = CliRunner()


def test_list_shows_the_iss_environment():
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "iss" in result.stdout.lower()


def test_generate_writes_a_run_directory(tmp_path):
    out = tmp_path / "run"
    result = runner.invoke(
        app,
        ["generate", "--out", str(out), "--split", "train:2:0", "--steps", "8",
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
        ["generate", "--out", str(out), "--split", "train:2:0", "--steps", "8",
         "--policy", "orbit", "--num-envs", "2", "--driver", "vector", "--no-lerobot"],
    )
    card = json.loads((out / "dataset_card.json").read_text())
    assert card["splits"]["train"]["policy_type"] == "orbit"

    from owm_envs.envs.common.policies import PolicyConfig

    assert PolicyConfig.from_yaml(out / "policy_config.yaml").type == "orbit"


def test_both_drivers_are_selectable(tmp_path):
    for driver in ("vector", "scan"):
        out = tmp_path / driver
        result = runner.invoke(
            app,
            ["generate", "--out", str(out), "--split", "train:2:0", "--steps", "8",
             "--policy", "dock", "--num-envs", "2", "--driver", driver, "--no-lerobot"],
        )
        assert result.exit_code == 0, result.stdout
        assert (out / "summary.json").exists()


def test_resolve_driver_vector_path_does_not_double_augment():
    # ISSPolicySource applies the goal-error block itself on the vector path
    # (see policy_source.py); if _resolve_driver handed the ORIGINAL cfg to
    # ISSVectorEnv too, the env would already emit 25-dim observations and
    # the policy source would append a second block on top -- 37-dim, not 25.
    from owm_envs.cli import _resolve_driver
    from owm_envs.drivers.types import RolloutSpec

    cfg = ISSConfig(max_steps=10, observation={"goal_error": True})
    chosen = _resolve_driver("vector", cfg, PolicyConfig(type="dock"), num_envs=2)
    batch = chosen.driver.generate(RolloutSpec(num_episodes=2, max_steps=10, seed=0))
    assert batch.observations.shape[-1] == 25


def test_auto_driver_picks_the_fused_path_for_iss(tmp_path):
    out = tmp_path / "auto"
    result = runner.invoke(
        app,
        ["generate", "--out", str(out), "--split", "train:2:0", "--steps", "8",
         "--policy", "dock", "--num-envs", "2", "--driver", "auto", "--no-lerobot"],
    )
    assert result.exit_code == 0, result.stdout
    # ISS is JAX-traceable, so auto must resolve to the scan driver and say so.
    assert "scan" in result.stdout.lower()


def test_config_file_is_loaded_and_recorded(tmp_path):
    from owm_envs.envs.common.config import PhysicsConfig
    from owm_envs.envs.iss.config import ISSConfig

    cfg_path = tmp_path / "env.yaml"
    ISSConfig(physics=PhysicsConfig(start_radius_range_m=(175.0, 175.0))).to_yaml(cfg_path)

    out = tmp_path / "run"
    result = runner.invoke(
        app,
        ["generate", "--out", str(out), "--config", str(cfg_path), "--split", "train:2:0",
         "--steps", "8", "--policy", "dock", "--num-envs", "2", "--driver", "vector",
         "--no-lerobot"],
    )
    assert result.exit_code == 0, result.stdout
    assert ISSConfig.from_yaml(out / "env_config.yaml").physics.start_radius_range_m == (175.0, 175.0)


def test_a_toml_config_file_is_loaded_too(tmp_path):
    # The shipped env configs under configs/ are TOML, so --config that only
    # spoke YAML could not load the very files the repo ships to be run.
    from owm_envs.envs.common.config import PhysicsConfig
    from owm_envs.envs.iss.config import ISSConfig

    cfg_path = tmp_path / "env.toml"
    ISSConfig(physics=PhysicsConfig(start_radius_range_m=(175.0, 175.0))).to_toml(cfg_path)

    out = tmp_path / "run"
    result = runner.invoke(
        app,
        ["generate", "--out", str(out), "--config", str(cfg_path), "--split", "train:2:0",
         "--steps", "8", "--policy", "dock", "--num-envs", "2", "--driver", "vector",
         "--no-lerobot"],
    )
    assert result.exit_code == 0, result.stdout
    assert ISSConfig.from_yaml(out / "env_config.yaml").physics.start_radius_range_m == (175.0, 175.0)


@pytest.mark.parametrize("filename, text, expected", [
    ("env.txt", "dt: 0.05\n", "suffix"),
    ("missing.yaml", None, "not found"),
    ("env.yaml", "dt: sometimes\n", "dt"),
])
def test_an_unreadable_config_is_a_usage_error_not_a_traceback(
    tmp_path, filename, text, expected
):
    # The wrong extension, the wrong path and a value the schema rejects are
    # all the caller naming the wrong file.
    path = tmp_path / filename
    if text is not None:
        path.write_text(text)
    result = runner.invoke(app, [
        "generate", "--out", str(tmp_path / "run"), "--steps", "4",
        "--split", "train:1:0", "--config", str(path), "--no-lerobot",
    ])
    assert result.exit_code != 0
    assert not isinstance(result.exception, (OSError, ValueError))
    assert "--config" in result.output and expected in result.output


def test_summary_reports_the_episode_count_requested(tmp_path):
    out = tmp_path / "run"
    runner.invoke(
        app,
        ["generate", "--out", str(out), "--split", "train:3:0", "--steps", "8",
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
        ["generate", "--out", str(out), "--split", "train:2:0", "--steps", "8",
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
        ["generate", "--out", str(out), "--config", str(cfg_path), "--split", "train:2:0",
         "--steps", "8", "--policy", "dock", "--num-envs", "2", "--driver", "vector",
         "--no-lerobot"],
    )
    assert result.exit_code == 0, result.stdout
    assert json.loads((out / "dataset_card.json").read_text())["fps"] == 100


def test_explicit_fps_is_honoured_but_warns_when_it_contradicts_dt(tmp_path):
    out = tmp_path / "run"
    result = runner.invoke(
        app,
        ["generate", "--out", str(out), "--split", "train:2:0", "--steps", "8",
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
        ["generate", "--out", str(out), "--split", "train:2:0", "--steps", "8",
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
        ["generate", "--out", str(out), "--config", str(cfg_path), "--split", "train:2:0",
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
        ["generate", "--out", str(out), "--config", str(cfg_path), "--split", "train:1:0",
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
        ["generate", "--out", str(out), "--split", "train:2:0", "--steps", "8",
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
        ["generate", "--out", str(out), "--split", "train:1:0", "--steps", "4",
         "--policy", "dock", "--num-envs", "0", "--driver", "scan", "--no-lerobot"],
    )
    assert result.exit_code != 0
    assert "num_envs" in result.output.lower()
    # Failed before doing any rollout work or writing partial output.
    assert not out.exists()


def test_unknown_policy_is_rejected(tmp_path):
    result = runner.invoke(
        app,
        ["generate", "--out", str(tmp_path / "run"), "--split", "train:1:0", "--steps", "4",
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
        ["generate", "--out", str(out), "--split", "train:1:0", "--steps", "4",
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

    monkeypatch.setattr(video, "iter_batch_frames", boom)

    out = tmp_path / "run"
    result = runner.invoke(
        app,
        ["generate", "--out", str(out), "--split", "train:1:0", "--steps", "4",
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
        ["generate", "--out", str(tmp_path / "run"), "--split", "train:1:0", "--steps", "4",
         "--policy", "dock", "--num-envs", "1", "--driver", "vector",
         "--render", "--no-lerobot"],
    )
    assert result.exit_code != 0
    assert "no effect with --no-lerobot" in result.output
    assert not (tmp_path / "run").exists()


def test_parse_split_flags_builds_specs():
    splits = _parse_split_flags(["train:4:0", "val:2:1"], steps=150, observe="state", policy="random", ports="")
    assert splits == {
        "train": SplitSpec(num_episodes=4, max_steps=150, seed=0),
        "val": SplitSpec(num_episodes=2, max_steps=150, seed=1),
    }


def test_parse_split_flags_accepts_a_transition_target():
    splits = _parse_split_flags(["train:100000t:0", "val:20000t:1"], steps=150, observe="state", policy="random", ports="")
    assert splits == {
        "train": SplitSpec(min_transitions=100000, max_steps=150, seed=0),
        "val": SplitSpec(min_transitions=20000, max_steps=150, seed=1),
    }


def test_parse_split_flags_accepts_a_transition_target_with_a_policy():
    splits = _parse_split_flags(["train:100000t:0:union"], steps=150, observe="state", policy="random", ports="")
    assert splits["train"] == SplitSpec(
        min_transitions=100000, max_steps=150, seed=0,
        policy=PolicyConfig(type="union", observe="state"),
    )


def test_parse_split_flags_accepts_a_per_split_policy():
    splits = _parse_split_flags(["train:4:0:union", "val:2:1:dock"], steps=150, observe="state", policy="random", ports="")
    assert splits["train"].policy == PolicyConfig(type="union", observe="state")
    assert splits["val"].policy == PolicyConfig(type="dock", observe="state")


@pytest.mark.parametrize("bad", ["train", "train:4", "train:4:0:bogus",
                                 "train:4:0:dock:extra", ":4:0",
                                 "train:x:0", "train:4:y", "train:0:0",
                                 "train:t:0", "train:0t:0", "train:12tt:0",
                                 "train:t12:0", "train:-5t:0"])
def test_parse_split_flags_rejects_malformed_entries(bad):
    with pytest.raises(typer.BadParameter):
        _parse_split_flags([bad], steps=150, observe="state", policy="random", ports="")


def test_parse_split_flags_accepts_a_port_suffix_without_a_policy():
    splits = _parse_split_flags(["val:8:1::all"], steps=150, observe="state", policy="dock", ports="")
    assert splits["val"].policy == PolicyConfig(
        type="dock", observe="state", dock=DockParams(ports=("all",))
    )


@pytest.mark.parametrize("joined", ["harmony_fwd_pma2,poisk_zenith",
                                    "harmony_fwd_pma2+poisk_zenith"])
def test_parse_split_flags_accepts_comma_or_plus_joined_ports(joined):
    splits = _parse_split_flags([f"train:4:0:dock:{joined}"], steps=150, observe="state",
                                policy="random", ports="")
    names = tuple(port.name for port in splits["train"].policy.dock.ports)
    assert names == ("harmony_fwd_pma2", "poisk_zenith")


def test_parse_split_flags_rejects_an_unknown_port_naming_the_known_ones():
    with pytest.raises(typer.BadParameter, match="known ports are"):
        _parse_split_flags(["train:4:0:dock:not_a_port"], steps=150, observe="state",
                           policy="random", ports="")


def test_split_ports_override_the_run_level_dock_ports():
    splits = _parse_split_flags(
        ["train:4:0::poisk_zenith", "val:2:1"], steps=150, observe="state",
        policy="dock", ports="all",
    )
    assert tuple(p.name for p in splits["train"].policy.dock.ports) == ("poisk_zenith",)
    # An unqualified split records no override and inherits --dock-ports at
    # the call site instead.
    assert splits["val"].policy is None


def test_parse_split_flags_rejects_an_empty_policy_field():
    # `train:4:0:` is a typo, not a way to spell "inherit --policy"; dropping
    # the trailing ':' already does that.
    with pytest.raises(typer.BadParameter, match="POLICY is empty"):
        _parse_split_flags(["train:4:0:"], steps=150, observe="state", policy="random", ports="")
    with pytest.raises(typer.BadParameter, match="POLICY is empty"):
        _parse_split_flags(["train:4:0::"], steps=150, observe="state", policy="random", ports="")


def test_dock_ports_all_reaches_the_run_level_policy(tmp_path):
    out = tmp_path / "run"
    result = runner.invoke(app, [
        "generate", "--out", str(out), "--steps", "8", "--split", "train:2:0",
        "--policy", "dock", "--dock-ports", "all", "--no-lerobot",
    ])
    assert result.exit_code == 0, result.output
    gen_policy = PolicyConfig.from_yaml(out / "policy_config.yaml")
    assert tuple(p.name for p in gen_policy.dock.ports) == PORT_NAMES


def test_vector_driver_warns_for_a_single_port_set(tmp_path):
    # The vector env scores `docked` against DockConfig whatever ports the
    # policy targets, and one port is no safer than several: even PMA-2's
    # derived pose is 0.84 m off the shipped one.
    out = tmp_path / "run"
    result = runner.invoke(app, [
        "generate", "--out", str(out), "--steps", "8", "--split", "train:2:0",
        "--policy", "dock", "--dock-ports", "harmony_fwd_pma2",
        "--driver", "vector", "--no-lerobot",
    ])
    assert result.exit_code == 0, result.output
    assert "scores dock success against DockConfig" in result.output
    assert "harmony_fwd_pma2" in result.output


def test_parse_split_flags_rejects_duplicate_names():
    with pytest.raises(typer.BadParameter, match="duplicate"):
        _parse_split_flags(["train:4:0", "train:2:1"], steps=150, observe="state", policy="random", ports="")


def test_observe_flag_reaches_per_split_policies(tmp_path):
    out = tmp_path / "run"
    result = runner.invoke(app, [
        "generate", "--out", str(out), "--steps", "8",
        "--split", "train:2:0:dock", "--observe", "measurement", "--no-lerobot",
    ])
    assert result.exit_code == 0, result.output
    gen = GenerationConfig.from_yaml(out / "generation_config.yaml")
    assert gen.splits["train"].policy.observe == "measurement"


def test_one_invocation_writes_every_split(tmp_path):
    out = tmp_path / "run"
    result = runner.invoke(app, [
        "generate", "--out", str(out), "--steps", "8",
        "--split", "train:3:0", "--split", "val:2:1",
        "--no-lerobot",
    ])
    assert result.exit_code == 0, result.output
    summary = json.loads((out / "summary.json").read_text())
    assert summary["counts"]["train"]["episodes"] == 3
    assert summary["counts"]["val"]["episodes"] == 2


def test_stats_survive_a_val_split_in_the_same_run(tmp_path):
    out_single = tmp_path / "single"
    out_multi = tmp_path / "multi"
    for args in (
        ["generate", "--out", str(out_single), "--steps", "8",
         "--split", "train:3:0", "--no-lerobot"],
        ["generate", "--out", str(out_multi), "--steps", "8",
         "--split", "train:3:0", "--split", "val:2:1", "--no-lerobot"],
    ):
        assert runner.invoke(app, args).exit_code == 0
    # Same train seed -> byte-identical train-only stats regardless of the
    # extra val split. This is the clobbering bug this PR exists to fix.
    assert (out_single / "normalization_stats.json").read_text() == \
           (out_multi / "normalization_stats.json").read_text()


def test_split_without_train_is_rejected(tmp_path):
    result = runner.invoke(app, [
        "generate", "--out", str(tmp_path / "r"), "--steps", "8",
        "--split", "val:2:1", "--no-lerobot",
    ])
    assert result.exit_code != 0
    assert "train" in result.output


def test_gen_config_file_drives_the_run(tmp_path):
    gen = GenerationConfig(splits={
        "train": SplitSpec(num_episodes=2, max_steps=8, seed=0),
        "val": SplitSpec(num_episodes=1, max_steps=8, seed=1),
    }, num_envs=2)
    gen_path = tmp_path / "gen.yaml"
    gen.to_yaml(gen_path)
    out = tmp_path / "run"
    result = runner.invoke(app, [
        "generate", "--out", str(out), "--gen-config", str(gen_path), "--no-lerobot",
    ])
    assert result.exit_code == 0, result.output
    assert GenerationConfig.from_yaml(out / "generation_config.yaml") == gen


def test_gen_config_conflicts_with_generation_flags(tmp_path):
    gen_path = tmp_path / "gen.yaml"
    GenerationConfig().to_yaml(gen_path)
    result = runner.invoke(app, [
        "generate", "--out", str(tmp_path / "r"),
        "--gen-config", str(gen_path), "--split", "train:2:0", "--no-lerobot",
    ])
    assert result.exit_code != 0
    assert "exclusive" in result.output


def test_provenance_lands_in_the_dataset_card(tmp_path):
    out = tmp_path / "run"
    runner.invoke(app, ["generate", "--out", str(out), "--steps", "8",
                        "--split", "train:2:0", "--no-lerobot"])
    card = json.loads((out / "dataset_card.json").read_text())
    assert len(card["provenance"]["git_commit"]) == 40


def test_per_split_policy_is_rolled_and_recorded(tmp_path):
    out = tmp_path / "run"
    result = runner.invoke(app, [
        "generate", "--out", str(out), "--steps", "8", "--policy", "union",
        "--split", "train:2:0", "--split", "val:1:1:dock",
        "--no-lerobot",
    ])
    assert result.exit_code == 0, result.output
    card = json.loads((out / "dataset_card.json").read_text())
    assert card["splits"]["train"]["policy_type"] == "union"   # run-level default
    assert card["splits"]["val"]["policy_type"] == "dock"      # per-split override


def test_invalid_gen_config_file_is_rejected_cleanly(tmp_path):
    bad = tmp_path / "gen.yaml"
    bad.write_text("splits:\n  val:\n    num_episodes: 2\n    seed: 1\n")
    result = runner.invoke(app, [
        "generate", "--out", str(tmp_path / "r"), "--gen-config", str(bad), "--no-lerobot",
    ])
    assert result.exit_code != 0
    assert "train" in result.output


def test_negative_seed_split_flag_is_rejected_cleanly(tmp_path):
    result = runner.invoke(app, [
        "generate", "--out", str(tmp_path / "r"), "--steps", "8",
        "--split", "train:2:-1", "--no-lerobot",
    ])
    assert result.exit_code != 0


def test_missing_gen_config_file_is_rejected_cleanly(tmp_path):
    result = runner.invoke(app, [
        "generate", "--out", str(tmp_path / "r"),
        "--gen-config", str(tmp_path / "nope.yaml"), "--no-lerobot",
    ])
    assert result.exit_code != 0
    assert "gen-config" in result.output


def test_malformed_gen_config_yaml_is_rejected_cleanly(tmp_path):
    bad = tmp_path / "gen.yaml"
    bad.write_text("splits: [unclosed\n")
    result = runner.invoke(app, [
        "generate", "--out", str(tmp_path / "r"),
        "--gen-config", str(bad), "--no-lerobot",
    ])
    assert result.exit_code != 0
    assert "gen-config" in result.output


def test_noise_preset_flag_overrides_the_env_config(tmp_path):
    out = tmp_path / "run"
    result = runner.invoke(app, [
        "generate", "--out", str(out), "--steps", "8", "--split", "train:2:0",
        "--noise", "cooperative", "--no-lerobot",
    ])
    assert result.exit_code == 0, result.output
    recorded = ISSConfig.from_yaml(out / "env_config.yaml")
    assert recorded.sensor_noise == PRESETS["cooperative"]


def test_observe_flag_is_recorded_in_the_policy_config(tmp_path):
    out = tmp_path / "run"
    result = runner.invoke(app, [
        "generate", "--out", str(out), "--steps", "8", "--split", "train:2:0",
        "--policy", "dock", "--observe", "measurement", "--no-lerobot",
    ])
    assert result.exit_code == 0, result.output
    recorded = PolicyConfig.from_yaml(out / "policy_config.yaml")
    assert recorded.observe == "measurement"


def test_unknown_noise_preset_is_rejected(tmp_path):
    result = runner.invoke(app, [
        "generate", "--out", str(tmp_path / "r"), "--steps", "8",
        "--split", "train:2:0", "--noise", "bogus", "--no-lerobot",
    ])
    assert result.exit_code != 0


def test_goal_error_flag_overrides_the_env_config(tmp_path):
    out = tmp_path / "run"
    result = runner.invoke(app, [
        "generate", "--out", str(out), "--steps", "8", "--split", "train:2:0",
        "--goal-error", "--no-lerobot",
    ])
    assert result.exit_code == 0, result.output
    assert ISSConfig.from_yaml(out / "env_config.yaml").observation.goal_error is True


def test_transition_targeted_split_hits_the_target_and_is_recorded(tmp_path):
    out = tmp_path / "run"
    result = runner.invoke(app, [
        "generate", "--out", str(out), "--steps", "8",
        "--split", "train:30t:0", "--no-lerobot",
    ])
    assert result.exit_code == 0, result.output
    summary = json.loads((out / "summary.json").read_text())
    assert summary["counts"]["train"]["transitions"] >= 30
    card = json.loads((out / "dataset_card.json").read_text())
    assert card["splits"]["train"]["min_transitions"] == 30


def test_goal_error_flag_defaults_to_config(tmp_path):
    out = tmp_path / "run"
    result = runner.invoke(app, [
        "generate", "--out", str(out), "--steps", "8", "--split", "train:2:0", "--no-lerobot",
    ])
    assert result.exit_code == 0, result.output
    assert ISSConfig.from_yaml(out / "env_config.yaml").observation.goal_error is False


def test_gpu_index_is_resolved_before_any_rollout_or_render(tmp_path, monkeypatch):
    """Adapter choice must land before pygfx pins its one shared device."""
    pytest.importorskip("lerobot", reason="--render requires the datasets extra")
    seen = {}

    def fake_select(index):
        seen["index"] = index
        raise RuntimeError("stop-after-select")

    monkeypatch.setattr("owm_envs.render.device.select_gpu", fake_select)
    out = tmp_path / "run"
    result = runner.invoke(app, [
        "generate", "--out", str(out), "--steps", "8", "--split", "train:1:0",
        "--render", "--gpu-index", "1",
    ])
    assert seen["index"] == 1
    assert isinstance(result.exception, RuntimeError)
    assert not out.exists(), "selection must precede the rollout, render and writer"


def test_invalid_gpu_index_is_a_usage_error_not_a_traceback(tmp_path, monkeypatch):
    pytest.importorskip("lerobot", reason="--render requires the datasets extra")

    def refuse(index):
        raise ValueError("gpu index 9 out of range; available adapters: 0: NVIDIA A100")

    monkeypatch.setattr("owm_envs.render.device.select_gpu", refuse)
    result = runner.invoke(app, [
        "generate", "--out", str(tmp_path / "run"), "--steps", "8",
        "--split", "train:1:0", "--render", "--gpu-index", "9",
    ])
    assert result.exit_code != 0
    assert not isinstance(result.exception, ValueError)
    assert "out of range" in result.output and "NVIDIA A100" in result.output


def test_render_workers_fan_out_and_keep_the_parent_off_the_gpu(tmp_path, monkeypatch):
    """With a pool, every renderer lives in a worker: this process must not
    select an adapter, or it would hold a device the workers cannot use."""
    pytest.importorskip("lerobot", reason="--render requires the datasets extra")
    import owm_envs.datasets.video as video

    seen = {}

    def fake_iter(batch, cfg, keys=(), workers=1, gpu_index=None):
        seen["workers"] = workers
        seen["gpu_index"] = gpu_index
        raise RuntimeError("stop-after-fan-out")

    monkeypatch.setattr(video, "iter_batch_frames", fake_iter)
    monkeypatch.setattr(
        "owm_envs.render.device.select_gpu",
        lambda index: pytest.fail("the parent must not select a GPU with a worker pool"),
    )
    # The parent's bounds check is real and enumerates adapters; that it
    # rejects a bad index is tested below, and this test is about selection.
    monkeypatch.setattr("owm_envs.render.device.check_gpu_index", lambda index: None)
    result = runner.invoke(app, [
        "generate", "--out", str(tmp_path / "run"), "--steps", "4", "--split", "train:1:0",
        "--num-envs", "1", "--render", "--render-workers", "4", "--gpu-index", "1",
    ])
    assert isinstance(result.exception, RuntimeError), result.output
    assert seen == {"workers": 4, "gpu_index": 1}


def test_earth_textures_are_resolved_in_the_parent_before_the_worker_pool(
    tmp_path, monkeypatch
):
    """Every worker's scene resolves the Earth textures itself, and a miss
    downloads and bakes a full map from a 9.6 GB source. Resolving them once
    here leaves the workers three finished files to open: N concurrent decodes
    cannot exhaust memory, and no two workers can settle on different tiers and
    mix resolutions within one dataset."""
    pytest.importorskip("lerobot", reason="--render requires the datasets extra")
    import owm_envs.datasets.video as video

    order = []

    def fake_resolve(kind, *, allow_download=False):
        order.append((kind, allow_download))
        return tmp_path / f"earth_{kind}"

    def fake_iter(batch, cfg, keys=(), workers=1, gpu_index=None):
        order.append(("iter_batch_frames", workers))
        raise RuntimeError("stop-after-fan-out")

    monkeypatch.setattr("owm_envs.render.earth.earth_texture_path", fake_resolve)
    monkeypatch.setattr(video, "iter_batch_frames", fake_iter)
    monkeypatch.setattr("owm_envs.render.device.check_gpu_index", lambda index: None)
    result = runner.invoke(app, [
        "generate", "--out", str(tmp_path / "run"), "--steps", "4", "--split", "train:1:0",
        "--num-envs", "1", "--render", "--render-workers", "4",
    ])
    assert isinstance(result.exception, RuntimeError), result.output
    assert order == [("color", True), ("clouds", True), ("bump", True),
                     ("iter_batch_frames", 4)]


def _fpv_clips(length=4, size=32):
    """A stand-in episode of video. 32x32 and even-sized: the media tee really
    encodes these, and libx264 cannot subsample an odd frame."""
    import numpy as np

    from owm_envs.datasets.video import FPV_KEY

    return {FPV_KEY: np.zeros((length, size, size, 3), dtype=np.uint8)}


def test_render_frames_reach_the_writer_unmaterialized(tmp_path, monkeypatch):
    """The writer has to receive a lazy iterator: a list of every clip in a
    500k-frame split is ~98 GB."""
    pytest.importorskip("lerobot", reason="--render requires the datasets extra")
    from collections.abc import Sequence

    import owm_envs.datasets.lerobot_writer as writer

    seen = {}

    def fake_write(root, repo_id, batch, fps, task_name="iss_docking", frames=None):
        seen["is_sequence"] = isinstance(frames, Sequence)
        seen["pulled"] = sum(1 for _ in frames)
        return root

    monkeypatch.setattr(writer, "write_lerobot_split", fake_write)
    monkeypatch.setattr(
        "owm_envs.datasets.video.iter_batch_frames",
        lambda batch, cfg, keys=(), workers=1, gpu_index=None: iter(
            [_fpv_clips() for _ in range(batch.num_episodes)]
        ),
    )
    result = runner.invoke(app, [
        "generate", "--out", str(tmp_path / "run"), "--steps", "4", "--split", "train:2:0",
        "--num-envs", "1", "--render",
    ])
    assert result.exit_code == 0, result.output
    assert seen["is_sequence"] is False
    assert seen["pulled"] == 2


@pytest.mark.parametrize("render_flag", ["--render", "--no-render"])
def test_zero_render_workers_is_a_usage_error(tmp_path, render_flag):
    """Nonsense is nonsense whether or not the run renders: a value that can
    never be valid must be rejected, not quietly ignored on the path that
    happens not to read it."""
    result = runner.invoke(app, [
        "generate", "--out", str(tmp_path / "run"), "--steps", "4", "--split", "train:1:0",
        render_flag, "--render-workers", "0",
    ])
    assert result.exit_code != 0
    assert "--render-workers must be >= 1" in result.output


def test_render_line_names_frames_and_workers_without_a_time_estimate(
    tmp_path, monkeypatch
):
    """The line must not predict a duration. Measured throughput scales
    1.6-2.6x over 2-8 workers rather than linearly, and per-frame cost moves
    with resolution and scene, so any minutes figure would be wrong by more
    than it is worth."""
    pytest.importorskip("lerobot", reason="--render requires the datasets extra")
    import owm_envs.datasets.lerobot_writer as writer

    def fake_write(root, repo_id, batch, fps, task_name="iss_docking", frames=None):
        for _ in frames:
            pass
        return root

    monkeypatch.delenv("OWM_ENVS_GPU_INDEX", raising=False)
    monkeypatch.setattr(writer, "write_lerobot_split", fake_write)
    monkeypatch.setattr(
        "owm_envs.datasets.video.iter_batch_frames",
        lambda batch, cfg, keys=(), workers=1, gpu_index=None: iter(
            [_fpv_clips() for _ in range(batch.num_episodes)]
        ),
    )
    result = runner.invoke(app, [
        "generate", "--out", str(tmp_path / "run"), "--steps", "4", "--split", "train:2:0",
        "--num-envs", "1", "--render", "--render-workers", "2",
    ])
    assert result.exit_code == 0, result.output
    line = next(ln for ln in result.output.splitlines() if ln.startswith("[render]"))
    assert re.fullmatch(
        r"\[render\] train: \d+ frames, 7 video feature\(s\) \([\w, ]+\), 2 worker\(s\)",
        line,
    ), line


def test_bad_gpu_index_fails_before_the_rollout_with_a_worker_pool(tmp_path, monkeypatch):
    """A pool's workers select their adapter after the rollout has already
    run, so the parent bounds-checks the index up front -- a typo must cost
    seconds, not an hour of rollout followed by a broken pool."""
    pytest.importorskip("lerobot", reason="--render requires the datasets extra")

    def refuse(index):
        raise ValueError("gpu index 9 out of range; available adapters: 0: NVIDIA A100")

    monkeypatch.setattr("owm_envs.render.device.check_gpu_index", refuse)
    monkeypatch.setattr(
        "owm_envs.render.device.select_gpu",
        lambda index: pytest.fail("the parent must not select a GPU with a worker pool"),
    )
    out = tmp_path / "run"
    result = runner.invoke(app, [
        "generate", "--out", str(out), "--steps", "8", "--split", "train:1:0",
        "--render", "--render-workers", "4", "--gpu-index", "9",
    ])
    assert result.exit_code != 0
    assert not isinstance(result.exception, ValueError)
    assert "out of range" in result.output and "NVIDIA A100" in result.output
    assert not out.exists(), "the check must precede the rollout"


def test_gpu_index_is_untouched_when_not_rendering(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "owm_envs.render.device.select_gpu",
        lambda index: pytest.fail("must not select a GPU without --render"),
    )
    result = runner.invoke(app, [
        "generate", "--out", str(tmp_path / "run"), "--steps", "8",
        "--split", "train:1:0", "--no-lerobot",
    ])
    assert result.exit_code == 0, result.output


def _pushable_run(tmp_path):
    """The two artifacts `owm-envs push` reads before it uploads anything."""
    run = tmp_path / "run"
    run.mkdir()
    ISSConfig(sensor_noise=PRESETS["cooperative"]).to_yaml(run / "env_config.yaml")
    (run / "summary.json").write_text(json.dumps({"counts": {
        "train": {"episodes": 96, "transitions": 500_012},
        "val": {"episodes": 11, "transitions": 50_004},
    }}))
    return run


@pytest.fixture
def recorded_push(monkeypatch):
    """Records what reached push_run; nothing gets near the Hub."""
    pushed = {}

    def fake_push(run_dir, name=None, namespace=None, private=False):
        pushed.update(run_dir=run_dir, name=name, namespace=namespace, private=private)
        return f"{namespace}/{name}"

    monkeypatch.setattr("owm_envs.datasets.hub.push_run", fake_push)
    return pushed


def test_push_reports_the_dataset_url(tmp_path, recorded_push):
    run = _pushable_run(tmp_path)
    result = runner.invoke(
        app, ["push", str(run), "--namespace", "acct", "--private", "--yes"]
    )

    assert result.exit_code == 0, result.output
    assert "https://huggingface.co/datasets/acct/owm-iss-coop-nogoal-dt50ms" in result.output
    # The confirmed repo is passed back explicitly, so the upload cannot land
    # anywhere other than the target the confirmation named.
    assert recorded_push == {"run_dir": run, "name": "owm-iss-coop-nogoal-dt50ms",
                             "namespace": "acct", "private": True}

    # Neither flag reaches push_run as None, not as False: "public" is a
    # request to change an existing repo's visibility, "unset" is not.
    runner.invoke(app, ["push", str(run), "--namespace", "acct", "--yes"])
    assert recorded_push["private"] is None
    runner.invoke(app, ["push", str(run), "--namespace", "acct", "--public", "--yes"])
    assert recorded_push["private"] is False


def test_push_names_the_repo_and_the_split_sizes_before_confirming(
    tmp_path, monkeypatch, recorded_push
):
    # The repo name is derived from the env config alone, so a trial run and
    # the production run generated from the same variant TOML target the same
    # repo -- and the upload mirrors, deleting what is there. What is about to
    # be replaced, and by how much data, has to be on screen before the answer.
    monkeypatch.setattr("owm_envs.cli._stdin_is_interactive", lambda: True)
    result = runner.invoke(
        app, ["push", str(_pushable_run(tmp_path)), "--namespace", "acct"], input="y\n"
    )
    assert result.exit_code == 0, result.output
    prompt = result.output.split("Replace")[0]
    assert "acct/owm-iss-coop-nogoal-dt50ms" in prompt
    assert "train: 96 episodes, 500012 transitions" in prompt
    assert "val: 11 episodes, 50004 transitions" in prompt
    assert "MIRRORS" in prompt
    assert recorded_push["name"] == "owm-iss-coop-nogoal-dt50ms"


def test_push_uploads_nothing_when_the_prompt_is_declined(
    tmp_path, monkeypatch, recorded_push
):
    monkeypatch.setattr("owm_envs.cli._stdin_is_interactive", lambda: True)
    result = runner.invoke(
        app, ["push", str(_pushable_run(tmp_path)), "--namespace", "acct"], input="n\n"
    )
    assert result.exit_code != 0
    assert recorded_push == {}


def test_push_without_a_terminal_refuses_unless_yes_is_given(tmp_path, recorded_push):
    # A scripted or piped invocation cannot answer a prompt. Prompting anyway
    # would read EOF and abort obscurely; proceeding would mirror over a
    # production repo unconfirmed.
    result = runner.invoke(app, ["push", str(_pushable_run(tmp_path)), "--namespace", "acct"])
    assert result.exit_code != 0
    assert "--yes" in result.output
    assert recorded_push == {}


def test_push_rejects_a_directory_that_is_not_a_finished_run(tmp_path):
    result = runner.invoke(app, ["push", str(tmp_path)])
    assert result.exit_code != 0
    assert "did not finish" in result.output
    assert not isinstance(result.exception, FileNotFoundError)


def test_push_does_not_blame_the_run_for_a_failed_hub_call(tmp_path, monkeypatch,
                                                           recorded_push):
    # HfHubHTTPError descends from OSError, so a handler wide enough for a
    # truncated summary.json also catches a failed login -- and would report a
    # perfectly good run directory as unreadable.
    import huggingface_hub

    class _Unauthorized:
        def whoami(self):
            raise OSError("401 Client Error: Unauthorized for url: .../whoami-v2")

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: _Unauthorized())
    result = runner.invoke(app, ["push", str(_pushable_run(tmp_path)), "--yes"])
    assert result.exit_code != 0
    assert "cannot read the run" not in result.output
    assert recorded_push == {}


@pytest.mark.parametrize("filename, text", [
    ("summary.json", "{not json"),
    ("summary.json", '{"dataset_root": "x"}'),
    ("summary.json", "null"),
    ("env_config.yaml", "dt: sometimes\n"),
])
def test_push_rejects_a_run_whose_own_artifacts_do_not_read(
    tmp_path, filename, text, recorded_push
):
    # A truncated summary or a hand-edited env config is the same class of
    # mistake as naming an unfinished run, and it is read before anything is
    # uploaded -- so it must read as a usage error, not a traceback.
    run = _pushable_run(tmp_path)
    (run / filename).write_text(text)
    result = runner.invoke(app, ["push", str(run), "--namespace", "acct", "--yes"])
    assert result.exit_code != 0
    assert not isinstance(result.exception, (OSError, ValueError))
    assert "cannot read the run" in result.output
    assert recorded_push == {}


def _clips_for(keys, length=4, size=32):
    """A stand-in episode carrying exactly the keys the run asked for."""
    import numpy as np

    return {key: np.zeros((length, size, size, 3), dtype=np.uint8) for key in keys}


def _record_render_keys(monkeypatch, seen):
    """Stand in for the render pool and the writer, capturing the keys asked for.

    The clips carry the requested keys and no others: a fake that always
    returned an fpv clip would let a run that never rendered fpv reach the
    media tee looking as though it had.
    """
    import owm_envs.datasets.lerobot_writer as writer

    def fake_write(root, repo_id, batch, fps, task_name="iss_docking", frames=None):
        for _ in frames:
            pass
        return root

    monkeypatch.setattr(writer, "write_lerobot_split", fake_write)
    monkeypatch.setattr(
        "owm_envs.datasets.video.iter_batch_frames",
        lambda batch, cfg, keys=(), workers=1, gpu_index=None: (
            seen.update(keys=tuple(keys)),
            iter([_clips_for(keys) for _ in range(batch.num_episodes)]),
        )[1],
    )


def test_a_rendered_run_writes_every_view_by_default(tmp_path, monkeypatch):
    """The default is the whole set -- six cameras and the mosaic -- so a run
    that says nothing about views gets everything there is to look at."""
    pytest.importorskip("lerobot", reason="--render requires the datasets extra")
    from owm_envs.datasets.video import OUTPUT_KEYS

    seen = {}
    _record_render_keys(monkeypatch, seen)
    result = runner.invoke(app, [
        "generate", "--out", str(tmp_path / "run"), "--steps", "4", "--split", "train:1:0",
        "--num-envs", "1", "--render",
    ])
    assert result.exit_code == 0, result.output
    assert seen["keys"] == OUTPUT_KEYS
    assert len(seen["keys"]) == 7


def test_render_views_restricts_a_run_to_what_it_asked_for(tmp_path, monkeypatch):
    """The whole point of keeping the flag: seven video streams is the cost a
    lean training run declines, and it must get exactly the key training reads."""
    pytest.importorskip("lerobot", reason="--render requires the datasets extra")
    from owm_envs.datasets.video import FPV_KEY

    seen = {}
    _record_render_keys(monkeypatch, seen)
    result = runner.invoke(app, [
        "generate", "--out", str(tmp_path / "run"), "--steps", "4", "--split", "train:1:0",
        "--num-envs", "1", "--render", "--render-views", "fpv",
    ])
    assert result.exit_code == 0, result.output
    assert seen["keys"] == (FPV_KEY,)


def test_a_run_without_the_fpv_view_still_completes(tmp_path, monkeypatch):
    """Asking for another view alone is a legitimate run, not a broken one:
    the media tee copies whatever it is handed rather than reaching for fpv,
    so the media tree follows the run's views and grows no others."""
    pytest.importorskip("lerobot", reason="--render requires the datasets extra")

    seen = {}
    _record_render_keys(monkeypatch, seen)
    out = tmp_path / "run"
    result = runner.invoke(app, [
        "generate", "--out", str(out), "--steps", "4", "--split", "train:1:0",
        "--num-envs", "1", "--render", "--render-views", "iss_top",
    ])
    assert result.exit_code == 0, result.output
    assert seen["keys"] == ("observation.images.iss_top",)
    assert [p.name for p in (out / "media").iterdir()] == ["iss_top"]
    assert (out / "media" / "iss_top" / "train" / "ep_0000.mp4").exists()


def test_a_rendered_run_writes_a_media_tree_for_every_view(tmp_path, monkeypatch):
    """The default renders all seven, so a run directory carries seven trees of
    per-episode copies alongside the dataset's own chunk files, and `push`
    ships them with it."""
    pytest.importorskip("lerobot", reason="--render requires the datasets extra")
    from owm_envs.datasets.video import OUTPUT_KEYS

    seen = {}
    _record_render_keys(monkeypatch, seen)
    out = tmp_path / "run"
    result = runner.invoke(app, [
        "generate", "--out", str(out), "--steps", "4", "--split", "train:2:0",
        "--num-envs", "1", "--render",
    ])
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in (out / "media").iterdir()) == sorted(
        key.rsplit(".", 1)[-1] for key in OUTPUT_KEYS
    )
    for key in OUTPUT_KEYS:
        view = key.rsplit(".", 1)[-1]
        clips = sorted(p.name for p in (out / "media" / view / "train").glob("*.mp4"))
        assert clips == ["ep_0000.mp4", "ep_0001.mp4"], f"{view} is missing per-episode clips"


def test_the_as_run_config_records_the_views_the_dataset_was_built_with(tmp_path, monkeypatch):
    """A run directory has to say which cameras its video came from: the same
    environment and policy can produce datasets with different feature sets,
    and nothing else in the directory distinguishes them."""
    pytest.importorskip("lerobot", reason="--render requires the datasets extra")
    from owm_envs.datasets.video import COMPOSITE_KEY, FPV_KEY

    seen = {}
    _record_render_keys(monkeypatch, seen)
    out = tmp_path / "run"
    result = runner.invoke(app, [
        "generate", "--out", str(out), "--steps", "4", "--split", "train:1:0",
        "--num-envs", "1", "--render", "--render-views", "composite,fpv",
    ])
    assert result.exit_code == 0, result.output
    # The recorded selection and the one actually rendered are the same thing,
    # which is the property that makes the record worth anything.
    assert seen["keys"] == (FPV_KEY, COMPOSITE_KEY)
    recorded = GenerationConfig.from_yaml(out / "generation_config.yaml")
    assert recorded.render_views == ["fpv", "composite"]


def test_a_gen_config_supplies_the_views(tmp_path, monkeypatch):
    """The recipe is the committed-config path, so the views have to come from
    it rather than from a flag the operator would have to remember to repeat."""
    pytest.importorskip("lerobot", reason="--render requires the datasets extra")

    recipe = tmp_path / "gen.yaml"
    GenerationConfig(
        splits={"train": SplitSpec(num_episodes=1, max_steps=4, seed=0)},
        num_envs=1,
        render_views="iss_top",
    ).to_yaml(recipe)

    seen = {}
    _record_render_keys(monkeypatch, seen)
    result = runner.invoke(app, [
        "generate", "--out", str(tmp_path / "run"), "--gen-config", str(recipe), "--render",
    ])
    assert result.exit_code == 0, result.output
    assert seen["keys"] == ("observation.images.iss_top",)


def test_render_views_is_exclusive_with_a_gen_config(tmp_path, monkeypatch):
    """The recipe carries its own selection, so a flag beside it would be two
    answers to one question -- the same rule the other recipe flags follow.

    Rejected before the GPU is touched, with --render passed: selecting an
    adapter pins one for the whole process, which is real cost to spend on an
    invocation that was never going to run.
    """
    pytest.importorskip("lerobot", reason="--render requires the datasets extra")

    recipe = tmp_path / "gen.yaml"
    GenerationConfig(
        splits={"train": SplitSpec(num_episodes=1, max_steps=4, seed=0)}
    ).to_yaml(recipe)

    probed = []
    monkeypatch.setattr("owm_envs.render.device.select_gpu", lambda index: probed.append(index))
    monkeypatch.setattr("owm_envs.render.device.check_gpu_index", lambda index: probed.append(index))

    result = runner.invoke(app, [
        "generate", "--out", str(tmp_path / "run"), "--gen-config", str(recipe),
        "--render-views", "fpv", "--render",
    ])
    assert result.exit_code != 0
    assert "--render-views" in result.output
    assert probed == [], "probed the GPU for an invocation that was a usage error"


def test_an_unknown_view_is_caught_even_beside_all(tmp_path):
    """'all' must not swallow a typo sitting next to it: the run would silently
    render something other than what was asked for."""
    result = runner.invoke(app, [
        "generate", "--out", str(tmp_path / "run"), "--steps", "4", "--split", "train:1:0",
        "--render", "--render-views", "all,dragon_fvp",
    ])
    assert result.exit_code != 0
    assert "dragon_fvp" in result.output


def test_render_views_takes_a_list_and_orders_it_canonically(tmp_path, monkeypatch):
    """Two runs asking for the same set must declare the same features, so the
    order is the module's rather than the order the names were typed in."""
    pytest.importorskip("lerobot", reason="--render requires the datasets extra")
    from owm_envs.datasets.video import FPV_KEY

    seen = {}
    _record_render_keys(monkeypatch, seen)
    result = runner.invoke(app, [
        "generate", "--out", str(tmp_path / "run"), "--steps", "4", "--split", "train:1:0",
        "--num-envs", "1", "--render", "--render-views", "composite,fpv",
    ])
    assert result.exit_code == 0, result.output
    assert seen["keys"] == (FPV_KEY, "observation.images.composite")


def test_an_unknown_render_view_is_a_usage_error(tmp_path):
    """Caught before the rollout: the value is only read once an hour of
    rollout is already spent, and a typo must not cost that."""
    result = runner.invoke(app, [
        "generate", "--out", str(tmp_path / "run"), "--steps", "4", "--split", "train:1:0",
        "--render", "--render-views", "DRAGON_FPV",
    ])
    assert result.exit_code != 0
    assert "--render-views" in result.output and "dragon_fpv" in result.output


def _env_config_with_ports(tmp_path, *names: str):
    from owm_envs.envs.iss.config import DockConfig, ISSConfig as Cfg

    path = tmp_path / "env.yaml"
    Cfg(dock=DockConfig(ports=names)).to_yaml(path)
    return path


def test_env_dock_ports_the_policy_does_not_share_is_a_usage_error(tmp_path):
    """Generation flies the POLICY's ports. An env config that names its own
    and a policy that names none would quietly write single-target data, so
    the disagreement is refused rather than ignored -- as ISSVectorEnv refuses
    it. Caught before the rollout, not after an hour of it."""
    cfg_path = _env_config_with_ports(tmp_path, "zvezda_aft", "poisk_zenith")
    result = runner.invoke(
        app,
        ["generate", "--out", str(tmp_path / "run"), "--config", str(cfg_path),
         "--split", "train:1:0", "--steps", "4", "--policy", "dock", "--num-envs", "1",
         "--driver", "scan", "--no-lerobot"],
    )
    assert result.exit_code != 0
    assert "dock ports disagree" in result.output
    # Names both sides, so the fix does not need a source dive.
    assert "zvezda_aft" in result.output and "--dock-ports" in result.output
    assert not (tmp_path / "run").exists()


def test_matching_dock_ports_on_both_sides_generate_normally(tmp_path):
    cfg_path = _env_config_with_ports(tmp_path, "zvezda_aft", "poisk_zenith")
    out = tmp_path / "run"
    result = runner.invoke(
        app,
        ["generate", "--out", str(out), "--config", str(cfg_path),
         "--split", "train:2:0", "--steps", "8", "--policy", "dock", "--num-envs", "2",
         "--driver", "scan", "--no-lerobot", "--dock-ports", "zvezda_aft,poisk_zenith"],
    )
    assert result.exit_code == 0, result.output
    assert ISSConfig.from_yaml(out / "env_config.yaml").dock.ports[0].name == "zvezda_aft"


def test_an_env_config_naming_no_ports_is_left_alone(tmp_path):
    """Every config written before DockConfig.ports existed: it says nothing
    about ports, so a policy port set is not a disagreement with it."""
    from owm_envs.envs.iss.config import ISSConfig as Cfg

    cfg_path = tmp_path / "env.yaml"
    Cfg().to_yaml(cfg_path)
    out = tmp_path / "run"
    result = runner.invoke(
        app,
        ["generate", "--out", str(out), "--config", str(cfg_path),
         "--split", "train:2:0", "--steps", "8", "--policy", "dock", "--num-envs", "2",
         "--driver", "scan", "--no-lerobot", "--dock-ports", "all"],
    )
    assert result.exit_code == 0, result.output


def test_a_later_splits_port_mismatch_stops_before_any_split_generates(
    tmp_path, monkeypatch
):
    """The check is pre-flight, not just-in-time.

    Splits generate in order, so checking each one as it comes up would find a
    disagreement in the last split only after every earlier split had already
    paid for its full rollout. `train` agrees here and `val` does not; nothing
    may generate.
    """
    import owm_envs.cli as cli_module

    resolved = []
    real_resolve = cli_module._resolve_driver

    def recording_resolve(*args, **kwargs):
        resolved.append(args)
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(cli_module, "_resolve_driver", recording_resolve)

    cfg_path = _env_config_with_ports(tmp_path, "zvezda_aft")
    result = runner.invoke(
        app,
        ["generate", "--out", str(tmp_path / "run"), "--config", str(cfg_path),
         "--split", "train:2:0:dock:zvezda_aft", "--split", "val:2:1:dock:poisk_zenith",
         "--steps", "8", "--num-envs", "2", "--driver", "scan", "--no-lerobot"],
    )
    assert result.exit_code != 0
    # Names the split that disagrees, not the one that was fine.
    assert "'val'" in result.output and "dock ports disagree" in result.output
    # The load-bearing assertion: no driver was ever built, so no rollout ran.
    assert resolved == []
    assert not (tmp_path / "run").exists()
