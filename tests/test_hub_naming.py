"""Hub repo naming and the dataset card, derived from a run's own artifacts.

Nothing here touches the network. Naming and the card are pure reads of a run
directory, and `push_run`'s Hub calls are exercised against a stand-in api
object -- what is being checked there is the call sequence, not the transfer.

The naming tests are load-bearing beyond this module: the six committed
configs/iss/env/*.toml must map onto exactly the six published dataset names, so a
config edit that changes what a dataset IS cannot leave it published under the
old name.
"""

import json
from pathlib import Path

import huggingface_hub
import pytest
import yaml
from pydantic import ValidationError

from owm_envs.datasets.hub import (
    _dataset_card,
    _run_env,
    _segments,
    _state_doc,
    dataset_name,
    hub_namespace,
    push_preview,
    push_run,
)
from owm_envs.datasets.stats import GenerationConfig
from owm_envs.envs import ENV_REGISTRY
from owm_envs.envs.common.config import BaseTaskConfig, ObservationConfig
from owm_envs.envs.common.policies import PolicyConfig
from owm_envs.envs.common.sensing import PRESETS, SensorNoiseConfig
from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss_hcw.config import HCWConfig
from owm_envs.envs.iss_numerical.config import NumericalConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
ISS = CONFIGS / "iss"

# The six variants and the dataset each one becomes. Spelled out literally
# rather than derived: deriving them from the same tag tables the naming code
# uses would make this test agree with any renaming, which is the one thing it
# exists to catch.
# Keyed by the file's stem under configs/iss/env/, which no longer repeats the
# env: the directory carries it.
VARIANT_NAMES = {
    "nonoise_nogoal": "owm-iss-nonoise-nogoal-dt50ms",
    "nonoise_goal": "owm-iss-nonoise-goal-dt50ms",
    "coop_nogoal": "owm-iss-coop-nogoal-dt50ms",
    "coop_goal": "owm-iss-coop-goal-dt50ms",
    "noncoop_nogoal": "owm-iss-noncoop-nogoal-dt50ms",
    "noncoop_goal": "owm-iss-noncoop-goal-dt50ms",
}

COMMIT = "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c"

# The schema a run with truth and no video writes, as lerobot records it in
# each split's meta/info.json.
FEATURES = {
    "observation_vector": {"dtype": "float32", "shape": [25], "names": None},
    "action": {"dtype": "float32", "shape": [6], "names": None},
    "reward": {"dtype": "float32", "shape": [1, 1], "names": None},
    "is_last": {"dtype": "bool", "shape": [1], "names": None},
    "terminated": {"dtype": "bool", "shape": [1], "names": None},
    "truncated": {"dtype": "bool", "shape": [1], "names": None},
    "policy_id": {"dtype": "int64", "shape": [1, 1], "names": None},
    "dock_target": {"dtype": "float32", "shape": [1, 7], "names": None},
    "state_vector": {"dtype": "float64", "shape": [13], "names": None},
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}

COUNTS = {
    "train": {"episodes": 96, "transitions": 500_012, "terminated": 71,
              "truncated": 25, "seconds": 25000.6, "minutes": 416.677, "hours": 6.94461},
    "val": {"episodes": 11, "transitions": 50_004, "terminated": 8,
            "truncated": 3, "seconds": 2500.2, "minutes": 41.67, "hours": 0.6945},
}


def _write_run(
    tmp_path: Path,
    env_cfg: BaseTaskConfig | None = None,
    *,
    env: str = "iss",
    features: dict | None = None,
    video: bool = False,
    finished: bool = True,
) -> Path:
    """A run directory holding exactly the artifacts RunMetadata.write leaves."""
    run = tmp_path / "run"
    run.mkdir()
    env_cfg = env_cfg or ISSConfig.from_toml(ISS / "env" / "noncoop_goal.toml")
    env_cfg.to_yaml(run / "env_config.yaml")
    GenerationConfig.from_yaml(ISS / "gen" / "trial.yaml").to_yaml(
        run / "generation_config.yaml"
    )
    PolicyConfig().to_yaml(run / "policy_config.yaml")
    (run / "normalization_stats.json").write_text("{}")
    (run / "dataset_card.json").write_text(json.dumps({
        "env": env,
        "fps": round(1.0 / env_cfg.dt),
        "dt": env_cfg.dt,
        "splits": {
            name: {"episodes": c["episodes"], "transitions": c["transitions"],
                   "seed": 0 if name == "train" else 1, "max_steps": 7200,
                   "min_transitions": 500_000, "num_episodes_requested": None,
                   "policy_type": "union" if name == "train" else "dock",
                   "union_weights": [0.3, 0.35, 0.35]}
            for name, c in COUNTS.items()
        },
        "provenance": {"owm_envs_version": "0.1.0", "git_commit": COMMIT,
                       "git_dirty": False},
    }))
    for split in COUNTS:
        meta = run / split / "meta"
        meta.mkdir(parents=True)
        (meta / "info.json").write_text(json.dumps({
            "fps": round(1.0 / env_cfg.dt),
            "video_path": "videos/{video_key}/file-000.mp4" if video else None,
            "features": features if features is not None else FEATURES,
        }))
        data = run / split / "data" / "chunk-000"
        data.mkdir(parents=True)
        (data / "file-000.parquet").write_bytes(b"")
    if finished:
        (run / "summary.json").write_text(json.dumps({
            "dataset_root": str(run), "counts": COUNTS, "normalization_stats": {}
        }))
    return run


def _card(run: Path) -> str:
    env_spec, env_cfg, config_parsed = _run_env(run)
    return _dataset_card(
        dataset_name(env_cfg, env=env_spec.name), run, env_spec, env_cfg, config_parsed
    )


def _column(card: str, name: str) -> str:
    """The schema table's row for one column, so a claim can be pinned to the
    cell that makes it rather than to the card as a whole -- which is what
    lets a test say a phrase is absent from one column while present in
    another."""
    (row,) = [line for line in card.splitlines() if line.startswith(f"| `{name}` |")]
    return row


def test_names_cover_the_matrix():
    assert dataset_name(ISSConfig(), env="iss") == "owm-iss-nonoise-nogoal-dt50ms"
    assert dataset_name(
        ISSConfig(sensor_noise=PRESETS["cooperative"],
                  observation=ObservationConfig(goal_error=True)),
        env="iss",
    ) == "owm-iss-coop-goal-dt50ms"
    assert dataset_name(
        ISSConfig(sensor_noise=PRESETS["noncooperative"]), env="iss"
    ) == "owm-iss-noncoop-nogoal-dt50ms"


def test_custom_noise_is_named_custom():
    cfg = ISSConfig(sensor_noise=SensorNoiseConfig(enabled=True, sigma_pos_m=9.9))
    assert dataset_name(cfg, env="iss") == "owm-iss-custom-nogoal-dt50ms"


def test_dt_tag_scales():
    assert dataset_name(ISSConfig(dt=0.1), env="iss") == "owm-iss-nonoise-nogoal-dt100ms"


def test_the_env_name_is_a_component_not_a_prefix():
    assert dataset_name(ISSConfig(), env="lunar") == "owm-lunar-nonoise-nogoal-dt50ms"


@pytest.mark.parametrize("config_name, expected", sorted(VARIANT_NAMES.items()))
def test_each_committed_variant_names_its_published_dataset(config_name, expected):
    assert dataset_name(
        ISSConfig.from_toml(ISS / "env" / f"{config_name}.toml"), env="iss"
    ) == expected


def test_the_six_variants_get_six_distinct_names():
    # Two variants collapsing onto one name would publish one over the other.
    names = {
        dataset_name(ISSConfig.from_toml(ISS / "env" / f"{c}.toml"), env="iss")
        for c in VARIANT_NAMES
    }
    assert len(names) == len(VARIANT_NAMES)


def test_a_run_is_named_and_parsed_through_the_env_that_generated_it(tmp_path):
    # Every step of the publish path used to be hardcoded to iss, which an
    # iss-hcw run cannot survive: configs forbid extra keys, so parsing its
    # as-run config as an ISSConfig fails outright on the reference orbit it
    # carries -- and had it parsed, the repo would have been named after the
    # wrong environment.
    run = _write_run(
        tmp_path, HCWConfig(sensor_noise=PRESETS["cooperative"]), env="iss-hcw"
    )

    env_spec, env_cfg, config_parsed = _run_env(run)
    assert env_spec.name == "iss-hcw"
    assert isinstance(env_cfg, HCWConfig)
    assert config_parsed
    assert push_preview(run)[0] == "owm-iss-hcw-coop-nogoal-dt50ms"
    # The card's own load example names the split the writer actually wrote.
    assert 'LeRobotDataset("iss-hcw/train"' in _card(run)


@pytest.mark.parametrize("recorded_env", [None, "iss-future"])
def test_a_run_the_registry_cannot_place_publishes_as_iss(tmp_path, recorded_env):
    # A run directory written before the card recorded an env, or one naming
    # an env this build does not register: as far as anything here can tell
    # both are iss runs, and reading them that way beats refusing to publish
    # them at all.
    run = _write_run(tmp_path)
    card = json.loads((run / "dataset_card.json").read_text())
    if recorded_env is None:
        del card["env"]
    else:
        card["env"] = recorded_env
    (run / "dataset_card.json").write_text(json.dumps(card))

    assert push_preview(run)[0] == "owm-iss-noncoop-goal-dt50ms"


def test_a_card_naming_an_unknown_env_says_so_and_still_loads_its_own_split(tmp_path):
    # The two names deliberately diverge: the repo is named after the iss
    # fallback, because that is the config class that actually parsed the
    # as-run config, while the load example names the split directory really
    # on disk, which the writer built from the card's own env. What must not
    # happen is a reader taking the iss state layout below for this dataset's.
    run = _write_run(tmp_path, env="iss-future")

    card = _card(run)
    assert "generated by the `iss-future` environment" in card
    assert "does not register" in card
    assert 'LeRobotDataset("iss-future/train"' in card
    assert "# owm-iss-noncoop-goal-dt50ms" in card


def test_an_unknown_envs_own_config_fields_do_not_crash_the_publish_path(tmp_path):
    # The realistic unknown-env run: a future env's config carries sections
    # ISSConfig does not declare, and configs forbid extra keys -- so reading
    # it as an ISSConfig raised, from inside the very path whose job was to
    # WARN that the env is unknown. The card has to come out instead, saying
    # that neither the env nor its config could be placed.
    run = _write_run(
        tmp_path, HCWConfig(dt=0.1, sensor_noise=PRESETS["cooperative"]), env="iss-future"
    )

    env_spec, env_cfg, config_parsed = _run_env(run)
    assert env_spec.name == "iss"
    assert not config_parsed
    # The fields iss shares with it still come from the run, which is what
    # keeps the repo name the run's own rather than a default's.
    assert env_cfg.dt == 0.1
    assert env_cfg.sensor_noise == PRESETS["cooperative"]
    assert push_preview(run)[0] == "owm-iss-coop-nogoal-dt100ms"

    card = _card(run)
    assert "generated by the `iss-future` environment" in card
    assert "did not parse as an `iss` config either" in card


def test_an_unknown_envs_unreadable_config_is_refused_rather_than_defaulted(tmp_path):
    # Dropping the keys iss does not declare is the whole of the tolerance
    # above. A SHARED field rejected on its value means a config this version
    # genuinely cannot read, and quietly substituting defaults there would
    # publish a card -- and a repo name built from the same defaults --
    # describing a run that never happened, with nothing on the card able to
    # say which values were the run's own.
    run = _write_run(tmp_path, HCWConfig(), env="iss-future")
    config = yaml.safe_load((run / "env_config.yaml").read_text())
    config["dt"] = -1.0
    (run / "env_config.yaml").write_text(yaml.safe_dump(config))

    with pytest.raises(ValidationError, match="dt"):
        _run_env(run)


def test_a_card_for_a_known_env_carries_no_unknown_env_warning(tmp_path):
    assert "does not register" not in _card(_write_run(tmp_path))


def test_a_run_with_no_card_at_all_is_refused_by_preview_and_push(tmp_path, api):
    # push_preview tolerating a missing card while push_run crashed on it was
    # the worst of both: the preview named a repo and printed the sizes it was
    # about to mirror over, and only then did the push die. A run this old
    # predates metadata the card is the only source of, so both refuse it, with
    # the same error naming the file.
    run = _write_run(tmp_path)
    (run / "dataset_card.json").unlink()

    with pytest.raises(FileNotFoundError, match="dataset_card.json missing"):
        push_preview(run)
    with pytest.raises(FileNotFoundError, match="hand-write dataset_card.json"):
        push_run(run)
    assert api.calls == []


# An iss-hcw run's own schema: a 15-wide state behind a 2-element epoch
# prefix, and no goal-error block, which is what the shipped config asks for.
HCW_FEATURES = {
    **FEATURES,
    "observation_vector": {"dtype": "float32", "shape": [15], "names": None},
    "state_vector": {"dtype": "float64", "shape": [15], "names": None},
}


def _hcw_run(tmp_path: Path) -> Path:
    """A synthetic run under the committed iss-hcw config, which is what an
    iss-hcw dataset would actually be generated from."""
    return _write_run(
        tmp_path,
        HCWConfig.from_toml(CONFIGS / "iss-hcw" / "env" / "default.toml"),
        env="iss-hcw",
        features=HCW_FEATURES,
    )


def test_the_card_describes_the_state_the_env_actually_carries(tmp_path):
    # iss-hcw stores 15 elements, the first two an absolute epoch. A card
    # describing the iss 13 would omit them entirely and then call the whole
    # row station-relative, which the epoch is not.
    card = _card(_hcw_run(tmp_path))
    assert "epoch as [Julian day, seconds of day] (2;" in card
    assert (
        "body rate (3, rad/s) -- position, velocity, attitude and body rate "
        "station-relative"
    ) in card
    assert "rad/s), all station-relative" not in card


# An iss-numerical run's own schema under the shipped default config
# (observation.mode="relative", no goal-error block): a 15-wide mode-shaped
# observation behind a 21-wide raw state.
NUMERICAL_FEATURES = {
    **FEATURES,
    "observation_vector": {"dtype": "float32", "shape": [15], "names": None},
    "state_vector": {"dtype": "float64", "shape": [21], "names": None},
}


def _numerical_run(tmp_path: Path) -> Path:
    """A synthetic run under the committed iss-numerical config, which is
    what an iss-numerical dataset would actually be generated from."""
    return _write_run(
        tmp_path,
        NumericalConfig.from_toml(CONFIGS / "iss-numerical" / "env" / "default.toml"),
        env="iss-numerical",
        features=NUMERICAL_FEATURES,
    )


def test_the_card_names_iss_numericals_observation_mode_instead_of_a_false_identity(tmp_path):
    # iss-numerical's observation is reshaped by observation.mode (here
    # "relative") rather than being the raw 21D state column for column, so
    # the card must not claim the fixed-slice obs-minus-truth identity every
    # other env's card does, and must instead say which mode reshaped it.
    # The raw layout belongs to the truth column, which is the one column it
    # is true of.
    card = _card(_numerical_run(tmp_path))
    state = _column(card, "state_vector")
    assert "chief ECI position and velocity (6, m, m/s)" in state
    assert "`relative` frame `observation.mode` selects" in state
    assert "- state_vector[" not in card


def test_the_card_describes_iss_numericals_observation_in_the_mode_it_was_recorded_in(tmp_path):
    # The (15,) shape cell beside this column is the shipped mode's, and the
    # raw 21-wide layout describing the truth column asserts four things this
    # row does not carry: a chief block, ECI position and velocity, a
    # body->ECI quaternion and inertial rates. Rendering it here would
    # contradict the width printed next to it.
    observation = _column(_card(_numerical_run(tmp_path)), "observation_vector")
    assert "`relative` frame `observation.mode` selects" in observation
    assert "13-element canonical relative view" in observation
    assert "body to world" in observation
    assert "chief" not in observation
    assert "ECI" not in observation
    # And the policy clause has to be this env's: the scripted policy reads
    # the measured RAW state the env publishes, never this reshaped row.
    assert "The scripted policy did not read this row at all" in observation
    assert "read the canonical relative view out of it" not in observation


@pytest.mark.parametrize(
    "mode, expected, absent",
    [
        ("absolute", "body to ECI/inertial", "body to world"),
        ("chaser_absolute", "the chief measured from the chaser", "body to world"),
        ("chief_absolute", "ROTATING world frame", "body to world"),
    ],
)
def test_each_observation_mode_gets_its_own_column_description(tmp_path, mode, expected, absent):
    # One mode's prose standing in for another's is the same falsehood as the
    # raw layout standing in for all four: chaser_absolute and chief_absolute
    # are both 21 wide and both differ from the state, and only the
    # description says how.
    cfg = NumericalConfig.from_toml(CONFIGS / "iss-numerical" / "env" / "default.toml")
    cfg = cfg.model_copy(update={"observation": cfg.observation.model_copy(update={"mode": mode})})
    features = {
        **NUMERICAL_FEATURES,
        "observation_vector": {"dtype": "float32", "shape": [21], "names": None},
    }
    run = _write_run(tmp_path, cfg, env="iss-numerical", features=features)

    observation = _column(_card(run), "observation_vector")
    assert f"`{mode}` frame `observation.mode` selects" in observation
    assert expected in observation
    assert absent not in observation


def test_the_card_gives_the_epochs_timescale_and_recorded_grain(tmp_path):
    # The simulator carries the epoch at float64 and the dataset records it at
    # float32, so what a consumer can resolve off these two columns is not what
    # the run integrated. Naming the timescale matters for the same reason: the
    # columns are only useful against an ephemeris.
    card = _card(_hcw_run(tmp_path))
    assert "UTC, the day number exact" in card
    assert "coarsens through the day from well under 0.5 ms to 7.8 ms" in card
    # Coarse, but not so coarse that two frames of a 20 Hz run collide. The
    # claim is about what the STEP spans, not about the gap between two
    # rounded timestamps, which is a whole number of ulps either side of it;
    # and 6.4 is the span at the coarsest ulp of the day, so it is a floor.
    assert "a 50 ms step spans at least 6.4 of those ulps" in card


def test_the_card_gives_the_noise_identity_at_this_envs_offsets(tmp_path):
    # The obs-minus-truth identity holds over the relative view, which sits
    # behind the epoch prefix here. Quoting the iss offsets would print an
    # expression whose two operands are not even the same width.
    card = _card(_hcw_run(tmp_path))
    assert "`observation_vector[2:8] - state_vector[2:8]`" in card
    assert "`observation_vector[12:15] - state_vector[12:15]`" in card
    # Never one range spanning the quaternion columns, at these offsets or
    # at the iss ones.
    assert "observation_vector[2:15]" not in card
    assert "observation_vector[:13]" not in card
    # And it has to say why the epoch columns are exempt rather than leave a
    # reader to wonder whether the sensor model touched them -- the epoch
    # phrase is the one named as untouched, so the tail of that phrase is what
    # ties the two together.
    assert "epoch as [Julian day, seconds of day] (2;" in card
    assert "consecutive frames stay distinct) -- is identical in the two channels" in card


@pytest.mark.parametrize("write", [_write_run, _hcw_run], ids=["iss", "iss-hcw"])
def test_the_card_confines_the_noise_identity_to_the_additive_channels(tmp_path, write):
    # Subtracting the four quaternion columns recovers no sigma: attitude
    # error is a rotation COMPOSED onto the true attitude, and then
    # sign-resolved onto its hemisphere. A card promising a difference there
    # would send anyone measuring the model back off the data to a number that
    # means nothing.
    card = _card(write(tmp_path))
    assert "Over the ADDITIVE channels -- position and velocity, and body rate --" in card
    assert "`quat_multiply(quat_conjugate(state_quat), observation_quat)`" in card


def test_the_card_says_what_the_policy_actually_consumed(tmp_path):
    # The observation row is what was RECORDED. What the scripted policy read
    # is the canonical view sliced out of it -- never the epoch columns -- and
    # under observe=state it is not this row at all.
    card = _card(_hcw_run(tmp_path))
    assert "the MEASURED state the policy acted on" not in card
    assert "epoch columns were never policy input" in card
    assert "`--observe state` flew the policy on the true state" in card


def test_the_card_states_the_starts_and_dynamics_of_its_own_env(tmp_path):
    card = _card(_hcw_run(tmp_path))
    assert "from starts between 80 m and 120 m out to a station docking port," in card
    assert "Clohessy-Wiltshire relative dynamics" in card
    assert "rigid-body free-flyer" not in card


def test_the_iss_card_still_reads_as_it_did(tmp_path):
    # The three sentences above are generated now rather than written out, and
    # the published iss cards are regenerated from this code -- so the iss
    # wording of each has to come back unchanged. Pinned exactly, newlines
    # included: the intro's line break moved when the two values became
    # interpolations (they cannot both wrap where the fixed text did), and
    # this is what says where it sits now rather than leaving the next move
    # to go unnoticed.
    card = _card(_write_run(tmp_path))
    assert (
        "position (3, m), velocity (3, m/s), attitude quaternion (4, w-first, body to "
        "world) and body rate (3, rad/s), all station-relative"
    ) in card
    assert "`observation_vector[0:6] - state_vector[0:6]`" in card
    assert "`observation_vector[10:13] - state_vector[10:13]`" in card
    assert (
        "manoeuvring from starts between 100 m and 500 m out to a station docking port,\n"
        "under rigid-body free-flyer dynamics and against the station's 313-box "
        "collision hull."
    ) in card


@pytest.mark.parametrize("env_name", sorted(ENV_REGISTRY))
def test_every_registered_env_can_document_its_own_state(env_name):
    # The state description is generated from the layout, so an env whose
    # layout this cannot describe -- a segment with no prose of its own, or
    # columns no segment covers -- either publishes a card silently missing
    # part of its state or dies at push time, with a whole dataset already
    # generated behind it. Registering the env is what should surface that.
    layout = ENV_REGISTRY[env_name].layout
    doc = _state_doc(layout)
    documented = sum(sl.stop - sl.start for _, sl in _segments(layout))
    assert documented == layout.state_dim, (
        f"{env_name}: the card documents {documented} of {layout.state_dim} "
        f"state columns -- {doc}"
    )


def test_the_card_gives_the_viewer_one_config_per_split(tmp_path):
    card = _card(_write_run(tmp_path))
    for split in COUNTS:
        assert f"  - config_name: {split}\n    data_files: {split}/data/**/*.parquet" in card


def test_the_card_documents_the_truth_channel(tmp_path):
    card = _card(_write_run(tmp_path))
    assert "`state_vector`" in card
    # The obs-minus-truth identity is the whole reason the column is written.
    assert "observation_vector[0:6] - state_vector[0:6]" in card


def test_the_card_omits_the_truth_channel_when_the_run_has_none(tmp_path):
    features = {k: v for k, v in FEATURES.items() if k != "state_vector"}
    card = _card(_write_run(tmp_path, features=features))
    assert "state_vector" not in card


def test_the_card_reports_the_split_sizes(tmp_path):
    card = _card(_write_run(tmp_path))
    assert "| `train` | 96 | 500012 |" in card
    assert "| `val` | 11 | 50004 |" in card


def test_the_card_describes_the_policy_split(tmp_path):
    card = _card(_write_run(tmp_path))
    # The shipped recipe: a union-policy train split over five ports, a
    # dock-policy val split over all eight.
    assert "union" in card and "harmony_fwd_pma2" in card
    for held_out in ("harmony_zenith_cbm", "poisk_zenith"):
        assert held_out in card


def test_the_card_carries_the_sensor_noise_block(tmp_path):
    card = _card(_write_run(tmp_path))
    assert "[sensor_noise]" in card
    assert "sigma_pos_frac_of_range = 0.01" in card
    assert "sigma_vel_m_s = 0.03" in card


def test_a_noise_free_run_says_the_channels_agree(tmp_path):
    cfg = ISSConfig.from_toml(ISS / "env" / "nonoise_nogoal.toml")
    features = {**FEATURES, "observation_vector": {"dtype": "float32", "shape": [13],
                                                   "names": None}}
    card = _card(_write_run(tmp_path, cfg, features=features))
    assert "noise-free" in card
    # A run with no goal-error block must not claim one.
    assert "goal-error block" not in card


def test_the_card_asserts_no_licence(tmp_path):
    # The station geometry these episodes fly against, and the video rendered
    # from it, are derived from third-party assets whose upstream terms were
    # never recorded (see the repository README). A `license:` key in the
    # frontmatter would be an assertion nobody can back.
    frontmatter = _card(_write_run(tmp_path)).split("---")[1]
    assert "license" not in frontmatter


def test_the_card_names_the_asset_sources_and_their_unrecorded_terms(tmp_path):
    card = _card(_write_run(tmp_path))
    assert "## Assets and attribution" in card
    for source in ("ISS_base.glb", "Sketchfab", "CGTrader", "NASA SVS"):
        assert source in card, f"{source} unattributed"
    assert "never recorded" in card


def test_the_card_names_the_source_commit(tmp_path):
    assert COMMIT in _card(_write_run(tmp_path))


def test_the_card_gives_a_regeneration_command(tmp_path):
    card = _card(_write_run(tmp_path))
    assert "owm-envs generate" in card
    # The as-run configs ride along in the repo, so the command names them
    # rather than guessing which committed variant produced the run.
    assert "--env-config env_config.yaml" in card
    assert "--gen-config generation_config.yaml" in card
    assert "--render" not in card


def test_the_regeneration_command_renders_when_the_run_has_video(tmp_path):
    features = {**FEATURES, "observation.images.fpv": {
        "dtype": "video", "shape": [256, 256, 3],
        "names": ["height", "width", "channels"]}}
    card = _card(_write_run(tmp_path, features=features, video=True))
    assert "--render" in card
    assert "observation.images.fpv" in card


def test_a_run_with_no_lerobot_splits_is_not_publishable(tmp_path):
    run = _write_run(tmp_path)
    (run / "train" / "meta" / "info.json").unlink()
    with pytest.raises(FileNotFoundError, match="no LeRobot"):
        _card(run)


class _FakeApi:
    """Stands in for HfApi, recording the call sequence and what it could see."""

    def __init__(self):
        self.calls: list[str] = []
        self.create_kwargs: dict = {}
        self.settings_kwargs: dict = {}
        self.upload_kwargs: dict = {}
        self.card_at_upload: str | None = None

    def whoami(self) -> dict:
        self.calls.append("whoami")
        return {"name": "acct"}

    def create_repo(self, repo_id, **kwargs):
        self.calls.append("create_repo")
        self.create_kwargs = {"repo_id": repo_id, **kwargs}

    def update_repo_settings(self, repo_id, **kwargs):
        self.calls.append("update_repo_settings")
        self.settings_kwargs = {"repo_id": repo_id, **kwargs}

    def upload_folder(self, **kwargs):
        self.calls.append("upload_folder")
        self.upload_kwargs = kwargs
        readme = Path(kwargs["folder_path"]) / "README.md"
        self.card_at_upload = readme.read_text() if readme.exists() else None


@pytest.fixture
def api(monkeypatch) -> _FakeApi:
    fake = _FakeApi()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: fake)
    return fake


def test_push_creates_the_repo_then_uploads_a_carded_folder(tmp_path, api):
    run = _write_run(tmp_path)
    repo_id = push_run(run)

    assert repo_id == "acct/owm-iss-noncoop-goal-dt50ms"
    # The repo has to exist before anything is uploaded into it, and the card
    # has to be on disk before the upload or it is not in that upload.
    assert api.calls == ["whoami", "create_repo", "upload_folder"]
    assert api.create_kwargs == {"repo_id": repo_id, "repo_type": "dataset",
                                 "private": False, "exist_ok": True}
    assert api.upload_kwargs["repo_id"] == repo_id
    assert api.upload_kwargs["repo_type"] == "dataset"
    assert Path(api.upload_kwargs["folder_path"]) == run
    assert api.card_at_upload is not None
    assert repo_id.split("/")[1] in api.card_at_upload


def test_push_mirrors_the_run_rather_than_merging_into_what_is_there(tmp_path, api):
    # A regenerated run must REPLACE its repo. Merging would leave the previous
    # run's parquet behind, still matching the split's viewer glob, so one repo
    # would serve two runs' frames under a README describing only the newer.
    push_run(_write_run(tmp_path))
    assert api.upload_kwargs["delete_patterns"] == "*"


def test_push_honours_an_explicit_name_and_namespace(tmp_path, api):
    run = _write_run(tmp_path)
    assert push_run(run, name="trial", namespace="org") == "org/trial"
    # A given namespace must not cost a whoami round-trip.
    assert "whoami" not in api.calls


@pytest.mark.parametrize("private", [True, False])
def test_push_applies_an_asked_for_visibility_before_uploading(tmp_path, api, private):
    push_run(_write_run(tmp_path), private=private)
    # create_repo's `private` is ignored for a repo that already exists, so the
    # setting has to be applied on its own -- and applied before the data
    # lands, or a repo asked to be private is briefly world-readable.
    assert api.calls == ["whoami", "create_repo", "update_repo_settings", "upload_folder"]
    assert api.settings_kwargs == {"repo_id": "acct/owm-iss-noncoop-goal-dt50ms",
                                   "repo_type": "dataset", "private": private}


def test_push_leaves_an_existing_repos_visibility_alone_by_default(tmp_path, api):
    # Neither flag given: a new repo is public, and a repo someone deliberately
    # made private is not exposed by a routine re-push.
    push_run(_write_run(tmp_path))
    assert "update_repo_settings" not in api.calls
    assert api.create_kwargs["private"] is False


def test_push_preview_names_the_repo_and_the_counts_a_push_would_replace(tmp_path, api):
    # The caller has to be able to see what an upload would mirror over before
    # it happens; asking must itself touch nothing, the Hub included -- a run
    # that cannot be read has to be distinguishable from a login that failed.
    repo_name, counts = push_preview(_write_run(tmp_path))
    assert repo_name == "owm-iss-noncoop-goal-dt50ms"
    assert counts["train"]["episodes"] == 96
    assert counts["val"]["transitions"] == 50_004
    assert api.calls == []


def test_push_preview_refuses_a_run_that_did_not_finish(tmp_path, api):
    with pytest.raises(FileNotFoundError, match="did not finish"):
        push_preview(_write_run(tmp_path, finished=False))


def test_push_preview_refuses_a_summary_holding_no_counts(tmp_path, api):
    # A summary.json that parses but carries no counts is not one this version
    # wrote; indexing it blind would raise a bare KeyError.
    run = _write_run(tmp_path)
    (run / "summary.json").write_text(json.dumps({"dataset_root": str(run)}))
    with pytest.raises(ValueError, match="no split counts"):
        push_preview(run)


def test_the_namespace_defaults_to_the_token_account(api):
    assert hub_namespace() == "acct"
    assert api.calls == ["whoami"]
    # A namespace that was given costs no round-trip at all.
    assert hub_namespace("org") == "org"
    assert api.calls == ["whoami"]


def test_push_refuses_a_run_that_did_not_finish(tmp_path, api):
    run = _write_run(tmp_path, finished=False)
    with pytest.raises(FileNotFoundError, match="did not finish"):
        push_run(run)
    assert api.calls == []
