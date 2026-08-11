"""The committed variant configs must stay derivable from code truth.

The six configs/iss/env/*.toml files are the cross product of the sensor-noise
presets and goal-error observation, and the two configs/iss/gen/*.yaml
files are the shipped docking recipe resized. The three
configs/iss-numerical/env/*_goal.toml files are the same noise presets with
goal-error always on, and the two configs/iss-numerical/gen/*.yaml files
are the same recipe retargeted at iss-numerical. All thirteen are written by
scripts/write_iss_variant_configs.py and nothing regenerates them at run time, so
an edit to iss_default.toml, iss_numerical_default.toml, to PRESETS, or to
generation_default.yaml would otherwise leave them silently stale -- and a
stale file here is a published dataset generated under a configuration no
longer in the repo.

Two kinds of assertion, deliberately: that each file still equals what the
writer produces today, and -- separately, spelled out literally -- that the
tags in each filename still mean what they say. The first alone cannot catch
a swapped entry in NOISE_TAGS, because the writer and the files would move
together. The names are load-bearing downstream: they become the published
dataset names.
"""

import sys
from pathlib import Path

import pytest

from owm_envs.datasets.stats import GenerationConfig
from owm_envs.envs.common.sensing import PRESETS
from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss_numerical.config import NumericalConfig

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
ISS = CONFIGS / "iss"
NUMERICAL = CONFIGS / "iss-numerical"

# The writer is a script under scripts/, not a module of the installed
# package, so this guard has to reach it by path to compare the files against
# the very code that produced them.
sys.path.insert(0, str(REPO / "scripts"))

import write_iss_variant_configs as wvc  # noqa: E402


@pytest.fixture(scope="module")
def base() -> ISSConfig:
    return ISSConfig.from_toml(ISS / "env" / "default.toml")


@pytest.mark.parametrize("goal_tag", list(wvc.GOAL_TAGS))
@pytest.mark.parametrize("noise_tag", list(wvc.NOISE_TAGS))
def test_each_variant_still_equals_what_the_writer_produces(noise_tag, goal_tag, base):
    committed = ISSConfig.from_toml(ISS / "env" / f"{noise_tag}_{goal_tag}.toml")
    assert committed == wvc.variant_config(base, noise_tag, goal_tag)


@pytest.mark.parametrize("goal_tag", list(wvc.GOAL_TAGS))
@pytest.mark.parametrize("noise_tag", list(wvc.NOISE_TAGS))
def test_the_variants_change_nothing_but_noise_and_goal_error(noise_tag, goal_tag, base):
    # The point of deriving every variant from one default is that a
    # cross-variant comparison isolates the two axes. Anything else drifting
    # -- dt, the dock pose, the reward weights -- would confound that.
    committed = ISSConfig.from_toml(ISS / "env" / f"{noise_tag}_{goal_tag}.toml")
    stripped = {"sensor_noise", "observation"}
    assert committed.model_dump(exclude=stripped) == base.model_dump(exclude=stripped)


@pytest.mark.parametrize(
    "noise_tag, preset_name",
    [("nonoise", "off"), ("coop", "cooperative"), ("noncoop", "noncooperative")],
)
@pytest.mark.parametrize("goal_tag", list(wvc.GOAL_TAGS))
def test_the_noise_tag_in_the_filename_means_that_preset(noise_tag, preset_name, goal_tag):
    # Spelled out rather than read from wvc.NOISE_TAGS: a typo'd or swapped
    # entry there would rename a dataset without changing its contents, and
    # the equality test above would follow it happily.
    cfg = ISSConfig.from_toml(ISS / "env" / f"{noise_tag}_{goal_tag}.toml")
    assert cfg.sensor_noise == PRESETS[preset_name]
    assert cfg.sensor_noise.enabled is (noise_tag != "nonoise")


@pytest.mark.parametrize("goal_tag, expected", [("nogoal", False), ("goal", True)])
@pytest.mark.parametrize("noise_tag", list(wvc.NOISE_TAGS))
def test_the_goal_tag_in_the_filename_means_that_observation(noise_tag, goal_tag, expected):
    cfg = ISSConfig.from_toml(ISS / "env" / f"{noise_tag}_{goal_tag}.toml")
    assert cfg.observation.goal_error is expected


@pytest.mark.parametrize("goal_tag", list(wvc.GOAL_TAGS))
@pytest.mark.parametrize("noise_tag", list(wvc.NOISE_TAGS))
def test_every_variant_bounds_the_domain_at_750_metres(noise_tag, goal_tag):
    # Spelled out rather than left to the whole-config comparison above: the
    # bound decides how long a runaway episode runs, so all six published
    # datasets have to share it or their episode-length distributions differ
    # on an axis their names do not name.
    cfg = ISSConfig.from_toml(ISS / "env" / f"{noise_tag}_{goal_tag}.toml")
    assert cfg.max_range_m == 750.0


@pytest.fixture(scope="module")
def numerical_base() -> NumericalConfig:
    return NumericalConfig.from_toml(NUMERICAL / "env" / "default.toml")


@pytest.mark.parametrize("noise_tag", list(wvc.NUMERICAL_NOISE_TAGS))
def test_each_numerical_variant_still_equals_what_the_writer_produces(noise_tag, numerical_base):
    committed = NumericalConfig.from_toml(NUMERICAL / "env" / f"{noise_tag}_goal.toml")
    assert committed == wvc.numerical_variant_config(numerical_base, noise_tag)


@pytest.mark.parametrize(
    "noise_tag, preset_name",
    [("nonoise", "off"), ("coop", "cooperative"), ("noncoop", "noncooperative")],
)
def test_the_numerical_noise_tag_in_the_filename_means_that_preset(noise_tag, preset_name):
    # Spelled out rather than read from wvc.NUMERICAL_NOISE_TAGS: a typo'd or
    # swapped entry there would rename a dataset without changing its
    # contents, and the equality test above would follow it happily.
    cfg = NumericalConfig.from_toml(NUMERICAL / "env" / f"{noise_tag}_goal.toml")
    assert cfg.sensor_noise == PRESETS[preset_name]
    assert cfg.sensor_noise.enabled is (noise_tag != "nonoise")


@pytest.mark.parametrize("noise_tag", list(wvc.NUMERICAL_NOISE_TAGS))
def test_every_numerical_variant_bounds_the_domain_at_750_metres(noise_tag):
    cfg = NumericalConfig.from_toml(NUMERICAL / "env" / f"{noise_tag}_goal.toml")
    assert cfg.max_range_m == 750.0


def test_numerical_variants_start_in_the_wide_shell():
    for tag in ("nonoise", "coop", "noncoop"):
        cfg = NumericalConfig.from_toml(NUMERICAL / "env" / f"{tag}_goal.toml")
        assert cfg.orbit.start_radius_range_m == (100.0, 225.0)


def test_numerical_variants_span_a_week_of_epochs():
    for tag in ("nonoise", "coop", "noncoop"):
        cfg = NumericalConfig.from_toml(NUMERICAL / "env" / f"{tag}_goal.toml")
        assert cfg.orbit.epoch_offset_range_s == (0.0, 604_800.0)


def test_numerical_variants_all_observe_the_goal():
    for tag in ("nonoise", "coop", "noncoop"):
        cfg = NumericalConfig.from_toml(NUMERICAL / "env" / f"{tag}_goal.toml")
        assert cfg.observation.goal_error is True
        assert cfg.observation.mode == "relative"


def test_numerical_variants_differ_only_in_sensor_noise():
    configs = {
        tag: NumericalConfig.from_toml(NUMERICAL / "env" / f"{tag}_goal.toml")
        for tag in ("nonoise", "coop", "noncoop")
    }
    dumps = {
        tag: cfg.model_dump(exclude={"sensor_noise"}) for tag, cfg in configs.items()
    }
    assert dumps["nonoise"] == dumps["coop"] == dumps["noncoop"]
    assert configs["nonoise"].sensor_noise.enabled is False
    assert configs["coop"].sensor_noise.enabled is True
    assert configs["noncoop"].sensor_noise.sigma_pos_frac_of_range > 0.0


@pytest.fixture(scope="module")
def default_gen() -> GenerationConfig:
    return GenerationConfig.from_yaml(ISS / "gen" / "default.yaml")


def test_the_500k_recipe_is_sized_in_transitions():
    gen = GenerationConfig.from_yaml(ISS / "gen" / "500k.yaml")
    assert gen.splits["train"].min_transitions == 500_000
    assert gen.splits["val"].min_transitions == 50_000
    # Sized in transitions means sized in transitions: an episode count left
    # behind would make the file one SplitSpec rejects, and the six published
    # datasets would differ in size from each other rather than in noise.
    assert gen.splits["train"].num_episodes is None
    assert gen.splits["val"].num_episodes is None
    assert gen.num_envs == 64


def test_the_trial_recipe_is_a_handful_of_episodes():
    gen = GenerationConfig.from_yaml(ISS / "gen" / "trial.yaml")
    assert (gen.splits["train"].num_episodes, gen.splits["val"].num_episodes) == (64, 2)
    assert gen.splits["train"].min_transitions is None
    assert gen.splits["val"].min_transitions is None


@pytest.mark.parametrize("name", ["500k", "trial"])
def test_the_recipes_keep_train_and_val_on_different_seeds(name):
    gen = GenerationConfig.from_yaml(ISS / "gen" / f"{name}.yaml")
    assert (gen.splits["train"].seed, gen.splits["val"].seed) == (0, 1)


@pytest.mark.parametrize("name", ["500k", "trial"])
def test_the_recipes_preserve_the_shipped_policy_split(name, default_gen):
    # Resizing a split must not disturb the held-out-port design: train is a
    # union policy over the five non-zenith ports, val docks at all eight.
    # Those policies survive only if the nested config round-trips through
    # YAML faithfully, pinned poses and all, so compare them whole.
    gen = GenerationConfig.from_yaml(ISS / "gen" / f"{name}.yaml")
    for split in ("train", "val"):
        assert gen.splits[split].policy == default_gen.splits[split].policy
        assert gen.splits[split].max_steps == default_gen.splits[split].max_steps
    assert gen.splits["train"].policy.type == "union"
    assert gen.splits["val"].policy.type == "dock"
    train_ports = {p.name for p in gen.splits["train"].policy.dock.ports}
    val_ports = {p.name for p in gen.splits["val"].policy.dock.ports}
    assert train_ports < val_ports


@pytest.mark.parametrize("name", ["500k", "trial"])
def test_the_recipes_are_byte_stable_under_regeneration(name):
    # Rerunning the writer must be a no-op on an up-to-date checkout;
    # otherwise every regeneration shows a spurious diff and the guard above
    # stops meaning anything.
    path = ISS / "gen" / f"{name}.yaml"
    assert GenerationConfig.from_yaml(path).to_yaml() == path.read_text()


def test_the_numerical_500k_recipe_is_sized_in_transitions():
    gen = GenerationConfig.from_yaml(NUMERICAL / "gen" / "500k.yaml")
    assert gen.splits["train"].min_transitions == 500_000
    assert gen.splits["val"].min_transitions == 50_000
    assert gen.splits["train"].num_episodes is None
    assert gen.splits["val"].num_episodes is None
    assert gen.num_envs == 64


def test_the_numerical_trial_recipe_is_a_handful_of_episodes():
    gen = GenerationConfig.from_yaml(NUMERICAL / "gen" / "trial.yaml")
    assert (gen.splits["train"].num_episodes, gen.splits["val"].num_episodes) == (64, 2)
    assert gen.splits["train"].min_transitions is None
    assert gen.splits["val"].min_transitions is None


@pytest.mark.parametrize("name", ["500k", "trial"])
def test_the_numerical_recipes_keep_train_and_val_on_different_seeds(name):
    gen = GenerationConfig.from_yaml(ISS / "gen" / f"{name}.yaml")
    assert (gen.splits["train"].seed, gen.splits["val"].seed) == (0, 1)


@pytest.mark.parametrize("name", ["500k", "trial"])
def test_the_numerical_recipes_preserve_the_shipped_policy_split(name, default_gen):
    # Resizing a split must not disturb the held-out-port design, nor the
    # controller gains -- only the orbit policy's radius range, which the
    # numerical recipes widen deliberately, is allowed to differ from the
    # shipped iss recipe.
    gen = GenerationConfig.from_yaml(ISS / "gen" / f"{name}.yaml")
    for split in ("train", "val"):
        assert gen.splits[split].policy.model_dump(exclude={"orbit"}) == (
            default_gen.splits[split].policy.model_dump(exclude={"orbit"})
        )
        assert gen.splits[split].policy.orbit.model_dump(exclude={"radius_range_m"}) == (
            default_gen.splits[split].policy.orbit.model_dump(exclude={"radius_range_m"})
        )
        assert gen.splits[split].max_steps == default_gen.splits[split].max_steps
    assert gen.splits["train"].policy.type == "union"
    assert gen.splits["val"].policy.type == "dock"
    train_ports = {p.name for p in gen.splits["train"].policy.dock.ports}
    val_ports = {p.name for p in gen.splits["val"].policy.dock.ports}
    assert train_ports < val_ports


@pytest.mark.parametrize("name", ["500k", "trial"])
def test_the_numerical_recipes_are_byte_stable_under_regeneration(name):
    path = NUMERICAL / "gen" / f"{name}.yaml"
    assert GenerationConfig.from_yaml(path).to_yaml() == path.read_text()


def test_numerical_recipes_target_the_numerical_env():
    for name in ("trial", "500k"):
        gen = GenerationConfig.from_yaml(NUMERICAL / "gen" / f"{name}.yaml")
        assert gen.env == "iss-numerical"
        for split in gen.splits.values():
            assert split.policy.orbit.radius_range_m == (80.0, 130.0)


def test_numerical_recipes_keep_ports_held_out_of_training():
    gen = GenerationConfig.from_yaml(NUMERICAL / "gen" / "500k.yaml")
    train = {p.name for p in gen.splits["train"].policy.dock.ports}
    val = {p.name for p in gen.splits["val"].policy.dock.ports}
    assert val - train == {"harmony_zenith_cbm", "poisk_zenith", "unity_nadir_cbm"}
