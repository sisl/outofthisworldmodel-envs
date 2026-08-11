"""Write the committed ISS variant configs and generation recipes.

Writes the six iss variants (the cross product of the sensor-noise presets --
off / cooperative / noncooperative -- and goal-error observation -- off / on
-- every one derived from configs/iss/env/default.toml so the physics stays
identical across all six and a comparison between two published datasets
isolates the axis their names differ on) and their two generation recipes,
plus the three iss-numerical variants (the same noise presets with goal-error
always on, derived from configs/iss-numerical/env/default.toml) and their two
generation recipes.

The generation recipes are configs/iss/gen/default.yaml resized: the
shipped held-out-port design (a union-policy train split over the five
non-zenith ports, a dock-policy val split over all eight) is carried through
untouched, only the env, the split sizes and the lane count change -- and,
for the numerical recipes, the orbit policy's radius range.

Rerun after changing iss_default.toml, iss_numerical_default.toml,
generation_default.yaml, or PRESETS; tests/test_variant_configs.py fails when
the committed files drift from what this writes.

Usage:
    uv run --extra datasets python scripts/write_iss_variant_configs.py
"""

from pathlib import Path

from owm_envs.datasets.stats import GenerationConfig, SplitSpec
from owm_envs.envs.common.config import ObservationConfig
from owm_envs.envs.common.sensing import PRESETS
from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss_numerical.config import NumericalConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
ISS = CONFIGS / "iss"
NUMERICAL = CONFIGS / "iss-numerical"

# Filename tag -> the thing it names. These tags become the published dataset
# names, so they are part of the interface, not an abbreviation scheme.
NOISE_TAGS = {"nonoise": "off", "coop": "cooperative", "noncoop": "noncooperative"}
GOAL_TAGS = {"nogoal": False, "goal": True}

# The numerical suite publishes the three noise variants only. Goal-error is
# on for all of them: these datasets exist to train RL baselines, and a policy
# with no goal in its observation cannot be told which port it is flying to.
NUMERICAL_NOISE_TAGS = NOISE_TAGS

# The orbit policy's commanded circle, unchanged from the iss recipe.
# Revolution closure is bounded by period, which grows with radius: at the
# 0.75 speed-fraction floor, the largest radius that still closes one
# revolution inside the 360 s horizon is ~148 m, so 130 m is already near
# that ceiling and widening the range would break the guarantee rather than
# protect it.
NUMERICAL_ORBIT_RADIUS_RANGE = (80.0, 130.0)

# One lane per episode is wasteful and one lane for 500k transitions is slow;
# 64 is the width both recipes roll out at.
NUM_ENVS = 64


def variant_config(base: ISSConfig, noise_tag: str, goal_tag: str) -> ISSConfig:
    """`base` with the noise preset and goal-error setting the tags name."""
    return base.model_copy(
        update={
            "sensor_noise": PRESETS[NOISE_TAGS[noise_tag]],
            "observation": ObservationConfig(goal_error=GOAL_TAGS[goal_tag]),
        }
    )


def resized(spec: SplitSpec, size_field: str, value: int) -> SplitSpec:
    """`spec` sized by `size_field` alone, with the other size field cleared.

    Rebuilt through the constructor rather than `model_copy(update=...)`:
    model_copy assigns without revalidating, so it would leave a split with
    both num_episodes and min_transitions set -- exactly the combination
    SplitSpec's validator exists to reject, written out to a file that then
    fails to load. Going through the constructor runs that validator here,
    where the writer can still fail loudly.
    """
    sizes = {"num_episodes": None, "min_transitions": None, size_field: value}
    return SplitSpec(**{**spec.model_dump(), **sizes})


def generation_config(
    train_field: str, train_value: int, val_field: str, val_value: int
) -> GenerationConfig:
    """The shipped recipe with both splits resized and widened to NUM_ENVS."""
    default = GenerationConfig.from_yaml(ISS / "gen" / "default.yaml")
    return GenerationConfig(
        **{
            **default.model_dump(),
            "splits": {
                "train": resized(default.splits["train"], train_field, train_value),
                "val": resized(default.splits["val"], val_field, val_value),
            },
            "num_envs": NUM_ENVS,
        }
    )


def numerical_variant_config(base: NumericalConfig, noise_tag: str) -> NumericalConfig:
    """`base` with the noise preset the tag names and goal-error observations.

    The observation model is copied rather than replaced: iss-numerical's is a
    subtype carrying `mode`, and a plain `ObservationConfig` would drop it.
    """
    return base.model_copy(
        update={
            "sensor_noise": PRESETS[NUMERICAL_NOISE_TAGS[noise_tag]],
            "observation": base.observation.model_copy(update={"goal_error": True}),
        }
    )


def numerical_generation_config(
    train_field: str, train_value: int, val_field: str, val_value: int
) -> GenerationConfig:
    """The shipped held-out-port recipe, retargeted at iss-numerical.

    The port design carries through untouched -- a union-policy train split on
    five ports, a dock-policy val split on all eight, so validation measures
    the held-out approaches. Only the env, the split sizes, the lane count and
    the orbit policy's radius range change.
    """
    default = GenerationConfig.from_yaml(ISS / "gen" / "default.yaml")
    splits = {}
    for name, field, value in (
        ("train", train_field, train_value),
        ("val", val_field, val_value),
    ):
        spec = resized(default.splits[name], field, value)
        policy = spec.policy.model_copy(
            update={
                "orbit": spec.policy.orbit.model_copy(
                    update={"radius_range_m": NUMERICAL_ORBIT_RADIUS_RANGE}
                )
            }
        )
        splits[name] = SplitSpec(**{**spec.model_dump(), "policy": policy})
    return GenerationConfig(
        **{
            **default.model_dump(),
            "env": "iss-numerical",
            "splits": splits,
            "num_envs": NUM_ENVS,
        }
    )


def main() -> None:
    base = ISSConfig.from_toml(ISS / "env" / "default.toml")
    for noise_tag in NOISE_TAGS:
        for goal_tag in GOAL_TAGS:
            path = ISS / "env" / f"{noise_tag}_{goal_tag}.toml"
            variant_config(base, noise_tag, goal_tag).to_toml(path)
            print(f"wrote {path}")

    for path, gen in (
        (
            ISS / "gen" / "500k.yaml",
            generation_config("min_transitions", 500_000, "min_transitions", 50_000),
        ),
        (
            ISS / "gen" / "trial.yaml",
            generation_config("num_episodes", 64, "num_episodes", 2),
        ),
    ):
        gen.to_yaml(path)
        print(f"wrote {path}")

    numerical_base = NumericalConfig.from_toml(NUMERICAL / "env" / "default.toml")
    for noise_tag in NUMERICAL_NOISE_TAGS:
        path = NUMERICAL / "env" / f"{noise_tag}_goal.toml"
        numerical_variant_config(numerical_base, noise_tag).to_toml(path)
        print(f"wrote {path}")

    for path, gen in (
        (
            NUMERICAL / "gen" / "500k.yaml",
            numerical_generation_config("min_transitions", 500_000, "min_transitions", 50_000),
        ),
        (
            NUMERICAL / "gen" / "trial.yaml",
            numerical_generation_config("num_episodes", 64, "num_episodes", 2),
        ),
    ):
        gen.to_yaml(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
