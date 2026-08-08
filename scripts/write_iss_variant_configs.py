"""Write the six committed ISS variant configs and the two generation recipes.

The variants are the cross product of the sensor-noise presets (off /
cooperative / noncooperative) and goal-error observation (off / on), every
one derived from configs/iss_default.toml so the physics stays identical
across all six and a comparison between two published datasets isolates the
axis their names differ on.

The generation recipes are configs/generation_default.yaml resized: the
shipped held-out-port design (a union-policy train split over the five
non-zenith ports, a dock-policy val split over all eight) is carried through
untouched, only the split sizes and the lane count change.

Rerun after changing iss_default.toml, generation_default.yaml, or PRESETS;
tests/test_variant_configs.py fails when the committed files drift from what
this writes.

Usage:
    uv run --extra datasets python scripts/write_iss_variant_configs.py
"""

from pathlib import Path

from owm_envs.datasets.stats import GenerationConfig, SplitSpec
from owm_envs.envs.iss.config import ISSConfig, ObservationConfig
from owm_envs.envs.iss.sensing import PRESETS

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

# Filename tag -> the thing it names. These tags become the published dataset
# names, so they are part of the interface, not an abbreviation scheme.
NOISE_TAGS = {"nonoise": "off", "coop": "cooperative", "noncoop": "noncooperative"}
GOAL_TAGS = {"nogoal": False, "goal": True}

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
    default = GenerationConfig.from_yaml(CONFIGS / "generation_default.yaml")
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


def main() -> None:
    base = ISSConfig.from_toml(CONFIGS / "iss_default.toml")
    for noise_tag in NOISE_TAGS:
        for goal_tag in GOAL_TAGS:
            path = CONFIGS / f"iss_{noise_tag}_{goal_tag}.toml"
            variant_config(base, noise_tag, goal_tag).to_toml(path)
            print(f"wrote {path}")

    for path, gen in (
        (
            CONFIGS / "generation_500k.yaml",
            generation_config("min_transitions", 500_000, "min_transitions", 50_000),
        ),
        (
            CONFIGS / "generation_trial.yaml",
            generation_config("num_episodes", 64, "num_episodes", 2),
        ),
    ):
        gen.to_yaml(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
