import numpy as np
import pytest
import yaml

from owm_envs.envs.iss.config import (
    ISSConfig,
    PhysicsConfig,
    RewardWeights,
    default_collision_boxes_path,
    load_collision_boxes,
)


def test_default_config_matches_expected_values():
    cfg = ISSConfig()
    assert cfg.dt == 0.05
    assert cfg.max_steps == 2000
    assert cfg.physics.mass == 12000.0
    assert cfg.physics.inertia_diag == (80000.0, 80000.0, 50000.0)
    assert cfg.physics.start_radius_m == 100.0
    # 9x actuator limits -- deliberately unphysical, a dataset-variety knob.
    assert cfg.control.limit_force_n == 18000.0
    assert cfg.control.limit_torque_nm == 90000.0
    assert cfg.dock.enabled is True
    assert cfg.dock.max_distance_m == 0.1
    assert cfg.dock.max_velocity_m_s == 0.5


def test_config_is_frozen():
    from pydantic import ValidationError

    cfg = ISSConfig()
    with pytest.raises(ValidationError):
        cfg.dt = 0.1


def test_config_roundtrips_through_yaml(tmp_path):
    original = ISSConfig(
        dt=0.02, max_steps=500, physics=PhysicsConfig(start_radius_m=250.0)
    )
    path = tmp_path / "run_config.yaml"
    original.to_yaml(path)
    assert ISSConfig.from_yaml(path) == original


def test_nested_reward_weights_survive_the_roundtrip(tmp_path):
    original = ISSConfig(reward_weights=RewardWeights(position=2.0, collision=10.0))
    path = tmp_path / "run_config.yaml"
    original.to_yaml(path)
    loaded = ISSConfig.from_yaml(path)
    assert loaded.reward_weights.position == 2.0
    assert loaded.reward_weights.collision == 10.0
    assert loaded == original


def test_reward_goal_position_defaults_to_none():
    assert ISSConfig().reward_goal_position is None


def test_reward_goal_position_survives_the_roundtrip(tmp_path):
    original = ISSConfig(reward_goal_position=(0.0, 0.0, 0.0))
    path = tmp_path / "run_config.yaml"
    original.to_yaml(path)
    loaded = ISSConfig.from_yaml(path)
    assert loaded.reward_goal_position == (0.0, 0.0, 0.0)
    assert loaded == original


def test_collision_boxes_path_explicit_none_survives_toml_roundtrip(tmp_path):
    # collision_boxes_path defaults to the shipped geometry's filename, not
    # None, and lives inside the nested `physics` table. The explicit-null
    # dotted-path machinery in ConfigModel must still record it correctly
    # through that extra level of nesting, or a caller who opted out of
    # collision geometry would silently get the default geometry back.
    original = ISSConfig(physics=PhysicsConfig(collision_boxes_path=None))
    path = tmp_path / "run_config.toml"
    original.to_toml(path)
    loaded = ISSConfig.from_toml(path)
    assert loaded.physics.collision_boxes_path is None
    assert loaded == original


def test_shipped_default_config_file_matches_code_defaults():
    # configs/iss_default.toml is the committed, versioned record of the
    # defaults. If someone changes a default in code without regenerating it,
    # this fails -- which is the point.
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    assert ISSConfig.from_toml(repo_root / "configs" / "iss_default.toml") == ISSConfig()


def test_invalid_config_is_rejected_at_load(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("dt: 0.05\ninertia_diag: [1.0, 2.0]\n")  # needs 3 entries
    with pytest.raises(Exception):
        ISSConfig.from_yaml(path)


def test_load_collision_boxes_reads_the_shipped_iss_geometry():
    centers, half_extents = load_collision_boxes(default_collision_boxes_path())
    assert centers.shape == (318, 3)
    assert half_extents.shape == (318, 3)
    assert centers.dtype == np.float32
    assert np.all(half_extents >= 0.0)


def test_load_collision_boxes_halves_the_size_field():
    # YAML stores full `size`; the dynamics needs half-extents.
    boxes = [{"center": [1.0, 2.0, 3.0], "size": [4.0, 6.0, 8.0]}]
    centers, half_extents = load_collision_boxes(boxes)
    np.testing.assert_allclose(centers, [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(half_extents, [[2.0, 3.0, 4.0]])


def test_load_collision_boxes_none_gives_empty_arrays():
    centers, half_extents = load_collision_boxes(None)
    assert centers.shape == (0, 3)
    assert half_extents.shape == (0, 3)


def test_load_collision_boxes_missing_file_raises(tmp_path):
    # Silently returning an empty set here would make a misconfigured path
    # look like "no collision" at runtime. Fail loudly instead.
    with pytest.raises(FileNotFoundError):
        load_collision_boxes(str(tmp_path / "nope.yaml"))


@pytest.mark.parametrize("mass", [0.0, -1.0, float("inf"), float("nan")])
def test_non_positive_or_non_finite_mass_is_rejected(mass):
    # Regression test: mass used to accept 0.0/negative, then _eom's
    # force-over-mass division produced inf/reversed-sign acceleration on the
    # first non-zero-force step instead of failing at config load. inf/NaN
    # are equally meaningless as a divisor.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PhysicsConfig(mass=mass)


@pytest.mark.parametrize(
    "inertia_diag",
    [
        (0.0, 80_000.0, 50_000.0),
        (-1.0, 80_000.0, 50_000.0),
        (float("inf"), 80_000.0, 50_000.0),
        (float("nan"), 80_000.0, 50_000.0),
    ],
)
def test_non_positive_or_non_finite_inertia_component_is_rejected(inertia_diag):
    # Same failure mode as mass: the angular EOM divides by each component.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PhysicsConfig(inertia_diag=inertia_diag)


@pytest.mark.parametrize("radius", [-1.0, float("inf"), float("nan")])
def test_negative_or_non_finite_collision_radius_is_rejected(radius):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PhysicsConfig(dragon_collision_radius_m=radius)


@pytest.mark.parametrize("radius", [-1.0, float("inf"), float("nan")])
def test_negative_or_non_finite_start_radius_is_rejected(radius):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PhysicsConfig(start_radius_m=radius)


def test_zero_collision_radius_and_start_radius_are_still_allowed():
    # These are degenerate but not physically impossible, unlike mass/inertia
    # dividing by zero -- only negative values are meaningless for a radius.
    PhysicsConfig(dragon_collision_radius_m=0.0, start_radius_m=0.0)


def test_non_positive_mass_is_rejected_at_toml_load(tmp_path):
    from pydantic import ValidationError

    text = ISSConfig().to_toml().replace("mass = 12000.0", "mass = -1.0")
    path = tmp_path / "bad.toml"
    path.write_text(text)
    with pytest.raises(ValidationError):
        ISSConfig.from_toml(path)


def test_non_positive_mass_is_rejected_at_yaml_load(tmp_path):
    from pydantic import ValidationError

    text = ISSConfig().to_yaml().replace("mass: 12000.0", "mass: -1.0")
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ValidationError):
        ISSConfig.from_yaml(path)
