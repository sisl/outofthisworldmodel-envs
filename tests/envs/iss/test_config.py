import warnings

import numpy as np
import pytest
import yaml

from owm_envs.envs.iss.config import (
    DEFAULT_COLLISION_BOXES_FILENAME,
    ISSConfig,
    PhysicsConfig,
    RewardWeights,
    _RESOURCES_DIR,
    default_collision_boxes_path,
    load_collision_boxes,
)


def test_default_config_matches_expected_values():
    cfg = ISSConfig()
    assert cfg.dt == 0.05
    assert cfg.max_steps == 7200
    assert cfg.physics.mass == 12000.0
    assert cfg.physics.inertia_diag == (80000.0, 80000.0, 50000.0)
    assert cfg.physics.start_radius_range_m == (100.0, 500.0)
    # Draco-class actuator limits: ~4x400 N per axis, couple torque ~2000 N*m.
    assert cfg.control.limit_force_n == 1600.0
    assert cfg.control.limit_torque_nm == 2000.0
    assert cfg.dock.enabled is True
    assert cfg.dock.max_distance_m == 0.1
    assert cfg.dock.max_velocity_m_s == 0.5


def test_max_range_defaults_to_750_metres():
    assert ISSConfig().max_range_m == 750.0


def test_max_range_explicit_none_survives_toml_roundtrip(tmp_path):
    # max_range_m defaults to 750.0, not None, so an explicit None has to go
    # through ConfigModel's explicit-null machinery: omitting it the way TOML
    # omits any None would silently resurrect the default bound on load and
    # terminate episodes a caller deliberately let run unbounded.
    original = ISSConfig(max_range_m=None)
    path = tmp_path / "run_config.toml"
    original.to_toml(path)
    loaded = ISSConfig.from_toml(path)
    assert loaded.max_range_m is None
    assert loaded == original


@pytest.mark.parametrize("max_range_m", [0.0, -1.0, float("inf"), float("nan")])
def test_non_positive_or_non_finite_max_range_is_rejected(max_range_m):
    # A zero or negative bound would put every reachable state outside the
    # domain, ending each episode on its first step; inf/NaN describe no
    # boundary at all -- None is how the bound is turned off.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ISSConfig(max_range_m=max_range_m)


def test_config_is_frozen():
    from pydantic import ValidationError

    cfg = ISSConfig()
    with pytest.raises(ValidationError):
        cfg.dt = 0.1


def test_config_roundtrips_through_yaml(tmp_path):
    original = ISSConfig(
        dt=0.02, max_steps=500, physics=PhysicsConfig(start_radius_range_m=(250.0, 250.0))
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
    assert centers.shape == (313, 3)
    assert half_extents.shape == (313, 3)
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


def test_load_collision_boxes_none_warns():
    # A silently collision-free environment can never terminate on
    # collision, so opting out via None must say so.
    with pytest.warns(UserWarning, match="no collision geometry"):
        load_collision_boxes(None)


def test_load_collision_boxes_missing_file_raises(tmp_path):
    # Silently returning an empty set here would make a misconfigured path
    # look like "no collision" at runtime. Fail loudly instead.
    with pytest.raises(FileNotFoundError):
        load_collision_boxes(str(tmp_path / "nope.yaml"))


def test_load_collision_boxes_relative_path_resolves_against_cwd(tmp_path, monkeypatch):
    # A user's own geometry, referenced by a relative path, must resolve
    # against wherever they're running from -- not get silently redirected
    # into the package's resources directory, where it doesn't exist.
    boxes = [{"center": [1.0, 2.0, 3.0], "size": [2.0, 2.0, 2.0]}]
    (tmp_path / "my_boxes.yaml").write_text(yaml.safe_dump(boxes))
    monkeypatch.chdir(tmp_path)

    centers, half_extents = load_collision_boxes("my_boxes.yaml")

    np.testing.assert_allclose(centers, [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(half_extents, [[1.0, 1.0, 1.0]])


def test_load_collision_boxes_relative_path_still_finds_shipped_default(
    tmp_path, monkeypatch
):
    # The shipped default filename must keep resolving to the packaged
    # geometry, from an arbitrary cwd that doesn't itself contain the file.
    monkeypatch.chdir(tmp_path)
    centers, half_extents = load_collision_boxes(DEFAULT_COLLISION_BOXES_FILENAME)
    assert centers.shape == (313, 3)
    assert half_extents.shape == (313, 3)


def test_load_collision_boxes_cwd_wins_over_package_on_name_collision(
    tmp_path, monkeypatch
):
    # If a user's cwd happens to hold a file with the same name as the
    # shipped default, their file must win -- otherwise a relative path
    # would be ambiguous and could silently load the wrong geometry.
    boxes = [{"center": [0.0, 0.0, 0.0], "size": [1.0, 1.0, 1.0]}]
    (tmp_path / DEFAULT_COLLISION_BOXES_FILENAME).write_text(yaml.safe_dump(boxes))
    monkeypatch.chdir(tmp_path)

    centers, half_extents = load_collision_boxes(DEFAULT_COLLISION_BOXES_FILENAME)

    assert centers.shape == (1, 3)
    np.testing.assert_allclose(centers, [[0.0, 0.0, 0.0]])


def test_load_collision_boxes_missing_relative_path_names_both_locations(
    tmp_path, monkeypatch
):
    # Neither the cwd nor the package resources directory has this file --
    # the error should say where it looked, not just that it failed.
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError) as excinfo:
        load_collision_boxes("nope.yaml")
    message = str(excinfo.value)
    assert str(tmp_path / "nope.yaml") in message
    assert str(_RESOURCES_DIR / "nope.yaml") in message


def test_load_collision_boxes_absolute_path_still_works(tmp_path):
    boxes = [{"center": [4.0, 5.0, 6.0], "size": [2.0, 2.0, 2.0]}]
    path = tmp_path / "abs_boxes.yaml"
    path.write_text(yaml.safe_dump(boxes))

    centers, half_extents = load_collision_boxes(str(path))

    np.testing.assert_allclose(centers, [[4.0, 5.0, 6.0]])
    np.testing.assert_allclose(half_extents, [[1.0, 1.0, 1.0]])


def test_load_collision_boxes_empty_file_gives_empty_arrays_without_warning(tmp_path):
    # An empty YAML file parses to None, same as the sentinel for "no path
    # configured" -- but here a real path *was* given, it just holds zero
    # boxes. That must not be misattributed to the None-path warning.
    path = tmp_path / "empty.yaml"
    path.write_text("")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        centers, half_extents = load_collision_boxes(str(path))
    assert centers.shape == (0, 3)
    assert half_extents.shape == (0, 3)


@pytest.mark.parametrize("dt", [0.0, -0.05, float("inf"), float("nan")])
def test_non_positive_or_non_finite_dt_is_rejected(dt):
    # dt scales every RK4 stage, so 0.0 advances nothing and a negative value
    # integrates backwards. It is also the divisor behind the recorded frame
    # rate 1/dt, where 0.0 raises instead of yielding a wrong rate.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ISSConfig(dt=dt)


@pytest.mark.parametrize("mass", [0.0, -1.0, float("inf"), float("nan")])
def test_non_positive_or_non_finite_mass_is_rejected(mass):
    # mass is a divisor in _eom's force-over-mass acceleration: 0.0 gives inf
    # and a negative value reverses the sign, so both must be rejected at
    # config load rather than surfacing as a bad acceleration later. inf/NaN
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


@pytest.mark.parametrize(
    "radius_range",
    [(-1.0, 1.0), (float("inf"), float("inf")), (float("nan"), 1.0), (200.0, 100.0)],
)
def test_negative_reversed_or_non_finite_start_radius_range_is_rejected(radius_range):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PhysicsConfig(start_radius_range_m=radius_range)


def test_zero_collision_radius_and_start_radius_are_still_allowed():
    # These are degenerate but not physically impossible, unlike mass/inertia
    # dividing by zero -- only negative values are meaningless for a radius.
    PhysicsConfig(dragon_collision_radius_m=0.0, start_radius_range_m=(0.0, 0.0))


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
