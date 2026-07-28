import numpy as np
import pytest
import yaml

from owm_envs.envs.iss.config import (
    ISSConfig,
    RewardWeights,
    default_collision_boxes_path,
    load_collision_boxes,
)


def test_default_config_matches_iss2_values():
    cfg = ISSConfig()
    assert cfg.dt == 0.05
    assert cfg.max_steps == 2000
    assert cfg.mass == 12000.0
    assert cfg.inertia_diag == (80000.0, 80000.0, 50000.0)
    assert cfg.start_radius_m == 100.0
    # 9x actuator limits from iss2 -- deliberately unphysical, a dataset-variety knob.
    assert cfg.control_limit_force_n == 18000.0
    assert cfg.control_limit_torque_nm == 90000.0
    assert cfg.dock_enabled is True
    assert cfg.dock_max_distance_m == 0.1
    assert cfg.dock_max_velocity_m_s == 0.5


def test_config_is_frozen():
    from pydantic import ValidationError

    cfg = ISSConfig()
    with pytest.raises(ValidationError):
        cfg.dt = 0.1


def test_config_roundtrips_through_yaml(tmp_path):
    original = ISSConfig(dt=0.02, max_steps=500, start_radius_m=250.0)
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


def test_shipped_default_config_file_matches_code_defaults():
    # configs/iss_default.yaml is the committed, versioned record of the
    # defaults. If someone changes a default in code without regenerating it,
    # this fails -- which is the point.
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    assert ISSConfig.from_yaml(repo_root / "configs" / "iss_default.yaml") == ISSConfig()


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
    # seamstress warned and silently returned an empty set, which makes a
    # misconfigured path look like "no collision" at runtime. Fail loudly instead.
    with pytest.raises(FileNotFoundError):
        load_collision_boxes(str(tmp_path / "nope.yaml"))
