import json

import numpy as np
import pytest

from owm_envs.datasets.trajectory import (
    META_KEYS,
    SHORT_VIEW_NAMES,
    Trajectory,
    load_trajectory,
    save_trajectory,
)


def test_round_trip_preserves_arrays_and_meta(short_trajectory, tmp_path):
    save_trajectory(short_trajectory, tmp_path)
    assert (tmp_path / "trajectory.npz").exists()
    assert (tmp_path / "meta.json").exists()
    loaded = load_trajectory(tmp_path)
    for name in ("epoch", "state", "rel_view", "measured_state", "observation",
                 "action_norm", "action_phys", "reward", "collision", "dock_target"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(short_trajectory, name))
    assert loaded.meta == short_trajectory.meta
    assert loaded.steps == short_trajectory.steps == len(short_trajectory.action_norm)
    assert loaded.dt == short_trajectory.meta["dt"]


def test_meta_json_is_human_readable(short_trajectory, tmp_path):
    save_trajectory(short_trajectory, tmp_path)
    text = (tmp_path / "meta.json").read_text()
    assert json.loads(text)["port"] == "harmony_fwd_pma2"
    assert "\n" in text  # indented, not one line


def test_validate_rejects_length_mismatch(short_trajectory):
    broken = Trajectory(**{**short_trajectory.__dict__, "reward": short_trajectory.reward[:-1]})
    with pytest.raises(ValueError, match="reward"):
        broken.validate()


def test_validate_rejects_missing_meta_key(short_trajectory):
    meta = dict(short_trajectory.meta)
    del meta["outcome"]
    broken = Trajectory(**{**short_trajectory.__dict__, "meta": meta})
    with pytest.raises(ValueError, match="outcome"):
        broken.validate()


def test_validate_rejects_unknown_env(short_trajectory):
    meta = {**short_trajectory.meta, "env": "not-an-env"}
    broken = Trajectory(**{**short_trajectory.__dict__, "meta": meta})
    with pytest.raises(ValueError, match="not-an-env"):
        broken.validate()


def test_meta_keys_cover_the_spec():
    assert set(META_KEYS) == {
        "method", "port", "seed", "env", "env_config", "dt", "rate_hz",
        "action_repeat", "steps", "outcome", "ever_collided", "min_range_m",
        "start_fingerprint", "lighting", "produced_by",
    }


def test_short_view_names():
    assert SHORT_VIEW_NAMES["fpv"] == "fpv"
    assert SHORT_VIEW_NAMES["dragon_iso"] == "iso"
    assert SHORT_VIEW_NAMES["composite"] == "composite"
