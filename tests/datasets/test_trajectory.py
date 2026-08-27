import hashlib
import json

import numpy as np
import pytest

from owm_envs.datasets.trajectory import (
    META_KEYS,
    SHORT_VIEW_NAMES,
    Trajectory,
    load_trajectory,
    save_trajectory,
    start_fingerprint,
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


def _broken(traj, **fields) -> Trajectory:
    return Trajectory(**{**traj.__dict__, **fields})


def test_validate_rejects_narrowed_state(short_trajectory):
    broken = _broken(short_trajectory, state=short_trajectory.state.astype(np.float32))
    with pytest.raises(ValueError, match="state must be stored as float64"):
        broken.validate()


def test_validate_rejects_widened_observation(short_trajectory):
    broken = _broken(short_trajectory, observation=short_trajectory.observation.astype(np.float64))
    with pytest.raises(ValueError, match="observation must be stored as float32"):
        broken.validate()


def test_validate_rejects_wrong_state_width(short_trajectory):
    state = short_trajectory.state
    broken = _broken(short_trajectory, state=np.concatenate([state, state[:, :1]], axis=1))
    with pytest.raises(ValueError, match="21-element state"):
        broken.validate()


def test_validate_rejects_flat_state(short_trajectory):
    broken = _broken(short_trajectory, state=short_trajectory.state.reshape(-1))
    with pytest.raises(ValueError, match="state must be 2-D"):
        broken.validate()


def test_validate_rejects_epoch_that_is_not_the_state_prefix(short_trajectory):
    broken = _broken(short_trajectory, epoch=short_trajectory.epoch + 1.0)
    with pytest.raises(ValueError, match="epoch does not match"):
        broken.validate()


def test_validate_rejects_non_positive_dt(short_trajectory):
    broken = _broken(short_trajectory, meta={**short_trajectory.meta, "dt": 0.0})
    with pytest.raises(ValueError, match="dt=0.0 must be finite and > 0"):
        broken.validate()


def test_validate_rejects_dt_that_disagrees_with_the_config(short_trajectory):
    broken = _broken(short_trajectory, meta={**short_trajectory.meta, "dt": 0.1})
    with pytest.raises(ValueError, match="disagrees with env_config"):
        broken.validate()


def test_validate_rejects_a_method_that_is_not_a_file_stem(short_trajectory):
    broken = _broken(short_trajectory, meta={**short_trajectory.meta, "method": "../escape"})
    with pytest.raises(ValueError, match="bare file stem"):
        broken.validate()


def test_validate_rejects_a_wrong_start_fingerprint(short_trajectory):
    meta = {**short_trajectory.meta, "start_fingerprint": "0" * 16}
    broken = _broken(short_trajectory, meta=meta)
    with pytest.raises(ValueError, match="start_fingerprint"):
        broken.validate()


def test_validate_rejects_unnormalised_actions(short_trajectory):
    actions = short_trajectory.action_norm.copy()
    actions[0, 0] = 2.5
    broken = _broken(short_trajectory, action_norm=actions)
    with pytest.raises(ValueError, match="action_norm reaches 2.5"):
        broken.validate()


def test_validate_rejects_non_finite_rewards(short_trajectory):
    reward = short_trajectory.reward.copy()
    reward[0] = np.nan
    broken = _broken(short_trajectory, reward=reward)
    with pytest.raises(ValueError, match="reward holds non-finite"):
        broken.validate()


def test_start_fingerprint_is_the_digest_of_the_raw_float64_bytes():
    state = np.arange(21, dtype=np.float64)
    digest = start_fingerprint(state)
    assert digest == hashlib.sha256(state.tobytes()).hexdigest()[:16]
    assert len(digest) == 16
    assert digest == start_fingerprint(state.tolist())
    assert digest != start_fingerprint(state + 1.0)


def test_fixture_fingerprint_matches_its_own_start(short_trajectory):
    assert short_trajectory.meta["start_fingerprint"] == start_fingerprint(
        short_trajectory.state[0]
    )


def test_validate_rejects_an_env_config_that_cannot_rebuild_the_env(short_trajectory):
    """The inline config is the file's promise that the scene can be rebuilt."""
    config = {**short_trajectory.meta["env_config"], "control": "not-a-control-block"}
    meta = {**short_trajectory.meta, "env_config": config}
    broken = _broken(short_trajectory, meta=meta)
    with pytest.raises(ValueError, match="does not rebuild iss-numerical's config"):
        broken.validate()
