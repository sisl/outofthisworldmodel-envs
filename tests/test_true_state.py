"""True-state recording through pack_episodes and TrajectoryBatch."""
import numpy as np
import pytest

from owm_envs.drivers.types import TrajectoryBatch, pack_episodes


def _episode(length: int, obs_dim: int = 13, act_dim: int = 6) -> dict:
    rng = np.random.default_rng(length)
    return {
        "obs": rng.normal(size=(length, obs_dim)).astype(np.float32),
        "true_obs": rng.normal(size=(length, 13)).astype(np.float32),
        "act": rng.normal(size=(length, act_dim)).astype(np.float32),
        "rew": rng.normal(size=(length,)).astype(np.float32),
        "terminated": True,
        "truncated": False,
        "policy_id": 0,
        "dock_target": np.zeros(7, dtype=np.float32),
    }


def test_pack_records_true_observations():
    batch = pack_episodes(
        [_episode(4), _episode(2)], obs_dim=13, act_dim=6,
        records_policy_ids=False, records_dock_targets=True,
        records_true_observations=True,
    )
    assert batch.true_observations is not None
    assert batch.true_observations.shape == (2, 4, 13)
    assert np.all(batch.true_observations[1, 2:] == 0)  # zero padding


def test_pack_without_flag_leaves_none():
    batch = pack_episodes(
        [_episode(3)], obs_dim=13, act_dim=6,
        records_policy_ids=False, records_dock_targets=True,
    )
    assert batch.true_observations is None


def test_validate_rejects_wrong_episode_count():
    batch = pack_episodes(
        [_episode(3)], obs_dim=13, act_dim=6,
        records_policy_ids=False, records_dock_targets=True,
        records_true_observations=True,
    )
    bad = TrajectoryBatch(
        observations=batch.observations, actions=batch.actions,
        rewards=batch.rewards, lengths=batch.lengths,
        terminated=batch.terminated, truncated=batch.truncated,
        policy_ids=None, dock_targets=batch.dock_targets,
        true_observations=np.zeros((2, 3, 13), dtype=np.float32),
    )
    with pytest.raises(ValueError):
        bad.validate()
