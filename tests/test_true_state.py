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


def test_pack_preserves_true_observation_values():
    episodes = [_episode(4), _episode(2)]
    batch = pack_episodes(
        episodes, obs_dim=13, act_dim=6,
        records_policy_ids=False, records_dock_targets=True,
        records_true_observations=True,
    )
    for i, episode in enumerate(episodes):
        length = episode["obs"].shape[0]
        np.testing.assert_array_equal(
            batch.true_observations[i, :length], episode["true_obs"]
        )


@pytest.mark.parametrize("bad_shape", [(13,), (1, 13)])
def test_pack_rejects_broadcastable_true_obs(bad_shape):
    """A (13,) or (1, 13) true_obs would broadcast over every timestep and
    produce a batch that validates but holds duplicated state."""
    episode = _episode(4)
    episode["true_obs"] = np.zeros(bad_shape, dtype=np.float32)
    with pytest.raises(ValueError, match="true_obs"):
        pack_episodes(
            [episode], obs_dim=13, act_dim=6,
            records_policy_ids=False, records_dock_targets=True,
            records_true_observations=True,
        )


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


def _rebuild(batch: TrajectoryBatch, true_observations: np.ndarray) -> TrajectoryBatch:
    return TrajectoryBatch(
        observations=batch.observations, actions=batch.actions,
        rewards=batch.rewards, lengths=batch.lengths,
        terminated=batch.terminated, truncated=batch.truncated,
        policy_ids=None, dock_targets=batch.dock_targets,
        true_observations=true_observations,
    )


@pytest.mark.parametrize("shape", [(2, 5, 13), (2, 4, 7)])
def test_validate_rejects_wrong_true_observation_shape(shape):
    batch = pack_episodes(
        [_episode(4), _episode(2)], obs_dim=13, act_dim=6,
        records_policy_ids=False, records_dock_targets=True,
        records_true_observations=True,
    )
    bad = _rebuild(batch, np.zeros(shape, dtype=np.float32))
    with pytest.raises(ValueError, match="true_observations"):
        bad.validate()


def test_validate_rejects_nonzero_true_observation_padding():
    batch = pack_episodes(
        [_episode(4), _episode(2)], obs_dim=13, act_dim=6,
        records_policy_ids=False, records_dock_targets=True,
        records_true_observations=True,
    )
    dirty = batch.true_observations.copy()
    dirty[1, 3] = 1.0
    with pytest.raises(ValueError, match="padding"):
        _rebuild(batch, dirty).validate()
