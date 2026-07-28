import numpy as np
import pytest

from owm_envs.drivers.types import RolloutSpec, TrajectoryBatch


def make_batch(num_episodes=2, max_len=5, obs_dim=13, act_dim=6, lengths=None):
    lengths = np.array(lengths if lengths is not None else [max_len] * num_episodes, dtype=np.int32)
    return TrajectoryBatch(
        observations=np.zeros((num_episodes, max_len, obs_dim), dtype=np.float32),
        actions=np.zeros((num_episodes, max_len, act_dim), dtype=np.float32),
        rewards=np.zeros((num_episodes, max_len), dtype=np.float32),
        lengths=lengths,
        terminated=np.zeros(num_episodes, dtype=bool),
        truncated=np.zeros(num_episodes, dtype=bool),
        policy_ids=None,
    )


def test_reports_episode_count_and_total_transitions():
    batch = make_batch(num_episodes=3, max_len=10, lengths=[10, 4, 7])
    assert batch.num_episodes == 3
    assert batch.total_transitions == 21


def test_validate_accepts_a_consistent_batch():
    make_batch().validate()


def test_validate_rejects_mismatched_episode_counts():
    batch = make_batch(num_episodes=2)
    bad = TrajectoryBatch(
        observations=batch.observations,
        actions=batch.actions,
        rewards=batch.rewards,
        lengths=np.array([5], dtype=np.int32),  # only 1, should be 2
        terminated=batch.terminated,
        truncated=batch.truncated,
        policy_ids=None,
    )
    with pytest.raises(ValueError, match="episode count"):
        bad.validate()


def test_validate_rejects_length_exceeding_max_len():
    batch = make_batch(num_episodes=1, max_len=5, lengths=[9])
    with pytest.raises(ValueError, match="exceeds"):
        batch.validate()


def test_validate_rejects_an_episode_flagged_both_terminated_and_truncated():
    batch = make_batch(num_episodes=1)
    bad = TrajectoryBatch(
        observations=batch.observations,
        actions=batch.actions,
        rewards=batch.rewards,
        lengths=batch.lengths,
        terminated=np.array([True]),
        truncated=np.array([True]),
        policy_ids=None,
    )
    with pytest.raises(ValueError, match="both"):
        bad.validate()


def test_validate_rejects_nonzero_padding():
    # Padding past `lengths` must be zero -- otherwise the lerobot writer would
    # silently emit garbage frames if it ever mis-slices.
    batch = make_batch(num_episodes=1, max_len=5, lengths=[3])
    batch.observations[0, 4, 0] = 1.0
    with pytest.raises(ValueError, match="padding"):
        batch.validate()


def test_rollout_spec_is_frozen():
    spec = RolloutSpec(num_episodes=4, max_steps=100, seed=0)
    with pytest.raises(Exception):
        spec.num_episodes = 8
