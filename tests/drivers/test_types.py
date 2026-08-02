import numpy as np
import pytest

from owm_envs.drivers.types import RolloutSpec, TrajectoryBatch, pack_episodes


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
    # Each episode's last observation is terminal, paired with a zero-pad
    # action, not a real transition: (10-1) + (4-1) + (7-1) = 18, not 21.
    assert batch.total_transitions == 18


def test_total_transitions_is_zero_for_a_single_observation_episode():
    # A length-1 episode has no step taken, so it contributes 0 transitions,
    # not -1.
    batch = make_batch(num_episodes=1, max_len=5, lengths=[1])
    assert batch.total_transitions == 0


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


def test_rollout_spec_rejects_both_num_episodes_and_min_transitions():
    with pytest.raises(ValueError, match="exactly one"):
        RolloutSpec(num_episodes=4, max_steps=100, seed=0, min_transitions=10)


def test_rollout_spec_rejects_neither_num_episodes_nor_min_transitions():
    with pytest.raises(ValueError, match="exactly one"):
        RolloutSpec(max_steps=100, seed=0)


def test_rollout_spec_accepts_min_transitions_mode():
    spec = RolloutSpec(max_steps=5, seed=0, min_transitions=10)
    assert spec.min_transitions == 10
    assert spec.num_episodes is None


def test_rollout_spec_rejects_a_non_positive_min_transitions():
    with pytest.raises(ValueError, match="min_transitions"):
        RolloutSpec(max_steps=5, seed=0, min_transitions=0)


def test_epoch_offsets_defaults_to_none():
    assert make_batch().epoch_offsets is None


def test_validate_accepts_a_batch_with_epoch_offsets():
    batch = make_batch(num_episodes=2)
    with_offsets = TrajectoryBatch(
        observations=batch.observations,
        actions=batch.actions,
        rewards=batch.rewards,
        lengths=batch.lengths,
        terminated=batch.terminated,
        truncated=batch.truncated,
        policy_ids=None,
        epoch_offsets=np.array([1.5, 2.5], dtype=np.float32),
    )
    with_offsets.validate()


def test_validate_rejects_mismatched_epoch_offsets_length():
    batch = make_batch(num_episodes=2)
    bad = TrajectoryBatch(
        observations=batch.observations,
        actions=batch.actions,
        rewards=batch.rewards,
        lengths=batch.lengths,
        terminated=batch.terminated,
        truncated=batch.truncated,
        policy_ids=None,
        epoch_offsets=np.array([1.5], dtype=np.float32),  # only 1, should be 2
    )
    with pytest.raises(ValueError, match="epoch_offsets"):
        bad.validate()


def _episode(length=3, epoch_offset=0.0):
    return {
        "obs": np.zeros((length, 13), dtype=np.float32),
        "act": np.zeros((length, 6), dtype=np.float32),
        "rew": np.zeros((length,), dtype=np.float32),
        "terminated": False,
        "truncated": True,
        "policy_id": 0,
        "epoch_offset": epoch_offset,
    }


def test_pack_episodes_leaves_epoch_offsets_none_by_default():
    batch = pack_episodes([_episode(), _episode()], obs_dim=13, act_dim=6, records_policy_ids=False)
    assert batch.epoch_offsets is None


def test_pack_episodes_carries_epoch_offsets_when_recorded():
    episodes = [_episode(epoch_offset=12.5), _episode(epoch_offset=-3.0)]
    batch = pack_episodes(
        episodes, obs_dim=13, act_dim=6, records_policy_ids=False, records_epoch_offsets=True
    )
    assert batch.epoch_offsets is not None
    assert batch.epoch_offsets.dtype == np.float32
    np.testing.assert_allclose(batch.epoch_offsets, [12.5, -3.0])
