import numpy as np
import pytest

from owm_envs.drivers.types import TrajectoryBatch

lerobot = pytest.importorskip("lerobot", reason="lerobot is an optional extra")

from owm_envs.datasets.lerobot_writer import write_lerobot_split  # noqa: E402


def small_batch():
    obs = np.zeros((2, 5, 13), dtype=np.float32)
    act = np.zeros((2, 5, 6), dtype=np.float32)
    obs[0, :, 0] = np.arange(5)
    obs[1, :3, 0] = np.arange(3)
    return TrajectoryBatch(
        observations=obs,
        actions=act,
        rewards=np.zeros((2, 5), dtype=np.float32),
        lengths=np.array([5, 3], dtype=np.int32),
        terminated=np.array([False, True]),
        truncated=np.array([True, False]),
        policy_ids=None,
    )


def test_writes_a_dataset_directory(tmp_path):
    out = write_lerobot_split(tmp_path / "train", "iss/train", small_batch(), fps=24)
    assert out.exists()
    assert any(out.iterdir())


def test_writes_only_real_frames_not_padding(tmp_path):
    # Episode 0 is 5 frames, episode 1 is 3 -> 8 total, NOT 10.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    write_lerobot_split(tmp_path / "train", "iss/train", small_batch(), fps=24)
    ds = LeRobotDataset("iss/train", root=tmp_path / "train")
    assert ds.num_frames == 8
    assert ds.num_episodes == 2


def test_roundtrips_observation_values(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    batch = small_batch()
    write_lerobot_split(tmp_path / "train", "iss/train", batch, fps=24)
    ds = LeRobotDataset("iss/train", root=tmp_path / "train")
    first = ds[0]["observation_vector"]
    np.testing.assert_allclose(np.asarray(first), batch.observations[0, 0], rtol=1e-5, atol=1e-5)


def test_rejects_a_batch_that_does_not_validate(tmp_path):
    batch = small_batch()
    bad = TrajectoryBatch(
        observations=batch.observations,
        actions=batch.actions,
        rewards=batch.rewards,
        lengths=np.array([99, 3], dtype=np.int32),  # exceeds padded width
        terminated=batch.terminated,
        truncated=batch.truncated,
        policy_ids=None,
    )
    with pytest.raises(ValueError):
        write_lerobot_split(tmp_path / "bad", "iss/bad", bad, fps=24)


def batch_with_metadata():
    """Two episodes with distinct, non-zero reward/terminated/truncated/policy_id
    values so a round-trip can be checked for real, not just for shape."""
    obs = np.zeros((2, 5, 13), dtype=np.float32)
    act = np.zeros((2, 5, 6), dtype=np.float32)
    obs[0, :, 0] = np.arange(5)
    obs[1, :3, 0] = np.arange(3)
    rewards = np.zeros((2, 5), dtype=np.float32)
    rewards[0, :] = np.arange(5, dtype=np.float32) * 0.1
    rewards[1, :3] = np.arange(3, dtype=np.float32) + 10.0
    return TrajectoryBatch(
        observations=obs,
        actions=act,
        rewards=rewards,
        lengths=np.array([5, 3], dtype=np.int32),
        terminated=np.array([False, True]),  # episode 1 terminated
        truncated=np.array([True, False]),  # episode 0 truncated
        policy_ids=np.array([2, 0], dtype=np.int32),
    )


def test_roundtrips_reward_terminated_truncated_policy_id(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    batch = batch_with_metadata()
    write_lerobot_split(tmp_path / "train", "iss/train", batch, fps=24)
    ds = LeRobotDataset("iss/train", root=tmp_path / "train")

    frame = 0
    for episode in range(batch.num_episodes):
        length = int(batch.lengths[episode])
        for t in range(length):
            row = ds[frame]
            assert float(row["reward"].reshape(())) == pytest.approx(float(batch.rewards[episode, t]))
            assert bool(row["terminated"]) == bool(batch.terminated[episode])
            assert bool(row["truncated"]) == bool(batch.truncated[episode])
            assert int(row["policy_id"].reshape(())) == int(batch.policy_ids[episode])
            frame += 1


def test_selects_terminated_episode_frames_via_feature(tmp_path):
    """The reason this schema extension exists: filter frames by outcome
    without needing to join back to the original TrajectoryBatch."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    batch = batch_with_metadata()  # episode 0 truncated (5 frames), episode 1 terminated (3 frames)
    write_lerobot_split(tmp_path / "train", "iss/train", batch, fps=24)
    ds = LeRobotDataset("iss/train", root=tmp_path / "train")

    terminated_frames = [i for i in range(ds.num_frames) if bool(ds[i]["terminated"])]
    assert len(terminated_frames) == int(batch.lengths[1])

    # And they are exactly episode 1's real frames: obs[1, :3, 0] == [0, 1, 2].
    selected_markers = sorted(float(ds[i]["observation_vector"][0]) for i in terminated_frames)
    assert selected_markers == [0.0, 1.0, 2.0]


def test_observation_and_action_schema_unchanged(tmp_path):
    """quickdraw-style consumers only look at these two features; the schema
    extension must not rename or reshape either of them."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    write_lerobot_split(tmp_path / "train", "iss/train", batch_with_metadata(), fps=24)
    ds = LeRobotDataset("iss/train", root=tmp_path / "train")

    assert ds.features["observation_vector"]["dtype"] == "float32"
    assert ds.features["observation_vector"]["shape"] == (13,)
    assert ds.features["action"]["dtype"] == "float32"
    assert ds.features["action"]["shape"] == (6,)
