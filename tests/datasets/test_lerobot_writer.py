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
