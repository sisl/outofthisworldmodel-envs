import numpy as np
import pytest

from owm_envs.drivers.types import TrajectoryBatch

pytest.importorskip("lerobot", reason="lerobot is an optional extra")

from owm_envs.datasets.lerobot_writer import write_lerobot_split  # noqa: E402


def small_batch(lengths=(4, 3), width=5):
    n = len(lengths)
    obs = np.zeros((n, width, 13), dtype=np.float32)
    act = np.zeros((n, width, 6), dtype=np.float32)
    for i, length in enumerate(lengths):
        obs[i, :length, 0] = np.arange(length)
    return TrajectoryBatch(
        observations=obs,
        actions=act,
        rewards=np.zeros((n, width), dtype=np.float32),
        lengths=np.array(lengths, dtype=np.int32),
        terminated=np.array([True] + [False] * (n - 1)),
        truncated=np.array([False] + [True] * (n - 1)),
        policy_ids=None,
    )


def fake_frames(lengths=(4, 3), size=32):
    """Stand-in for rendered clips -- no GPU needed to test the writer."""
    return [
        np.random.default_rng(i).integers(0, 255, (length, size, size, 3), dtype=np.uint8)
        for i, length in enumerate(lengths)
    ]


def test_video_feature_is_absent_when_no_frames_are_given(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    write_lerobot_split(tmp_path / "a", "iss/a", small_batch(), fps=24)
    ds = LeRobotDataset("iss/a", root=tmp_path / "a")
    assert "observation.images.fpv" not in ds.features


def test_video_feature_is_written_when_frames_are_given(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    write_lerobot_split(
        tmp_path / "b", "iss/b", small_batch(), fps=24, frames=fake_frames(), fpv_size=32
    )
    ds = LeRobotDataset("iss/b", root=tmp_path / "b")
    assert "observation.images.fpv" in ds.features
    assert ds.num_frames == 7  # 4 + 3, padding excluded


def test_frame_count_mismatch_raises(tmp_path):
    # A clip shorter than its episode would silently misalign video with state.
    bad = [np.zeros((2, 32, 32, 3), np.uint8), np.zeros((3, 32, 32, 3), np.uint8)]
    with pytest.raises(ValueError, match="length"):
        write_lerobot_split(
            tmp_path / "c", "iss/c", small_batch(), fps=24, frames=bad, fpv_size=32
        )


def test_vector_features_are_unchanged_when_video_is_added(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    write_lerobot_split(
        tmp_path / "d", "iss/d", small_batch(), fps=24, frames=fake_frames(), fpv_size=32
    )
    ds = LeRobotDataset("iss/d", root=tmp_path / "d")
    assert ds.features["observation_vector"]["shape"] == (13,)
    assert ds.features["action"]["shape"] == (6,)
