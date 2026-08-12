import numpy as np
import pytest

from owm_envs.drivers.types import TrajectoryBatch

pytest.importorskip("lerobot", reason="lerobot is not installed")

from owm_envs.datasets.lerobot_writer import write_lerobot_split  # noqa: E402
from owm_envs.datasets.video import COMPOSITE_KEY, FPV_KEY  # noqa: E402

FPV = FPV_KEY


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


def fake_frames(lengths=(4, 3), size=32, keys=(FPV,)):
    """Stand-in for rendered clips -- no GPU needed to test the writer."""
    return [
        {
            key: np.random.default_rng(i + j).integers(
                0, 255, (length, size, size, 3), dtype=np.uint8
            )
            for j, key in enumerate(keys)
        }
        for i, length in enumerate(lengths)
    ]


def test_video_feature_is_absent_when_no_frames_are_given(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    write_lerobot_split(tmp_path / "a", "iss/a", small_batch(), fps=24)
    ds = LeRobotDataset("iss/a", root=tmp_path / "a")
    assert "observation.images.fpv" not in ds.features


def test_video_feature_is_written_when_frames_are_given(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    write_lerobot_split(tmp_path / "b", "iss/b", small_batch(), fps=24, frames=fake_frames())
    ds = LeRobotDataset("iss/b", root=tmp_path / "b")
    assert "observation.images.fpv" in ds.features
    assert ds.num_frames == 7  # 4 + 3, padding excluded


def test_frame_count_mismatch_raises(tmp_path):
    # A clip shorter than its episode would silently misalign video with state.
    bad = [{FPV: np.zeros((2, 32, 32, 3), np.uint8)}, {FPV: np.zeros((3, 32, 32, 3), np.uint8)}]
    with pytest.raises(ValueError, match="length"):
        write_lerobot_split(tmp_path / "c", "iss/c", small_batch(), fps=24, frames=bad)


def test_vector_features_are_unchanged_when_video_is_added(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    write_lerobot_split(tmp_path / "d", "iss/d", small_batch(), fps=24, frames=fake_frames())
    ds = LeRobotDataset("iss/d", root=tmp_path / "d")
    assert ds.features["observation_vector"]["shape"] == (13,)
    assert ds.features["action"]["shape"] == (6,)


def test_video_feature_shape_matches_a_nonsquare_clip(tmp_path):
    # The declared feature shape comes from the clip itself: the renderer
    # returns (H, W, 3), and H and W differ whenever the render config is
    # non-square. A feature shape derived from image_width alone instead
    # would fail every add_frame() call for such a config.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    height, width = 96, 160
    frames = [
        {FPV: np.random.default_rng(i).integers(
            0, 255, (length, height, width, 3), dtype=np.uint8
        )}
        for i, length in enumerate((4, 3))
    ]
    write_lerobot_split(tmp_path / "e", "iss/e", small_batch(), fps=24, frames=frames)
    ds = LeRobotDataset("iss/e", root=tmp_path / "e")
    assert ds.features["observation.images.fpv"]["shape"] == (height, width, 3)
    # Reading a frame back must not raise: add_frame()'s shape validator
    # checks every incoming frame against the declared feature shape, so a
    # mismatch between the two would surface right here.
    assert ds[0]["observation.images.fpv"] is not None
    assert ds.num_frames == 7


def test_mismatched_clip_frame_shapes_raise(tmp_path):
    # Episode 0's clip shape sets the declared feature shape; a later episode
    # with a different H/W would silently corrupt the video feature if this
    # weren't caught.
    bad = [{FPV: np.zeros((4, 96, 160, 3), np.uint8)}, {FPV: np.zeros((3, 64, 64, 3), np.uint8)}]
    with pytest.raises(ValueError, match="shape"):
        write_lerobot_split(tmp_path / "f", "iss/f", small_batch(), fps=24, frames=bad)


def test_a_video_feature_is_declared_for_every_key(tmp_path):
    # The training view and the review mosaic are independently encoded video
    # features sharing the vector frames they were rendered from.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    keys = (FPV, COMPOSITE_KEY)
    frames = fake_frames(keys=keys)
    write_lerobot_split(tmp_path / "g", "iss/g", small_batch(), fps=24, frames=frames)
    ds = LeRobotDataset("iss/g", root=tmp_path / "g")
    assert set(keys) <= set(ds.features)
    for key in keys:
        assert ds[0][key] is not None
    assert ds.num_frames == 7


def test_an_episode_missing_a_view_raises(tmp_path):
    # Episode 0 fixes the schema, so an episode that drops a view would leave
    # a declared feature with nothing behind it for those frames.
    keys = (FPV, COMPOSITE_KEY)
    frames = fake_frames(keys=keys)
    del frames[1][keys[1]]
    with pytest.raises(ValueError, match="video features"):
        write_lerobot_split(tmp_path / "h", "iss/h", small_batch(), fps=24, frames=frames)
