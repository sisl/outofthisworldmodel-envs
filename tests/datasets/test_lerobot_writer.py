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


def test_batch_without_dock_targets_writes_nan_rows(tmp_path):
    # A source that cannot supply a target still owes every frame a value, and
    # the feature must stay in the schema so the columns do not vary per run.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    batch = small_batch()
    assert batch.dock_targets is None
    write_lerobot_split(tmp_path / "train", "iss/train", batch, fps=24)
    ds = LeRobotDataset("iss/train", root=tmp_path / "train")
    assert np.isnan(np.asarray(ds[0]["dock_target"])).all()


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
    """Two episodes with distinct, non-zero
    reward/terminated/truncated/policy_id/dock_target values so a round-trip
    can be checked for real, not just for shape."""
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
        dock_targets=np.array(
            [[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
             [-4.5, 0.25, 9.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32
        ),
    )


def test_roundtrips_reward_terminated_truncated_policy_id_and_dock_target(tmp_path):
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
            np.testing.assert_allclose(
                np.asarray(row["dock_target"]).reshape(7), batch.dock_targets[episode], atol=1e-6
            )
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


def test_is_last_marks_exactly_the_final_frame_of_each_episode(tmp_path):
    """The end-of-episode signal a frame-wise consumer needs.

    `terminated`/`truncated` are broadcast across every frame of an episode,
    so neither identifies WHERE the episode ends. `is_last` does, and it marks
    the one frame per episode whose action is the zero pad rather than a real
    action.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    batch = batch_with_metadata()  # 5 frames then 3 frames
    write_lerobot_split(tmp_path / "train", "iss/train", batch, fps=20)
    ds = LeRobotDataset("iss/train", root=tmp_path / "train")

    last_frames = [i for i in range(ds.num_frames) if bool(ds[i]["is_last"])]
    assert last_frames == [4, 7]
    assert len(last_frames) == batch.num_episodes

    # Dropping those frames leaves exactly the usable transitions.
    assert ds.num_frames - len(last_frames) == batch.total_transitions


def test_episode_lengths_recoverable_from_written_dataset(tmp_path):
    """True episode length survives the write, by two independent routes."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    batch = batch_with_metadata()
    write_lerobot_split(tmp_path / "train", "iss/train", batch, fps=20)
    ds = LeRobotDataset("iss/train", root=tmp_path / "train")

    expected = [int(n) for n in batch.lengths]

    # Route 1: lerobot's own per-episode bookkeeping.
    assert list(ds.meta.episodes["length"]) == expected

    # Route 2: counting frames up to and including each is_last marker.
    lengths, run = [], 0
    for i in range(ds.num_frames):
        run += 1
        if bool(ds[i]["is_last"]):
            lengths.append(run)
            run = 0
    assert lengths == expected


def test_terminal_frame_carries_zero_pad_action(tmp_path):
    """is_last identifies the frame that must be dropped: its action is the
    zero pad written past the terminal state, not an action that was taken."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    batch = batch_with_metadata()
    # Every real action is non-zero, so a zero action can only be the pad.
    batch.actions[0, :4] = 1.0
    batch.actions[1, :2] = 1.0
    write_lerobot_split(tmp_path / "train", "iss/train", batch, fps=20)
    ds = LeRobotDataset("iss/train", root=tmp_path / "train")

    for i in range(ds.num_frames):
        row = ds[i]
        action_is_pad = not np.any(np.asarray(row["action"]))
        assert action_is_pad == bool(row["is_last"])


def test_observation_and_action_schema_unchanged(tmp_path):
    """observation_vector and action are the two features a trajectory
    consumer reads; the outcome-metadata features must not rename or reshape
    either of them."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    write_lerobot_split(tmp_path / "train", "iss/train", batch_with_metadata(), fps=24)
    ds = LeRobotDataset("iss/train", root=tmp_path / "train")

    assert ds.features["observation_vector"]["dtype"] == "float32"
    assert ds.features["observation_vector"]["shape"] == (13,)
    assert ds.features["action"]["dtype"] == "float32"
    assert ds.features["action"]["shape"] == (6,)


def test_finalizes_the_dataset_without_relying_on_garbage_collection(tmp_path, monkeypatch):
    """The writer must call `finalize()` itself.

    lerobot only writes the parquet footer metadata when the writer is
    finalized; without it the split on disk is not a loadable dataset. A
    dataset that is merely dropped happens to finalize through the writer's
    `__del__`, so a split written this way looks fine as long as nothing keeps
    the object alive -- and stops being written the moment something does
    (a reference held by a caller, a traceback, a delayed collection).
    Holding that reference here is what tells the two apart.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    created = []
    real_create = LeRobotDataset.create

    def capturing_create(*args, **kwargs):
        dataset = real_create(*args, **kwargs)
        created.append(dataset)
        return dataset

    monkeypatch.setattr(LeRobotDataset, "create", capturing_create)
    write_lerobot_split(tmp_path / "train", "iss/train", small_batch(), fps=24)
    assert created, "writer did not create a dataset"

    reloaded = LeRobotDataset("iss/train", root=tmp_path / "train")
    assert reloaded.num_frames == 8
    assert reloaded.num_episodes == 2
