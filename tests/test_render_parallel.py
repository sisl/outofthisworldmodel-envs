"""Parallel rendering must be bit-identical to sequential, and the writer
must accept a lazy clip iterator."""

import gc
import weakref

import numpy as np
import pytest

from owm_envs.drivers.types import TrajectoryBatch

from .test_true_state import _scan_batch

pytest.importorskip("pygfx", reason="rendering is an optional extra")

from owm_envs.datasets.video import iter_batch_frames, render_batch_frames  # noqa: E402


class _FakeRenderer:
    """Stands in for ISSRenderer so the ordering and lifetime tests need no
    GPU or scene assets -- the clip it returns encodes the state it was given,
    which is what the ordering assertions read back."""

    instances = 0
    closes = 0

    def __init__(self, cfg):
        _FakeRenderer.instances += 1
        self.cfg = cfg

    def render(self, state, action=None, view="DRAGON_FPV"):
        frame = np.zeros((self.cfg.image_height, self.cfg.image_width, 3), dtype=np.uint8)
        frame[0, 0, 0] = int(state[0])
        return frame

    def close(self):
        _FakeRenderer.closes += 1


class _Cfg:
    image_width = 4
    image_height = 4


def _fake_batch(lengths=(3, 2, 4)):
    """A batch whose observation dim 0 is a unique per-episode marker."""
    n = len(lengths)
    width = max(lengths)
    obs = np.zeros((n, width, 13), dtype=np.float32)
    for i, length in enumerate(lengths):
        obs[i, :length, 0] = i + 1
    return TrajectoryBatch(
        observations=obs,
        actions=np.zeros((n, width, 6), dtype=np.float32),
        rewards=np.zeros((n, width), dtype=np.float32),
        lengths=np.array(lengths, dtype=np.int32),
        terminated=np.array([True] + [False] * (n - 1)),
        truncated=np.array([False] + [True] * (n - 1)),
        policy_ids=None,
    )


@pytest.fixture
def fake_renderer(monkeypatch):
    _FakeRenderer.instances = 0
    _FakeRenderer.closes = 0
    monkeypatch.setattr("owm_envs.render.renderer.ISSRenderer", _FakeRenderer)
    return _FakeRenderer


def test_single_worker_yields_clips_in_episode_order(fake_renderer):
    batch = _fake_batch()
    clips = list(iter_batch_frames(batch, _Cfg()))
    assert [clip.shape[0] for clip in clips] == [3, 2, 4]
    assert [int(clip[0, 0, 0, 0]) for clip in clips] == [1, 2, 3]


def test_single_worker_reuses_one_renderer_and_closes_it(fake_renderer):
    list(iter_batch_frames(_fake_batch(), _Cfg()))
    assert fake_renderer.instances == 1
    assert fake_renderer.closes == 1


def test_abandoning_the_iterator_still_closes_the_renderer(fake_renderer):
    # The writer may raise part-way through consuming the clips, and a
    # renderer holds ~200 MB of GPU buffers, so it has to be released even
    # when the iterator is never exhausted.
    clips = iter_batch_frames(_fake_batch(), _Cfg())
    next(clips)
    clips.close()
    assert fake_renderer.closes == 1


def test_parallel_matches_sequential():
    """Needs a real GPU renderer: the point is that a worker process renders
    the same episode to the same bytes as the in-process path."""
    batch = _scan_batch("off", goal_error=False)
    from owm_envs.render.iss_scene import RenderConfig

    cfg = RenderConfig(image_width=64, image_height=64)
    sequential = render_batch_frames(batch, cfg)
    parallel = list(iter_batch_frames(batch, cfg, workers=2))
    assert len(parallel) == len(sequential)
    for seq, par in zip(sequential, parallel):
        np.testing.assert_array_equal(seq, par)


def _writer():
    pytest.importorskip("lerobot", reason="lerobot is an optional extra")
    from owm_envs.datasets.lerobot_writer import write_lerobot_split

    return write_lerobot_split


def _clips(lengths, size=32):
    """32x32, not smaller: SVT-AV1 raises SIGFPE encoding a tiny frame."""
    return [np.zeros((int(length), size, size, 3), dtype=np.uint8) for length in lengths]


def test_writer_accepts_iterator(tmp_path):
    write_lerobot_split = _writer()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    batch = _fake_batch()
    write_lerobot_split(
        tmp_path / "train", "iss/train", batch, fps=20, frames=iter(_clips(batch.lengths))
    )
    dataset = LeRobotDataset("iss/train", root=tmp_path / "train")
    assert dataset.features["observation.images.fpv"]["shape"] == (32, 32, 3)
    assert dataset.num_frames == int(batch.lengths.sum())


def test_writer_releases_each_clip_before_pulling_the_next(tmp_path):
    """Streaming is the whole point: holding every clip is the ~98 GB
    materialization this replaces, so a consumed clip must be collectable
    while the next episodes are still being written."""
    write_lerobot_split = _writer()

    batch = _fake_batch()
    refs: list[weakref.ref] = []

    def clips():
        for length in batch.lengths:
            clip = np.zeros((int(length), 32, 32, 3), dtype=np.uint8)
            refs.append(weakref.ref(clip))
            yield clip
            del clip
            gc.collect()
            # Everything but the clip just handed over is finished with.
            assert [i for i, ref in enumerate(refs[:-1]) if ref() is not None] == []

    write_lerobot_split(tmp_path / "lazy", "iss/lazy", batch, fps=20, frames=clips())
    assert len(refs) == batch.num_episodes


def test_writer_rejects_an_iterator_shorter_than_the_batch(tmp_path):
    write_lerobot_split = _writer()

    batch = _fake_batch()
    with pytest.raises(ValueError, match="ran out"):
        write_lerobot_split(
            tmp_path / "short", "iss/short", batch, fps=20,
            frames=iter(_clips(batch.lengths)[:-1]),
        )


def test_writer_rejects_an_iterator_longer_than_the_batch(tmp_path):
    write_lerobot_split = _writer()

    batch = _fake_batch()
    with pytest.raises(ValueError, match="more clips"):
        write_lerobot_split(
            tmp_path / "long", "iss/long", batch, fps=20,
            frames=iter(_clips(list(batch.lengths) + [1])),
        )


def test_writer_rejects_a_mismatched_clip_length_from_an_iterator(tmp_path):
    write_lerobot_split = _writer()

    batch = _fake_batch()
    bad = _clips(batch.lengths)
    bad[1] = np.zeros((bad[1].shape[0] + 1, 32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="length"):
        write_lerobot_split(
            tmp_path / "badlen", "iss/badlen", batch, fps=20, frames=iter(bad)
        )


def test_writer_rejects_a_mismatched_frame_shape_from_an_iterator(tmp_path):
    write_lerobot_split = _writer()

    batch = _fake_batch()
    bad = _clips(batch.lengths)
    bad[1] = np.zeros((bad[1].shape[0], 64, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape"):
        write_lerobot_split(
            tmp_path / "badshape", "iss/badshape", batch, fps=20, frames=iter(bad)
        )
