import gc
import weakref

import numpy as np
import pytest

pytest.importorskip("pygfx", reason="rendering is an optional extra")

from owm_envs.datasets.video import (  # noqa: E402
    COMPOSITE_KEY,
    COMPOSITE_VIEWS,
    FPV_KEY,
    render_batch_frames,
    render_episode_frames,
)
from owm_envs.drivers.types import TrajectoryBatch  # noqa: E402


class _FakeRenderer:
    """Stands in for ISSRenderer so these tests don't need real GLB/texture
    assets -- only construction and close() call counts matter here."""

    instances = 0
    closes = 0

    def __init__(self, cfg):
        _FakeRenderer.instances += 1
        self.cfg = cfg

    def render_views(self, state, action=None, views=("DRAGON_FPV",)):
        frame = np.zeros((self.cfg.image_height, self.cfg.image_width, 3), dtype=np.uint8)
        return {view: frame.copy() for view in views}

    def close(self):
        _FakeRenderer.closes += 1


class _Cfg:
    image_width = 8
    image_height = 8


def small_batch(lengths=(3, 2)):
    n = len(lengths)
    width = max(lengths)
    return TrajectoryBatch(
        observations=np.zeros((n, width, 13), dtype=np.float32),
        actions=np.zeros((n, width, 6), dtype=np.float32),
        rewards=np.zeros((n, width), dtype=np.float32),
        lengths=np.array(lengths, dtype=np.int32),
        terminated=np.array([True] + [False] * (n - 1)),
        truncated=np.array([False] + [True] * (n - 1)),
        policy_ids=None,
    )


@pytest.fixture(autouse=True)
def _reset_and_patch(monkeypatch):
    _FakeRenderer.instances = 0
    _FakeRenderer.closes = 0
    monkeypatch.setattr("owm_envs.render.renderer.ISSRenderer", _FakeRenderer)


def test_render_batch_frames_builds_one_renderer_for_the_whole_batch():
    # One renderer serves the whole batch: constructing an ISSRenderer loads
    # the GLB/texture/cubemap assets and uploads ~200 MB to the GPU, costing
    # ~3.6 s. A fresh renderer per episode would repeat that N times for an
    # N-episode batch.
    batch = small_batch()
    frames = render_batch_frames(batch, _Cfg())
    assert len(frames) == batch.num_episodes
    assert _FakeRenderer.instances == 1
    assert _FakeRenderer.closes == 1


def test_render_episode_frames_reuses_a_passed_in_renderer():
    batch = small_batch()
    renderer = _FakeRenderer(_Cfg())
    render_episode_frames(batch, 0, _Cfg(), renderer=renderer)
    render_episode_frames(batch, 1, _Cfg(), renderer=renderer)
    assert _FakeRenderer.instances == 1  # only the one built above
    assert _FakeRenderer.closes == 0  # the caller owns closing a passed-in renderer


def test_render_episode_frames_without_a_renderer_builds_and_closes_its_own():
    # Standalone use: no renderer given, so one is built and closed just for
    # this call.
    batch = small_batch()
    render_episode_frames(batch, 0, _Cfg())
    assert _FakeRenderer.instances == 1
    assert _FakeRenderer.closes == 1


def test_the_default_renders_only_the_egocentric_training_view():
    # Every extra feature costs a full render per frame and a full video
    # stream, so the default must stay what training consumes.
    clips = render_episode_frames(small_batch(), 0, _Cfg())
    assert list(clips) == [FPV_KEY]


def test_the_composite_adds_one_feature_the_size_of_a_single_view():
    # Six views in one frame, not six frames: the mosaic is the same shape as
    # the training view, which is what keeps it affordable to store.
    clips = render_episode_frames(small_batch(), 0, _Cfg(), composite=True)
    assert set(clips) == {FPV_KEY, COMPOSITE_KEY}
    assert clips[COMPOSITE_KEY].shape == clips[FPV_KEY].shape


def test_the_composite_asks_for_every_view_once_per_frame():
    # One pose per frame serves all six cameras; asking per view would redo
    # the posing six times for the same instant.
    calls = []

    class _CountingRenderer(_FakeRenderer):
        def render_views(self, state, action=None, views=("DRAGON_FPV",)):
            calls.append(tuple(views))
            return super().render_views(state, action, views)

    batch = small_batch()
    renderer = _CountingRenderer(_Cfg())
    render_episode_frames(batch, 0, _Cfg(), composite=True, renderer=renderer)
    assert calls == [COMPOSITE_VIEWS] * int(batch.lengths[0])


def test_the_composite_tiles_every_view_into_distinct_regions():
    # A mosaic that dropped or duplicated a tile would still be the right
    # shape, so check the tiles carry the views they were given.
    from owm_envs.datasets.video import tile_views

    rendered = {
        view: np.full((8, 8, 3), index * 40 + 10, dtype=np.uint8)
        for index, view in enumerate(COMPOSITE_VIEWS)
    }
    frame = tile_views(rendered, 8, 9)
    tiles = [
        frame[row * 4 : row * 4 + 4, col * 3 : col * 3 + 3]
        for row in range(2)
        for col in range(3)
    ]
    assert [int(tile.mean().round()) for tile in tiles] == [
        index * 40 + 10 for index in range(len(COMPOSITE_VIEWS))
    ]


def test_the_composite_leaves_no_uncovered_strip():
    # 256 does not divide by three. Flooring to a common tile size covers 255
    # of the columns and leaves the last one black; the default 512 px width
    # has the same remainder.
    from owm_envs.datasets.video import tile_views

    rendered = {
        view: np.full((8, 8, 3), index + 1, dtype=np.uint8)
        for index, view in enumerate(COMPOSITE_VIEWS)
    }
    frame = tile_views(rendered, 130, 256)
    assert frame.shape == (130, 256, 3)
    assert frame.min() > 0, "part of the mosaic was left uncovered"
    assert set(np.unique(frame).tolist()) <= {index + 1 for index in range(6)}


def test_the_media_tee_releases_each_episode_before_pulling_the_next(tmp_path):
    """The writer's one-episode video bound runs through here now, and with a
    composite beside the training view there is twice as much of it to hold.

    Checked from the producing side: the tee only asks for the next episode
    once it has let go of the last, so by the time this generator is resumed
    every clip it has handed over must be collectable.
    """
    from owm_envs.datasets.video import FPV_KEY, tee_episode_clips

    refs: list[weakref.ref] = []

    def clips():
        for _ in range(3):
            clip = np.zeros((4, 32, 32, 3), dtype=np.uint8)
            refs.append(weakref.ref(clip))
            yield {FPV_KEY: clip}
            del clip
            gc.collect()
            alive = [i for i, ref in enumerate(refs) if ref() is not None]
            assert alive == [], f"episodes {alive} still held when the next was asked for"

    # Consumed the way the writer consumes: each episode is let go of before
    # the next is asked for. A `for` loop would keep its own binding alive
    # across the request and measure the consumer rather than the tee.
    tee = tee_episode_clips(clips(), tmp_path / "media", fps=20)
    while True:
        try:
            episode = next(tee)
        except StopIteration:
            break
        del episode
    assert len(refs) == 3


def test_closing_the_media_tee_closes_the_render_pool_behind_it(tmp_path):
    """The writer closes the iterator it was handed, which is now the tee. If
    the tee did not pass that on, the pool would only come down when the tee's
    own frame was collected -- which a held traceback delays, and which nothing
    but CPython's refcounting would do at all.

    This test keeps its own reference to the source, so dropping the tee cannot
    stand in for the tee actually closing it.
    """
    from owm_envs.datasets.video import FPV_KEY, tee_episode_clips

    closed: list[bool] = []

    def source():
        try:
            for _ in range(3):
                yield {FPV_KEY: np.zeros((4, 32, 32, 3), dtype=np.uint8)}
        finally:
            closed.append(True)

    clips = source()
    tee = tee_episode_clips(clips, tmp_path / "media", fps=20)
    next(tee)
    tee.close()
    assert closed == [True], "the tee did not close the iterator behind it"
