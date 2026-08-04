import gc
import weakref

import numpy as np
import pytest

pytest.importorskip("pygfx", reason="rendering is an optional extra")

from owm_envs.datasets.video import (  # noqa: E402
    COMPOSITE_KEY,
    COMPOSITE_VIEWS,
    FPV_KEY,
    OUTPUT_KEYS,
    VIEW_KEYS,
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
    clips = render_episode_frames(
        small_batch(), 0, _Cfg(), keys=(FPV_KEY, COMPOSITE_KEY)
    )
    assert set(clips) == {FPV_KEY, COMPOSITE_KEY}
    assert clips[COMPOSITE_KEY].shape == clips[FPV_KEY].shape


def test_every_named_view_can_be_written_as_its_own_feature():
    # The full set: six cameras under their own keys plus the mosaic. Each is
    # a video stream of its own, which is what the flag exists to let a run
    # trade away.
    clips = render_episode_frames(small_batch(), 0, _Cfg(), keys=OUTPUT_KEYS)
    assert tuple(clips) == OUTPUT_KEYS
    assert {clip.shape for clip in clips.values()} == {(3, 8, 8, 3)}


def test_each_per_view_feature_carries_its_own_camera():
    # Six keys of the right shape would look identical to six copies of the
    # training view, so check the pixels reach the key named after them.
    class _PerViewRenderer(_FakeRenderer):
        def render_views(self, state, action=None, views=("DRAGON_FPV",)):
            return {
                view: np.full((8, 8, 3), COMPOSITE_VIEWS.index(view) + 1, dtype=np.uint8)
                for view in views
            }

    clips = render_episode_frames(
        small_batch(), 0, _Cfg(), keys=OUTPUT_KEYS, renderer=_PerViewRenderer(_Cfg())
    )
    for index, view in enumerate(COMPOSITE_VIEWS):
        assert set(np.unique(clips[VIEW_KEYS[view]]).tolist()) == {index + 1}


def test_only_the_cameras_a_run_asked_for_are_drawn():
    # The per-frame cost is draws, and a run that wants one view must not pay
    # for six. The composite is the exception -- it is not a camera and needs
    # all of them -- which the test above pins.
    calls = []

    class _CountingRenderer(_FakeRenderer):
        def render_views(self, state, action=None, views=("DRAGON_FPV",)):
            calls.append(tuple(views))
            return super().render_views(state, action, views)

    batch = small_batch()
    render_episode_frames(
        batch,
        0,
        _Cfg(),
        keys=("observation.images.iss_top",),
        renderer=_CountingRenderer(_Cfg()),
    )
    assert calls == [("ISS_TOP",)] * int(batch.lengths[0])


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
    render_episode_frames(
        batch, 0, _Cfg(), keys=(FPV_KEY, COMPOSITE_KEY), renderer=renderer
    )
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


def test_a_failing_debug_clip_does_not_abort_the_dataset_write(tmp_path, monkeypatch):
    """The per-episode mp4 is auxiliary -- nothing reads it back -- so letting
    an encoder or filesystem failure on one out of the tee would abort the
    split write and strand the episodes already on disk, spending the dataset
    on a debug convenience.

    Driven through the real writer because that is what the failure would take
    down. The boundary is per episode rather than a switch thrown on the first
    failure: the clips after the bad one are still worth having.
    """
    pytest.importorskip("lerobot", reason="lerobot is an optional extra")

    import imageio.v3 as iio
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from owm_envs.datasets.lerobot_writer import write_lerobot_split
    from owm_envs.datasets.video import tee_episode_clips

    encode = iio.imwrite

    def failing_encode(path, *args, **kwargs):
        if str(path).endswith("ep_0001.mp4"):
            raise OSError("encoder went away")
        return encode(path, *args, **kwargs)

    monkeypatch.setattr("imageio.v3.imwrite", failing_encode)

    lengths = (3, 2, 4)
    media = tmp_path / "media"
    clips = ({FPV_KEY: np.zeros((length, 32, 32, 3), dtype=np.uint8)} for length in lengths)
    with pytest.warns(UserWarning, match=r"ep_0001\.mp4.*encoder went away"):
        write_lerobot_split(
            tmp_path / "split",
            "iss/split",
            small_batch(lengths),
            fps=20,
            frames=tee_episode_clips(clips, media, fps=20),
        )

    ds = LeRobotDataset("iss/split", root=tmp_path / "split")
    assert ds.num_episodes == len(lengths), "the failed clip took an episode out of the dataset"
    assert ds.num_frames == sum(lengths)
    # Read back rather than trust the counts: the episode whose debug clip
    # failed still has to decode from the dataset's own video, and so does the
    # one written after it.
    starts = [sum(lengths[:i]) for i in range(len(lengths))]
    for episode, start in enumerate(starts):
        frame = ds[start][FPV_KEY]
        assert frame.shape[-2:] == (32, 32), f"episode {episode} did not decode"

    assert sorted(path.name for path in media.glob("*.mp4")) == [
        "ep_0000.mp4",
        "ep_0002.mp4",
    ], "the clip after the failure was not written"
