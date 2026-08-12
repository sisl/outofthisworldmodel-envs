"""Parallel rendering must be bit-identical to sequential, and the writer
must accept a lazy clip iterator."""

import gc
import subprocess
import sys
import textwrap
import time
import weakref

import numpy as np
import pytest

from owm_envs.drivers.types import TrajectoryBatch

from .test_true_state import _scan_batch

pytest.importorskip("pygfx", reason="rendering is an optional extra")

from owm_envs.datasets.video import (  # noqa: E402
    FPV_KEY,
    iter_batch_frames,
    render_adapter_for,
    render_batch_frames,
)
from owm_envs.envs.iss.config import ISSConfig  # noqa: E402

FPV = FPV_KEY

# The environment these batches came from: the render pipeline builds its
# adapter from the pair rather than being handed one, because a spawned
# worker can only be sent something it can rebuild an adapter from.
ISS = {"env_name": "iss", "env_cfg": ISSConfig()}
ISS_STATE_DIM = 13


class _FakeRenderer:
    """Stands in for ISSRenderer so the ordering and lifetime tests need no
    GPU or scene assets -- the clip it returns encodes the state it was given,
    which is what the ordering assertions read back."""

    instances = 0
    closes = 0

    def __init__(self, cfg):
        _FakeRenderer.instances += 1
        self.cfg = cfg

    def render_views(self, inputs, views=("DRAGON_FPV",)):
        frame = np.zeros((self.cfg.image_height, self.cfg.image_width, 3), dtype=np.uint8)
        frame[0, 0, 0] = int(inputs.position_world[0])
        return {view: frame.copy() for view in views}

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
    clips = [episode[FPV] for episode in iter_batch_frames(batch, _Cfg(), **ISS)]
    assert [clip.shape[0] for clip in clips] == [3, 2, 4]
    assert [int(clip[0, 0, 0, 0]) for clip in clips] == [1, 2, 3]


def test_single_worker_reuses_one_renderer_and_closes_it(fake_renderer):
    list(iter_batch_frames(_fake_batch(), _Cfg(), **ISS))
    assert fake_renderer.instances == 1
    assert fake_renderer.closes == 1


def test_abandoning_the_iterator_still_closes_the_renderer(fake_renderer):
    # The writer may raise part-way through consuming the clips, and a
    # renderer holds ~200 MB of GPU buffers, so it has to be released even
    # when the iterator is never exhausted.
    clips = iter_batch_frames(_fake_batch(), _Cfg(), **ISS)
    next(clips)
    clips.close()
    assert fake_renderer.closes == 1


def test_a_worker_never_downloads_its_own_earth_textures(monkeypatch):
    """`generate` resolves all three Earth textures before starting the pool.
    A worker that resolved with downloads enabled would undo that: when the
    parent's fetch failed the source is still absent, so every worker would
    retry the multi-gigabyte download at once, and workers whose retries
    disagreed would render one dataset at two different Earth resolutions."""
    import owm_envs.datasets.video as video

    seen = {}

    class _Recorder:
        def __init__(self, cfg, *, download_textures=True):
            seen["download_textures"] = download_textures

    monkeypatch.setattr("owm_envs.render.renderer.ISSRenderer", _Recorder)
    monkeypatch.setattr("owm_envs.render.device.select_gpu", lambda index: None)
    from owm_envs.render.iss_scene import RenderConfig

    video._worker_init(
        RenderConfig().model_dump_json(), (FPV,), None, "iss", ISSConfig().model_dump_json()
    )
    assert seen["download_textures"] is False


def test_a_worker_builds_its_own_render_adapter(monkeypatch):
    """An adapter reaches a worker as the (env_name, config) it is built from,
    never as a callable: the pool is a spawn one, the child re-imports this
    module from scratch, and what the parent holds is bound to a config the
    child does not have. The config crosses as JSON for the same reason the
    RenderConfig does, so this rebuilds through that round trip.
    """
    import owm_envs.datasets.video as video

    monkeypatch.setattr("owm_envs.render.renderer.ISSRenderer", lambda cfg, **kw: None)
    monkeypatch.setattr("owm_envs.render.device.select_gpu", lambda index: None)
    from owm_envs.render.iss_scene import RenderConfig

    video._worker_init(
        RenderConfig().model_dump_json(), (FPV,), None, "iss", ISSConfig().model_dump_json()
    )
    posed = video._WORKER_ADAPTER(np.arange(13.0), np.zeros(6))
    np.testing.assert_array_equal(posed.position_world, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(posed.quaternion_bw, [6.0, 7.0, 8.0, 9.0])


def test_a_goal_augmented_hcw_batch_is_posed_at_its_own_state_width(monkeypatch):
    """No truth channel, so the pose comes off the observation -- and an
    iss-hcw observation is a 15-wide state (a 2-element epoch prefix ahead of
    the 13D view) followed by the 12-dim goal-error block. What an adapter is
    owed is its env's whole state row, so the width of the row is asserted
    here and not only the pose that came out of it: iss-hcw's own adapter
    happens to read nothing past element 12, so a row truncated to iss's 13
    would render identically today and stop doing so the moment a layout puts
    anything the renderer reads further out. Both paths take their width from
    the registry; this pins the in-process one, which is the one that used to
    compute the right source and then ignore it.
    """
    import owm_envs.datasets.video as video

    from owm_envs.envs.common.epoch_state import epoch_prefix
    from owm_envs.envs.common.goal import GOAL_ERROR_DIM
    from owm_envs.envs.common.orbit import ReferenceOrbit
    from owm_envs.envs.iss_hcw.config import HCWConfig

    posed = []

    class _Capturing(_FakeRenderer):
        def render_views(self, inputs, views=("DRAGON_FPV",)):
            posed.append(inputs)
            return super().render_views(inputs, views)

    monkeypatch.setattr("owm_envs.render.renderer.ISSRenderer", _Capturing)
    cfg = HCWConfig()
    state_dim = 15
    row = np.zeros(state_dim + GOAL_ERROR_DIM, dtype=np.float32)
    row[:2] = np.asarray(epoch_prefix(ReferenceOrbit(cfg.orbit).epoch0), dtype=np.float32)
    row[2:state_dim] = np.arange(1.0, 14.0)
    # A goal block that would be unmistakable if it ever reached a pose.
    row[state_dim:] = 99.0

    batch = TrajectoryBatch(
        observations=row.reshape(1, 1, -1),
        actions=np.zeros((1, 1, 6), dtype=np.float32),
        rewards=np.zeros((1, 1), dtype=np.float32),
        lengths=np.array([1], dtype=np.int32),
        terminated=np.array([True]),
        truncated=np.array([False]),
        policy_ids=None,
    )
    assert batch.true_state is None

    # The real adapter, wrapped so the row it was handed can be inspected.
    real = render_adapter_for("iss-hcw", cfg)
    handed = []

    def recording(state, action=None):
        handed.append(np.asarray(state).copy())
        return real(state, action)

    monkeypatch.setattr(video, "render_adapter_for", lambda env_name, env_cfg: recording)
    list(iter_batch_frames(batch, _Cfg(), env_name="iss-hcw", env_cfg=cfg))

    assert len(handed) == 1
    np.testing.assert_array_equal(handed[0], row[:state_dim])

    expected = real(row[:state_dim], np.zeros(6, dtype=np.float32))
    assert len(posed) == 1
    np.testing.assert_array_equal(posed[0].position_world, expected.position_world)
    np.testing.assert_array_equal(posed[0].quaternion_bw, expected.quaternion_bw)
    np.testing.assert_array_equal(
        posed[0].lighting.sun_direction_world, expected.lighting.sun_direction_world
    )
    assert posed[0].lighting.illumination == expected.lighting.illumination
    assert posed[0].lighting.chief_distance_m == expected.lighting.chief_distance_m


def test_parallel_matches_sequential():
    """Needs a real GPU renderer: the point is that a worker process renders
    the same episode as the in-process path.

    To within one unit per channel, not to the byte. A worker forces JAX onto
    the CPU because it replays stored states and steps no dynamics, while the
    process that renders in-process holds whichever backend it started with;
    the two lower the posing arithmetic differently, and a pose that differs
    in its last bits moves a lit edge across a quantisation boundary here and
    there. What the tolerance still pins down is that both paths draw the same
    scene from the same states: a worker posing from the wrong row, or at the
    wrong layout, is off by far more than a unit.
    """
    batch = _scan_batch("off", goal_error=False)
    from owm_envs.render.iss_scene import RenderConfig

    cfg = RenderConfig(image_width=64, image_height=64)
    sequential = render_batch_frames(
        batch, cfg, adapter=render_adapter_for(**ISS), state_dim=ISS_STATE_DIM
    )
    parallel = list(iter_batch_frames(batch, cfg, workers=2, **ISS))
    assert len(parallel) == len(sequential)
    for seq, par in zip(sequential, parallel):
        assert set(seq) == set(par)
        for key in seq:
            # Signed, so a worker frame darker than the in-process one does
            # not wrap around to 255 and pass.
            drift = np.abs(seq[key].astype(np.int16) - par[key].astype(np.int16))
            assert drift.max() <= 1, (
                f"{key}: {int((drift > 1).sum())} of {drift.size} channels differ "
                f"by more than one unit (worst {int(drift.max())}), which is more "
                f"than backend rounding: the two paths are drawing different scenes"
            )


def test_abandoning_a_pool_iterator_returns_promptly():
    """The writer can raise part-way through the clips. Shutting the pool
    down then waits only for the episodes already running, never for the
    ones still queued."""
    batch = _scan_batch("off", goal_error=False)
    from owm_envs.render.iss_scene import RenderConfig

    clips = iter_batch_frames(
        batch, RenderConfig(image_width=64, image_height=64), workers=2, **ISS
    )
    next(clips)
    start = time.perf_counter()
    clips.close()
    assert time.perf_counter() - start < 60


def test_a_worker_that_cannot_start_fails_instead_of_hanging():
    """A worker whose start-up raises -- a mistyped --gpu-index, a card with
    no VRAM left -- must end the render. `multiprocessing.Pool` respawns such
    a worker for ever while the parent blocks on a result that never comes,
    which would hang a ten-hour render instead of failing it.

    Run in a subprocess so a regression here cannot leave this suite behind a
    process-spawning loop.
    """
    script = textwrap.dedent(
        """
        import numpy as np
        from owm_envs.datasets.video import iter_batch_frames
        from owm_envs.envs.iss.config import ISSConfig
        from owm_envs.drivers.types import TrajectoryBatch
        from owm_envs.render.iss_scene import RenderConfig

        batch = TrajectoryBatch(
            observations=np.zeros((2, 2, 13), dtype=np.float32),
            actions=np.zeros((2, 2, 6), dtype=np.float32),
            rewards=np.zeros((2, 2), dtype=np.float32),
            lengths=np.array([2, 2], dtype=np.int32),
            terminated=np.array([True, False]),
            truncated=np.array([False, True]),
            policy_ids=None,
        )
        cfg = RenderConfig(image_width=64, image_height=64)
        list(iter_batch_frames(batch, cfg, workers=2, gpu_index=9999,
                               env_name="iss", env_cfg=ISSConfig()))
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert done.returncode != 0
    assert "gpu index 9999" in done.stdout + done.stderr


def _writer():
    pytest.importorskip("lerobot", reason="lerobot is an optional extra")
    from owm_envs.datasets.lerobot_writer import write_lerobot_split

    return write_lerobot_split


def _clips(lengths, size=32, keys=(FPV,)):
    """32x32, not smaller: SVT-AV1 raises SIGFPE encoding a tiny frame."""
    return [
        {key: np.zeros((int(length), size, size, 3), dtype=np.uint8) for key in keys}
        for length in lengths
    ]


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
            yield {FPV: clip}
            del clip
            gc.collect()
            # Every clip handed over so far, including the one for the
            # episode just written: the writer asks for the next clip only
            # once it has finished with the last, so peak is one episode of
            # video, not two.
            assert [i for i, ref in enumerate(refs) if ref() is not None] == []

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
    bad[1] = {FPV: np.zeros((bad[1][FPV].shape[0] + 1, 32, 32, 3), dtype=np.uint8)}
    with pytest.raises(ValueError, match="length"):
        write_lerobot_split(
            tmp_path / "badlen", "iss/badlen", batch, fps=20, frames=iter(bad)
        )


def test_writer_closes_the_clip_source_when_an_episode_fails(tmp_path):
    """A write that dies part-way leaves the clip generator suspended, and the
    raised exception's traceback pins the writer's frame -- so the render pool
    behind that generator would stay up until the exception was discarded,
    which for a CLI error is process exit. Closing the source is what runs the
    pool's own shutdown. `excinfo` holds that traceback here on purpose: it is
    what stops refcount collection from standing in for the close.
    """
    write_lerobot_split = _writer()

    batch = _fake_batch()
    closed = []

    def clips():
        try:
            yield {FPV: np.zeros((int(batch.lengths[0]), 32, 32, 3), dtype=np.uint8)}
            yield {FPV: np.zeros((int(batch.lengths[1]) + 1, 32, 32, 3), dtype=np.uint8)}
        finally:
            closed.append(True)

    with pytest.raises(ValueError) as excinfo:
        write_lerobot_split(
            tmp_path / "closed", "iss/closed", batch, fps=20, frames=clips()
        )
    assert "episode 1" in str(excinfo.value)
    assert closed == [True]


def test_writer_closes_the_clip_source_when_the_first_episode_fails(tmp_path):
    """Episode 0 fails while the hand-back generator is still suspended on the
    peeked clip, before it ever delegates to the source. Closing the hand-back
    alone leaves the source open, and the caller still holds it -- the CLI
    keeps its `iter_batch_frames(...)` binding for the whole split -- so
    nothing else collects it either.
    """
    write_lerobot_split = _writer()

    batch = _fake_batch()
    closed = []

    def clips():
        try:
            yield {FPV: np.zeros((int(batch.lengths[0]) + 1, 32, 32, 3), dtype=np.uint8)}
        finally:
            closed.append(True)

    source = clips()
    with pytest.raises(ValueError) as excinfo:
        write_lerobot_split(
            tmp_path / "first", "iss/first", batch, fps=20, frames=source
        )
    assert "episode 0" in str(excinfo.value)
    assert closed == [True]


def test_writer_closes_the_clip_source_when_the_dataset_cannot_be_created(
    tmp_path, monkeypatch
):
    """Declaring the video feature needs a real clip, so by the time the
    dataset is created the pool is already up and holding every renderer it
    will ever hold. A create that fails -- unwritable root, a directory
    already there -- must not leave them running."""
    write_lerobot_split = _writer()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    batch = _fake_batch()
    closed = []

    def clips():
        try:
            yield from _clips(batch.lengths)
        finally:
            closed.append(True)

    def refuse(*args, **kwargs):
        raise OSError("root is not writable")

    monkeypatch.setattr(LeRobotDataset, "create", refuse)
    source = clips()
    with pytest.raises(OSError):
        write_lerobot_split(tmp_path / "nope", "iss/nope", batch, fps=20, frames=source)
    assert closed == [True]


def test_writer_closes_the_clip_source_when_the_feature_cannot_be_declared(tmp_path):
    """The same window, reached without patching anything: a clip whose frames
    are not (H, W, C) fails the feature unpack, which is after the peek has
    started the pool and before any dataset exists."""
    write_lerobot_split = _writer()

    batch = _fake_batch()
    closed = []

    def clips():
        try:
            yield {FPV: np.zeros((int(batch.lengths[0]), 32, 32), dtype=np.uint8)}
        finally:
            closed.append(True)

    source = clips()
    with pytest.raises(ValueError):
        write_lerobot_split(tmp_path / "flat", "iss/flat", batch, fps=20, frames=source)
    assert closed == [True]


def test_a_clip_source_that_fails_to_close_does_not_mask_the_write_error(tmp_path):
    """Cleanup must not replace the failure being reported. The write error is
    what names the real fault; a pool that also fails on the way down must not
    be what a ten-hour render is left holding."""
    write_lerobot_split = _writer()

    batch = _fake_batch()

    def clips():
        try:
            yield {FPV: np.zeros((int(batch.lengths[0]), 32, 32, 3), dtype=np.uint8)}
            yield {FPV: np.zeros((int(batch.lengths[1]) + 1, 32, 32, 3), dtype=np.uint8)}
        finally:
            raise RuntimeError("the pool failed to shut down")

    with pytest.raises(ValueError, match="episode 1"):
        write_lerobot_split(
            tmp_path / "mask", "iss/mask", batch, fps=20, frames=clips()
        )


def test_writer_rejects_a_mismatched_frame_shape_from_an_iterator(tmp_path):
    write_lerobot_split = _writer()

    batch = _fake_batch()
    bad = _clips(batch.lengths)
    bad[1] = {FPV: np.zeros((bad[1][FPV].shape[0], 64, 64, 3), dtype=np.uint8)}
    with pytest.raises(ValueError, match="shape"):
        write_lerobot_split(
            tmp_path / "badshape", "iss/badshape", batch, fps=20, frames=iter(bad)
        )
