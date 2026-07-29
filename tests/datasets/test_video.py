import numpy as np
import pytest

pytest.importorskip("pygfx", reason="rendering is an optional extra")

from owm_envs.datasets.video import render_batch_frames, render_episode_frames  # noqa: E402
from owm_envs.drivers.types import TrajectoryBatch  # noqa: E402


class _FakeRenderer:
    """Stands in for ISSRenderer so these tests don't need real GLB/texture
    assets -- only construction and close() call counts matter here."""

    instances = 0
    closes = 0

    def __init__(self, cfg):
        _FakeRenderer.instances += 1
        self.cfg = cfg

    def render(self, state, action=None, view="DRAGON_FPV"):
        return np.zeros((self.cfg.image_height, self.cfg.image_width, 3), dtype=np.uint8)

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
    # The bug this guards against: video.py used to construct a fresh
    # ISSRenderer per episode, paying the full GLB/texture/cubemap load cost
    # (~3.6 s plus ~200 MB of GPU uploads) N times for an N-episode batch.
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
    # Backward-compatible standalone use: no renderer given, so one is built
    # and closed just for this call, as before.
    batch = small_batch()
    render_episode_frames(batch, 0, _Cfg())
    assert _FakeRenderer.instances == 1
    assert _FakeRenderer.closes == 1
