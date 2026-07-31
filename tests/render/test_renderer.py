import numpy as np
import pytest

pytest.importorskip("pygfx", reason="rendering is an optional extra")
pytest.importorskip("trimesh", reason="GLB loading needs trimesh")

from owm_envs.render.renderer import ISSRenderer  # noqa: E402
from owm_envs.render.iss_scene import RenderConfig  # noqa: E402

VIEWS = ["DRAGON_ISO", "DRAGON_TOP", "DRAGON_FPV", "ISS_ISO", "ISS_TOP", "ISS_FPV"]


def a_state(pos=(100.0, 0.0, 0.0)):
    return np.array([*pos, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)


@pytest.fixture(scope="module")
def renderer():
    r = ISSRenderer(RenderConfig(image_width=128, image_height=128))
    yield r
    r.close()


@pytest.fixture(scope="module")
def nonsquare_renderer():
    r = ISSRenderer(RenderConfig(image_width=160, image_height=96))
    yield r
    r.close()


def test_renders_an_rgb_array_of_the_configured_size(nonsquare_renderer):
    # Non-square dimensions pin height/width ordering, not just total pixel
    # count -- a stray `pixel_ratio` left at its pygfx default of 2 (see
    # renderer.py) would double both axes and still pass a square-only check.
    frame = nonsquare_renderer.render(a_state())
    assert frame.shape == (96, 160, 3)
    assert frame.dtype == np.uint8


@pytest.mark.parametrize("view", VIEWS)
def test_every_view_renders(renderer, view):
    frame = renderer.render(a_state(), view=view)
    assert frame.shape[2] == 3


def test_views_are_pairwise_distinct(renderer):
    # A uniform grey wash -- e.g. every view pointed at nothing -- would pass
    # a "some pixels are non-black" check while still being visually useless.
    # Requiring every pair of the six views to actually differ catches that;
    # the observed minimum pairwise mean-absolute difference across the six
    # views is 4.4, well above the diff > 1.0 threshold used here.
    frames = {view: renderer.render(a_state(), view=view) for view in VIEWS}
    for i, a in enumerate(VIEWS):
        for b in VIEWS[i + 1 :]:
            diff = np.abs(frames[a].astype(np.int16) - frames[b].astype(np.int16)).mean()
            assert diff > 1.0, f"{a} and {b} rendered near-identical frames (diff={diff})"


def test_different_states_produce_different_frames(renderer):
    near = renderer.render(a_state(pos=(30.0, 0.0, 0.0)))
    far = renderer.render(a_state(pos=(300.0, 0.0, 0.0)))
    assert not np.array_equal(near, far)


def test_unknown_view_raises(renderer):
    with pytest.raises(ValueError, match="view"):
        renderer.render(a_state(), view="NADIR")


def test_views_returns_all_six_placements(renderer):
    views = renderer.views(a_state())
    assert set(views) == set(VIEWS)


def test_render_after_close_raises_runtime_error():
    # A closed renderer must fail with a clear, actionable error rather than
    # an AttributeError from internally nulled-out state.
    r = ISSRenderer(RenderConfig(image_width=64, image_height=64))
    r.close()
    with pytest.raises(RuntimeError, match="closed"):
        r.render(a_state())
