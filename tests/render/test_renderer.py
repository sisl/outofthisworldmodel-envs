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


def test_renders_an_rgb_array_of_the_configured_size(renderer):
    frame = renderer.render(a_state())
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8


@pytest.mark.parametrize("view", VIEWS)
def test_every_view_renders(renderer, view):
    frame = renderer.render(a_state(), view=view)
    assert frame.shape[2] == 3


@pytest.mark.parametrize("view", VIEWS)
def test_every_view_produces_a_non_blank_frame(renderer, view):
    # A view pointed at nothing renders pure black. That is the failure mode a
    # shape-only assertion misses entirely.
    frame = renderer.render(a_state(), view=view)
    assert int((frame.sum(axis=-1) > 10).sum()) > 100, f"{view} rendered a blank frame"


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
