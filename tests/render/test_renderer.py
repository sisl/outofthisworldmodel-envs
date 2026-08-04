import numpy as np
import pytest

pytest.importorskip("pygfx", reason="rendering is an optional extra")
pytest.importorskip("trimesh", reason="GLB loading needs trimesh")

import jax.numpy as jnp  # noqa: E402
from owm_envs.core.quaternion import quat_from_rotmat  # noqa: E402
from owm_envs.render.iss_scene import RenderConfig  # noqa: E402
from owm_envs.render.renderer import (  # noqa: E402
    _MAX_DEPTH_RANGE_RATIO,
    ISSRenderer,
    _build_views,
    _with_scene_far,
)

VIEWS = ["DRAGON_ISO", "DRAGON_TOP", "DRAGON_FPV", "ISS_ISO", "ISS_TOP", "ISS_FPV"]


def a_state(pos=(100.0, 0.0, 0.0)):
    return np.array([*pos, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)


def _state_on_the_limb(pos=(0.0, 60.0, 0.0)):
    """A Dragon attitude that puts Earth's limb on the FPV frame's midline.

    The FPV camera looks along body +Z, so the attitude is built from the
    world-frame axes that direction has to land on: forward depressed below
    the local horizontal by the limb's own depression angle, and the local
    outward normal as up.
    """
    cfg = RenderConfig()
    orbit_radius = cfg.earth_radius_m + cfg.iss_altitude_m
    depression = np.arccos(cfg.earth_radius_m / orbit_radius)
    # +Z is away from Earth, so depressing the ray means tilting it towards -Z.
    forward = np.array([np.cos(depression), 0.0, -np.sin(depression)])
    up = np.array([0.0, 0.0, 1.0]) - forward[2] * forward
    up /= np.linalg.norm(up)
    right = np.cross(up, forward)
    quat = np.asarray(
        quat_from_rotmat(jnp.asarray(np.stack([right, up, forward], axis=1), dtype=jnp.float32)),
        dtype=np.float32,
    )
    return np.array([*pos, 0, 0, 0, *quat, 0, 0, 0], dtype=np.float32)


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


def _earth_horizon_slant_m(cfg):
    """Distance from the scene origin to Earth's limb -- the farthest point of
    the planet's surface any camera at the station can see."""
    orbit_radius = cfg.earth_radius_m + cfg.iss_altitude_m
    return float(np.sqrt(orbit_radius**2 - cfg.earth_radius_m**2))


@pytest.mark.parametrize("view", VIEWS)
def test_every_view_can_see_the_whole_earth(view):
    # Earth's limb is 2352 km away; a far plane short of that cuts the globe
    # off mid-surface and lets the starfield through the hole.
    cfg = RenderConfig()
    scene_view = _with_scene_far(_build_views(cfg, a_state())[view], cfg)
    assert scene_view.far > _earth_horizon_slant_m(cfg)


@pytest.mark.parametrize("view", VIEWS)
def test_no_view_asks_for_a_far_plane_its_near_plane_cannot_express(view):
    # A far plane past `near * 2**24` is not merely imprecise, it is inert:
    # the float32 projection rounds it away and clips at the smaller distance
    # anyway. Asking for one hides that the near plane is what needs moving.
    cfg = RenderConfig()
    scene_view = _with_scene_far(_build_views(cfg, a_state())[view], cfg)
    assert scene_view.far <= scene_view.near * _MAX_DEPTH_RANGE_RATIO


def test_earth_fills_the_lower_half_of_an_fpv_frame_pointed_at_it(renderer):
    # The regression: with an FPV near plane of 0.05 m the reachable far plane
    # collapsed to ~1000 km, so the surface past that -- everything from the
    # limb inwards to a ring well short of it -- was clipped and the starfield
    # showed through the hole. The camera here is pitched so Earth's limb sits
    # on the horizontal midline, which puts the whole lower half on the planet
    # and the farthest surface of all in the band just below the midline.
    # Earth is far brighter than the starfield, so its presence is measurable
    # as coverage: that band went from 82% lit under the bug to 99.8% lit.
    frame = renderer.render(_state_on_the_limb(), view="DRAGON_FPV")
    height = frame.shape[0]
    band = frame[height // 2 : height // 2 + height // 8]
    lit = float((band.mean(axis=-1) > 40).mean())
    assert lit > 0.97, f"only {lit:.2%} of the surface below the limb is lit"


def test_render_after_close_raises_runtime_error():
    # A closed renderer must fail with a clear, actionable error rather than
    # an AttributeError from internally nulled-out state.
    r = ISSRenderer(RenderConfig(image_width=64, image_height=64))
    r.close()
    with pytest.raises(RuntimeError, match="closed"):
        r.render(a_state())
