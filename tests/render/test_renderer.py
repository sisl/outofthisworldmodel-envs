import dataclasses

import numpy as np
import pygfx as gfx
import pytest

pytest.importorskip("pygfx", reason="rendering is an optional extra")
pytest.importorskip("trimesh", reason="GLB loading needs trimesh")

import jax.numpy as jnp  # noqa: E402
from owm_envs.core.quaternion import quat_from_rotmat  # noqa: E402
from owm_envs.render.iss_scene import RenderConfig, _collect_meshes  # noqa: E402
from owm_envs.render.renderer import (  # noqa: E402
    _MAX_DEPTH_RANGE_RATIO,
    ISSRenderer,
    _build_views,
    _dragon_fpv_pose_world,
    _with_scene_far,
)

VIEWS = ["DRAGON_ISO", "DRAGON_TOP", "DRAGON_FPV", "ISS_ISO", "ISS_TOP", "ISS_FPV"]


def a_state(pos=(100.0, 0.0, 0.0)):
    return np.array([*pos, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)


def _state_pitched(depression_deg, pos=(0.0, 60.0, 0.0)):
    """A Dragon attitude with the FPV boresight `depression_deg` below the
    local horizontal.

    The FPV camera looks along body +Z, so the attitude is built from the
    world-frame axes that direction has to land on; +Z is away from Earth, so
    depressing the boresight tilts it towards -Z. Negative values look above
    the horizontal, which leaves only the far, near-limb part of the planet in
    frame.
    """
    d = np.deg2rad(depression_deg)
    forward = np.array([np.cos(d), 0.0, -np.sin(d)])
    up = np.array([0.0, 0.0, 1.0]) - forward[2] * forward
    up /= np.linalg.norm(up)
    right = np.cross(up, forward)
    quat = np.asarray(
        quat_from_rotmat(jnp.asarray(np.stack([right, up, forward], axis=1), dtype=jnp.float32)),
        dtype=np.float32,
    )
    return np.array([*pos, 0, 0, 0, *quat, 0, 0, 0], dtype=np.float32)


def _limb_depression_deg(cfg):
    """How far below the local horizontal Earth's limb sits, from orbit."""
    return float(np.degrees(np.arccos(cfg.earth_radius_m / _orbit_radius_m(cfg))))


def _orbit_radius_m(cfg):
    return cfg.earth_radius_m + cfg.iss_altitude_m


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
    return float(np.sqrt(_orbit_radius_m(cfg) ** 2 - cfg.earth_radius_m**2))


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


def test_a_custom_cameras_oversized_far_is_capped_too():
    # `render_view` takes any CameraView, so the cap has to bind on a far the
    # caller chose as well -- widening one that is too small and cutting one
    # that the projection would round away regardless.
    cfg = RenderConfig()
    greedy = dataclasses.replace(
        _build_views(cfg, a_state())["DRAGON_FPV"], near=0.5, far=1e12
    )
    assert _with_scene_far(greedy, cfg).far == 0.5 * _MAX_DEPTH_RANGE_RATIO


def test_earth_fills_the_lower_half_of_an_fpv_frame_pointed_at_it(renderer):
    # The regression: with an FPV near plane of 0.05 m the reachable far plane
    # collapsed to ~1000 km, so the surface past that -- everything from the
    # limb inwards to a ring well short of it -- was clipped and the starfield
    # showed through the hole. The camera here is pitched so Earth's limb sits
    # on the horizontal midline, which puts the whole lower half on the planet
    # and the farthest surface of all in the band just below the midline.
    # Earth is far brighter than the starfield, so its presence is measurable
    # as coverage: that band went from 11% lit under the bug to 100% lit.
    #
    # Counted only over the pixels a ray-cast puts on the planet. The band
    # straddles the limb by a pixel or two, and counting raw brightness there
    # would let a bright patch of sky stand in for surface -- which is exactly
    # what happened while the starfield was undimmed.
    cfg = RenderConfig()
    on_the_limb = _state_pitched(_limb_depression_deg(cfg))
    frame = renderer.render(on_the_limb, view="DRAGON_FPV")
    height = frame.shape[0]
    band = slice(height // 2, height // 2 + height // 8)
    on_earth = _earth_hit_mask(cfg, on_the_limb, *frame.shape[:2])[band]
    lit = (frame[band].mean(axis=-1) > 40) & on_earth
    covered = float(lit.sum()) / int(on_earth.sum())
    assert covered > 0.97, f"only {covered:.2%} of the surface below the limb is lit"



def _earth_hit_mask(cfg, state, height, width):
    """Per pixel: does this FPV camera ray reach Earth's surface?

    Geometry's own answer to where the planet must appear, independent of what
    was drawn -- which is what makes "Earth is missing here" measurable rather
    than eyeballed.
    """
    eye, forward, up = (v.astype(np.float64) for v in _dragon_fpv_pose_world(cfg, state))
    up -= forward * (up @ forward)
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)

    half = np.tan(np.deg2rad(cfg.dragon_fpv_fov_y_deg) / 2.0)
    ys = (1.0 - 2.0 * (np.arange(height) + 0.5) / height) * half
    xs = (2.0 * (np.arange(width) + 0.5) / width - 1.0) * half * (width / height)
    gx, gy = np.meshgrid(xs, ys)
    dirs = forward + gx[..., None] * right + gy[..., None] * up
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    centre = np.array([0.0, 0.0, -(cfg.earth_radius_m + cfg.iss_altitude_m)])
    oc = eye - centre
    b = dirs @ oc
    c = float(oc @ oc) - cfg.earth_radius_m**2
    disc = b * b - c
    return (disc > 0.0) & (-b - np.sqrt(np.maximum(disc, 0.0)) > 0.0)


@pytest.fixture(scope="module")
def no_glow_renderer():
    # The atmospheric glow is bright and sits in front of the limb, so it
    # lights pixels whose surface was clipped away. Off, the only thing that
    # can light a pixel the planet covers is the planet.
    cfg = RenderConfig(image_width=128, image_height=128, show_earth_glow=False)
    r = ISSRenderer(cfg)
    yield cfg, r
    r.close()


def _missing_earth_fraction(cfg, renderer, state):
    frame = renderer.render(state, view="DRAGON_FPV")
    hit = _earth_hit_mask(cfg, state, *frame.shape[:2])
    expected = int(hit.sum())
    assert expected > 0, "the test framing must put some of Earth on screen"
    return float((hit & ~(frame.mean(axis=-1) > 40)).sum()) / expected


def test_earth_stays_whole_and_steady_across_small_attitude_steps(no_glow_renderer):
    """The flicker half of the clipping bug.

    The clip distance is fixed in camera space, so it lands on a ring of the
    planet whose radius does not move with the camera. Small attitude steps
    sweep that ring across the surface, and the patch that survives it changes
    from frame to frame far faster than the camera does -- content popping in
    and out rather than panning. This framing starts looking above the local
    horizontal, so only the far, near-limb part of the planet is in shot: the
    part the old near plane could not reach.

    Measured over these eight frames: an FPV near plane of 0.05 m left up to
    4.00% of the planet's on-screen area unrendered and swung by 2.00 points
    between consecutive frames; at 0.5 m that is 0.09% and 0.05%.
    """
    cfg, renderer = no_glow_renderer
    fractions = np.array(
        [_missing_earth_fraction(cfg, renderer, _state_pitched(-18.0 + i * 1.5)) for i in range(8)]
    )
    assert fractions.max() < 0.01, f"Earth missing from up to {fractions.max():.2%} of its area"
    jumps = np.abs(np.diff(fractions))
    assert jumps.max() < 0.005, f"missing-Earth area jumped {jumps.max():.2%} between frames"


def test_clouds_are_never_drawn_over_a_clipped_surface():
    """The cloud shell sits 12 km above the surface, so along any ray it is
    reached first. A clip distance that fell between the two drew cloud with
    nothing underneath it -- the clouds-without-surface banding. Rendering the
    same state with clouds on and off isolates exactly those pixels.
    """
    state = _state_pitched(-13.5)
    cfg = RenderConfig(image_width=128, image_height=128, show_earth_glow=False)
    lit = {}
    for clouds in (True, False):
        r = ISSRenderer(cfg.model_copy(update={"show_earth_clouds": clouds}))
        try:
            lit[clouds] = r.render(state, view="DRAGON_FPV").mean(axis=-1) > 40
        finally:
            r.close()

    hit = _earth_hit_mask(cfg, state, 128, 128)
    orphaned = float((hit & lit[True] & ~lit[False]).sum()) / int(hit.sum())
    assert orphaned < 0.001, f"{orphaned:.2%} of Earth is cloud over a clipped surface"


def _hull_mask(renderer, state):
    """Which pixels the capsule's own body covers, found by toggling it."""
    dragon = renderer._iss_scene.dragon
    dragon.visible = True
    with_hull = renderer.render(state, view="DRAGON_FPV")
    dragon.visible = False
    without = renderer.render(state, view="DRAGON_FPV")
    dragon.visible = True
    return np.abs(with_hull.astype(np.int16) - without.astype(np.int16)).max(axis=-1) > 16


def test_the_fpv_camera_cannot_see_through_its_own_nose_cone(renderer):
    """The other side of the near plane's squeeze.

    The FPV camera is mounted inside the capsule, 0.4026 m behind the nearest
    hull surface in its own frustum, so a near plane pushed much past that
    slices the cone open and the planet shows through the hole. Rather than
    pick a pixel count out of the air, this measures the hull against the same
    scene rendered with a near plane too small to clip anything. At this
    resolution the camera keeps 98% of that reference anywhere in 0.20-0.40 m
    -- the residual is mask-threshold noise -- against 87.1% at 0.45 m and
    65.3% at 0.50 m.
    """
    cfg = RenderConfig(image_width=128, image_height=128, fpv_camera_near_m=0.02)
    unclipped = ISSRenderer(cfg)
    try:
        state = _state_pitched(_limb_depression_deg(cfg))
        reference = _hull_mask(unclipped, state)
    finally:
        unclipped.close()

    assert reference.mean() > 0.05, "the cone should cover a real part of this framing"
    kept = float((_hull_mask(renderer, state) & reference).sum()) / int(reference.sum())
    assert kept > 0.95, f"the near plane clipped away {1 - kept:.1%} of the capsule's hull"


def _cloud_mesh(renderer):
    """The cloud deck, which is the surface group's second and last child."""
    children = renderer._iss_scene._earth_surface_group.children
    assert len(children) == 2, "expected the surface group to hold the globe and the deck"
    return children[1]


def _cloud_mask(renderer, state, cloud):
    """Which pixels the cloud deck is responsible for, found by toggling it."""
    cloud.visible = True
    on = renderer.render(state, view="DRAGON_FPV")
    cloud.visible = False
    off = renderer.render(state, view="DRAGON_FPV")
    cloud.visible = True
    return np.abs(on.astype(np.int16) - off.astype(np.int16)).max(axis=-1) > 24


def test_the_cloud_deck_is_drawn_between_the_globe_and_the_station(renderer):
    """The ordering the deck's correctness rests on, asserted directly.

    With no depth test of its own, nothing but the sort key keeps the deck off
    the station -- and the station's queue is whatever its glTF materials
    happened to load as, not something this package sets. A future asset that
    loaded them differently would silently put the deck on top of the station,
    which the pixel tests could only catch where the two overlap.
    """
    scene = renderer._iss_scene
    globe, deck = scene._earth_surface_group.children
    station = _collect_meshes(scene.iss) + _collect_meshes(scene.dragon)
    assert station, "expected the station and capsule to carry meshes"

    glow = [
        mesh
        for mesh in _collect_meshes(scene._earth_surface_group.parent)
        if mesh not in (globe, deck)
    ]
    assert glow, "expected the atmospheric shells to be present"

    assert globe.material.render_queue < deck.material.render_queue
    assert deck.material.render_queue < min(m.material.render_queue for m in glow)
    assert max(m.material.render_queue for m in glow) < min(
        m.material.render_queue for m in station
    )
    # The shells still depth-test -- being hidden behind the globe is what
    # makes them a limb rather than a wash -- but on `<=`, so a fragment whose
    # depth saturates to 1.0 past the depth range still draws against the
    # cleared buffer instead of losing `<` to it.
    assert all(m.material.depth_test for m in glow)
    assert all(m.material.depth_compare == "<=" for m in glow)
    assert not any(m.material.depth_write for m in glow)
    assert deck.material.depth_test is False
    assert deck.material.depth_write is False
    # Front faces are the near hemisphere from outside the shell; both sides
    # would let the deck's far half paint over its near half.
    assert deck.material.side == gfx.VisibleSide.front


def test_the_cloud_deck_does_not_flicker_between_frames(renderer):
    """The deck sits 12 km above the surface, which no depth buffer reachable
    from a docking near plane can resolve -- one float32 depth step is ~21 km
    at the nadir range and ~650 km at the limb. Depth-testing the deck against
    the surface was therefore rounding noise: it threw most of the deck away
    and changed its mind about which pixels as the camera moved, so cloud
    patches blinked on and off over the oceans.

    These six frames are 0.05 degrees apart -- far less than a pixel of cloud
    motion -- so the deck has to stay put. Measured over them: composited by
    depth, 65.3% of cloud pixels changed state between neighbouring frames;
    composited by draw order, 5.3%, which is the genuine sub-pixel drift.
    """
    cloud = _cloud_mesh(renderer)
    masks = [_cloud_mask(renderer, _state_pitched(62.0 + i * 0.05), cloud) for i in range(6)]

    coverage = np.array([m.mean() for m in masks])
    # A deck that is not drawn at all would hold perfectly still.
    assert coverage.min() > 0.02, f"only {coverage.min():.2%} of the frame is cloud"

    flipped = []
    for before, after in zip(masks, masks[1:]):
        flipped.append(float((before ^ after).sum()) / int((before | after).sum()))
    assert max(flipped) < 0.15, f"{max(flipped):.1%} of cloud pixels flipped between frames"


def test_the_station_occludes_the_cloud_deck(renderer):
    """Drawing the deck without a depth test leaves draw order alone to keep it
    behind the station, so this checks that order really does hold in a view
    where surface, deck and station all stack up. The station's silhouette
    comes from toggling it with the planet hidden, so both renders it is
    measured from still have the skybox behind them.
    """
    scene = renderer._iss_scene
    # Straight down onto the station from 26 m, so Earth fills everything the
    # station does not.
    state = _state_pitched(90.0, pos=(0.0, 0.0, 26.0))
    planet = [
        child
        for child in scene.scene.children
        if not isinstance(child, gfx.Light) and child not in (scene.iss, scene.dragon)
    ]

    try:
        cloud_pixels = _cloud_mask(renderer, state, _cloud_mesh(renderer))
        for child in planet:
            child.visible = False
        with_station = renderer.render(state, view="DRAGON_FPV")
        scene.iss.visible = scene.dragon.visible = False
        sky_only = renderer.render(state, view="DRAGON_FPV")
    finally:
        for child in planet:
            child.visible = True
        scene.iss.visible = scene.dragon.visible = True

    station = np.abs(with_station.astype(np.int16) - sky_only.astype(np.int16)).max(axis=-1) > 12
    assert station.mean() > 0.2, "the station should fill much of this framing"
    assert cloud_pixels.sum() > 0, "the deck should be visible past the station"

    # Not exactly zero: the two renders antialias the station's edge against
    # different backdrops, and this station is mostly edge -- a lattice of
    # trusses and panels. Real bleed would scale with the silhouette's area
    # rather than with its perimeter.
    bleed = float((cloud_pixels & station).sum()) / int(station.sum())
    assert bleed < 0.005, f"{bleed:.2%} of the station was painted over by cloud"


def _erode(mask, iterations):
    out = mask.copy()
    for _ in range(iterations):
        e = out.copy()
        e[1:] &= out[:-1]
        e[:-1] &= out[1:]
        e[:, 1:] &= out[:, :-1]
        e[:, :-1] &= out[:, 1:]
        out = e
    return out


def test_the_atmosphere_is_a_limb_not_a_wash_over_the_planet(renderer):
    """What the glow shells' depth test is for.

    They are additive back-face spheres, so the fragment drawn for each is on
    the FAR side of the shell. Over the planet those fall behind the globe and
    the depth test drops them; only around the limb, where the ray misses the
    surface entirely, do they survive and accumulate. Drop the depth test to
    stop the far shells being clipped and the effect inverts into a blue wash
    over the whole disc -- measured, +7.2 levels over the disc interior where
    it should be adding nothing.

    The opposite failure is the shells being clipped rather than occluded: on
    a `<` compare their far fragments saturate to a depth of exactly 1.0 and
    lose to the cleared buffer, which thins the limb from +32.7 levels to
    +11.2. Both directions are pinned here, so neither can be traded for the
    other.
    """
    cfg = RenderConfig(image_width=160, image_height=160)
    state = _state_pitched(_limb_depression_deg(cfg) + 25.0)
    shells = [
        mesh
        for mesh in _collect_meshes(renderer._iss_scene._earth_surface_group.parent)
        if mesh not in tuple(renderer._iss_scene._earth_surface_group.children)
    ]
    assert shells, "expected the atmospheric shells to be present"

    try:
        with_glow = renderer.render(state, view="DRAGON_FPV").astype(np.float64)
        for shell in shells:
            shell.visible = False
        without = renderer.render(state, view="DRAGON_FPV").astype(np.float64)
    finally:
        for shell in shells:
            shell.visible = True

    added = with_glow - without
    on_earth = _earth_hit_mask(cfg, state, *added.shape[:2])
    # Eroded hard, so "interior" excludes the limb the glow belongs to.
    interior = _erode(on_earth, 12)
    assert interior.any(), "the framing must show a solid piece of the disc"
    assert added[interior].mean() < 0.5, "the atmosphere is washing over the planet"
    limb = added[~on_earth].mean()
    assert limb > 20.0, f"the limb is only lit to +{limb:.1f}; clipped shells?"


def _sky_only(renderer, state):
    """Render with every scene object hidden, so the frame is pure starfield."""
    scene = renderer._iss_scene.scene
    hidden = [child for child in scene.children if not isinstance(child, gfx.Light)]
    try:
        for child in hidden:
            child.visible = False
        return renderer.render(state, view="DRAGON_FPV")
    finally:
        for child in hidden:
            child.visible = True


def _linear(frame):
    """sRGB bytes -> linear radiance, which is what the knob scales."""
    f = frame.astype(np.float64) / 255.0
    return float(np.where(f <= 0.04045, f / 12.92, ((f + 0.055) / 1.055) ** 2.4).mean())


def test_star_brightness_scales_the_starfield(renderer):
    """The star map is a survey composite, exposed to show the Milky Way rather
    than as a camera stopped down for a sunlit station would see it. The knob
    is an exposure multiplier in linear light, applied through the skybox
    material's opacity over an already-black buffer.
    """
    # Well above the limb, so nothing but sky is in shot either way.
    state = _state_pitched(-60.0)
    at_default = _sky_only(renderer, state)

    frames = {}
    for brightness in (1.0, 0.0):
        r = ISSRenderer(
            RenderConfig(image_width=128, image_height=128, star_brightness=brightness)
        )
        try:
            frames[brightness] = _sky_only(r, state)
        finally:
            r.close()

    assert frames[0.0].max() == 0, "a brightness of 0 must leave the sky black"
    assert frames[1.0].max() > 0, "a brightness of 1 must leave the sky lit"

    default = RenderConfig().star_brightness
    assert default < 1.0, "the shipped default must dim the survey map"
    # Not merely dimmer: dimmer by the factor asked for.
    ratio = _linear(at_default) / _linear(frames[1.0])
    assert default * 0.9 < ratio < default * 1.1, f"asked for {default}, got {ratio:.3f}"


def test_render_views_matches_rendering_each_view_on_its_own(renderer):
    together = renderer.render_views(a_state(), views=VIEWS)
    assert list(together) == VIEWS
    for view in VIEWS:
        np.testing.assert_array_equal(together[view], renderer.render(a_state(), view=view))


def test_render_views_poses_the_scene_once_for_all_six(renderer, monkeypatch):
    # Posing is per-state, not per-camera. Six views that each re-posed the
    # capsule would do the same work six times for the same frame -- the whole
    # reason multi-view rendering is worth having as its own call.
    poses = []
    real_update = renderer._iss_scene.update
    monkeypatch.setattr(
        renderer._iss_scene,
        "update",
        lambda state, action=None: (poses.append(1), real_update(state, action))[1],
    )
    renderer.render_views(a_state(), views=VIEWS)
    assert len(poses) == 1


def test_render_views_rejects_an_unknown_view(renderer):
    with pytest.raises(ValueError, match="view"):
        renderer.render_views(a_state(), views=["DRAGON_FPV", "NADIR"])


def test_render_after_close_raises_runtime_error():
    # A closed renderer must fail with a clear, actionable error rather than
    # an AttributeError from internally nulled-out state.
    r = ISSRenderer(RenderConfig(image_width=64, image_height=64))
    r.close()
    with pytest.raises(RuntimeError, match="closed"):
        r.render(a_state())


# Four consecutive poses from val episode 1 of the run that showed the defect
# (logs/trial_coop_goal), copied out of its stored `state_vector` rows rather
# than invented: the artifact only appears at particular limb geometries, and
# a pose chosen for looking convenient did not reproduce it.
_LIMB_ARTIFACT_POSES = (
    (90.759438, -39.391617, -8.915079, -0.762598, 2.067935, 0.498665,
     0.654533, -0.243694, -0.699601, 0.150863, 0.305657, -0.533946, 0.241813),
    (85.982445, -32.487934, -8.551683, -4.352481, 4.753126, -0.535502,
     0.281803, -0.115844, -0.793397, 0.526962, 0.194769, -0.341299, 0.143757),
    (85.763176, -32.248665, -8.579208, -4.418416, 4.81745, -0.565676,
     0.273789, -0.112852, -0.792779, 0.532733, 0.190809, -0.333066, 0.139999),
    (85.540672, -32.006161, -8.608126, -4.481843, 4.88244, -0.591207,
     0.265948, -0.10991, -0.792086, 0.538323, 0.186894, -0.324926, 0.136296),
)


def _grow(mask, radius):
    out = mask.copy()
    for _ in range(radius):
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def _limb_gap_fraction(frame):
    """Share of the atmosphere band bitten out by near-black gaps.

    The band is the teal pixels. A gap is darkness with both the band and the
    lit surface close by -- sky past the band's outer edge is just as dark but
    has no surface beside it, which is what keeps the starfield out of the
    count.
    """
    rgb = frame.astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    band = (b > 60) & (b > r + 25) & (g > r + 10)
    lit = (rgb.max(axis=-1) > 60) & ~band
    dark = rgb.max(axis=-1) < 30
    gaps = dark & _grow(band, 4) & _grow(lit, 4)
    # Not `max(..., 1)`: a band that vanished entirely would divide into zero
    # gaps and read as perfect. The caller checks this is a real band first.
    return float(band.sum()), float(gaps.sum()) / max(int(band.sum()), 1)


@pytest.fixture(scope="module")
def earth_only_renderer():
    """A renderer with the station and capsule hidden.

    Their silhouettes against the band are dark, lit-adjacent and band-adjacent
    -- what `_limb_gap_fraction` looks for, and not a defect. Hiding them
    leaves only Earth and its atmosphere to be measured.
    """
    r = ISSRenderer(RenderConfig(image_width=512, image_height=512))
    scene = r._iss_scene
    for child in scene.scene.children:
        if isinstance(child, gfx.Light):
            continue
        if child is scene.iss or child is scene.dragon:
            child.visible = False
    yield r
    r.close()


@pytest.mark.parametrize("pose", _LIMB_ARTIFACT_POSES)
def test_the_atmosphere_band_has_no_holes_bitten_out_of_it(earth_only_renderer, pose):
    """Black polygons used to appear inside the atmosphere band, moving as the
    pose turned.

    Only the sliver of each glow shell outside the globe's silhouette is ever
    drawn, and the shells' facets were coarser than that sliver is wide, so
    they left gaps showing the near-black sky behind. Nothing was drawn in
    them -- rendered against a red background, they came back red.

    Measured over the reported episode, the band lost up to 0.96% of itself to
    these; at the shipped tessellation it loses none. The poses are four
    consecutive frames, so this also covers the defect moving under small
    rotations.
    """
    frame = earth_only_renderer.render(np.array(pose, dtype=np.float32), view="DRAGON_FPV")
    band, gaps = _limb_gap_fraction(frame)
    # A band that is not drawn at all has no holes in it either.
    assert band > 5000, f"only {band:.0f} px of atmosphere band in frame"
    # Tight enough to name the shipped tessellation rather than merely rule out
    # the reported defect: 96x48 measured 0.96% here and 192x96 still 0.04%,
    # against none at the 256x128 that ships.
    assert gaps < 0.0002, f"{gaps:.2%} of the atmosphere band is punched out"

def test_render_views_rejects_a_bare_view_name(renderer):
    # A string is a sequence of characters, so this would otherwise be
    # reported as an unknown view called "D".
    with pytest.raises(TypeError, match="sequence of view names"):
        renderer.render_views(a_state(), views="DRAGON_FPV")
