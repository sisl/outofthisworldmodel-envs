import warnings

import numpy as np
import pytest

pytest.importorskip("pygfx", reason="rendering is an optional extra")
pytest.importorskip("trimesh", reason="GLB loading needs trimesh")

import imageio.v3 as iio  # noqa: E402  -- ships with the render extra, like pygfx
import owm_envs.render.iss_scene as scene_module  # noqa: E402
from PIL import Image  # noqa: E402
from owm_envs.render.earth import MAP_WIDTHS  # noqa: E402
from owm_envs.render.inputs import Lighting  # noqa: E402
from owm_envs.render.iss_scene import (  # noqa: E402
    _FILL_INTENSITY_FRACTION,
    ISSScene,
    RenderConfig,
    _earth_globe_geometry,
    _earth_normal_map,
    _load_cloud_texture,
    _load_rgb_texture,
    _normal_map_from_height,
)


def state_at(pos, quat=(1.0, 0.0, 0.0, 0.0)):
    return np.array([*pos, 0, 0, 0, *quat, 0, 0, 0], dtype=np.float32)


def _texture_limit_warnings(caught):
    # pygfx and NumPy emit unrelated deprecation warnings from inside texture
    # creation, so these tests filter for the one warning they are about.
    return [w for w in caught if "texture limit" in str(w.message)]


@pytest.fixture(scope="module")
def scene():
    # Building the scene loads ~100 MB of assets; do it once for the module.
    return ISSScene(RenderConfig())


def test_scene_contains_objects(scene):
    assert len(scene.scene.children) > 0


def test_background_skybox_exists(scene):
    assert scene.background is not None


def test_update_moves_the_dragon_to_the_state_position(scene):
    scene.update(state_at((100.0, 0.0, 0.0)))
    np.testing.assert_allclose(np.asarray(scene.dragon.local.position), [100.0, 0.0, 0.0], atol=1e-3)
    scene.update(state_at((0.0, -50.0, 10.0)))
    np.testing.assert_allclose(np.asarray(scene.dragon.local.position), [0.0, -50.0, 10.0], atol=1e-3)


def test_update_applies_the_state_attitude(scene):
    # 90 deg about z: q = [cos45, 0, 0, sin45].
    # Body +x must map to world [0, 1, 0], and body +y to world [-1, 0, 0].
    #
    # quat_to_rotmat (owm_envs.core.quaternion) deliberately transposes
    # astrojax's rotation matrix to turn its reference->body DCM convention
    # into the body->world active-rotation sense this scene needs. Checking
    # the exact direction, not just that the matrix changed, is what
    # distinguishes that sense from its transpose: without the transpose the
    # capsule would rotate the wrong way in every frame, with body +x
    # mapping to world [0, -1, 0] instead, which a "did it change" check
    # would not catch.
    s = float(np.sin(np.pi / 4))
    c = float(np.cos(np.pi / 4))
    scene.update(state_at((0.0, 0.0, 0.0), quat=(c, 0.0, 0.0, s)))
    rotation = np.asarray(scene.dragon.local.rotation_matrix)[:3, :3]
    body_x_in_world = rotation @ np.array([1.0, 0.0, 0.0])
    body_y_in_world = rotation @ np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(body_x_in_world, [0.0, 1.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(body_y_in_world, [-1.0, 0.0, 0.0], atol=1e-3)


def test_update_accepts_a_13d_state_not_14d(scene):
    # This port's state convention is 13D, not the 14D layout used elsewhere.
    scene.update(np.zeros(13, dtype=np.float32))
    with pytest.raises(ValueError, match="13-element state"):
        scene.update(np.zeros(9, dtype=np.float32))


@pytest.fixture
def lit_scene(scene):
    """The module-scoped scene, with everything `lighting=` moves put back
    afterwards so the other tests still see the as-built graph."""
    nodes = (
        scene._sun,
        scene._directional_light,
        scene._fill_light,
        scene._earth_group,
        scene._moon_group,
    )
    saved = [(tuple(node.local.position), getattr(node, "intensity", None)) for node in nodes]
    yield scene
    for node, (position, intensity) in zip(nodes, saved):
        node.local.position = position
        if intensity is not None:
            node.intensity = intensity
    scene._sun.visible = True
    # Also the flag that decides whether an unlit frame restores anything: a
    # scene left marked as lit would make the next test's `lighting=None` do
    # work the as-built graph does not expect.
    scene._lit = False


def _lighting(**overrides):
    kwargs = {
        "sun_direction_world": np.array([0.0, 0.0, 1.0]),
        "illumination": 0.25,
        "chief_distance_m": 6_795_000.0,
        "moon_vector_world": np.array([0.0, 3.6e8, 0.0]),
    }
    return Lighting(**(kwargs | overrides))


def _lighting_snapshot(scene):
    """Everything `_apply_lighting` touches, in one comparable value."""
    return {
        "sun": tuple(scene._sun.local.position),
        "sun_visible": scene._sun.visible,
        "light": tuple(scene._directional_light.local.position),
        "intensity": scene._directional_light.intensity,
        "fill": tuple(scene._fill_light.local.position),
        "fill_intensity": scene._fill_light.intensity,
        "earth": tuple(scene._earth_group.local.position),
        "moon": tuple(scene._moon_group.local.position),
    }


def test_update_without_lighting_moves_nothing(scene):
    # Every frame rendered before per-frame lighting existed goes through this
    # path, so it has to stay a no-op for everything but the capsule.
    before = _lighting_snapshot(scene)
    scene.update(np.zeros(13, dtype=np.float32))
    assert _lighting_snapshot(scene) == before


def test_update_without_lighting_restores_the_static_configuration(lit_scene):
    # The other direction of the no-op above, and the one a per-frame pixel
    # comparison never reaches: once a frame HAS been lit, an unlit frame must
    # put the scene back rather than inherit that ephemeris. A renderer is
    # shared across a whole split and can be handed frames from more than one
    # env, so the alternative is an env with no ephemeris of its own lit by
    # whatever iss-hcw last asked for. Eclipsed lighting, so the sun's
    # visibility has to come back too.
    before = _lighting_snapshot(lit_scene)
    lit_scene.update(np.zeros(13, dtype=np.float32), lighting=_lighting(illumination=0.0))
    assert _lighting_snapshot(lit_scene) != before

    lit_scene.update(np.zeros(13, dtype=np.float32))
    assert _lighting_snapshot(lit_scene) == before


def test_the_sun_disc_is_hidden_in_umbra(lit_scene):
    # Dimming the lights is not enough: the proxy sun sits at
    # sun_visual_distance_m (1e6 m), nearer than the Earth's surface along an
    # oblique ray, so nothing occludes it and an eclipsed frame would draw the
    # sun in front of the night side -- measured, 55 of 1115 samples over one
    # ISS orbit. Penumbra keeps it, since the station can still see the disc.
    lit_scene.update(np.zeros(13, dtype=np.float32), lighting=_lighting(illumination=0.0))
    assert lit_scene._sun.visible is False

    lit_scene.update(np.zeros(13, dtype=np.float32), lighting=_lighting(illumination=0.25))
    assert lit_scene._sun.visible is True


def test_the_scene_graph_holds_the_nodes_update_moves(scene):
    # `update` reaches the sun, lights, Earth and moon through attributes, so
    # the graph has to keep holding those very objects -- an attribute pointing
    # at a node that was never added would move nothing at all. Pinned in build
    # order, which is also the order the lights reach the shader, and including
    # the one light no attribute holds: the ambient fill is easy to drop while
    # promoting its neighbours.
    children = list(scene.scene.children)
    assert children[:4] == [scene.iss, scene._earth_group, scene._moon_group, scene._sun]
    assert type(children[4]).__name__ == "AmbientLight"
    assert children[5:] == [scene._directional_light, scene._fill_light, scene.dragon]


def test_update_applies_lighting(lit_scene):
    cfg = lit_scene.cfg
    lit_scene.update(np.zeros(13, dtype=np.float32), lighting=_lighting())

    sun_position = [0.0, 0.0, cfg.sun_visual_distance_m]
    np.testing.assert_allclose(lit_scene._sun.local.position, sun_position)
    np.testing.assert_allclose(lit_scene._directional_light.local.position, sun_position)
    assert np.isclose(lit_scene._directional_light.intensity, 0.25 * cfg.directional_light_intensity)
    assert np.isclose(
        lit_scene._fill_light.intensity,
        0.25 * _FILL_INTENSITY_FRACTION * cfg.directional_light_intensity,
    )
    np.testing.assert_allclose(lit_scene._earth_group.local.position, [0.0, 0.0, -6_795_000.0])
    # The moon hangs off the earth centre: earth_center + the geocentric vector.
    np.testing.assert_allclose(lit_scene._moon_group.local.position, [0.0, 3.6e8, -6_795_000.0])


def test_update_keeps_the_moon_at_its_true_distance(lit_scene):
    # orbit.moon_vector_world hands over direction AND distance so the moon
    # swings +/-7% in apparent size over a month. Renormalising it to
    # earth_moon_distance_m -- as the static config placement does -- would
    # freeze it at the mean and throw that away.
    perigee_m = 3.633e8
    lit_scene.update(
        np.zeros(13, dtype=np.float32),
        lighting=_lighting(moon_vector_world=np.array([0.0, perigee_m, 0.0])),
    )
    earth_center = np.asarray(lit_scene._earth_group.local.position)
    moon = np.asarray(lit_scene._moon_group.local.position)
    assert np.isclose(np.linalg.norm(moon - earth_center), perigee_m, rtol=1e-6)
    assert not np.isclose(np.linalg.norm(moon - earth_center), lit_scene.cfg.earth_moon_distance_m, rtol=1e-3)


def test_update_keeps_the_sun_at_its_visual_distance(lit_scene):
    # The disc's radius is baked from sun_visual_distance_m, so its angular
    # diameter is only right while it sits at exactly that distance.
    cfg = lit_scene.cfg
    before = np.asarray(lit_scene._sun.geometry.positions.data).copy()
    lit_scene.update(
        np.zeros(13, dtype=np.float32),
        lighting=_lighting(sun_direction_world=np.array([0.0, -6.0, 0.0])),
    )
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(lit_scene._sun.local.position)), cfg.sun_visual_distance_m, rtol=1e-6
    )
    np.testing.assert_array_equal(np.asarray(lit_scene._sun.geometry.positions.data), before)


def test_shadow_camera_follows_the_key_light(lit_scene):
    # Nothing here repositions the shadow camera: pygfx re-derives it from the
    # light's world position on every frame (LightShadow._update_matrix). This
    # pins that, because if it ever stopped being true the shadows would keep
    # falling from the light's build-time direction with nothing to show it.
    light = lit_scene._directional_light
    lit_scene.update(np.zeros(13, dtype=np.float32), lighting=_lighting())
    light.shadow._update_matrix(light)
    np.testing.assert_allclose(
        np.asarray(light.shadow.camera.world.position),
        [0.0, 0.0, lit_scene.cfg.sun_visual_distance_m],
        rtol=1e-6,
    )


def test_render_config_round_trips_through_toml(tmp_path):
    cfg = RenderConfig(image_width=256, image_height=128)
    path = tmp_path / "render.toml"
    cfg.to_toml(path)
    assert RenderConfig.from_toml(path) == cfg


def test_scene_does_not_hardcode_a_high_resolution_source_name():
    # Source filenames belong to earth.py's tier table; a copy here would go
    # stale the moment a source is replaced.
    import owm_envs.render.iss_scene as scene_module

    assert "EarthColorMap-80k" not in open(scene_module.__file__).read()


def test_normal_map_from_height_gradient():
    height = np.tile(np.linspace(0.0, 1.0, 64, dtype=np.float32), (64, 1))
    normals = _normal_map_from_height(height, strength=8.0)
    assert normals.shape == (64, 64, 3)
    assert normals.dtype == np.float32
    norms = np.linalg.norm(normals, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    # A pure x-gradient tilts normals along x only, never y.
    assert np.abs(normals[..., 0]).max() > 0.01
    assert np.abs(normals[1:-1, 1:-1, 1]).max() < 1e-5
    # z stays positive: the surface never folds past vertical.
    assert normals[..., 2].min() > 0.0
    # Ground rising toward +x tilts the normal back toward -x. With the sign
    # flipped the whole planet would light as if its terrain were inverted.
    assert normals[1:-1, 1:-1, 0].max() < 0.0


def test_normal_map_from_height_tilts_along_y_for_a_latitude_gradient():
    height = np.tile(np.linspace(0.0, 1.0, 64, dtype=np.float32)[:, None], (1, 64))
    normals = _normal_map_from_height(height, strength=8.0)
    assert np.abs(normals[1:-1, 1:-1, 0]).max() < 1e-5
    # Positive, unlike the x channel: pygfx's getTangentFrame hands the shader
    # a negated bitangent, so it subtracts this channel for us. Getting this
    # backwards lights every north-south slope as its own mirror image.
    assert normals[1:-1, 1:-1, 1].min() > 0.0


def test_normal_map_from_height_wraps_across_the_longitude_seam():
    # Rotating the map in longitude must simply rotate the normals: the
    # antimeridian is not a special place on the planet. Letting np.gradient
    # fall back to a one-sided difference at the array edge makes it one, and
    # on the real bump map that mis-tilts the seam by up to 48 degrees.
    rng = np.random.default_rng(0)
    height = rng.random((16, 32), dtype=np.float32)
    normals = _normal_map_from_height(height, strength=8.0)
    rolled = _normal_map_from_height(np.roll(height, 7, axis=1), strength=8.0)
    np.testing.assert_allclose(np.roll(normals, 7, axis=1), rolled, atol=1e-6)


def test_earth_globe_geometry_spans_the_whole_sphere():
    radius = 100.0
    geom = _earth_globe_geometry(
        radius=radius, subpoint_lon_deg=0.0, subpoint_lat_deg=0.0, width_segments=16, height_segments=8
    )
    positions = np.asarray(geom.positions.data)
    np.testing.assert_allclose(np.linalg.norm(positions, axis=1), radius, rtol=1e-5)
    np.testing.assert_allclose(positions.min(axis=0), [-radius, -radius, -radius], atol=radius * 1e-4)
    np.testing.assert_allclose(positions.max(axis=0), [radius, radius, radius], atol=radius * 1e-4)

    # Equirectangular UVs must use the full map, or the texture is cropped.
    texcoords = np.asarray(geom.texcoords.data)
    np.testing.assert_allclose(texcoords.min(axis=0), [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(texcoords.max(axis=0), [1.0, 1.0], atol=1e-6)


def test_earth_globe_geometry_puts_the_subpoint_on_the_scene_axis():
    # The Earth group sits directly below the scene origin along -Z, so the
    # configured sub-satellite point has to land on the globe's local +Z.
    # 8x4 segments put lon -90 / lat +45 exactly on a grid vertex.
    radius = 100.0
    geom = _earth_globe_geometry(
        radius=radius, subpoint_lon_deg=-90.0, subpoint_lat_deg=45.0, width_segments=8, height_segments=4
    )
    positions = np.asarray(geom.positions.data)
    texcoords = np.asarray(geom.texcoords.data)

    subpoint = int(np.argmin(np.linalg.norm(texcoords - np.array([0.25, 0.25]), axis=1)))
    np.testing.assert_allclose(positions[subpoint], [0.0, 0.0, radius], atol=1e-4 * radius)

    # Local north is +Y, so the pole lands on the spin axis this scene rotates
    # the surface about -- see ISSScene._earth_spin_axis_local.
    pole = int(np.argmin(texcoords[:, 1]))
    lat = np.deg2rad(45.0)
    np.testing.assert_allclose(positions[pole] / radius, [0.0, np.cos(lat), np.sin(lat)], atol=1e-5)


def test_earth_globe_triangles_face_outward():
    # Backface culling hides the far hemisphere only if the visible side is
    # the front side; a flipped winding renders the globe inside-out.
    geom = _earth_globe_geometry(
        radius=1.0, subpoint_lon_deg=-122.0, subpoint_lat_deg=37.0, width_segments=16, height_segments=8
    )
    positions = np.asarray(geom.positions.data)
    indices = np.asarray(geom.indices.data).reshape(-1, 3)
    a, b, c = positions[indices[:, 0]], positions[indices[:, 1]], positions[indices[:, 2]]
    face_normals = np.cross(b - a, c - a)
    centroids = (a + b + c) / 3.0
    # The pole rows collapse to triangles with no orientation to check; float32
    # noise leaves them a sliver of area, so cut on a fraction of a real one.
    areas = np.linalg.norm(face_normals, axis=1)
    real = areas > 1e-4 * areas.max()
    outward = np.einsum("ij,ij->i", face_normals[real] / areas[real, None], centroids[real])
    assert outward.min() > 0.5


def test_earth_globe_seam_columns_coincide_in_space():
    # u=0 and u=1 are the same meridian; the duplicate column exists only so
    # both texture edges have a vertex. If they drifted apart the globe would
    # have a slit in it.
    width_segments, height_segments = 16, 8
    geom = _earth_globe_geometry(
        radius=100.0,
        subpoint_lon_deg=-122.0,
        subpoint_lat_deg=37.0,
        width_segments=width_segments,
        height_segments=height_segments,
    )
    positions = np.asarray(geom.positions.data).reshape(height_segments + 1, width_segments + 1, 3)
    np.testing.assert_allclose(positions[:, 0], positions[:, -1], atol=1e-4)


def test_normal_map_texture_encodes_normals_into_the_zero_one_range(tmp_path):
    # mesh.wgsl decodes a normal map as `sample * 2 - 1`, and only in a linear
    # colorspace -- an sRGB texture here would bend every normal.
    path = tmp_path / "flat.png"
    iio.imwrite(path, np.full((16, 32), 128, dtype=np.uint8))
    texture_map = _earth_normal_map(path, show=True, strength=8.0)
    arr = np.asarray(texture_map.texture.data)
    assert arr.dtype == np.uint8
    assert texture_map.texture.colorspace == "physical"
    np.testing.assert_allclose(arr[..., 0], 128, atol=1)
    np.testing.assert_allclose(arr[..., 1], 128, atol=1)
    assert (arr[..., 2] == 255).all()


def test_normal_map_texture_survives_the_round_trip_through_the_shader_decode(tmp_path):
    # Decode exactly as mesh.wgsl does and check the texture still points where
    # _normal_map_from_height aimed it -- the flat case above cannot catch a
    # swapped or negated channel.
    rng = np.random.default_rng(0)
    height = rng.integers(0, 255, (16, 32), dtype=np.uint8)
    path = tmp_path / "rough.png"
    iio.imwrite(path, height)

    encoded = np.asarray(_earth_normal_map(path, show=True, strength=8.0).texture.data)
    decoded = encoded.astype(np.float32) / 255.0 * 2.0 - 1.0
    expected = _normal_map_from_height(height.astype(np.float32) / 255.0, 8.0)
    # 8-bit quantisation, doubled by the decode, is the whole error budget.
    assert np.abs(decoded - expected).max() < 0.01


def test_missing_bump_map_warns_rather_than_failing(tmp_path):
    with pytest.warns(UserWarning, match="bump map missing"):
        assert _earth_normal_map(tmp_path / "absent.png", show=True, strength=8.0) is None


def test_disabled_bump_map_is_silent(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _earth_normal_map(tmp_path / "absent.png", show=False, strength=8.0) is None
    assert not [w for w in caught if "bump map missing" in str(w.message)]


def test_oversized_texture_is_downscaled_to_the_device_limit(tmp_path, monkeypatch):
    # The full-globe colour map is 16384 px wide, above the 8192 px
    # max-texture-dimension-2d that plenty of adapters report.
    monkeypatch.setattr(scene_module, "_max_texture_size", lambda: 32)
    path = tmp_path / "oversized.png"
    iio.imwrite(path, np.zeros((80, 160, 3), dtype=np.uint8))
    with pytest.warns(UserWarning, match="texture limit"):
        texture = _load_rgb_texture(path)
    assert texture.size[:2] == (20, 10)


def test_texture_at_exactly_the_device_limit_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(scene_module, "_max_texture_size", lambda: 32)
    path = tmp_path / "exact.png"
    iio.imwrite(path, np.zeros((32, 32, 3), dtype=np.uint8))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        texture = _load_rgb_texture(path)
    assert texture.size[:2] == (32, 32)
    assert not _texture_limit_warnings(caught)


def test_every_earth_texture_loader_honours_the_device_limit(tmp_path, monkeypatch):
    # All three Earth maps are full-globe sized, so guarding only the colour
    # map would still fail the upload on a smaller adapter.
    monkeypatch.setattr(scene_module, "_max_texture_size", lambda: 32)
    clouds, bump = tmp_path / "clouds.png", tmp_path / "bump.png"
    iio.imwrite(clouds, np.zeros((64, 128, 3), dtype=np.uint8))
    iio.imwrite(bump, np.zeros((64, 128), dtype=np.uint8))

    with pytest.warns(UserWarning, match="texture limit"):
        assert _load_cloud_texture(clouds, opacity=0.8).size[:2] == (32, 16)
    with pytest.warns(UserWarning, match="texture limit"):
        assert _earth_normal_map(bump, show=True, strength=8.0).texture.size[:2] == (32, 16)


def test_downscaling_never_collapses_a_dimension_to_zero(tmp_path, monkeypatch):
    # The reduction factor comes from the longer side, so a sliver of an image
    # can drive the shorter one below a pixel.
    monkeypatch.setattr(scene_module, "_max_texture_size", lambda: 4)
    path = tmp_path / "sliver.png"
    iio.imwrite(path, np.zeros((64, 1, 3), dtype=np.uint8))
    with pytest.warns(UserWarning, match="texture limit"):
        texture = _load_rgb_texture(path)
    assert min(texture.size[:2]) >= 1


def _bomb_warnings(caught):
    return [w for w in caught if issubclass(w.category, Image.DecompressionBombWarning)]


@pytest.mark.parametrize(
    "load",
    [
        _load_rgb_texture,
        lambda path: _load_cloud_texture(path, opacity=0.8),
        lambda path: _earth_normal_map(path, show=True, strength=8.0),
    ],
)
def test_a_map_above_pillows_pixel_cap_loads_without_a_bomb_warning(tmp_path, monkeypatch, load):
    # A real full-globe map is 16384x8192 = 134 Mpx, over Pillow's 89.5 Mpx
    # default. Writing one here would cost hundreds of megabytes, so the cap is
    # lowered instead: what the loader must do is raise whatever cap is in
    # force, and that is the same code path at either size.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 16)
    path = tmp_path / "big.png"
    iio.imwrite(path, np.zeros((64, 128, 3), dtype=np.uint8))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load(path)
    assert not _bomb_warnings(caught)


def test_the_raised_pixel_cap_covers_the_largest_full_globe_map():
    # The cap is derived from the widths owm_envs.render.earth downsamples to,
    # so raising one of those cannot silently put a shipped map back over it.
    widest = max(MAP_WIDTHS.values())
    assert scene_module._MAX_MAP_PIXELS >= widest * (widest // 2)


def test_earth_is_a_full_globe_with_a_normal_map(scene):
    earth_mesh = scene._earth_surface_group.children[0]
    positions = np.asarray(earth_mesh.geometry.positions.data)
    # A cropped patch never reaches the far side of the planet.
    assert positions[:, 2].min() < -0.99 * scene.cfg.earth_radius_m
    assert earth_mesh.material.normal_map is not None
    np.testing.assert_allclose(earth_mesh.material.normal_scale, (2.5, 2.5))


def test_shipped_asset_matches_the_pinned_world_frame():
    # The station's world placement is a constant of the environment: the
    # collision hull and the dock poses are authored against it. Pinning the
    # shipped asset's world-frame extent means no asset swap, upright-rotation
    # change or offset edit can drift it silently. ISS_RECENTRE_OFFSET stays
    # frozen at the value measured on the retired ISS_stationary.glb -- moving
    # it would move the station under the hull and every dock pose -- so
    # ISS_base.glb's recentred vertex mean is pinned slightly off zero: the
    # PMM's vertices left the set while the offset did not follow.
    from owm_envs.render import asset_path
    from owm_envs.render.iss_scene import iss_vertices_world

    points = iss_vertices_world(asset_path("international-space-station", RenderConfig().iss_asset))
    assert points.shape == (329035, 3)
    np.testing.assert_allclose(points.mean(axis=0), [-0.004376, 0.135138, 0.085027], atol=1e-5)
    np.testing.assert_allclose(points.min(axis=0), [-55.761206, -25.293193, -31.652401], atol=1e-4)
    np.testing.assert_allclose(points.max(axis=0), [56.227223, 33.333488, 37.131761], atol=1e-4)
