import numpy as np
import pytest

pytest.importorskip("pygfx", reason="rendering is an optional extra")
pytest.importorskip("trimesh", reason="GLB loading needs trimesh")

from owm_envs.render.iss_scene import ISSScene, RenderConfig  # noqa: E402


def state_at(pos, quat=(1.0, 0.0, 0.0, 0.0)):
    return np.array([*pos, 0, 0, 0, *quat, 0, 0, 0], dtype=np.float32)


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


def test_render_config_round_trips_through_toml(tmp_path):
    cfg = RenderConfig(image_width=256, image_height=128)
    path = tmp_path / "render.toml"
    cfg.to_toml(path)
    assert RenderConfig.from_toml(path) == cfg


def test_no_reference_to_the_missing_bump_texture():
    # No bump/normal-map source exists for this asset set; the code path
    # must be gone entirely, not merely disabled. Checks every module in the
    # render package, not just this one -- a reference left in a sibling
    # module would otherwise go uncaught.
    import pkgutil

    import owm_envs.render as render_package
    import owm_envs.render.iss_scene as scene_module

    for module_info in pkgutil.walk_packages(render_package.__path__, prefix="owm_envs.render."):
        module = __import__(module_info.name, fromlist=["_"])
        if not hasattr(module, "__file__") or module.__file__ is None:
            continue
        source = open(module.__file__).read()
        assert "bump" not in source.lower(), f"{module_info.name} still references bump"

    assert "EarthColorMap-80k" not in open(scene_module.__file__).read()


def test_shipped_asset_matches_the_pinned_world_frame():
    # The station's world placement is a constant of the environment: the
    # collision hull and the dock poses are authored against it. Pinning the
    # shipped asset's world-frame extent (and that the recentred vertex mean
    # is zero, which is how ISS_RECENTRE_OFFSET was measured) means no asset
    # swap, upright-rotation change or offset edit can drift it silently.
    from owm_envs.render import asset_path
    from owm_envs.render.iss_scene import iss_vertices_world

    points = iss_vertices_world(asset_path("international-space-station", RenderConfig().iss_asset))
    assert points.shape == (325374, 3)
    np.testing.assert_allclose(points.mean(axis=0), [0.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(points.min(axis=0), [-55.761206, -25.293193, -31.652401], atol=1e-4)
    np.testing.assert_allclose(points.max(axis=0), [56.227223, 33.333488, 37.131761], atol=1e-4)
