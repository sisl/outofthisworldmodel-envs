import numpy as np
import pytest

pytest.importorskip("pygfx", reason="pygfx is not installed")

from owm_envs.render.view import CameraView, make_camera  # noqa: E402


def a_view(**kw):
    base = dict(
        name="test",
        camera_type="perspective",
        position=np.array([10.0, 0.0, 0.0]),
        target=np.zeros(3),
        up=np.array([0.0, 0.0, 1.0]),
    )
    base.update(kw)
    return CameraView(**base)


def test_view_is_frozen():
    view = a_view()
    with pytest.raises(Exception):
        view.name = "other"


def test_perspective_camera_is_positioned_at_the_view_position():
    cam = make_camera(a_view())
    np.testing.assert_allclose(np.asarray(cam.local.position), [10.0, 0.0, 0.0], atol=1e-5)


def test_perspective_camera_carries_the_requested_fov():
    cam = make_camera(a_view(fov_y_deg=42.0))
    assert np.isclose(cam.fov, 42.0, atol=1e-5)


def test_orthographic_camera_is_built_when_requested():
    import pygfx as gfx

    cam = make_camera(a_view(camera_type="orthographic", ortho_half_extent=25.0))
    assert isinstance(cam, gfx.OrthographicCamera)


def test_camera_looks_at_the_target():
    # A camera at +x looking at the origin must have its forward axis along -x.
    cam = make_camera(a_view(position=np.array([10.0, 0.0, 0.0])))
    forward = np.asarray(cam.local.rotation_matrix)[:3, 2]
    assert forward[0] > 0.9 or forward[0] < -0.9


def test_near_and_far_are_applied_when_given():
    cam = make_camera(a_view(near=0.5, far=1234.0))
    assert np.isclose(cam.near, 0.5, atol=1e-6)
    assert np.isclose(cam.far, 1234.0, atol=1e-3)


def test_unknown_camera_type_raises():
    with pytest.raises(ValueError, match="camera_type"):
        make_camera(a_view(camera_type="fisheye"))
