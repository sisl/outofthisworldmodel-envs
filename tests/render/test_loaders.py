import numpy as np
import pytest

pytest.importorskip("pygfx", reason="pygfx is not installed")
pytest.importorskip("trimesh", reason="GLB loading needs trimesh")

from owm_envs.render import asset_path  # noqa: E402
from owm_envs.render.loaders import load_cubemap_from_faces, load_glb_scene  # noqa: E402


def test_loads_the_iss_mesh():
    group = load_glb_scene(asset_path("international-space-station", "ISS_base.glb"))
    assert len(group.children) > 0


def test_loads_the_dragon_mesh():
    group = load_glb_scene(asset_path("spacex-dragon-capsule", "spacex_dragon_2_exterior.glb"))
    assert len(group.children) > 0


def test_scale_is_applied():
    path = asset_path("spacex-dragon-capsule", "spacex_dragon_2_exterior.glb")
    small = load_glb_scene(path, scale=1.0)
    big = load_glb_scene(path, scale=4.0)
    assert np.asarray(big.local.scale)[0] > np.asarray(small.local.scale)[0]


def test_loads_the_starmap_cubemap():
    tex = load_cubemap_from_faces(asset_path("nasa_starmap_2020"))
    assert tex is not None


def test_missing_cubemap_face_raises_a_named_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="px"):
        load_cubemap_from_faces(tmp_path)


def test_missing_glb_raises(tmp_path):
    with pytest.raises(Exception):
        load_glb_scene(tmp_path / "nope.glb")
