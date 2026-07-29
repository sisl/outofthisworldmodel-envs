from pathlib import Path

import pytest

from owm_envs.render import asset_path, resources_dir

EXPECTED = [
    ("international-space-station", "ISS_stationary.glb"),
    ("earth", "patches", "earth_color_patch.jpg"),
    ("earth", "patches", "earth_clouds_patch.jpg"),
    ("moon", "moon_small.glb"),
    ("spacex-dragon-capsule", "spacex_dragon_2_exterior.glb"),
]
CUBEMAP_FACES = ["px", "nx", "py", "ny", "pz", "nz"]


def test_resources_dir_exists():
    assert resources_dir().is_dir()


@pytest.mark.parametrize("parts", EXPECTED)
def test_expected_assets_resolve(parts):
    assert asset_path(*parts).exists()


@pytest.mark.parametrize("face", CUBEMAP_FACES)
def test_starmap_cubemap_is_complete(face):
    # A missing face makes the skybox loader fail deep inside pygfx with a
    # confusing error, so check all six up front.
    assert asset_path("nasa_starmap_2020", f"{face}.png").exists()


@pytest.mark.parametrize("parts", EXPECTED)
def test_assets_are_real_files_not_unfetched_lfs_pointers(parts):
    # An unfetched LFS pointer is a ~130-byte text file starting with this line.
    # Without this check the failure surfaces as an unintelligible parse error.
    head = asset_path(*parts).open("rb").read(64)
    assert not head.startswith(b"version https://git-lfs"), (
        f"{parts} is an unfetched git-lfs pointer; run `git lfs pull`"
    )


def test_missing_asset_raises_a_helpful_error():
    with pytest.raises(FileNotFoundError, match="not found"):
        asset_path("earth", "does_not_exist.tif")
