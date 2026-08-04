"""Earth full-map texture resolution: three tiers, artifact names, bake shapes."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from owm_envs.render import earth
from owm_envs.render.earth import _bake_map, _ensure_earth_source, earth_texture_path


@pytest.fixture
def fake_resources(tmp_path, monkeypatch):
    monkeypatch.setattr(earth, "resources_dir", lambda: tmp_path)
    monkeypatch.setattr(earth, "MAP_WIDTHS", {"color": 64, "clouds": 64, "bump": 32})
    (tmp_path / "earth" / "sources").mkdir(parents=True)
    (tmp_path / "earth" / "maps").mkdir(parents=True)
    return tmp_path


def _write_source(path: Path, mode: str) -> None:
    rng = np.random.default_rng(0)
    if mode == "L":
        arr = rng.integers(0, 255, (90, 180), dtype=np.uint8)
    else:
        arr = rng.integers(0, 255, (90, 180, 3), dtype=np.uint8)
    Image.fromarray(arr, mode=mode).save(path)


def test_bump_bakes_grayscale_png_from_source(fake_resources):
    _write_source(fake_resources / "earth" / "sources" / "Earth-40K-Bump.tif", "L")
    path = earth_texture_path("bump")
    assert path.name == "earth_bump_full.png"
    assert path.exists()
    img = Image.open(path)
    assert img.mode == "L"
    assert img.size == (32, 16)  # full equirect, 2:1


def test_color_bakes_full_jpg(fake_resources):
    _write_source(fake_resources / "earth" / "sources" / "EarthColorMap-80k.tif", "RGB")
    path = earth_texture_path("color")
    assert path.name == "earth_color_full.jpg"
    assert Image.open(path).size == (64, 32)


@pytest.mark.parametrize(
    ("kind", "source_name", "map_name", "mode"),
    [
        ("color", "EarthColorMap-80k.tif", "earth_color_full.jpg", "RGB"),
        ("clouds", "Earth-40K-Clouds.tif", "earth_clouds_full.jpg", "RGB"),
        ("bump", "Earth-40K-Bump.tif", "earth_bump_full.png", "L"),
    ],
)
def test_every_kind_bakes_its_own_source_to_its_own_artifact(
    fake_resources, kind, source_name, map_name, mode
):
    # Pins the whole per-kind table: a source renamed in one dict and not the
    # others resolves to the wrong file, or bakes in the wrong colour mode.
    _write_source(fake_resources / "earth" / "sources" / source_name, mode)
    path = earth_texture_path(kind)
    assert path == fake_resources / "earth" / "maps" / map_name
    assert Image.open(path).mode == mode


def test_production_bake_targets_are_pinned():
    # Re-baking at a different size silently changes every rendered frame's
    # ground texel density, so the shipped widths are part of the contract.
    assert earth.MAP_WIDTHS == {"color": 16384, "clouds": 16384, "bump": 8192}
    kinds = set(earth._SOURCE_NAMES)
    assert set(earth.MAP_WIDTHS) == kinds
    assert set(earth._MAP_NAMES) == kinds
    assert set(earth._BAKE_MODE) == kinds


def test_committed_map_short_circuits_bake(fake_resources):
    baked = fake_resources / "earth" / "maps" / "earth_color_full.jpg"
    Image.new("RGB", (8, 4)).save(baked)
    assert earth_texture_path("color") == baked


def test_missing_everything_returns_baked_path(fake_resources):
    # No source, no baked map: returns the (nonexistent) baked path, no raise.
    path = earth_texture_path("clouds")
    assert path.name == "earth_clouds_full.jpg"
    assert not path.exists()


def test_unknown_texture_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        earth_texture_path("infrared")


def test_download_failure_falls_back_to_the_baked_map(fake_resources, monkeypatch):
    # The configured mirror currently returns HTTP 403 (account cap exceeded),
    # so this is the path that actually executes today. It must not raise.
    def boom(*args, **kwargs):
        raise OSError("simulated 403: cap exceeded")

    monkeypatch.setattr("urllib.request.urlretrieve", boom)
    with pytest.warns(UserWarning, match="baked Earth map"):
        path = earth_texture_path("color", allow_download=True)
    assert path.name == "earth_color_full.jpg"


def test_downloader_returns_none_on_failure_rather_than_raising(fake_resources, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr("urllib.request.urlretrieve", boom)
    with pytest.warns(UserWarning):
        assert _ensure_earth_source("EarthColorMap-80k.tif") is None


def test_downloader_leaves_no_part_file_behind(fake_resources, monkeypatch):
    # A truncated download must not leave a .part file that a later run mistakes
    # for real data.
    def boom(url, filename):
        Path(filename).write_bytes(b"partial")
        raise OSError("connection reset")

    monkeypatch.setattr("urllib.request.urlretrieve", boom)
    with pytest.warns(UserWarning):
        _ensure_earth_source("EarthColorMap-80k.tif")
    assert list((fake_resources / "earth" / "sources").glob("*.part")) == []


def test_tier2_miss_warns_when_source_dir_has_unmatched_files(fake_resources):
    # A source directory that exists but contains no matching file warns
    # rather than silently falling through to the baked map.
    (fake_resources / "earth" / "sources" / "some_other_file.tif").write_bytes(b"wrong name")
    with pytest.warns(UserWarning, match="tier-2 bake"):
        assert _bake_map("clouds") is None


def test_tier2_miss_is_silent_when_source_dir_is_empty(fake_resources, recwarn):
    # An empty (or absent) sources/ directory is the common case -- no
    # high-res source was ever provided, so there is nothing to warn about.
    assert _bake_map("clouds") is None
    assert len(recwarn) == 0
