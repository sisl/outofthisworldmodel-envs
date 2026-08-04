"""Earth full-map texture resolution: three tiers, artifact names, bake shapes."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from owm_envs.render import earth, resources_dir
from owm_envs.render.earth import _bake_map, _ensure_earth_source, earth_texture_path

KINDS = ["color", "clouds", "bump"]


@pytest.fixture
def fake_resources(tmp_path, monkeypatch):
    monkeypatch.setattr(earth, "resources_dir", lambda: tmp_path)
    monkeypatch.setattr(earth, "MAP_WIDTHS", {"color": 64, "clouds": 64, "bump": 32})
    (tmp_path / "earth" / "sources").mkdir(parents=True)
    (tmp_path / "earth" / "maps").mkdir(parents=True)
    return tmp_path


def _write_source(path: Path, mode: str, fmt: str | None = None) -> None:
    rng = np.random.default_rng(0)
    if mode == "L":
        arr = rng.integers(0, 255, (90, 180), dtype=np.uint8)
    else:
        arr = rng.integers(0, 255, (90, 180, 3), dtype=np.uint8)
    Image.fromarray(arr, mode=mode).save(path, format=fmt)


def _write_fallback(resources: Path, kind: str, size=(8, 4)) -> Path:
    path = resources / "earth" / "maps" / earth._FALLBACK_NAMES[kind]
    Image.new("L" if kind == "bump" else "RGB", size).save(path)
    return path


def _refuse_network(*args, **kwargs):
    raise AssertionError("resolution reached the network")


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
    assert earth.FALLBACK_WIDTHS == {"color": 8192, "clouds": 4096, "bump": 2048}
    kinds = set(earth._SOURCE_NAMES)
    assert set(earth.MAP_WIDTHS) == kinds
    assert set(earth._MAP_NAMES) == kinds
    assert set(earth._BAKE_MODE) == kinds
    assert set(earth.FALLBACK_WIDTHS) == kinds
    assert set(earth._FALLBACK_NAMES) == kinds


def test_full_map_wins_over_both_the_source_and_the_fallback(fake_resources, monkeypatch):
    monkeypatch.setattr("urllib.request.urlretrieve", _refuse_network)
    full = fake_resources / "earth" / "maps" / "earth_color_full.jpg"
    Image.new("RGB", (8, 4)).save(full)
    _write_source(fake_resources / "earth" / "sources" / "EarthColorMap-80k.tif", "RGB")
    _write_fallback(fake_resources, "color")

    assert earth_texture_path("color", allow_download=True) == full
    assert Image.open(full).size == (8, 4)  # untouched: no re-bake over a present map


def test_a_source_bakes_the_full_map_in_preference_to_the_fallback(fake_resources):
    _write_source(fake_resources / "earth" / "sources" / "EarthColorMap-80k.tif", "RGB")
    _write_fallback(fake_resources, "color")
    path = earth_texture_path("color")
    assert path.name == "earth_color_full.jpg"
    assert Image.open(path).size == (64, 32)


@pytest.mark.parametrize("kind", KINDS)
def test_no_full_map_and_no_source_resolves_to_the_fallback(fake_resources, kind):
    fallback = _write_fallback(fake_resources, kind)
    path = earth_texture_path(kind)
    assert path == fallback
    assert path.exists()


def test_missing_everything_returns_the_fallback_path(fake_resources):
    # No source, no full map, no fallback: returns the (nonexistent) fallback
    # path rather than raising, so a caller can render without the texture.
    path = earth_texture_path("clouds")
    assert path.name == "earth_clouds_fallback.jpg"
    assert not path.exists()


def test_unknown_texture_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        earth_texture_path("infrared")


def test_present_source_short_circuits_the_download(fake_resources, monkeypatch):
    # The default render path asks for a download; a machine that already has
    # the sources side-loaded must never touch the network.
    monkeypatch.setattr("urllib.request.urlretrieve", _refuse_network)
    _write_source(fake_resources / "earth" / "sources" / "EarthColorMap-80k.tif", "RGB")
    path = earth_texture_path("color", allow_download=True)
    assert path.name == "earth_color_full.jpg"
    assert Image.open(path).size == (64, 32)


def test_download_failure_falls_back_to_the_committed_fallback(fake_resources, monkeypatch):
    # The configured mirror currently returns HTTP 403 (account cap exceeded),
    # so this is the path that actually executes today. It must not raise.
    def boom(*args, **kwargs):
        raise OSError("simulated 403: cap exceeded")

    monkeypatch.setattr("urllib.request.urlretrieve", boom)
    fallback = _write_fallback(fake_resources, "color")
    with pytest.warns(UserWarning, match="fallback Earth map"):
        path = earth_texture_path("color", allow_download=True)
    assert path == fallback


def test_a_downloaded_source_is_baked_into_the_full_map(fake_resources, monkeypatch):
    def fetch(url, filename):
        # urlretrieve writes to a `.part` name, so the encoder is explicit here.
        assert url.endswith("EarthColorMap-80k.tif")
        _write_source(Path(filename), "RGB", fmt="TIFF")

    monkeypatch.setattr("urllib.request.urlretrieve", fetch)
    _write_fallback(fake_resources, "color")
    path = earth_texture_path("color", allow_download=True)
    assert path.name == "earth_color_full.jpg"
    assert Image.open(path).size == (64, 32)


@pytest.mark.parametrize("kind", KINDS)
def test_committed_fallbacks_resolve_with_no_sources_and_no_network(kind, tmp_path, monkeypatch):
    # Fresh-clone regression: neither the gitignored full maps nor the
    # multi-gigabyte sources are in the repository, so the committed fallbacks
    # alone have to satisfy every texture lookup -- offline.
    real_maps = resources_dir() / "earth" / "maps"
    maps = tmp_path / "earth" / "maps"
    maps.mkdir(parents=True)
    for name in earth._FALLBACK_NAMES.values():
        (maps / name).symlink_to(real_maps / name)
    monkeypatch.setattr(earth, "resources_dir", lambda: tmp_path)
    monkeypatch.setattr("urllib.request.urlretrieve", _refuse_network)

    assert earth_texture_path(kind).exists()

    def offline(*args, **kwargs):
        raise OSError("no route to host")

    monkeypatch.setattr("urllib.request.urlretrieve", offline)
    with pytest.warns(UserWarning):
        assert earth_texture_path(kind, allow_download=True).exists()


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
    # rather than silently falling through to the fallback map.
    (fake_resources / "earth" / "sources" / "some_other_file.tif").write_bytes(b"wrong name")
    with pytest.warns(UserWarning, match="tier-2 bake"):
        assert _bake_map("clouds") is None


def test_tier2_miss_is_silent_when_source_dir_is_empty(fake_resources, recwarn):
    # An empty (or absent) sources/ directory is the common case -- no
    # high-res source was ever provided, so there is nothing to warn about.
    assert _bake_map("clouds") is None
    assert len(recwarn) == 0
