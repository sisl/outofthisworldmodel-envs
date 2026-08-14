"""Earth full-map texture resolution: four tiers, artifact names, map shapes."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from owm_envs.render import asset_hub, earth, resources_dir
from owm_envs.render.earth import _downsample_map, earth_texture_path

KINDS = ["color", "clouds", "bump"]


@pytest.fixture
def fake_resources(tmp_path, monkeypatch):
    monkeypatch.setattr(earth, "resources_dir", lambda: tmp_path)
    # Tier 2 fetches through `asset_hub.download_asset`, which resolves its
    # own `earth_dir()` from its own `resources_dir` binding -- redirecting
    # only `earth`'s would leave a fetch landing under the real resources.
    monkeypatch.setattr(asset_hub, "resources_dir", lambda: tmp_path)
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


@pytest.fixture
def network_calls(monkeypatch):
    """Records attempted downloads (and fails them) so a test can assert none."""
    calls = []

    def record(**kwargs):
        calls.append(kwargs["filename"])
        raise OSError("network access is not allowed in tests")

    monkeypatch.setattr("owm_envs.render.asset_hub.hf_hub_download", record)
    return calls


def test_bump_downsamples_grayscale_png_from_source(fake_resources):
    _write_source(fake_resources / "earth" / "sources" / "Earth-40K-Bump.tif", "L")
    path = earth_texture_path("bump")
    assert path.name == "earth_bump_full.png"
    assert path.exists()
    img = Image.open(path)
    assert img.mode == "L"
    assert img.size == (32, 16)  # full equirect, 2:1


def test_color_downsamples_full_jpg(fake_resources):
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
def test_every_kind_downsamples_its_own_source_to_its_own_artifact(
    fake_resources, kind, source_name, map_name, mode
):
    # Pins the whole per-kind table: a source renamed in one dict and not the
    # others resolves to the wrong file, or writes the wrong colour mode.
    _write_source(fake_resources / "earth" / "sources" / source_name, mode)
    path = earth_texture_path(kind)
    assert path == fake_resources / "earth" / "maps" / map_name
    assert Image.open(path).mode == mode


def test_production_downsample_targets_are_pinned():
    # Re-downsampling at a different size silently changes every rendered frame's
    # ground texel density, so the shipped widths are part of the contract.
    assert earth.MAP_WIDTHS == {"color": 16384, "clouds": 16384, "bump": 8192}
    assert earth.FALLBACK_WIDTHS == {"color": 8192, "clouds": 4096, "bump": 2048}
    kinds = set(earth._SOURCE_NAMES)
    assert set(earth.MAP_WIDTHS) == kinds
    assert set(earth._MAP_NAMES) == kinds
    assert set(earth._MAP_MODES) == kinds
    assert set(earth.FALLBACK_WIDTHS) == kinds
    assert set(earth._FALLBACK_NAMES) == kinds


def test_full_map_wins_over_both_the_source_and_the_fallback(fake_resources, network_calls):
    full = fake_resources / "earth" / "maps" / "earth_color_full.jpg"
    Image.new("RGB", (8, 4)).save(full)
    _write_source(fake_resources / "earth" / "sources" / "EarthColorMap-80k.tif", "RGB")
    _write_fallback(fake_resources, "color")

    assert earth_texture_path("color", allow_download=True) == full
    assert Image.open(full).size == (8, 4)  # untouched: no re-downsample over a present map
    assert network_calls == []


def test_a_source_downsamples_the_full_map_in_preference_to_the_fallback(fake_resources):
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


def test_a_missing_fallback_warns_about_git_lfs(fake_resources):
    # The fallback is committed through git-lfs, so the realistic way to lose
    # it is a clone without lfs. Returning the path silently would surface as
    # an image-decoder error deep inside the texture loader instead.
    with pytest.warns(UserWarning, match="git lfs"):
        path = earth_texture_path("clouds")
    assert path.name == "earth_clouds_fallback.jpg"
    assert not path.exists()


def test_unknown_texture_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        earth_texture_path("infrared")


@pytest.mark.parametrize("kind", KINDS)
def test_relpaths_mirror_the_local_tree(kind):
    # The repo tree and the local tree are the same shape; if they drift, a
    # download lands somewhere tier 1 will never look.
    assert earth.map_relpath(kind) == f"maps/{earth._MAP_NAMES[kind]}"
    assert earth.source_relpath(kind) == f"sources/{earth._SOURCE_NAMES[kind]}"
    assert (resources_dir() / "earth" / earth.map_relpath(kind)).parent.name == "maps"


def test_texture_kinds_covers_the_whole_per_kind_table():
    assert set(earth.TEXTURE_KINDS) == set(earth._SOURCE_NAMES)


def test_download_failure_falls_back_to_the_committed_fallback(fake_resources, monkeypatch):
    # Tier 2 is an upgrade over a fallback that always exists, so an
    # unreachable Hub must warn and fall through rather than raise.
    def boom(**kwargs):
        raise OSError("simulated 503")

    monkeypatch.setattr("owm_envs.render.asset_hub.hf_hub_download", boom)
    fallback = _write_fallback(fake_resources, "color")
    with pytest.warns(UserWarning, match="could not fetch"):
        path = earth_texture_path("color", allow_download=True)
    assert path == fallback


def test_a_downloaded_map_is_returned_at_the_tier_one_path(fake_resources, monkeypatch):
    def fetch(*, repo_id, repo_type, filename, local_dir):
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 4)).save(dest)
        return str(dest)

    monkeypatch.setattr("owm_envs.render.asset_hub.hf_hub_download", fetch)
    _write_fallback(fake_resources, "color")
    path = earth_texture_path("color", allow_download=True)
    assert path == fake_resources / "earth" / "maps" / "earth_color_full.jpg"
    assert Image.open(path).size == (8, 4)


def test_the_hosted_map_is_preferred_over_downsampling_a_local_source(
    fake_resources, monkeypatch
):
    # 14 MB of finished JPEG beats 1.6 GB of TIFF plus minutes of resampling
    # for the same bytes. `owm-envs earth regenerate` is how a maintainer
    # forces the downsample of a source they just replaced.
    def fetch(*, repo_id, repo_type, filename, local_dir):
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 4)).save(dest)
        return str(dest)

    monkeypatch.setattr("owm_envs.render.asset_hub.hf_hub_download", fetch)
    _write_source(fake_resources / "earth" / "sources" / "EarthColorMap-80k.tif", "RGB")
    path = earth_texture_path("color", allow_download=True)
    assert Image.open(path).size == (8, 4)  # the fetched map, not a (64, 32) downsample


def test_a_source_still_downsamples_when_the_hub_is_unreachable(fake_resources, monkeypatch):
    # Tier 3 is what an offline maintainer machine falls back to.
    def boom(**kwargs):
        raise OSError("offline")

    monkeypatch.setattr("owm_envs.render.asset_hub.hf_hub_download", boom)
    _write_source(fake_resources / "earth" / "sources" / "EarthColorMap-80k.tif", "RGB")
    with pytest.warns(UserWarning, match="could not fetch"):
        path = earth_texture_path("color", allow_download=True)
    assert Image.open(path).size == (64, 32)


@pytest.mark.parametrize("kind", KINDS)
def test_committed_fallbacks_resolve_with_no_sources_and_no_network(
    kind, tmp_path, monkeypatch, network_calls
):
    # Fresh-clone regression: neither the gitignored full maps nor the
    # multi-gigabyte sources are in the repository, so the committed fallbacks
    # alone have to satisfy every texture lookup -- offline.
    real_maps = resources_dir() / "earth" / "maps"
    maps = tmp_path / "earth" / "maps"
    maps.mkdir(parents=True)
    for name in earth._FALLBACK_NAMES.values():
        (maps / name).symlink_to(real_maps / name)
    monkeypatch.setattr(earth, "resources_dir", lambda: tmp_path)

    assert earth_texture_path(kind).exists()
    assert network_calls == []

    with pytest.warns(UserWarning):
        assert earth_texture_path(kind, allow_download=True).exists()
    assert len(network_calls) == 1


def test_tier2_miss_warns_when_source_dir_has_unmatched_files(fake_resources):
    # A source directory that exists but contains no matching file warns
    # rather than silently falling through to the fallback map.
    (fake_resources / "earth" / "sources" / "some_other_file.tif").write_bytes(b"wrong name")
    with pytest.warns(UserWarning, match="tier-3 downsample"):
        assert _downsample_map("clouds") is None


def test_tier2_miss_is_silent_when_source_dir_is_empty(fake_resources, recwarn):
    # An empty (or absent) sources/ directory is the common case -- no
    # high-res source was ever provided, so there is nothing to warn about.
    assert _downsample_map("clouds") is None
    assert len(recwarn) == 0
