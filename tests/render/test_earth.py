from pathlib import Path

import pytest

from owm_envs.render.earth import _bake_patch, _ensure_earth_source, earth_texture_path


@pytest.mark.parametrize("kind", ["color", "clouds"])
def test_baked_patch_resolves_without_network(kind):
    # The default path must need no network and no high-res source on disk.
    path = earth_texture_path(kind)
    assert path.exists()


def test_download_failure_falls_back_to_the_baked_patch(monkeypatch, tmp_path):
    # The configured mirror currently returns HTTP 403 (account cap exceeded),
    # so this is the path that actually executes today. It must not raise.
    #
    # _source_dir is monkeypatched to an empty tmp_path so this test doesn't
    # depend on the real resources/earth/sources/ directory being empty: if a
    # maintainer ever successfully downloads a high-res source locally,
    # _ensure_earth_source would return early and the warning this test
    # expects would never fire.
    def boom(*args, **kwargs):
        raise OSError("simulated 403: cap exceeded")

    monkeypatch.setattr("urllib.request.urlretrieve", boom)
    monkeypatch.setattr("owm_envs.render.earth._source_dir", lambda: tmp_path)
    with pytest.warns(UserWarning, match="baked Earth patch"):
        path = earth_texture_path("color", allow_download=True)
    assert path.exists()


def test_downloader_returns_none_on_failure_rather_than_raising(monkeypatch, tmp_path):
    # See test_download_failure_falls_back_to_the_baked_patch above for why
    # _source_dir must be monkeypatched here too.
    def boom(*args, **kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr("urllib.request.urlretrieve", boom)
    monkeypatch.setattr("owm_envs.render.earth._source_dir", lambda: tmp_path)
    with pytest.warns(UserWarning):
        assert _ensure_earth_source("EarthColorMap-80k.tif") is None


def test_downloader_leaves_no_part_file_behind(monkeypatch, tmp_path):
    # A truncated download must not leave a .part file that a later run mistakes
    # for real data.
    def boom(url, filename):
        Path(filename).write_bytes(b"partial")
        raise OSError("connection reset")

    monkeypatch.setattr("urllib.request.urlretrieve", boom)
    monkeypatch.setattr("owm_envs.render.earth._source_dir", lambda: tmp_path)
    with pytest.warns(UserWarning):
        _ensure_earth_source("EarthColorMap-80k.tif")
    assert list(tmp_path.glob("*.part")) == []


def test_unknown_texture_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        earth_texture_path("infrared")


def test_tier2_miss_warns_when_source_dir_has_unmatched_files(monkeypatch, tmp_path):
    # A source directory that exists but contains no matching file warns
    # rather than silently falling through to the baked patch.
    (tmp_path / "some_other_file.tif").write_bytes(b"not the expected name")
    monkeypatch.setattr("owm_envs.render.earth._source_dir", lambda: tmp_path)
    with pytest.warns(UserWarning, match="tier-2 bake"):
        assert _bake_patch("clouds") is None


def test_tier2_miss_is_silent_when_source_dir_is_empty(monkeypatch, tmp_path, recwarn):
    # An empty (or absent) sources/ directory is the common case -- no
    # high-res source was ever provided, so there is nothing to warn about.
    monkeypatch.setattr("owm_envs.render.earth._source_dir", lambda: tmp_path)
    assert _bake_patch("clouds") is None
    assert len(recwarn) == 0
