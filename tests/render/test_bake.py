"""The Earth map baker must live inside the installed package.

A `scripts/` path resolved relative to this repository's checkout layout
does not exist in a `pip install`ed copy, so loading `bake_full_map` that way
would make the download-then-bake path (tier 3 in earth.py) crash on
success. It must be an ordinary importable module under `src/`.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import owm_envs
from owm_envs.render.bake import bake_full_map


def _write_source(path, size, mode="RGB"):
    rng = np.random.default_rng(0)
    w, h = size
    shape = (h, w) if mode == "L" else (h, w, 3)
    Image.fromarray(rng.integers(0, 255, shape, dtype=np.uint8), mode=mode).save(path)


def test_bake_full_map_is_importable_from_the_package():
    assert callable(bake_full_map)


def test_bake_full_map_source_lives_under_src():
    # Guards against an importlib/spec_from_file_location trick that reaches
    # outside the installed package (e.g. into a scripts/ directory that a
    # wheel never ships).
    package_root = owm_envs.__file__
    assert "src/owm_envs" in package_root.replace("\\", "/") or "/owm_envs/" in package_root

    source_file = inspect.getsourcefile(bake_full_map)
    assert source_file is not None
    assert "owm_envs" in source_file.replace("\\", "/")
    assert "scripts" not in source_file.replace("\\", "/")


def test_bake_resamples_any_source_to_a_2to1_equirectangular_map(tmp_path):
    # Sources are not all exactly 2:1; the texture the sphere samples must be.
    source = tmp_path / "source.tif"
    _write_source(source, (101, 37))
    output = tmp_path / "nested" / "earth_color_full.jpg"
    bake_full_map(source, output, 64)
    assert Image.open(output).size == (64, 32)


def test_bake_writes_the_requested_mode(tmp_path):
    source = tmp_path / "source.tif"
    _write_source(source, (180, 90))
    output = tmp_path / "earth_bump_full.png"
    bake_full_map(source, output, 32, mode="L")
    assert Image.open(output).mode == "L"


def test_bake_encodes_by_extension_not_by_temp_name(tmp_path):
    # The atomic write goes through a temp path whose suffix Pillow cannot map
    # to a format, so the encoder has to be chosen from the final name.
    source = tmp_path / "source.tif"
    _write_source(source, (180, 90))
    jpg = tmp_path / "earth_color_full.jpg"
    png = tmp_path / "earth_bump_full.png"
    bake_full_map(source, jpg, 32)
    bake_full_map(source, png, 32, mode="L")
    assert Image.open(jpg).format == "JPEG"
    assert Image.open(png).format == "PNG"


def test_bake_leaves_no_temp_file_behind_on_success(tmp_path):
    source = tmp_path / "source.tif"
    _write_source(source, (180, 90))
    bake_full_map(source, tmp_path / "earth_color_full.jpg", 32)
    assert list(tmp_path.glob("*.part")) == []


def test_concurrent_bakes_do_not_share_a_temp_file(tmp_path, monkeypatch):
    # Two workers baking the same map must not write through one temp path:
    # whichever finishes first would replace the file the other is still
    # writing, and either one's cleanup would delete the other's data.
    source = tmp_path / "source.tif"
    _write_source(source, (180, 90))
    output = tmp_path / "earth_color_full.jpg"
    seen = []
    real_save = Image.Image.save

    def record(self, fp, *args, **kwargs):
        seen.append(Path(fp))
        return real_save(self, fp, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", record)
    bake_full_map(source, output, 32)
    bake_full_map(source, output, 32)

    assert len(seen) == 2
    assert seen[0] != seen[1]
    assert output not in seen


def test_bake_failure_leaves_neither_a_partial_nor_a_stale_map(tmp_path, monkeypatch):
    # Task 11 renders episodes in parallel workers; a half-written map picked up
    # by a sibling worker would corrupt frames instead of failing loudly.
    source = tmp_path / "source.tif"
    _write_source(source, (180, 90))
    output = tmp_path / "earth_color_full.jpg"

    def die(self, fp, *args, **kwargs):
        Path(fp).write_bytes(b"half an image")
        raise OSError("disk full")

    monkeypatch.setattr(Image.Image, "save", die)
    with pytest.raises(OSError, match="disk full"):
        bake_full_map(source, output, 32)
    assert not output.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_bake_does_not_clobber_an_existing_map_when_it_fails(tmp_path, monkeypatch):
    source = tmp_path / "source.tif"
    _write_source(source, (180, 90))
    output = tmp_path / "earth_color_full.jpg"
    bake_full_map(source, output, 32)
    good = output.read_bytes()

    def die(self, fp, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Image.Image, "save", die)
    with pytest.raises(OSError):
        bake_full_map(source, output, 64)
    assert output.read_bytes() == good
