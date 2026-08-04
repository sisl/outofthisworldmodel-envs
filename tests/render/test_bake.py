"""The Earth map baker must live inside the installed package.

A `scripts/` path resolved relative to this repository's checkout layout
does not exist in a `pip install`ed copy, so loading `bake_full_map` that way
would make the download-then-bake path (tier 3 in earth.py) crash on
success. It must be an ordinary importable module under `src/`.
"""

from __future__ import annotations

import inspect

import numpy as np
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
