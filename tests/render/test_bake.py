"""The Earth patch baker must live inside the installed package.

`bake_patch` used to be loaded from a `scripts/` path resolved relative to
this repository's checkout layout, which does not exist in a `pip install`ed
copy: the download-then-bake path (tier 3 in earth.py) would crash on
success. It must now be an ordinary importable module under `src/`.
"""

from __future__ import annotations

import inspect

import owm_envs
from owm_envs.render.bake import DEFAULT_ANGLE, DEFAULT_LAT, DEFAULT_LON, bake_patch


def test_bake_patch_is_importable_from_the_package():
    assert callable(bake_patch)
    assert isinstance(DEFAULT_LON, float)
    assert isinstance(DEFAULT_LAT, float)
    assert isinstance(DEFAULT_ANGLE, float)


def test_bake_patch_source_lives_under_src():
    # Guards against a re-introduced importlib/spec_from_file_location trick
    # that reaches outside the installed package (e.g. into a scripts/
    # directory that a wheel never ships).
    package_root = owm_envs.__file__
    assert "src/owm_envs" in package_root.replace("\\", "/") or "/owm_envs/" in package_root

    source_file = inspect.getsourcefile(bake_patch)
    assert source_file is not None
    assert "owm_envs" in source_file.replace("\\", "/")
    assert "scripts" not in source_file.replace("\\", "/")
