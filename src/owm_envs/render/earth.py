"""Earth texture resolution: baked patches, local re-baking, opt-in download.

Three-tier strategy:
1. A locally-baked or committed patch exists -- return it. This is the
   default and needs no network.
2. A high-resolution source is present on disk -- bake from it (reusing
   scripts/bake_earth_patches.py's crop logic).
3. If `allow_download` is set, try the mirror; on ANY failure fall back to
   tier 1.
"""

from __future__ import annotations

import importlib.util
import urllib.request
import warnings
from pathlib import Path
from typing import Literal

from owm_envs.render import resources_dir

_EARTH_ASSET_BASE_URL = "https://s3.us-west-004.backblazeb2.com/outofthisworldmodel-iss"

TextureKind = Literal["color", "clouds", "bump"]

_SOURCE_NAMES = {
    "color": "EarthColorMap-80k.tif",
    "clouds": "EarthCloudMap-80k.tif",
    "bump": "EarthBumpMap-80k.tif",
}


def _patches_dir() -> Path:
    return resources_dir() / "earth" / "patches"


def _source_dir() -> Path:
    return resources_dir() / "earth" / "sources"


def _load_bake_patch():
    """Import `bake_patch` from the standalone scripts/bake_earth_patches.py utility."""
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "bake_earth_patches.py"
    spec = importlib.util.spec_from_file_location("_owm_bake_earth_patches", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.bake_patch, module.DEFAULT_LON, module.DEFAULT_LAT, module.DEFAULT_ANGLE


def _bake_patch(kind: TextureKind) -> Path | None:
    """Tier 2: bake a fresh patch from a high-resolution source on disk, if present."""
    source = _source_dir() / _SOURCE_NAMES[kind]
    if not source.exists():
        return None

    bake_patch, lon, lat, angle = _load_bake_patch()
    output = _patches_dir() / f"earth_{kind}_patch.jpg"
    bake_patch(source, output, lon, lat, angle)
    return output


def _ensure_earth_source(name: str) -> Path | None:
    """Fetch a high-resolution Earth source. Returns None if unavailable.

    Never raises and never blocks rendering: the committed baked patches are the
    default path, and this is an optional quality upgrade.
    """
    dest = _source_dir() / name
    if dest.exists():
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{_EARTH_ASSET_BASE_URL}/{name}", tmp)
        tmp.rename(dest)  # atomic: a crash mid-download leaves no false complete file
        return dest
    except Exception as exc:  # HTTPError, URLError, OSError, cap exceeded...
        tmp.unlink(missing_ok=True)
        warnings.warn(f"could not fetch {name} ({exc}); using the baked Earth patch")
        return None


def earth_texture_path(kind: TextureKind, *, allow_download: bool = False) -> Path:
    """Resolve an Earth texture through the three-tier strategy.

    Always returns a usable path -- download failures fall back to the
    committed baked patch rather than raising.
    """
    if kind not in _SOURCE_NAMES:
        raise ValueError(f"unknown Earth texture kind: {kind!r}")

    baked = _patches_dir() / f"earth_{kind}_patch.jpg"

    if allow_download:
        source = _ensure_earth_source(_SOURCE_NAMES[kind])
        if source is not None:
            rebaked = _bake_patch(kind)
            if rebaked is not None:
                return rebaked
        return baked  # download unavailable (or bake failed) -- fall back to tier 1

    if baked.exists():
        return baked

    rebaked = _bake_patch(kind)  # tier 2: no baked patch, but a source may be on disk
    if rebaked is not None:
        return rebaked

    return baked
