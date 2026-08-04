"""Earth texture resolution: baked full-globe maps, local re-baking, opt-in download.

Three-tier strategy:
1. A locally-baked or committed map exists -- return it. This is the
   default and needs no network.
2. A high-resolution source is present on disk -- bake from it (via
   `owm_envs.render.bake`).
3. If `allow_download` is set, try the mirror; on ANY failure fall back to
   tier 1.
"""

from __future__ import annotations

import urllib.request
import warnings
from pathlib import Path
from typing import Literal

from owm_envs.render import resources_dir

_EARTH_ASSET_BASE_URL = "https://s3.us-west-004.backblazeb2.com/outofthisworldmodel-iss"

TextureKind = Literal["color", "clouds", "bump"]

_SOURCE_NAMES = {
    "color": "EarthColorMap-80k.tif",
    "clouds": "Earth-40K-Clouds.tif",
    "bump": "Earth-40K-Bump.tif",
}

# Full-globe equirectangular bake targets (width; height is width/2).
# color/clouds at 16384 keep ~2.4 km/texel at the equator; bump stays at
# 8192 because relief is low-frequency and the normal map is computed
# from gradients, not displayed directly.
MAP_WIDTHS = {"color": 16384, "clouds": 16384, "bump": 8192}

# Bump is PNG, not JPEG: the normal map is computed from height
# GRADIENTS, which JPEG block artifacts corrupt far more than they
# corrupt a directly-displayed image.
_MAP_NAMES = {
    "color": "earth_color_full.jpg",
    "clouds": "earth_clouds_full.jpg",
    "bump": "earth_bump_full.png",
}
_BAKE_MODE = {"color": "RGB", "clouds": "RGB", "bump": "L"}


def _maps_dir() -> Path:
    return resources_dir() / "earth" / "maps"


def _source_dir() -> Path:
    return resources_dir() / "earth" / "sources"


def _bake_map(kind: TextureKind) -> Path | None:
    """Tier 2: bake a fresh map from a high-resolution source on disk, if present."""
    source_dir = _source_dir()
    source = source_dir / _SOURCE_NAMES[kind]
    if not source.exists():
        if source_dir.is_dir() and any(source_dir.iterdir()):
            # A source directory exists but nothing in it matches the expected
            # filename -- silently skipping tier 2 here would leave a
            # maintainer's dropped-in file never picked up, with no clue why.
            warnings.warn(
                f"{source_dir} has files but none named {_SOURCE_NAMES[kind]!r}; "
                f"tier-2 bake for {kind!r} skipped"
            )
        return None

    from owm_envs.render.bake import bake_full_map

    output = _maps_dir() / _MAP_NAMES[kind]
    bake_full_map(source, output, MAP_WIDTHS[kind], mode=_BAKE_MODE[kind])
    return output


def _ensure_earth_source(name: str) -> Path | None:
    """Fetch a high-resolution Earth source. Returns None if unavailable.

    Never raises and never blocks rendering: the committed baked maps are the
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
        warnings.warn(f"could not fetch {name} ({exc}); using the baked Earth map")
        return None


def earth_texture_path(kind: TextureKind, *, allow_download: bool = False) -> Path:
    """Resolve an Earth texture through the three-tier strategy.

    Never raises on a download failure -- it falls back to the committed
    baked map. The returned path is where the map belongs, which does not
    exist when nothing is committed and no source is on disk; callers that
    can render without a texture must check.
    """
    if kind not in _SOURCE_NAMES:
        raise ValueError(f"unknown Earth texture kind: {kind!r}")

    baked = _maps_dir() / _MAP_NAMES[kind]

    if allow_download:
        source = _ensure_earth_source(_SOURCE_NAMES[kind])
        if source is not None:
            rebaked = _bake_map(kind)
            if rebaked is not None:
                return rebaked
        return baked  # download unavailable (or bake failed) -- fall back to tier 1

    if baked.exists():
        return baked

    rebaked = _bake_map(kind)  # tier 2: no baked map, but a source may be on disk
    if rebaked is not None:
        return rebaked

    return baked
