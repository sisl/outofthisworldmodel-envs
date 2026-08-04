"""Earth texture resolution: local full maps, downsampling, committed fallbacks.

Three-tier strategy:
1. A locally-downsampled full-resolution map exists -- return it. These are
   too large to commit, so they are gitignored and only present on a machine
   that has downsampled them.
2. A high-resolution source is present on disk -- downsample the full map
   from it (via `owm_envs.render.downsample`). When `allow_download` is set
   and the source is missing, fetch it from the mirror first; any failure
   warns and falls through.
3. The committed reasonable-resolution fallback map. Always present in a
   clone, so rendering works offline out of the box.
"""

from __future__ import annotations

import os
import urllib.request
import warnings
from pathlib import Path
from typing import Literal
from uuid import uuid4

from owm_envs.render import resources_dir

_EARTH_ASSET_BASE_URL = "https://s3.us-west-004.backblazeb2.com/outofthisworldmodel-iss"

TextureKind = Literal["color", "clouds", "bump"]

_SOURCE_NAMES = {
    "color": "EarthColorMap-80k.tif",
    "clouds": "Earth-40K-Clouds.tif",
    "bump": "Earth-40K-Bump.tif",
}

# Full-globe equirectangular downsample targets (width; height is width/2).
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
_MAP_MODES = {"color": "RGB", "clouds": "RGB", "bump": "L"}

# Committed fallbacks: the full maps above total ~46 MB, too much for git.
# These are the same globes at half (color) or a quarter (clouds, bump) the
# width -- ~7 MB in LFS, and enough for a recognisable Earth without any
# source on disk.
FALLBACK_WIDTHS = {"color": 8192, "clouds": 4096, "bump": 2048}
_FALLBACK_NAMES = {
    "color": "earth_color_fallback.jpg",
    "clouds": "earth_clouds_fallback.jpg",
    "bump": "earth_bump_fallback.png",
}


def _maps_dir() -> Path:
    return resources_dir() / "earth" / "maps"


def _source_dir() -> Path:
    return resources_dir() / "earth" / "sources"


def _downsample_map(kind: TextureKind) -> Path | None:
    """Tier 2: downsample a fresh map from a high-resolution source on disk, if present."""
    source_dir = _source_dir()
    source = source_dir / _SOURCE_NAMES[kind]
    if not source.exists():
        if source_dir.is_dir() and any(source_dir.iterdir()):
            # A source directory exists but nothing in it matches the expected
            # filename -- silently skipping tier 2 here would leave a
            # maintainer's dropped-in file never picked up, with no clue why.
            warnings.warn(
                f"{source_dir} has files but none named {_SOURCE_NAMES[kind]!r}; "
                f"tier-2 downsample for {kind!r} skipped"
            )
        return None

    from owm_envs.render.downsample import downsample_full_map

    output = _maps_dir() / _MAP_NAMES[kind]
    downsample_full_map(source, output, MAP_WIDTHS[kind], mode=_MAP_MODES[kind])
    return output


def _ensure_earth_source(name: str) -> Path | None:
    """Fetch a high-resolution Earth source. Returns None if unavailable.

    Never raises and never blocks rendering: the committed fallback maps are
    always there, and this is an optional quality upgrade.
    """
    dest = _source_dir() / name
    if dest.exists():
        return dest
    # Per-download name: parallel render workers each resolve their own
    # textures, and a shared temporary lets one worker's cleanup delete
    # another's live download.
    tmp = dest.with_suffix(f"{dest.suffix}.{uuid4().hex}.part")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        warnings.warn(f"fetching {name} from {_EARTH_ASSET_BASE_URL}; this is a large file")
        urllib.request.urlretrieve(f"{_EARTH_ASSET_BASE_URL}/{name}", tmp)
        os.replace(tmp, dest)  # atomic: a crash mid-download leaves no false complete file
        return dest
    except Exception as exc:  # HTTPError, URLError, OSError, cap exceeded...
        tmp.unlink(missing_ok=True)
        warnings.warn(f"could not fetch {name} ({exc}); using the fallback Earth map")
        return None


def earth_texture_path(kind: TextureKind, *, allow_download: bool = False) -> Path:
    """Resolve an Earth texture through the three-tier strategy.

    Never raises on a download failure -- it falls back to the committed
    fallback map, which exists in every clone.
    """
    if kind not in _SOURCE_NAMES:
        raise ValueError(f"unknown Earth texture kind: {kind!r}")

    full = _maps_dir() / _MAP_NAMES[kind]
    if full.exists():
        return full

    if allow_download and not (_source_dir() / _SOURCE_NAMES[kind]).exists():
        _ensure_earth_source(_SOURCE_NAMES[kind])  # warns and returns None on failure

    downsampled = _downsample_map(kind)
    if downsampled is not None:
        return downsampled

    fallback = _maps_dir() / _FALLBACK_NAMES[kind]
    if not fallback.exists():
        # The fallbacks are committed through git-lfs, so a clone without lfs
        # loses them; say so here rather than let an image decoder fail on a
        # pointer file several frames later.
        warnings.warn(f"{fallback} is missing; if this is a fresh clone, run `git lfs pull`")
    return fallback
