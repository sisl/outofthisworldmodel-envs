"""Earth texture resolution: hosted maps, local downsampling, committed fallbacks.

Four tiers, cheapest first:
1. A full-resolution map is already on disk -- return it. These are too large
   to commit, so they are gitignored and present only where one has been
   fetched or generated.
2. `allow_download` is set -- fetch the finished map from the asset dataset on
   the Hub (see `owm_envs.render.asset_hub`). ~48 MB for all three, landing
   at the tier-1 path. Any failure warns and falls through.
3. A high-resolution source is present on disk -- downsample the full map from
   it, via `owm_envs.render.downsample`. This is the offline maintainer path;
   `regenerate_map` forces it for a source that has just been replaced.
4. The committed reasonable-resolution fallback map. Always present in a
   clone, so rendering works offline out of the box.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

from owm_envs.render import resources_dir
from owm_envs.render.asset_hub import download_asset

TextureKind = Literal["color", "clouds", "bump"]

_SOURCE_NAMES = {
    "color": "EarthColorMap-80k.tif",
    "clouds": "Earth-40K-Clouds.tif",
    "bump": "Earth-40K-Bump.tif",
}

TEXTURE_KINDS: tuple[TextureKind, ...] = ("color", "clouds", "bump")

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


def _run_downsample(kind: TextureKind, source: Path) -> Path:
    from owm_envs.render.downsample import downsample_full_map

    output = _maps_dir() / _MAP_NAMES[kind]
    downsample_full_map(source, output, MAP_WIDTHS[kind], mode=_MAP_MODES[kind])
    return output


def _downsample_map(kind: TextureKind) -> Path | None:
    """Tier 3: downsample a fresh map from a high-resolution source on disk, if present."""
    source_dir = _source_dir()
    source = source_dir / _SOURCE_NAMES[kind]
    if not source.exists():
        if source_dir.is_dir() and any(source_dir.iterdir()):
            # A source directory exists but nothing in it matches the expected
            # filename -- silently skipping tier 3 here would leave a
            # maintainer's dropped-in file never picked up, with no clue why.
            warnings.warn(
                f"{source_dir} has files but none named {_SOURCE_NAMES[kind]!r}; "
                f"tier-3 downsample for {kind!r} skipped"
            )
        return None
    return _run_downsample(kind, source)


def regenerate_map(kind: TextureKind) -> Path:
    """Downsample the full map from the local source, replacing any existing map.

    Unlike tier 3 this is unconditional: it is how a maintainer who has just
    replaced a source gets a map from it, rather than the hosted one tier 2
    would otherwise serve.
    """
    source = _source_dir() / _SOURCE_NAMES[kind]
    if not source.exists():
        raise FileNotFoundError(
            f"{source} is missing; fetch it with `owm-envs earth pull-sources`"
        )
    return _run_downsample(kind, source)


def map_relpath(kind: TextureKind) -> str:
    """Repo-relative path of a full map in the asset dataset."""
    return f"maps/{_MAP_NAMES[kind]}"


def source_relpath(kind: TextureKind) -> str:
    """Repo-relative path of a high-resolution source in the asset dataset."""
    return f"sources/{_SOURCE_NAMES[kind]}"


def earth_texture_path(kind: TextureKind, *, allow_download: bool = False) -> Path:
    """Resolve an Earth texture through the four-tier strategy.

    Never raises on a download failure -- it falls back to the committed
    fallback map, which exists in every clone.
    """
    if kind not in _SOURCE_NAMES:
        raise ValueError(f"unknown Earth texture kind: {kind!r}")

    full = _maps_dir() / _MAP_NAMES[kind]
    if full.exists():
        return full

    if allow_download:
        downloaded = download_asset(map_relpath(kind))  # warns and returns None on failure
        if downloaded is not None:
            return downloaded

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
