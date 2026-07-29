"""CLI wrapper for baking a 50x50 degree Earth patch from a high-resolution
equirectangular source.

Run it whenever the patch centre or angle needs to change; the high-resolution
sources are large (gigabytes) and are not committed to this repository.

Usage:
    uv run --extra render python scripts/bake_earth_patches.py \\
        SOURCE.tif OUTPUT.jpg [--lon LON] [--lat LAT] [--angle ANGLE]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from owm_envs.render.bake import DEFAULT_ANGLE, DEFAULT_LAT, DEFAULT_LON, bake_patch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="high-resolution equirectangular source")
    parser.add_argument("output", type=Path, help="output JPEG path")
    parser.add_argument("--lon", type=float, default=DEFAULT_LON)
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT)
    parser.add_argument("--angle", type=float, default=DEFAULT_ANGLE)
    args = parser.parse_args()
    bake_patch(args.source, args.output, args.lon, args.lat, args.angle)


if __name__ == "__main__":
    main()
