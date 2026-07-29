"""Bake a 50x50 degree Earth patch from a high-resolution equirectangular source.

Standalone utility -- not part of the importable `owm_envs` package. Run it
whenever the patch centre or angle needs to change; the high-resolution
sources are large (gigabytes) and are not committed to this repository.

Usage:
    uv run --extra render python scripts/bake_earth_patches.py \\
        SOURCE.tif OUTPUT.jpg [--lon LON] [--lat LAT] [--angle ANGLE]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

# Pillow refuses to open gigapixel images without this.
Image.MAX_IMAGE_PIXELS = None

PATCH_SIZE = 8192
JPEG_QUALITY = 92

# Matches seamstress's configured patch centre and angle.
DEFAULT_LON = -122.1697
DEFAULT_LAT = 37.4275
DEFAULT_ANGLE = 50.0


def bake_patch(source: Path, output: Path, lon: float, lat: float, angle_deg: float) -> None:
    im = Image.open(source)
    w, h = im.size
    cx = int((lon + 180) / 360 * w)
    cy = int((90 - lat) / 180 * h)
    half_w = int(angle_deg / 360 * w / 2)
    half_h = int(angle_deg / 180 * h / 2)
    patch = im.crop((cx - half_w, cy - half_h, cx + half_w, cy + half_h))
    patch = patch.convert("RGB").resize((PATCH_SIZE, PATCH_SIZE), Image.LANCZOS)
    output.parent.mkdir(parents=True, exist_ok=True)
    patch.save(output, quality=JPEG_QUALITY)


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
