"""Bake a 50x50 degree Earth patch from a high-resolution equirectangular source.

Used by `owm_envs.render.earth` for tier-2 rebaking, and by
`scripts/bake_earth_patches.py` as a standalone CLI. The high-resolution
sources are large (gigabytes) and are not committed to this repository.

Requires the `render` extra (Pillow) -- import this module lazily from
anywhere that must work without it.
"""

from __future__ import annotations

from pathlib import Path

PATCH_SIZE = 8192
JPEG_QUALITY = 92

# This package's configured patch centre and angle -- the region of Earth
# visible from the ISS's simulated orbit, and the angular width/height of the
# crop taken around it.
DEFAULT_LON = -122.1697
DEFAULT_LAT = 37.4275
DEFAULT_ANGLE = 50.0


def bake_patch(source: Path, output: Path, lon: float, lat: float, angle_deg: float) -> None:
    from PIL import Image

    # Pillow refuses to open gigapixel images without this.
    Image.MAX_IMAGE_PIXELS = None

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
