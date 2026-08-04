"""Bake full-globe equirectangular Earth maps from high-resolution sources.

Used by `owm_envs.render.earth` for tier-2 baking, and by
`scripts/bake_earth_maps.py` as a standalone CLI. The high-resolution
sources are large (gigabytes) and are not committed to this repository.

Requires the `render` extra (Pillow) -- import this module lazily from
anywhere that must work without it.
"""

from __future__ import annotations

from pathlib import Path

JPEG_QUALITY = 92


def bake_full_map(source: Path, output: Path, max_width: int, mode: str = "RGB") -> None:
    from PIL import Image

    # Pillow refuses to open gigapixel images without this.
    Image.MAX_IMAGE_PIXELS = None

    im = Image.open(source)
    target = (int(max_width), int(max_width) // 2)
    im = im.convert(mode)
    if im.size != target:
        im = im.resize(target, Image.LANCZOS)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in (".jpg", ".jpeg"):
        im.save(output, quality=JPEG_QUALITY)
    else:
        im.save(output)
