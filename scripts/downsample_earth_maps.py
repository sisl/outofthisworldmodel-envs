"""CLI wrapper for downsampling the full-globe equirectangular Earth maps.

Run it whenever a target width changes or a source is replaced; the
high-resolution sources are large (gigabytes) and are not committed to this
repository. It downsamples every texture kind from `resources/earth/sources/`
into `resources/earth/maps/`, both the gitignored full-resolution maps and the
committed fallbacks.

Usage:
    uv run --extra render python scripts/downsample_earth_maps.py
"""

from __future__ import annotations

from owm_envs.render.downsample import downsample_full_map
from owm_envs.render.earth import (
    _FALLBACK_NAMES,
    _MAP_MODES,
    _MAP_NAMES,
    _SOURCE_NAMES,
    FALLBACK_WIDTHS,
    MAP_WIDTHS,
    _maps_dir,
    _source_dir,
)


def main() -> None:
    for kind in sorted(_SOURCE_NAMES):
        source = _source_dir() / _SOURCE_NAMES[kind]
        if not source.exists():
            raise SystemExit(f"missing high-resolution source for {kind!r}: {source}")
        for names, widths in ((_MAP_NAMES, MAP_WIDTHS), (_FALLBACK_NAMES, FALLBACK_WIDTHS)):
            output = _maps_dir() / names[kind]
            print(f"downsampling {kind}: {source.name} -> {output.name} at {widths[kind]}px wide")
            downsample_full_map(source, output, widths[kind], mode=_MAP_MODES[kind])


if __name__ == "__main__":
    main()
