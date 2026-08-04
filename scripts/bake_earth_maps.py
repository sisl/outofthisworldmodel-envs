"""CLI wrapper for baking the full-globe equirectangular Earth maps.

Run it whenever a bake width changes or a source is replaced; the
high-resolution sources are large (gigabytes) and are not committed to this
repository. It bakes every texture kind from `resources/earth/sources/` into
`resources/earth/maps/`.

Usage:
    uv run --extra render python scripts/bake_earth_maps.py
"""

from __future__ import annotations

from owm_envs.render.bake import bake_full_map
from owm_envs.render.earth import (
    _BAKE_MODE,
    _MAP_NAMES,
    _SOURCE_NAMES,
    MAP_WIDTHS,
    _maps_dir,
    _source_dir,
)


def main() -> None:
    for kind in sorted(_SOURCE_NAMES):
        source = _source_dir() / _SOURCE_NAMES[kind]
        if not source.exists():
            raise SystemExit(f"missing high-resolution source for {kind!r}: {source}")
        output = _maps_dir() / _MAP_NAMES[kind]
        print(f"baking {kind}: {source.name} -> {output.name} at {MAP_WIDTHS[kind]}px wide")
        bake_full_map(source, output, MAP_WIDTHS[kind], mode=_BAKE_MODE[kind])


if __name__ == "__main__":
    main()
