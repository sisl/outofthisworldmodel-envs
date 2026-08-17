#!/usr/bin/env python3
"""Print how many episodes a LeRobot split can actually be loaded for; else 0.

Stdlib only, so the generation and push scripts can call it per directory
without paying an environment resolve each time.

Two things have to hold, and neither implies the other:

`meta/episodes` has to exist. lerobot writes the parquet footers and the
episode metadata only in `finalize()`, so a split that was never finalized is
not a loadable dataset -- loading it finds no episode metadata and falls
through to the Hub. Neither `meta/info.json` nor `meta/stats.json` shows this:
both exist from early in the write and are updated as episodes are saved, so a
run killed mid-render leaves both behind looking healthy.

The count in `meta/info.json` has to match what the run asked for, which the
caller compares. Finalization alone does not give that: the writer finalizes
on its failure path too, deliberately, so a failed run can leave a loadable
split holding fewer episodes than were wanted.
"""

import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
if not (root / "meta" / "episodes").exists():
    print(0)
    raise SystemExit(0)
try:
    print(int(json.loads((root / "meta" / "info.json").read_text())["total_episodes"]))
except (OSError, ValueError, KeyError, TypeError):
    print(0)
