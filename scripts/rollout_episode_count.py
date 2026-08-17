#!/usr/bin/env python3
"""Print how many episodes a finished rollout holds; 0 if it did not finish.

Stdlib only, so the generation and push scripts can call it per directory
without paying an environment resolve each time.

`rollout.json` is the marker because `rollout` writes it last, after the
render and the dataset write, so it exists only for a run that got all the
way through. Nothing lerobot writes can serve instead: `meta/info.json` is
created with the dataset and its episode count climbs as episodes are saved,
`meta/stats.json` is rewritten alongside it, and `meta/episodes` is flushed
every ten episodes rather than at the end. Each of those exists, and looks
healthy, in a run killed halfway.

The count still has to match what the caller asked for. A rollout that ends
short raises rather than writing a manifest, so in practice this is a
belt-and-braces check against a tree assembled by hand or by an older run.
"""

import json
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1]) / "rollout.json"
try:
    print(len(json.loads(manifest.read_text())["episodes"]))
except (OSError, ValueError, KeyError, TypeError):
    print(0)
