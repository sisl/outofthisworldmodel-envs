#!/usr/bin/env python3
"""Print how many episodes a finished rollout's dataset holds; else 0.

Every caller publishes a LeRobot dataset, so a rollout that wrote no split is
not a candidate however cleanly it finished -- counting its episodes would let
a push mirror a dataset repo that contains no dataset.

Stdlib only, so the generation and push scripts can call it per directory
without paying an environment resolve each time.

`rollout.json` is the marker because `rollout` writes it last, after the
render and the dataset write, so it exists only for a run that got all the way
through. Nothing lerobot writes can serve instead: `meta/info.json` is created
with the dataset and its episode count climbs as episodes are saved,
`meta/stats.json` is rewritten alongside it, and `meta/episodes` is flushed
every ten episodes rather than at the end. Each of those exists, and looks
healthy, in a run killed halfway.

`dataset_episodes` has to be present, and that is the point of reading it
rather than just counting the episode list. A manifest written before this
ordering existed was written BEFORE the render, so it carries the full episode
list whether or not its run ever finished -- and an interrupted one of those
is on disk indistinguishable from a complete run by its episode list alone.
Only a run that reached the end writes the key, so its absence means the
directory predates the guarantee and cannot be trusted to hold what it claims.

When the key is present it is the split's own episode count, read back from
disk after the write, or null for a `--no-lerobot` run that wrote no split.
Null counts as nothing here for the reason above: the run finished, but not
with the thing these scripts exist to publish.

So the count is printed only when the key is present, is not null, and agrees
with the episode list. Everything else -- no manifest, no key, a null, or a
disagreement -- is 0.
"""

import json
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1]) / "rollout.json"
try:
    run = json.loads(manifest.read_text())
    episodes = len(run["episodes"])
    written = run["dataset_episodes"]  # KeyError dates a pre-marker manifest
    print(episodes if written == episodes else 0)
except (OSError, ValueError, KeyError, TypeError):
    print(0)
