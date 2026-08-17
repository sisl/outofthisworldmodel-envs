#!/usr/bin/env python3
"""Print how many episodes a LeRobot split holds; 0 if it holds none.

Stdlib only, so the generation and push scripts can call it per directory
without paying an environment resolve each time.

`meta/info.json` exists from the moment the dataset is created, before a
single episode is written, so its presence says nothing about whether a run
finished -- a killed run leaves one behind reading zero episodes. The count
inside it is what separates a finished split from an abandoned one, and it is
what the scripts compare against the episode count they asked for.
"""

import json
import pathlib
import sys

info = pathlib.Path(sys.argv[1]) / "meta" / "info.json"
try:
    print(int(json.loads(info.read_text())["total_episodes"]))
except (OSError, ValueError, KeyError, TypeError):
    print(0)
